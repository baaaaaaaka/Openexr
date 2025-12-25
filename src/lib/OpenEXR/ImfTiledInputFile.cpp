//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class TiledInputFile
//
//-----------------------------------------------------------------------------

#include "ImfTiledInputFile.h"

#include "Iex.h"

#include "IlmThreadPool.h"
#if ILMTHREAD_THREADING_ENABLED
#    include "IlmThreadProcessGroup.h"
#    include <mutex>
#endif

#include "ImfFrameBuffer.h"
#include "ImfInputPartData.h"

// TODO: remove once TiledOutput is converted
#include "ImfTileOffsets.h"
#include "ImfTiledMisc.h"

#include "openexr_decode.h"
#include "openexr_part.h"

// Global flag to enable non-temporal writes for tile decoding.
// This is useful for ML data loaders where decoded data is immediately
// transferred to GPU memory and won't be read again soon.
// Thread-safe: uses atomic for thread safety.
#include <atomic>
static std::atomic<bool> g_useNonTemporalWrites{false};

// Thread-local storage for X-direction cropping during tile decode.
// This allows the decoder to skip unnecessary pixels, reducing memory bandwidth.
struct XCropConfig {
    int cropXMin = -1;  // -1 means no crop (full tile)
    int cropXMax = -1;
    int tileWidth = 0;  // Full tile width
};
static thread_local XCropConfig g_xCropConfig;

// Thread-local storage for Y-direction cropping during tile decode.
// This allows the decoder to skip unnecessary lines, reducing memory bandwidth.
struct YCropConfig {
    int cropYMin = -1;  // -1 means no crop (full tile)
    int cropYMax = -1;
};
static thread_local YCropConfig g_yCropConfig;

namespace OPENEXR_IMF_INTERNAL_NAMESPACE {
    
IMF_EXPORT void setNonTemporalWrites(bool enable)
{
    g_useNonTemporalWrites.store(enable, std::memory_order_relaxed);
}

IMF_EXPORT bool nonTemporalWrites()
{
    return g_useNonTemporalWrites.load(std::memory_order_relaxed);
}

// Set X crop region for tile decoding (thread-local)
// cropXMin/cropXMax are absolute pixel coordinates
// tileWidth is the full tile width
IMF_EXPORT void setTileXCrop(int cropXMin, int cropXMax, int tileWidth)
{
    g_xCropConfig.cropXMin = cropXMin;
    g_xCropConfig.cropXMax = cropXMax;
    g_xCropConfig.tileWidth = tileWidth;
}

// Clear X crop (use full tile)
IMF_EXPORT void clearTileXCrop()
{
    g_xCropConfig.cropXMin = -1;
    g_xCropConfig.cropXMax = -1;
    g_xCropConfig.tileWidth = 0;
}

// Set Y crop region for tile decoding (thread-local)
// cropYMin/cropYMax are absolute pixel coordinates
IMF_EXPORT void setTileYCrop(int cropYMin, int cropYMax)
{
    g_yCropConfig.cropYMin = cropYMin;
    g_yCropConfig.cropYMax = cropYMax;
}

// Clear Y crop (use full tile)
IMF_EXPORT void clearTileYCrop()
{
    g_yCropConfig.cropYMin = -1;
    g_yCropConfig.cropYMax = -1;
}

} // namespace OPENEXR_IMF_INTERNAL_NAMESPACE

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cstring>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#else
#include <windows.h>
#endif

// I/O Merging support for Lustre/GPFS optimization
// When enabled, multiple tile reads are merged into fewer large I/O operations
// This variable is also referenced by ImfScanLineInputFile.cpp
std::atomic<bool> g_enableIOMerge{true};

// Thread-local prefetch buffer and mapping for merged I/O
struct PrefetchBuffer {
    std::vector<uint8_t> data;
    // Map from chunk data_offset to position in prefetch buffer
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> offsetMap;  // offset -> (buffer_pos, size)
    bool active = false;
};
static thread_local PrefetchBuffer g_prefetchBuffer;

// Custom read function that uses prefetched data - zero-copy version
// For COMPRESSED data: sets packed_buffer pointer to prefetch buffer
static exr_result_t
prefetched_read_chunk(exr_decode_pipeline_t* decode)
{
    if (!g_prefetchBuffer.active) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    uint64_t dataOffset = decode->chunk.data_offset;
    auto it = g_prefetchBuffer.offsetMap.find(dataOffset);
    if (it == g_prefetchBuffer.offsetMap.end()) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    size_t bufferPos = it->second.first;
    
    // Verify buffer position is valid
    size_t packedSize = decode->chunk.packed_size;
    if (bufferPos + packedSize > g_prefetchBuffer.data.size()) {
        return EXR_ERR_OUT_OF_MEMORY;
    }
    
    // ZERO-COPY approach: Point packed_buffer directly to prefetch buffer
    // Set packed_alloc_size = 0 to prevent OpenEXR from freeing it
    decode->packed_buffer = g_prefetchBuffer.data.data() + bufferPos;
    decode->packed_alloc_size = 0;  // CRITICAL: Prevents exr_decoding_destroy from freeing
    
    return EXR_ERR_SUCCESS;
}

// Helper function to swap bytes to native endian (little-endian on x86)
static inline void swap_to_native16(uint8_t* ptr, size_t count)
{
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    uint16_t* p = reinterpret_cast<uint16_t*>(ptr);
    for (size_t i = 0; i < count; ++i) {
        p[i] = ((p[i] & 0xFF) << 8) | ((p[i] >> 8) & 0xFF);
    }
#else
    (void)ptr; (void)count; // No-op on little-endian
#endif
}

static inline void swap_to_native32(uint8_t* ptr, size_t count)
{
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    uint32_t* p = reinterpret_cast<uint32_t*>(ptr);
    for (size_t i = 0; i < count; ++i) {
        p[i] = ((p[i] & 0xFF) << 24) | ((p[i] & 0xFF00) << 8) |
               ((p[i] >> 8) & 0xFF00) | ((p[i] >> 24) & 0xFF);
    }
#else
    (void)ptr; (void)count; // No-op on little-endian
#endif
}

// Custom read function for UNCOMPRESSED data from prefetch buffer
// This mimics read_uncompressed_direct but reads from prefetch buffer instead of file
// Also handles Y-crop (user_line_begin_skip, user_line_end_ignore) and 
// X-crop (user_pixel_begin_skip, user_pixel_end_ignore)
//
// Memory copy analysis:
// - For uncompressed data, this function necessarily copies from prefetch buffer to output.
// - This is ONE memcpy per scanline, trading I/O calls for memory bandwidth.
// - On high-latency filesystems (Lustre/GPFS), this tradeoff is beneficial:
//   * Without IOMerge: N tiles × 1 pread each = N I/O round trips
//   * With IOMerge: 1 merged pread + N memcpy = 1 I/O round trip + fast memory ops
// - The memcpy overhead is typically <1% of I/O latency on network filesystems.
static exr_result_t
prefetched_read_uncompressed_direct(exr_decode_pipeline_t* decode)
{
    if (!g_prefetchBuffer.active) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    uint64_t dataOffset = decode->chunk.data_offset;
    auto it = g_prefetchBuffer.offsetMap.find(dataOffset);
    if (it == g_prefetchBuffer.offsetMap.end()) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    size_t bufferPos = it->second.first;
    size_t bufferSize = it->second.second;
    
    // Pointer to the start of chunk data in prefetch buffer
    const uint8_t* srcData = g_prefetchBuffer.data.data() + bufferPos;
    size_t srcOffset = 0;
    
    int height = decode->chunk.height;
    int start_y = decode->chunk.start_y;
    
    // Y-crop parameters
    int line_begin_skip = decode->user_line_begin_skip;
    int line_end_ignore = decode->user_line_end_ignore;
    
    // X-crop parameters  
    int pixel_begin_skip = decode->user_pixel_begin_skip;
    int pixel_end_ignore = decode->user_pixel_end_ignore;
    
    for (int y = 0; y < height; ++y)
    {
        // Check if this line should be skipped (Y-crop)
        bool skipLine = (y < line_begin_skip) || (y >= height - line_end_ignore);
        
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = &decode->channels[c];
            
            if (decc->height == 0) continue;
            
            // Full scanline width in bytes (for source data)
            size_t fullLineBytes = (size_t)decc->width * (size_t)decc->bytes_per_element;
            
            if (decc->y_samples > 1)
            {
                if (((start_y + y) % decc->y_samples) != 0) continue;
            }
            
            // Verify we have enough data in prefetch buffer
            if (srcOffset + fullLineBytes > bufferSize) {
                return EXR_ERR_OUT_OF_MEMORY;
            }
            
            if (skipLine || !decc->decode_to_ptr)
            {
                // Skip this line's data
                srcOffset += fullLineBytes;
                continue;
            }
            
            // Calculate output pointer
            uint8_t* cdata = decc->decode_to_ptr;
            
            int output_y = y - line_begin_skip;
            if (decc->y_samples > 1)
            {
                cdata += ((size_t)(output_y / decc->y_samples) * (size_t)decc->user_line_stride);
            }
            else
            {
                cdata += (size_t)output_y * (size_t)decc->user_line_stride;
            }
            
            // Apply X-crop
            int out_width = decc->width - pixel_begin_skip - pixel_end_ignore;
            if (out_width <= 0)
            {
                srcOffset += fullLineBytes;
                continue;
            }
            
            size_t skipBytes = (size_t)pixel_begin_skip * (size_t)decc->bytes_per_element;
            size_t copyBytes = (size_t)out_width * (size_t)decc->bytes_per_element;
            
            // Copy cropped portion from prefetch buffer to output
            // This memcpy is necessary because:
            // 1. Output may have different stride than source
            // 2. X-crop requires reading from middle of source line
            // 3. Prefetch buffer is temporary and will be cleared after readTiles()
            memcpy(cdata, srcData + srcOffset + skipBytes, copyBytes);
            srcOffset += fullLineBytes;
            
            // Byte swap to native endian if needed (no-op on little-endian x86)
            if (decc->bytes_per_element == 2)
                swap_to_native16(cdata, out_width);
            else if (decc->bytes_per_element == 4)
                swap_to_native32(cdata, out_width);
        }
    }
    
    return EXR_ERR_SUCCESS;
}

// Helper: read multiple ranges from file using a single open
static bool 
readMultipleRangesFromFile(const char* filename, 
                           const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                           uint8_t* buffer)
{
#ifndef _WIN32
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return false;
    
    size_t bufferPos = 0;
    for (const auto& r : ranges) {
        ssize_t bytesRead = pread(fd, buffer + bufferPos, r.second, static_cast<off_t>(r.first));
        if (bytesRead != static_cast<ssize_t>(r.second)) {
            close(fd);
            return false;
        }
        bufferPos += r.second;
    }
    
    close(fd);
    return true;
#else
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, 
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return false;
    
    size_t bufferPos = 0;
    for (const auto& r : ranges) {
        OVERLAPPED overlapped = {};
        overlapped.Offset = static_cast<DWORD>(r.first);
        overlapped.OffsetHigh = static_cast<DWORD>(r.first >> 32);
        
        DWORD bytesRead = 0;
        BOOL success = ReadFile(hFile, buffer + bufferPos, static_cast<DWORD>(r.second), &bytesRead, &overlapped);
        if (!success || bytesRead != r.second) {
            CloseHandle(hFile);
            return false;
        }
        bufferPos += r.second;
    }
    
    CloseHandle(hFile);
    return true;
#endif
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

// Enable/disable I/O merging for tile reading
IMF_EXPORT void setIOMerge(bool enable)
{
    g_enableIOMerge.store(enable, std::memory_order_relaxed);
}

IMF_EXPORT bool isMergeEnabled()
{
    return g_enableIOMerge.load(std::memory_order_relaxed);
}

namespace {

struct TileProcess
{
    ~TileProcess ()
    {
        if (!first) {
            exr_decoding_destroy (decoder.context, &decoder);
        }
    }

    void run_decode (
        exr_const_context_t ctxt,
        int pn,
        const FrameBuffer *outfb,
        const std::vector<Slice> &filllist);

    void update_pointers (
        const FrameBuffer *outfb,
        int fb_absX, int fb_absY,
        int t_absX, int t_absY);

    void run_fill (
        const FrameBuffer *outfb,
        int fb_absX, int fb_absY,
        int t_absX, int t_absY,
        const std::vector<Slice> &filllist);

    bool                  first = true;
    exr_chunk_info_t      cinfo;
    exr_decode_pipeline_t decoder;

    TileProcess*          next;
};

#if ILMTHREAD_THREADING_ENABLED
using TileProcessGroup = ILMTHREAD_NAMESPACE::ProcessGroup<TileProcess>;
#endif

} // empty namespace

//
// struct TiledInputFile::Data stores things that will be
// needed between calls to readTile()
//

struct TiledInputFile::Data
{
    Data (Context *ctxt, int pN, int nT)
    : _ctxt (ctxt)
    , partNumber (pN)
    , numThreads (nT)
    {}

    void initialize ()
    {
        if (_ctxt->storage (partNumber) != EXR_STORAGE_TILED)
            throw IEX_NAMESPACE::ArgExc ("File part is not a tiled part");

        if (EXR_ERR_SUCCESS != exr_get_tile_descriptor (
                *_ctxt,
                partNumber,
                &tile_x_size,
                &tile_y_size,
                &tile_level_mode,
                &tile_round_mode))
            throw IEX_NAMESPACE::ArgExc ("Unable to query tile descriptor");

        if (EXR_ERR_SUCCESS != exr_get_tile_levels (
                *_ctxt,
                partNumber,
                &num_x_levels,
                &num_y_levels))
            throw IEX_NAMESPACE::ArgExc ("Unable to query number of tile levels");
    }

    void readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly);

    Context* _ctxt;
    int partNumber;
    int numThreads;
    Header header;
    bool header_filled = false;

    uint32_t tile_x_size = 0;
    uint32_t tile_y_size = 0;
    exr_tile_level_mode_t tile_level_mode = EXR_TILE_LAST_TYPE;
    exr_tile_round_mode_t tile_round_mode = EXR_TILE_ROUND_LAST_TYPE;

    int32_t num_x_levels = 0;
    int32_t num_y_levels = 0;

    // TODO: remove once we can remove deprecated API
    std::vector<char> _tile_data_scratch;

    FrameBuffer frameBuffer;
    std::vector<Slice> fill_list;

    std::vector<std::string> _failures;

#if ILMTHREAD_THREADING_ENABLED
    std::mutex _mx;

    class TileBufferTask final : public ILMTHREAD_NAMESPACE::Task
    {
    public:
        TileBufferTask (
            ILMTHREAD_NAMESPACE::TaskGroup* group,
            Data*                   ifd,
            TileProcessGroup*       tileg,
            const FrameBuffer*      outfb,
            const exr_chunk_info_t& cinfo)
            : Task (group)
            , _outfb (outfb)
            , _ifd (ifd)
            , _tile (tileg->pop ())
            , _tile_group (tileg)
        {
            _tile->cinfo = cinfo;
        }

        ~TileBufferTask () override
        {
            _tile_group->push (_tile);
        }

        void execute () override;

    private:
        void run_decode ();

        const FrameBuffer* _outfb;
        Data*              _ifd;

        TileProcess*       _tile;
        TileProcessGroup*  _tile_group;
    };
#endif
};

TiledInputFile::TiledInputFile (
    const char*               filename,
    const ContextInitializer& ctxtinit,
    int                       numThreads)
    : _ctxt (filename, ctxtinit, Context::read_mode_t{})
    , _data (std::make_shared<Data> (&_ctxt, 0, numThreads))
{
    _data->initialize ();
}

TiledInputFile::TiledInputFile (const char fileName[], int numThreads)
    : TiledInputFile (
        fileName,
        ContextInitializer ()
        .silentHeaderParse (true)
        .strictHeaderValidation (false),
        numThreads)
{
}

TiledInputFile::TiledInputFile (
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, int numThreads)
    : TiledInputFile (
        is.fileName (),
        ContextInitializer ()
        .silentHeaderParse (true)
        .strictHeaderValidation (false)
        .setInputStream (&is),
        numThreads)
{
}

TiledInputFile::TiledInputFile (InputPartData* part)
    : _ctxt (part->context),
      _data (std::make_shared<Data> (&_ctxt, part->partNumber, part->numThreads))
{
    _data->initialize ();
}

const char*
TiledInputFile::fileName () const
{
    return _ctxt.fileName ();
}

const Header&
TiledInputFile::header () const
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    if (!_data->header_filled)
    {
        _data->header = _ctxt.header (_data->partNumber);
        _data->header_filled = true;
    }
    return _data->header;
}

int
TiledInputFile::version () const
{
    return _ctxt.version ();
}

void
TiledInputFile::setFrameBuffer (const FrameBuffer& frameBuffer)
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    _data->fill_list.clear ();

    for (FrameBuffer::ConstIterator j = frameBuffer.begin ();
         j != frameBuffer.end ();
         ++j)
    {
        const exr_attr_chlist_entry_t* curc = _ctxt.findChannel (
            _data->partNumber, j.name ());

        if (!curc)
        {
            _data->fill_list.push_back (j.slice ());
            continue;
        }

        if (curc->x_sampling != j.slice ().xSampling ||
            curc->y_sampling != j.slice ().ySampling)
            THROW (
                IEX_NAMESPACE::ArgExc,
                "X and/or y subsampling factors "
                "of \""
                    << j.name ()
                    << "\" channel "
                       "of input file \""
                    << fileName ()
                    << "\" are "
                       "not compatible with the frame buffer's "
                       "subsampling factors.");
    }

    _data->frameBuffer = frameBuffer;
}

const FrameBuffer&
TiledInputFile::frameBuffer () const
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    return _data->frameBuffer;
}

bool
TiledInputFile::isComplete () const
{
    return _ctxt.chunkTableValid (_data->partNumber);
}

void
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    //
    // Read a range of tiles from the file into the framebuffer
    //

    try
    {
        if (!isValidLevel (lx, ly))
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Level coordinate "
                "(" << lx
                    << ", " << ly
                    << ") "
                       "is invalid.");

        if (dx1 > dx2) std::swap (dx1, dx2);
        if (dy1 > dy2) std::swap (dy1, dy2);

        _data->readTiles (dx1, dx2, dy1, dy2, lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc& e)
    {
        REPLACE_EXC (
            e,
            "Error reading pixel data from image "
            "file \""
                << fileName () << "\". " << e.what ());
        throw;
    }
}

void
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    readTiles (dx1, dx2, dy1, dy2, l, l);
}

void
TiledInputFile::readTile (int dx, int dy, int lx, int ly)
{
    readTiles (dx, dx, dy, dy, lx, ly);
}

void
TiledInputFile::readTile (int dx, int dy, int l)
{
    readTile (dx, dy, l, l);
}

void
TiledInputFile::rawTileData (
    int&         dx,
    int&         dy,
    int&         lx,
    int&         ly,
    const char*& pixelData,
    int&         pixelDataSize)
{
    exr_chunk_info_t cinfo;
    if (EXR_ERR_SUCCESS == exr_read_tile_chunk_info (
            _ctxt, _data->partNumber, dx, dy, lx, ly, &cinfo))
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (_data->_mx);
#endif
        _data->_tile_data_scratch.resize (cinfo.packed_size);
        pixelDataSize = static_cast<int> (cinfo.packed_size);
        if (EXR_ERR_SUCCESS !=
            exr_read_chunk (_ctxt, _data->partNumber, &cinfo,
                            _data->_tile_data_scratch.data ()))
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                    << fileName () << "\". Unable to read raw tile data of "
                    << pixelDataSize << " bytes.");
        }
        pixelData = _data->_tile_data_scratch.data ();
        dx = cinfo.start_x;
        dy = cinfo.start_y;
        lx = cinfo.level_x;
        ly = cinfo.level_y;
    }
    else
    {
        if (!isValidTile (dx, dy, lx, ly))
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                << fileName () << "\". "
                << "Tried to read a tile outside "
                "the image file's data window.");
        }
        else
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading chunk information for tile from image "
                "file \""
                << fileName () << "\". Unable to read raw tile offset information.");
        }
    }
}

unsigned int
TiledInputFile::tileXSize () const
{
    return _data->tile_x_size;
}

unsigned int
TiledInputFile::tileYSize () const
{
    return _data->tile_y_size;
}

LevelMode
TiledInputFile::levelMode () const
{
    return (LevelMode)_data->tile_level_mode;
}

LevelRoundingMode
TiledInputFile::levelRoundingMode () const
{
    return (LevelRoundingMode)_data->tile_round_mode;
}

int
TiledInputFile::numLevels () const
{
    if (levelMode () == RIPMAP_LEVELS)
        THROW (
            IEX_NAMESPACE::LogicExc,
            "Error calling numLevels() on image "
            "file \""
                << fileName ()
                << "\" "
                   "(numLevels() is not defined for files "
                   "with RIPMAP level mode).");

    return _data->num_x_levels;
}

int
TiledInputFile::numXLevels () const
{
    return _data->num_x_levels;
}

int
TiledInputFile::numYLevels () const
{
    return _data->num_y_levels;
}

bool
TiledInputFile::isValidLevel (int lx, int ly) const
{
    if (lx < 0 || ly < 0) return false;

    if (levelMode () == MIPMAP_LEVELS && lx != ly) return false;

    if (lx >= numXLevels () || ly >= numYLevels ()) return false;

    return true;
}

int
TiledInputFile::levelWidth (int lx) const
{
    int32_t levw = 0;
    if (EXR_ERR_SUCCESS != exr_get_level_sizes (
            _ctxt, _data->partNumber, lx, 0, &levw, nullptr))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error calling levelWidth() on image "
            "file \""
                << fileName () << "\".");
    }
    return levw;
}

int
TiledInputFile::levelHeight (int ly) const
{
    int32_t levh = 0;
    if (EXR_ERR_SUCCESS != exr_get_level_sizes (
            _ctxt, _data->partNumber, 0, ly, nullptr, &levh))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error calling levelWidth() on image "
            "file \""
                << fileName () << "\".");
    }
    return levh;
}

int
TiledInputFile::numXTiles (int lx) const
{
    int32_t countx = 0;
    if (EXR_ERR_SUCCESS != exr_get_tile_counts (
            _ctxt, _data->partNumber, lx, 0, &countx, nullptr))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error calling numXTiles() on image "
            "file \""
                << fileName () << "\".");
    }
    return countx;
}

int
TiledInputFile::numYTiles (int ly) const
{
    int32_t county = 0;
    if (EXR_ERR_SUCCESS != exr_get_tile_counts (
            _ctxt, _data->partNumber, 0, ly, nullptr, &county))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error calling numYTiles() on image "
            "file \""
                << fileName () << "\".");
    }
    return county;
}

IMATH_NAMESPACE::Box2i
TiledInputFile::dataWindowForLevel (int l) const
{
    return dataWindowForLevel (l, l);
}

IMATH_NAMESPACE::Box2i
TiledInputFile::dataWindowForLevel (int lx, int ly) const
{
    int32_t levw = 0, levh = 0;
    if (EXR_ERR_SUCCESS != exr_get_level_sizes (
            _ctxt, _data->partNumber, lx, ly, &levw, &levh))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error calling dataWindowForLevel() on image "
            "file \""
                << fileName () << "\".");
    }
    exr_attr_box2i_t dw = _ctxt.dataWindow (_data->partNumber);
    return IMATH_NAMESPACE::Box2i (
        IMATH_NAMESPACE::V2i (dw.min.x, dw.min.y),
        IMATH_NAMESPACE::V2i (dw.min.x + levw - 1, dw.min.y + levh - 1));
}

IMATH_NAMESPACE::Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int l) const
{
    return dataWindowForTile (dx, dy, l, l);
}

IMATH_NAMESPACE::Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    try
    {
        if (!isValidTile (dx, dy, lx, ly))
            throw IEX_NAMESPACE::ArgExc ("Arguments not in valid range.");

        //exr_attr_box2i_t dw = _ctxt.dataWindow (_data->partNumber);
        auto dw = dataWindowForLevel (lx, ly);

        int32_t tileSizeX, tileSizeY;
        if (EXR_ERR_SUCCESS !=
            exr_get_tile_sizes (_ctxt, _data->partNumber, lx, ly, &tileSizeX, &tileSizeY))
          throw IEX_NAMESPACE::ArgExc ("Unable to query the data window.");

        dw.min.x += dx * tileSizeX;
        dw.min.y += dy * tileSizeY;
        int limX = dw.min.x + tileSizeX - 1;
        int limY = dw.min.y + tileSizeY - 1;
        limX = std::min (limX, dw.max.x);
        limY = std::min (limY, dw.max.y);

        return IMATH_NAMESPACE::Box2i (
            IMATH_NAMESPACE::V2i (dw.min.x, dw.min.y),
            IMATH_NAMESPACE::V2i (limX, limY));
    }
    catch (IEX_NAMESPACE::BaseExc& e)
    {
        REPLACE_EXC (
            e,
            "Error calling dataWindowForTile() on image "
            "file \""
                << fileName () << "\". " << e.what ());
        throw;
    }
}

bool
TiledInputFile::isValidTile (int dx, int dy, int lx, int ly) const
{
    int32_t countx = 0, county = 0;
    if (EXR_ERR_SUCCESS == exr_get_tile_counts (
            _ctxt, _data->partNumber, lx, ly, &countx, &county))
    {
        // get tile counts will check lx, ly for us
        return ((dx < countx && dx >= 0) &&
                (dy < county && dy >= 0));
    }
    return false;
}

namespace
{
struct tilepos
{
    uint64_t filePos;
    int      dx;
    int      dy;
    int      lx;
    int      ly;
    bool     operator< (const tilepos& other) const
    {
        return filePos < other.filePos;
    }
};
} // namespace

void
TiledInputFile::tileOrder (int dx[], int dy[], int lx[], int ly[]) const
{
    // TODO: remove once TiledOutputFile copy is converted
    switch (_ctxt.lineOrder (_data->partNumber))
    {
        case EXR_LINEORDER_RANDOM_Y:
            // calc below outside the nest
            break;

        case EXR_LINEORDER_DECREASING_Y:
        {
            dx[0] = 0;
            dy[0] = numYTiles (0) - 1;
            lx[0] = 0;
            ly[0] = 0;
            return;
        }
        case EXR_LINEORDER_INCREASING_Y:
            dx[0] = 0;
            dy[0] = 0;
            lx[0] = 0;
            ly[0] = 0;
            return;

        case EXR_LINEORDER_LAST_TYPE: /* invalid but should never be here */
        default:
            throw IEX_NAMESPACE::ArgExc ("Unknown LineOrder.");
    }

    size_t numAllTiles = 0;
    int numX = numXLevels ();
    int numY = numYLevels ();

    switch (levelMode ())
    {
        case ONE_LEVEL:
        case MIPMAP_LEVELS:
            for (int i_l = 0; i_l < numY; ++i_l)
                numAllTiles += size_t (numXTiles (i_l)) * size_t (numYTiles (i_l));
            break;

        case RIPMAP_LEVELS:
            for (int i_ly = 0; i_ly < numY; ++i_ly)
                for (int i_lx = 0; i_lx < numX; ++i_lx)
                    numAllTiles += size_t (numXTiles (i_lx)) * size_t (numYTiles (i_ly));
            break;

        default:
            throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }

    std::vector<tilepos> table;

    table.resize (numAllTiles);
    size_t tIdx = 0;
    switch (levelMode ())
    {
        case ONE_LEVEL:
        case MIPMAP_LEVELS:
            for (int i_l = 0; i_l < numY; ++i_l)
            {
                int nY = numYTiles (i_l);
                int nX = numXTiles (i_l);

                for ( int y = 0; y < nY; ++y )
                    for ( int x = 0; x < nX; ++x )
                    {
                        exr_chunk_info_t cinfo;
                        if (EXR_ERR_SUCCESS == exr_read_tile_chunk_info (
                                _ctxt, _data->partNumber, x, y, i_l, i_l, &cinfo))
                        {
                            tilepos &tp = table[tIdx++];
                            tp.filePos = cinfo.data_offset;
                            tp.dx = x;
                            tp.dy = y;
                            tp.lx = i_l;
                            tp.ly = i_l;
                        }
                        else
                        {
                            throw IEX_NAMESPACE::ArgExc ("Unable to get tile offset.");
                        }
                    }
            }
            break;

        case RIPMAP_LEVELS:
            for (int i_ly = 0; i_ly < numY; ++i_ly)
            {
                int nY = numYTiles (i_ly);
                for (int i_lx = 0; i_lx < numX; ++i_lx)
                {
                    int nX = numXTiles (i_lx);
                    for ( int y = 0; y < nY; ++y )
                        for ( int x = 0; x < nX; ++x )
                        {
                            exr_chunk_info_t cinfo;
                            if (EXR_ERR_SUCCESS == exr_read_tile_chunk_info (
                                    _ctxt, _data->partNumber, x, y, i_lx, i_ly, &cinfo))
                            {
                                tilepos &tp = table[tIdx++];
                                tp.filePos = cinfo.data_offset;
                                tp.dx = x;
                                tp.dy = y;
                                tp.lx = i_lx;
                                tp.ly = i_ly;
                            }
                            else
                            {
                                throw IEX_NAMESPACE::ArgExc ("Unable to get tile offset.");
                            }
                        }
                }
            }
            break;

        default: throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }

    std::sort (table.begin(), table.end ());

    for (size_t i = 0; i < numAllTiles; ++i)
    {
        const auto& tp = table[i];
        dx[i] = tp.dx;
        dy[i] = tp.dy;
        lx[i] = tp.lx;
        ly[i] = tp.ly;
    }
}

// Helper: merge adjacent byte ranges for I/O optimization
static std::vector<std::pair<uint64_t, uint64_t>> 
mergeIOranges(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t gapThreshold = 4096)
{
    if (ranges.empty()) return {};
    
    // Sort by offset
    std::sort(ranges.begin(), ranges.end());
    
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    merged.push_back(ranges[0]);
    
    for (size_t i = 1; i < ranges.size(); ++i) {
        auto& last = merged.back();
        const auto& curr = ranges[i];
        
        uint64_t lastEnd = last.first + last.second;
        
        // Merge if overlapping or gap is small
        if (curr.first <= lastEnd + gapThreshold) {
            uint64_t newEnd = std::max(lastEnd, curr.first + curr.second);
            last.second = newEnd - last.first;
        } else {
            merged.push_back(curr);
        }
    }
    
    return merged;
}

void TiledInputFile::Data::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    // CRITICAL: Clear prefetch state at the start of each call
    // This prevents cross-call state corruption
    g_prefetchBuffer.active = false;
    g_prefetchBuffer.offsetMap.clear();
    // Force deallocation of prefetch buffer to avoid heap corruption
    std::vector<uint8_t>().swap(g_prefetchBuffer.data);
    
    int nTiles = dx2 - dx1 + 1;
    nTiles *= dy2 - dy1 + 1;

    exr_chunk_info_t      cinfo;
    
    // ==================== I/O MERGING OPTIMIZATION ====================
    // When enabled, we batch-read chunk leaders AND pixel data in merged I/O.
    // This reduces pread calls from ~113 to ~3 for a 10x10 tile region.
    //
    // Strategy:
    // 1. Get chunk offset table (already cached, no I/O)
    // 2. Batch-read all chunk leaders in one I/O (~2KB for 100 tiles)
    // 3. Parse leaders in-memory to get packed_size
    // 4. Batch-read all pixel data in merged I/O
    
    bool usePrefetch = g_enableIOMerge.load(std::memory_order_relaxed) && nTiles > 1;
    std::vector<exr_chunk_info_t> allChunks;
    
    if (usePrefetch)
    {
        const char* filename = _ctxt->fileName();
        if (!filename || filename[0] == '\0') {
            usePrefetch = false;
        }
        else
        {
            // Get chunk offset table (already in memory, no I/O)
            uint64_t* chunkTable = nullptr;
            int32_t chunkCount = 0;
            exr_result_t rv = exr_get_chunk_table(*_ctxt, partNumber, &chunkTable, &chunkCount);
            
            // Get data window to compute numTilesX
            exr_attr_box2i_t dw;
            if (EXR_ERR_SUCCESS != exr_get_data_window(*_ctxt, partNumber, &dw)) {
                usePrefetch = false;
            }
            else if (rv != EXR_ERR_SUCCESS || !chunkTable) {
                usePrefetch = false;
            }
            else
            {
                // Compute chunk indices and collect offsets for batch leader read.
                // IMPORTANT: Chunk table indexing depends on tile level mode:
                // - ONE_LEVEL / MIPMAP: chunkIdx = sum(prev_levels) + ty * numTilesX(level) + tx
                // - RIPMAP: chunkIdx accounts for (levelx, levely) ordering
                //
                // If chunkIdx or per-tile width/height is wrong, the decoder can write past the
                // user buffer (heap corruption -> free(): invalid next size).
                exr_tile_level_mode_t levelMode = EXR_TILE_LAST_TYPE;
                exr_tile_round_mode_t roundMode = EXR_TILE_ROUND_LAST_TYPE;
                uint32_t descTileW = 0, descTileH = 0;
                int32_t levelsx = 0, levelsy = 0;
                int32_t levelPixelW = 0, levelPixelH = 0;
                int32_t levelTileW = 0, levelTileH = 0;
                if (EXR_ERR_SUCCESS != exr_get_tile_descriptor(*_ctxt, partNumber, &descTileW, &descTileH, &levelMode, &roundMode) ||
                    EXR_ERR_SUCCESS != exr_get_tile_levels(*_ctxt, partNumber, &levelsx, &levelsy) ||
                    EXR_ERR_SUCCESS != exr_get_level_sizes(*_ctxt, partNumber, lx, ly, &levelPixelW, &levelPixelH) ||
                    EXR_ERR_SUCCESS != exr_get_tile_sizes(*_ctxt, partNumber, lx, ly, &levelTileW, &levelTileH))
                {
                    usePrefetch = false;
                }

                std::vector<int32_t> tileCountX;
                std::vector<int32_t> tileCountY;
                if (usePrefetch)
                {
                    if (levelMode == EXR_TILE_RIPMAP_LEVELS)
                    {
                        tileCountX.assign((size_t)std::max(0, levelsx), 0);
                        tileCountY.assign((size_t)std::max(0, levelsy), 0);
                        int32_t tmpx = 0, tmpy = 0;
                        for (int l = 0; l < levelsx; ++l)
                        {
                            if (EXR_ERR_SUCCESS != exr_get_tile_counts(*_ctxt, partNumber, l, 0, &tileCountX[(size_t)l], &tmpy))
                            {
                                usePrefetch = false;
                                break;
                            }
                        }
                        for (int l = 0; l < levelsy; ++l)
                        {
                            if (EXR_ERR_SUCCESS != exr_get_tile_counts(*_ctxt, partNumber, 0, l, &tmpx, &tileCountY[(size_t)l]))
                            {
                                usePrefetch = false;
                                break;
                            }
                        }
                    }
                    else
                    {
                        int32_t maxLevels = levelsx;
                        if (maxLevels <= 0) maxLevels = 1;
                        tileCountX.assign((size_t)maxLevels, 0);
                        tileCountY.assign((size_t)maxLevels, 0);
                        for (int l = 0; l < maxLevels; ++l)
                        {
                            if (EXR_ERR_SUCCESS != exr_get_tile_counts(*_ctxt, partNumber, l, l, &tileCountX[(size_t)l], &tileCountY[(size_t)l]))
                            {
                                usePrefetch = false;
                                break;
                            }
                        }
                    }
                }

                auto computeChunkIdx = [&](int tx, int ty) -> int32_t {
                    if (!usePrefetch) return -1;
                    if (tx < 0 || ty < 0 || lx < 0 || ly < 0) return -1;
                    if (levelMode == EXR_TILE_RIPMAP_LEVELS)
                    {
                        if (lx >= levelsx || ly >= levelsy) return -1;
                        int32_t numx = tileCountX[(size_t)lx];
                        int32_t numy = tileCountY[(size_t)ly];
                        if (tx >= numx || ty >= numy) return -1;
                        int64_t chunkoff = 0;
                        // Sum all previous Y levels
                        for (int ylv = 0; ylv < ly; ++ylv)
                        {
                            int32_t rowy = tileCountY[(size_t)ylv];
                            for (int xlv = 0; xlv < levelsx; ++xlv)
                                chunkoff += (int64_t)tileCountX[(size_t)xlv] * (int64_t)rowy;
                        }
                        // Sum previous X levels in this Y level
                        for (int xlv = 0; xlv < lx; ++xlv)
                            chunkoff += (int64_t)tileCountX[(size_t)xlv] * (int64_t)numy;
                        chunkoff += (int64_t)ty * (int64_t)numx + (int64_t)tx;
                        if (chunkoff < 0 || chunkoff > INT32_MAX) return -1;
                        return (int32_t)chunkoff;
                    }
                    // ONE_LEVEL or MIPMAP_LEVELS (both require lx == ly)
                    if (lx != ly) return -1;
                    if (lx < 0 || lx >= (int)tileCountX.size()) return -1;
                    int32_t numx = tileCountX[(size_t)lx];
                    int32_t numy = tileCountY[(size_t)lx];
                    if (tx >= numx || ty >= numy) return -1;
                    int64_t chunkoff = 0;
                    for (int l = 0; l < lx; ++l)
                        chunkoff += (int64_t)tileCountX[(size_t)l] * (int64_t)tileCountY[(size_t)l];
                    chunkoff += (int64_t)ty * (int64_t)numx + (int64_t)tx;
                    if (chunkoff < 0 || chunkoff > INT32_MAX) return -1;
                    return (int32_t)chunkoff;
                };

                int numTilesX = 0;
                if (usePrefetch)
                {
                    if (levelMode == EXR_TILE_RIPMAP_LEVELS)
                        numTilesX = (lx >= 0 && lx < (int)tileCountX.size()) ? tileCountX[(size_t)lx] : 0;
                    else
                        numTilesX = (lx >= 0 && lx < (int)tileCountX.size()) ? tileCountX[(size_t)lx] : 0;
                    if (numTilesX <= 0) usePrefetch = false;
                }
                struct TileLeaderInfo {
                    int tx, ty;
                    int32_t chunkIdx;
                    uint64_t leaderOffset;
                };
                std::vector<TileLeaderInfo> tileInfos;
                tileInfos.reserve(nTiles);
                
                uint64_t minLeaderOffset = UINT64_MAX;
                uint64_t maxLeaderOffset = 0;
                
                for (int ty = dy1; ty <= dy2; ++ty) {
                    for (int tx = dx1; tx <= dx2; ++tx) {
                        TileLeaderInfo ti;
                        ti.tx = tx;
                        ti.ty = ty;
                        ti.chunkIdx = computeChunkIdx(tx, ty);
                        
                        if (ti.chunkIdx >= 0 && ti.chunkIdx < chunkCount) {
                            ti.leaderOffset = chunkTable[ti.chunkIdx];
                            if (ti.leaderOffset > 0) {
                                minLeaderOffset = std::min(minLeaderOffset, ti.leaderOffset);
                                maxLeaderOffset = std::max(maxLeaderOffset, ti.leaderOffset);
                                tileInfos.push_back(ti);
                            }
                        }
                    }
                }
                
                if (tileInfos.empty()) {
                    usePrefetch = false;
                }
                else
                {
                    // Leader size: 20 bytes (5 int32) for single-part
                    size_t leaderSize = 20;  // tx, ty, lx, ly, packed_size
                    
                    // Get compression type for chunk info
                    exr_compression_t compType;
                    if (EXR_ERR_SUCCESS != exr_get_compression(*_ctxt, partNumber, &compType)) {
                        usePrefetch = false;
                    }
                    else
                    {
                        // Get max unpacked size per chunk
                        uint64_t maxUnpackedSize = 0;
                        exr_get_chunk_unpacked_size(*_ctxt, partNumber, &maxUnpackedSize);
                        
                        size_t maxPackedSizeEstimate = (size_t)maxUnpackedSize;
                        
                        // ========== ROW-BASED I/O MERGING ==========
                        // Each row's tiles are contiguous in the file (row-major order).
                        // Read per-row ranges to minimize bandwidth while reducing I/O count.
                        // For 10x10 crop: 10 pread calls instead of 207.
                        
                        // Build per-row ranges
                        struct RowRange {
                            int ty;
                            uint64_t startOffset;
                            uint64_t endOffset;
                            size_t bufferOffset;
                        };
                        std::vector<RowRange> rowRanges;
                        
                        int currentRow = -1;
                        for (const auto& ti : tileInfos) {
                            uint64_t tileEnd = ti.leaderOffset + leaderSize + maxPackedSizeEstimate;
                            if (ti.ty != currentRow) {
                                // Start new row
                                RowRange rr;
                                rr.ty = ti.ty;
                                rr.startOffset = ti.leaderOffset;
                                rr.endOffset = tileEnd;
                                rr.bufferOffset = 0;  // Will be set later
                                rowRanges.push_back(rr);
                                currentRow = ti.ty;
                            } else {
                                // Extend current row
                                rowRanges.back().endOffset = std::max(rowRanges.back().endOffset, tileEnd);
                            }
                        }
                        
                        // Calculate buffer offsets and total size
                        size_t totalBufferSize = 0;
                        for (auto& rr : rowRanges) {
                            rr.bufferOffset = totalBufferSize;
                            totalBufferSize += (rr.endOffset - rr.startOffset);
                        }
                        
                        // Allocate buffer
                        g_prefetchBuffer.data.resize(totalBufferSize);
                        
                        // Read each row range with one pread per row
                        bool readSuccess = true;
                        int fd = ::open(filename, O_RDONLY);
                        if (fd < 0) {
                            readSuccess = false;
                        } else {
                            for (size_t ri = 0; ri < rowRanges.size(); ri++) {
                                const auto& rr = rowRanges[ri];
                                size_t size = rr.endOffset - rr.startOffset;
                                if(rr.bufferOffset + size > g_prefetchBuffer.data.size()) {
                                    readSuccess = false;
                                    break;
                                }
                                ssize_t bytesRead = ::pread64(fd, 
                                    g_prefetchBuffer.data.data() + rr.bufferOffset, 
                                    size, rr.startOffset);
                                if (bytesRead != (ssize_t)size) {
                                    readSuccess = false;
                                    break;
                                }
                            }
                            ::close(fd);
                        }
                        
                        if (!readSuccess) {
                            usePrefetch = false;
                        }
                        else
                        {
                            // Parse leaders and build allChunks + offsetMap
                            allChunks.reserve(nTiles);
                            g_prefetchBuffer.offsetMap.clear();
                            g_prefetchBuffer.offsetMap.reserve(nTiles);
                            
                            // Map each tile to its row range
                            size_t rowIdx = 0;
                            int tilesInRow = 0;
                            int tilesPerRow = dx2 - dx1 + 1;
                            
                            for (const auto& ti : tileInfos) {
                                // Move to next row if needed
                                if (tilesInRow >= tilesPerRow && rowIdx + 1 < rowRanges.size()) {
                                    rowIdx++;
                                    tilesInRow = 0;
                                }
                                
                                const auto& rr = rowRanges[rowIdx];
                                size_t bufferPos = rr.bufferOffset + (ti.leaderOffset - rr.startOffset);
                                
                                // Parse leader
                                int32_t* leader = reinterpret_cast<int32_t*>(
                                    g_prefetchBuffer.data.data() + bufferPos);
                                int32_t ldr_tx = leader[0];
                                int32_t ldr_ty = leader[1];
                                int32_t ldr_lx = leader[2];
                                int32_t ldr_ly = leader[3];
                                int32_t packed_size = leader[4];
                                
                                if (ldr_tx != ti.tx || ldr_ty != ti.ty || 
                                    ldr_lx != lx || ldr_ly != ly || packed_size <= 0) {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }
                                
                                // Build chunk info
                                exr_chunk_info_t ci;
                                ci.idx = ti.chunkIdx;
                                ci.type = (uint8_t)EXR_STORAGE_TILED;
                                ci.compression = (uint8_t)compType;
                                ci.start_x = ti.tx;
                                ci.start_y = ti.ty;
                                // Compute actual chunk width/height for edge tiles at this level.
                                // levelPixelW/H are the level dimensions; tiles are indexed from 0.
                                int32_t cx0 = (int32_t)ti.tx * levelTileW;
                                int32_t cy0 = (int32_t)ti.ty * levelTileH;
                                int32_t cwidth = levelTileW;
                                int32_t cheight = levelTileH;
                                if (cx0 + cwidth > levelPixelW) cwidth = levelPixelW - cx0;
                                if (cy0 + cheight > levelPixelH) cheight = levelPixelH - cy0;
                                if (cwidth <= 0 || cheight <= 0)
                                {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }
                                ci.width = cwidth;
                                ci.height = cheight;
                                ci.level_x = (uint8_t)lx;
                                ci.level_y = (uint8_t)ly;
                                ci.packed_size = packed_size;
                                ci.unpacked_size = maxUnpackedSize;
                                ci.data_offset = ti.leaderOffset + leaderSize;
                                ci.sample_count_data_offset = 0;
                                ci.sample_count_table_size = 0;
                                
                                allChunks.push_back(ci);
                                
                                // Map data_offset to buffer position
                                size_t dataBufferPos = bufferPos + leaderSize;
                                
                                g_prefetchBuffer.offsetMap[ci.data_offset] = {dataBufferPos, (size_t)packed_size};
                                
                                tilesInRow++;
                            }
                            
                            if (usePrefetch && !allChunks.empty()) {
                                g_prefetchBuffer.active = true;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: if batch read failed, collect chunks the traditional way
    if (usePrefetch && allChunks.empty())
    {
        usePrefetch = false;
    }
    // ==================== END I/O MERGING ====================

    // When IOMerge prefetch is active, we MUST use single-threaded processing
    // because g_prefetchBuffer is thread_local and worker threads won't see it.
    // The I/O merging benefit (fewer pread calls) is still achieved; we just
    // decode tiles sequentially after the merged read.
    bool useMultiThreading = (nTiles > 1 && numThreads > 1 && !usePrefetch);
    
#if ILMTHREAD_THREADING_ENABLED
    if (useMultiThreading)
    {
        TileProcessGroup tpg (numThreads);

        {
            ILMTHREAD_NAMESPACE::TaskGroup tg;

            // Note: usePrefetch is false here due to the condition above
            for (int ty = dy1; ty <= dy2; ++ty)
            {
                for (int tx = dx1; tx <= dx2; ++tx)
                {
                    exr_result_t rv = exr_read_tile_chunk_info (
                        *_ctxt, partNumber, tx, ty, lx, ly, &cinfo);
                    if (EXR_ERR_INCOMPLETE_CHUNK_TABLE == rv)
                    {
                        THROW (
                            IEX_NAMESPACE::InputExc,
                            "Tile (" << tx << ", " << ty << ", " << lx << ", " << ly
                            << ") is missing.");
                    }
                    else if (EXR_ERR_SUCCESS != rv)
                        throw IEX_NAMESPACE::InputExc ("Unable to query tile information");

                    ILMTHREAD_NAMESPACE::ThreadPool::addGlobalTask (
                        new TileBufferTask (&tg, this, &tpg, &frameBuffer, cinfo) );
                }
            }
        }

        tpg.throw_on_failure ();
    }
    else
#endif
    {
        TileProcess tp;

        if (usePrefetch && !allChunks.empty())
        {
            // Reuse collected chunk info
            for (size_t i = 0; i < allChunks.size(); ++i)
            {
                const auto& chunk = allChunks[i];
                
                tp.cinfo = chunk;
                tp.run_decode (
                    *_ctxt,
                    partNumber,
                    &frameBuffer,
                    fill_list);
            }
        }
        else
        {
            for (int ty = dy1; ty <= dy2; ++ty)
            {
                for (int tx = dx1; tx <= dx2; ++tx)
                {
                    exr_result_t rv = exr_read_tile_chunk_info (
                        *_ctxt, partNumber, tx, ty, lx, ly, &cinfo);
                    if (EXR_ERR_INCOMPLETE_CHUNK_TABLE == rv)
                    {
                        THROW (
                            IEX_NAMESPACE::InputExc,
                            "Tile (" << tx << ", " << ty << ", " << lx << ", " << ly
                            << ") is missing.");
                    }
                    else if (EXR_ERR_SUCCESS != rv)
                        throw IEX_NAMESPACE::InputExc ("Unable to query tile information");

                    tp.cinfo = cinfo;
                    tp.run_decode (
                        *_ctxt,
                        partNumber,
                        &frameBuffer,
                        fill_list);
                }
            }
        }
    }
    
    // Clean up prefetch state - must happen AFTER all TileProcess objects are destroyed
    // to ensure packed_buffer pointers are no longer used
    if (g_prefetchBuffer.active) {
        g_prefetchBuffer.active = false;
        g_prefetchBuffer.offsetMap.clear();
    }
}

////////////////////////////////////////

#if ILMTHREAD_THREADING_ENABLED
void TiledInputFile::Data::TileBufferTask::execute ()
{
    try
    {
        _tile->run_decode (
            *(_ifd->_ctxt),
            _ifd->partNumber,
            _outfb,
            _ifd->fill_list);
    }
    catch (std::exception &e)
    {
        _tile_group->record_failure (e.what ());
    }
    catch (...)
    {
        _tile_group->record_failure ("Unknown exception");
    }
}
#endif

////////////////////////////////////////

void TileProcess::run_decode (
    exr_const_context_t ctxt,
    int pn,
    const FrameBuffer *outfb,
    const std::vector<Slice> &filllist)
{
    int absX, absY, tileX, tileY;
    exr_attr_box2i_t dw;
    uint16_t prevFlags = 0;
    
    // stash the flag off to make sure to clean up in the event
    // of an exception by changing the flag after init...
    bool isfirst = first;
    if (first)
    {
        if (EXR_ERR_SUCCESS !=
            exr_decoding_initialize (ctxt, pn, &cinfo, &decoder))
        {
            throw IEX_NAMESPACE::IoExc ("Unable to initialize decode pipeline");
        }

        first = false;
    }
    else
    {
        if (EXR_ERR_SUCCESS !=
            exr_decoding_update (ctxt, pn, &cinfo, &decoder))
        {
            throw IEX_NAMESPACE::IoExc ("Unable to update decode pipeline");
        }
        prevFlags = decoder.decode_flags;
    }

    // Apply non-temporal writes flag if enabled globally
    if (g_useNonTemporalWrites.load(std::memory_order_relaxed))
        decoder.decode_flags |= EXR_DECODE_NON_TEMPORAL_WRITES;
    else
        decoder.decode_flags &= ~EXR_DECODE_NON_TEMPORAL_WRITES;

    if (EXR_ERR_SUCCESS != exr_get_data_window (ctxt, pn, &dw))
        throw IEX_NAMESPACE::ArgExc ("Unable to query the data window.");

    if (EXR_ERR_SUCCESS != exr_get_tile_sizes (
            ctxt, pn, cinfo.level_x, cinfo.level_y, &tileX, &tileY))
        throw IEX_NAMESPACE::ArgExc ("Unable to query the data window.");

    absX = dw.min.x + tileX * cinfo.start_x;
    absY = dw.min.y + tileY * cinfo.start_y;

    update_pointers (outfb, dw.min.x, dw.min.y, absX, absY);

    // Re-choose routines if flags changed (for non-temporal writes support)
    if (isfirst || prevFlags != decoder.decode_flags)
    {
        if (EXR_ERR_SUCCESS !=
            exr_decoding_choose_default_routines (ctxt, pn, &decoder))
        {
            throw IEX_NAMESPACE::IoExc ("Unable to choose decoder routines");
        }
    }

    // If prefetch buffer is active, use custom read function
    if (g_prefetchBuffer.active)
    {
        if (decoder.decompress_fn != nullptr)
        {
            // Compressed data: use prefetched_read_chunk which sets packed_buffer
            // The decompress_fn and unpack_and_convert_fn will handle the rest
            decoder.read_fn = &prefetched_read_chunk;
        }
        else
        {
            // Uncompressed data: use prefetched_read_uncompressed_direct
            // which reads directly from prefetch buffer to output channels
            decoder.read_fn = &prefetched_read_uncompressed_direct;
        }
    }

    if (EXR_ERR_SUCCESS != exr_decoding_run (ctxt, pn, &decoder))
        throw IEX_NAMESPACE::IoExc ("Unable to run decoder");

    run_fill (outfb, dw.min.x, dw.min.y, absX, absY, filllist);
}

////////////////////////////////////////

void TileProcess::update_pointers (const FrameBuffer *outfb, int fb_absX, int fb_absY, int t_absX, int t_absY)
{
    decoder.user_line_begin_skip = 0;
    decoder.user_line_end_ignore = 0;
    decoder.user_pixel_begin_skip = 0;
    decoder.user_pixel_end_ignore = 0;
    
    // Apply X crop if configured
    // t_absX is the absolute X position of the tile's left edge
    // cinfo.width is the tile width
    int tileXMin = t_absX;
    int tileXMax = t_absX + cinfo.width - 1;
    
    if (g_xCropConfig.cropXMin >= 0 && g_xCropConfig.cropXMax >= 0)
    {
        // Calculate how much to skip at the beginning and end of each line
        if (g_xCropConfig.cropXMin > tileXMin)
        {
            decoder.user_pixel_begin_skip = g_xCropConfig.cropXMin - tileXMin;
        }
        if (g_xCropConfig.cropXMax < tileXMax)
        {
            decoder.user_pixel_end_ignore = tileXMax - g_xCropConfig.cropXMax;
        }
    }
    
    // Apply Y crop if configured
    // t_absY is the absolute Y position of the tile's top edge
    // cinfo.height is the tile height
    int tileYMin = t_absY;
    int tileYMax = t_absY + cinfo.height - 1;
    
    if (g_yCropConfig.cropYMin >= 0 && g_yCropConfig.cropYMax >= 0)
    {
        // Calculate how many lines to skip at the beginning and end
        if (g_yCropConfig.cropYMin > tileYMin)
        {
            decoder.user_line_begin_skip = g_yCropConfig.cropYMin - tileYMin;
        }
        if (g_yCropConfig.cropYMax < tileYMax)
        {
            decoder.user_line_end_ignore = tileYMax - g_yCropConfig.cropYMax;
        }
    }

    for (int c = 0; c < decoder.channel_count; ++c)
    {
        exr_coding_channel_info_t& curchan = decoder.channels[c];
        uint8_t*                   ptr;
        const Slice*               fbslice;

        fbslice = outfb->findSlice (curchan.channel_name);

        if (curchan.height == 0 || !fbslice)
        {
            curchan.decode_to_ptr     = NULL;
            curchan.user_pixel_stride = 0;
            curchan.user_line_stride  = 0;
            continue;
        }

        if (fbslice->xSampling != 1 || fbslice->ySampling != 1)
            throw IEX_NAMESPACE::ArgExc ("Tiled data should not have subsampling.");

        int xOffset = fbslice->xTileCoords ? 0 : t_absX;
        int yOffset = fbslice->yTileCoords ? 0 : t_absY;
        
        // Adjust xOffset for X crop - the output pointer should point to where 
        // the first non-skipped pixel should go
        if (decoder.user_pixel_begin_skip > 0 && !fbslice->xTileCoords)
        {
            // When using X crop, the output buffer expects data to start at the crop position
            // So we add the skip amount to align the output correctly
            xOffset += decoder.user_pixel_begin_skip;
        }
        
        // Adjust yOffset for Y crop - the output pointer should point to where
        // the first non-skipped line should go
        if (decoder.user_line_begin_skip > 0 && !fbslice->yTileCoords)
        {
            // When using Y crop, the output buffer expects data to start at the crop position
            // So we add the skip amount to align the output correctly
            yOffset += decoder.user_line_begin_skip;
        }

        curchan.user_bytes_per_element = (fbslice->type == HALF) ? 2 : 4;
        curchan.user_data_type         = (exr_pixel_type_t)fbslice->type;
        curchan.user_pixel_stride      = fbslice->xStride;
        curchan.user_line_stride       = fbslice->yStride;

        ptr  = reinterpret_cast<uint8_t*> (fbslice->base);
        ptr += int64_t (xOffset) * int64_t (fbslice->xStride);
        ptr += int64_t (yOffset) * int64_t (fbslice->yStride);

        curchan.decode_to_ptr = ptr;
    }
}

////////////////////////////////////////

void TileProcess::run_fill (
    const FrameBuffer *outfb, int fb_absX, int fb_absY, int t_absX, int t_absY,
    const std::vector<Slice> &filllist)
{
    for (auto& s: filllist)
    {
        uint8_t* ptr;

        if (s.xSampling != 1 || s.ySampling != 1)
            throw IEX_NAMESPACE::ArgExc ("Tiled data should not have subsampling.");

        int xOffset = s.xTileCoords ? 0 : t_absX;
        int yOffset = s.yTileCoords ? 0 : t_absY;

        ptr  = reinterpret_cast<uint8_t*> (s.base);
        ptr += int64_t (xOffset) * int64_t (s.xStride);
        ptr += int64_t (yOffset) * int64_t (s.yStride);

        // TODO: update ImfMisc, lift fill type / value
        for ( int start = 0; start < cinfo.height; ++start )
        {
            if (start % s.ySampling) continue;

            uint8_t* outptr = ptr;
            for ( int sx = 0; sx < cinfo.width; ++sx )
            {
                if (sx % s.xSampling) continue;

                switch (s.type)
                {
                    case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
                    {
                        unsigned int fillVal = (unsigned int) (s.fillValue);
                        *(unsigned int*)outptr = fillVal;
                        break;
                    }

                    case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:
                    {
                        half fillVal = half (s.fillValue);
                        *(half*)outptr = fillVal;
                        break;
                    }

                    case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:
                    {
                        float fillVal = float (s.fillValue);
                        *(float*)outptr = fillVal;
                        break;
                    }
                    default:
                        throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
                }
                outptr += s.xStride;
            }

            ptr += s.yStride;
        }
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

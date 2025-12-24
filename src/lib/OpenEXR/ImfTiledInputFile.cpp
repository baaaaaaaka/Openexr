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
std::atomic<bool> g_enableIOMerge{false};

// Thread-local prefetch buffer and mapping for merged I/O
struct PrefetchBuffer {
    std::vector<uint8_t> data;
    // Map from chunk data_offset to position in prefetch buffer
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> offsetMap;  // offset -> (buffer_pos, size)
    bool active = false;
};
static thread_local PrefetchBuffer g_prefetchBuffer;

// Custom read function that uses prefetched data - zero-copy version
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
    
    // Zero-copy: point directly to prefetch buffer
    // The packed_buffer is only read by the decompressor, not modified
    // We mark alloc_size as 0 to prevent the decoder from freeing it
    if (decode->packed_buffer && decode->packed_alloc_size > 0 && decode->free_fn) {
        // Free any previously allocated buffer
        decode->free_fn(EXR_TRANSCODE_BUFFER_PACKED, decode->packed_buffer);
    }
    
    decode->packed_buffer = g_prefetchBuffer.data.data() + bufferPos;
    decode->packed_alloc_size = 0;  // Mark as not owned - prevents free
    
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
        if (!first)
            exr_decoding_destroy (decoder.context, &decoder);
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
    int nTiles = dx2 - dx1 + 1;
    nTiles *= dy2 - dy1 + 1;

    exr_chunk_info_t      cinfo;
    
    // ==================== I/O MERGING OPTIMIZATION ====================
    // If I/O merging is enabled and we have multiple tiles, prefetch all data
    // in merged I/O operations to reduce the number of system calls.
    // Also collect chunk info once to avoid redundant queries.
    bool usePrefetch = g_enableIOMerge.load(std::memory_order_relaxed) && nTiles > 1;
    std::vector<exr_chunk_info_t> allChunks;  // Reused if prefetch enabled
    
    if (usePrefetch)
    {
        // Phase 1: Collect all chunk info (will be reused in decode loop)
        allChunks.reserve(nTiles);
        std::vector<std::pair<uint64_t, uint64_t>> ioRanges;
        ioRanges.reserve(nTiles);
        
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
                
                allChunks.push_back(cinfo);
                ioRanges.push_back({cinfo.data_offset, cinfo.packed_size});
            }
        }
        
        // Phase 2: Merge adjacent ranges and calculate total size
        auto mergedRanges = mergeIOranges(ioRanges, 4096);
        
        size_t totalSize = 0;
        for (const auto& r : mergedRanges) {
            totalSize += r.second;
        }
        
        g_prefetchBuffer.data.resize(totalSize);
        g_prefetchBuffer.offsetMap.clear();
        g_prefetchBuffer.offsetMap.reserve(nTiles);
        
        // Phase 3: Read merged ranges (single file open)
        const char* filename = _ctxt->fileName();
        if (!filename || filename[0] == '\0') {
            usePrefetch = false;
            allChunks.clear();  // Will need to re-query
        }
        else
        {
            if (!readMultipleRangesFromFile(filename, mergedRanges, g_prefetchBuffer.data.data())) {
                throw IEX_NAMESPACE::IoExc ("Failed to prefetch tile data");
            }
            
            // Build offset map with single pass
            std::vector<std::pair<uint64_t, size_t>> rangeStartToBufferPos;
            rangeStartToBufferPos.reserve(mergedRanges.size());
            size_t bufferPos = 0;
            for (const auto& r : mergedRanges) {
                rangeStartToBufferPos.push_back({r.first, bufferPos});
                bufferPos += r.second;
            }
            
            for (const auto& chunk : allChunks) {
                auto it = std::upper_bound(rangeStartToBufferPos.begin(), rangeStartToBufferPos.end(),
                                          std::make_pair(chunk.data_offset, SIZE_MAX));
                if (it != rangeStartToBufferPos.begin()) {
                    --it;
                    g_prefetchBuffer.offsetMap[chunk.data_offset] = {
                        it->second + (chunk.data_offset - it->first), 
                        chunk.packed_size
                    };
                }
            }
            
            g_prefetchBuffer.active = true;
        }
    }
    // ==================== END I/O MERGING ====================

#if ILMTHREAD_THREADING_ENABLED
    if (nTiles > 1 && numThreads > 1)
    {
        TileProcessGroup tpg (numThreads);

        {
            ILMTHREAD_NAMESPACE::TaskGroup tg;

            if (usePrefetch && !allChunks.empty())
            {
                // Reuse collected chunk info
                for (const auto& chunk : allChunks)
                {
                    ILMTHREAD_NAMESPACE::ThreadPool::addGlobalTask (
                        new TileBufferTask (&tg, this, &tpg, &frameBuffer, chunk) );
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

                        ILMTHREAD_NAMESPACE::ThreadPool::addGlobalTask (
                            new TileBufferTask (&tg, this, &tpg, &frameBuffer, cinfo) );
                    }
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
            for (const auto& chunk : allChunks)
            {
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
    
    // Clean up prefetch state (keep buffer capacity for reuse)
    if (g_prefetchBuffer.active) {
        g_prefetchBuffer.active = false;
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
        decoder.read_fn = &prefetched_read_chunk;
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

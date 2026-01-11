//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class ScanLineInputFile
//
//-----------------------------------------------------------------------------

#include "ImfScanLineInputFile.h"

#include "Iex.h"

#include "IlmThreadPool.h"
#if ILMTHREAD_THREADING_ENABLED
#    include "IlmThreadProcessGroup.h"
#    include <mutex>
#endif

#include "ImfFrameBuffer.h"
#include "ImfInputPartData.h"

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <atomic>

#include "openexr_part.h"

// Match OpenEXRCore's internal sampling math (internal_util.h)
static inline int openexr_compute_sampled_height (int height, int y_sampling, int start_y)
{
    int nlines;
    if (y_sampling <= 1) return height;
    if (height == 1)
        nlines = (start_y % y_sampling) == 0 ? 1 : 0;
    else
    {
        int start, end;
        start = start_y % y_sampling;
        if (start != 0)
            start = start_y + (y_sampling - start);
        else
            start = start_y;

        end = start_y + height - 1;
        end -= (end < 0 ? -end : end) % y_sampling;

        if (start > end)
            nlines = start == start_y ? 1 : 0;
        else
            nlines = (end - start) / y_sampling + 1;
    }
    return nlines;
}

static inline int openexr_compute_sampled_width (int width, int x_sampling, int start_x)
{
    (void)start_x; // see OpenEXRCore: scanline x sampling assumes aligned start_x
    if (x_sampling <= 1) return width;
    return (width == 1) ? 1 : (width / x_sampling);
}

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

// Reference to global I/O merge flag and mode (defined in ImfTiledInputFile.cpp)
extern std::atomic<bool> g_enableIOMerge;
enum class IOMergeMode { ROW, SINGLE };
extern IOMergeMode g_iomergeMode;

// Thread-local prefetch buffer for scanline I/O merging
struct ScanLinePrefetchBuffer {
    std::vector<uint8_t> data;
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> offsetMap;  // offset -> (buffer_pos, size)
    bool active = false;
};
static thread_local ScanLinePrefetchBuffer g_scanlinePrefetchBuffer;

// Custom read function that uses prefetched scanline data - zero-copy version
// For COMPRESSED data: sets packed_buffer pointer to prefetch buffer
static exr_result_t
scanline_prefetched_read_chunk(exr_decode_pipeline_t* decode)
{
    if (!g_scanlinePrefetchBuffer.active) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    uint64_t dataOffset = decode->chunk.data_offset;
    auto it = g_scanlinePrefetchBuffer.offsetMap.find(dataOffset);
    if (it == g_scanlinePrefetchBuffer.offsetMap.end()) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    size_t bufferPos = it->second.first;
    size_t packed_size = decode->chunk.packed_size;
    
    // Validate buffer bounds
    if (bufferPos + packed_size > g_scanlinePrefetchBuffer.data.size()) {
        return EXR_ERR_OUT_OF_MEMORY;
    }
    
    // Zero-copy: point directly to prefetch buffer
    // Free any previously allocated buffer
    if (decode->packed_buffer && decode->packed_alloc_size > 0) {
        if (decode->free_fn)
            decode->free_fn(EXR_TRANSCODE_BUFFER_PACKED, decode->packed_buffer);
    }
    
    decode->packed_buffer = g_scanlinePrefetchBuffer.data.data() + bufferPos;
    decode->packed_alloc_size = 0;  // Mark as not owned - prevents free
    
    return EXR_ERR_SUCCESS;
}

// Helper function to swap bytes to native endian
static inline void scanline_swap_to_native16(uint8_t* ptr, size_t count)
{
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    uint16_t* p = reinterpret_cast<uint16_t*>(ptr);
    for (size_t i = 0; i < count; ++i) {
        p[i] = ((p[i] & 0xFF) << 8) | ((p[i] >> 8) & 0xFF);
    }
#else
    (void)ptr; (void)count;
#endif
}

static inline void scanline_swap_to_native32(uint8_t* ptr, size_t count)
{
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    uint32_t* p = reinterpret_cast<uint32_t*>(ptr);
    for (size_t i = 0; i < count; ++i) {
        p[i] = ((p[i] & 0xFF) << 24) | ((p[i] & 0xFF00) << 8) |
               ((p[i] >> 8) & 0xFF00) | ((p[i] >> 24) & 0xFF);
    }
#else
    (void)ptr; (void)count;
#endif
}

// Custom read function for UNCOMPRESSED scanline data from prefetch buffer
// Reads directly from prefetch buffer to output channels
static exr_result_t
scanline_prefetched_read_uncompressed_direct(exr_decode_pipeline_t* decode)
{
    if (!g_scanlinePrefetchBuffer.active) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    uint64_t dataOffset = decode->chunk.data_offset;
    auto it = g_scanlinePrefetchBuffer.offsetMap.find(dataOffset);
    if (it == g_scanlinePrefetchBuffer.offsetMap.end()) {
        return EXR_ERR_INVALID_ARGUMENT;
    }
    
    size_t bufferPos = it->second.first;
    size_t bufferSize = it->second.second;
    
    const uint8_t* srcData = g_scanlinePrefetchBuffer.data.data() + bufferPos;
    size_t srcOffset = 0;
    
    int height = decode->chunk.height;
    int start_y = decode->chunk.start_y;
    
    for (int y = 0; y < height; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = &decode->channels[c];
            
            if (decc->height == 0) continue;
            
            size_t toread = (size_t)decc->width * (size_t)decc->bytes_per_element;
            
            uint8_t* cdata = decc->decode_to_ptr;
            if (!cdata) {
                srcOffset += toread;
                continue;
            }
            
            if (decc->y_samples > 1)
            {
                if (((start_y + y) % decc->y_samples) != 0) continue;
                cdata += ((size_t)(y / decc->y_samples) * (size_t)decc->user_line_stride);
            }
            else
            {
                cdata += (size_t)y * (size_t)decc->user_line_stride;
            }
            
            if (srcOffset + toread > bufferSize) {
                return EXR_ERR_OUT_OF_MEMORY;
            }
            
            memcpy(cdata, srcData + srcOffset, toread);
            srcOffset += toread;
            
            if (decc->bytes_per_element == 2)
                scanline_swap_to_native16(cdata, decc->width);
            else if (decc->bytes_per_element == 4)
                scanline_swap_to_native32(cdata, decc->width);
        }
    }
    
    return EXR_ERR_SUCCESS;
}

// Helper: read multiple ranges from file using a single open
static bool 
scanlineReadMultipleRanges(const char* filename, 
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

// Helper: merge adjacent byte ranges
static std::vector<std::pair<uint64_t, uint64_t>> 
mergeScanlineIOranges(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t gapThreshold = 4096)
{
    if (ranges.empty()) return {};
    
    std::sort(ranges.begin(), ranges.end());
    
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    merged.push_back(ranges[0]);
    
    for (size_t i = 1; i < ranges.size(); ++i) {
        auto& last = merged.back();
        const auto& curr = ranges[i];
        
        uint64_t lastEnd = last.first + last.second;
        
        if (curr.first <= lastEnd + gapThreshold) {
            uint64_t newEnd = std::max(lastEnd, curr.first + curr.second);
            last.second = newEnd - last.first;
        } else {
            merged.push_back(curr);
        }
    }
    
    return merged;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

struct ScanLineProcess
{
    ~ScanLineProcess ()
    {
        if (!first) {
            exr_decoding_destroy (decoder.context, &decoder);
        }
    }

    void run_decode (
        exr_const_context_t ctxt,
        int pn,
        const FrameBuffer *outfb,
        int fbY,
        int fbLastY,
        const std::vector<Slice> &filllist);

    void run_unpack (
        exr_const_context_t ctxt,
        int pn,
        const FrameBuffer *outfb,
        int fbY,
        int fbLastY,
        const std::vector<Slice> &filllist);

    void update_pointers (
        const FrameBuffer *outfb,
        int fbY,
        int fbLastY);

    void run_fill (
        const FrameBuffer *outfb,
        int fbY,
        const std::vector<Slice> &filllist);

    exr_result_t          last_decode_err = EXR_ERR_UNKNOWN;
    bool                  first = true;
    exr_chunk_info_t      cinfo;
    exr_decode_pipeline_t decoder;
    // Cache default read_fn chosen by OpenEXRCore so we can restore it
    // when IOMerge prefetch is not active (readPixels calls can alternate).
    exr_result_t (*default_read_fn)(exr_decode_pipeline_t*) = nullptr;

    // requirement to use process group
    ScanLineProcess* next;
};

#if ILMTHREAD_THREADING_ENABLED
using ScanLineProcessGroup = ILMTHREAD_NAMESPACE::ProcessGroup<ScanLineProcess>;
#endif

} // empty namespace

struct ScanLineInputFile::Data
{
    Data (Context *ctxt, int pN, int nT)
    : _ctxt (ctxt)
    , partNumber (pN)
    , numThreads (nT)
    {}

    void initialize ()
    {
        if (_ctxt->storage (partNumber) != EXR_STORAGE_SCANLINE)
            throw IEX_NAMESPACE::ArgExc ("File part is not a scanline part");
    }

    Context* _ctxt;
    int partNumber;
    int numThreads;
    Header header;
    bool header_filled = false;

    // TODO: remove once we can remove deprecated API
    std::vector<char> _pixel_data_scratch;

    void readPixels (const FrameBuffer &fb, int scanLine1, int scanLine2);

    // only keep a single stash of a scanline for things which
    // are reading one-scanline at a time. if we try to keep a
    // multi-threaded stash of scanlines, memory grows too rapidly
    std::unique_ptr<ScanLineProcess> singleScan;
    std::unique_ptr<ScanLineProcess> checkoutScan ()
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (_mx);
#endif
        if (singleScan) {
            return std::move (singleScan);
        }
        auto newScan = std::make_unique<ScanLineProcess> ();
        return newScan;
    }
    void checkinScan (std::unique_ptr<ScanLineProcess> &sp)
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (_mx);
#endif
        singleScan = std::move (sp);
    }

    FrameBuffer frameBuffer;
    std::vector<Slice> fill_list;

#if ILMTHREAD_THREADING_ENABLED
    std::mutex _mx;

    class LineBufferTask final : public ILMTHREAD_NAMESPACE::Task
    {
    public:
        LineBufferTask (
            ILMTHREAD_NAMESPACE::TaskGroup* group,
            Data*                   ifd,
            ScanLineProcessGroup*   lineg,
            const FrameBuffer*      outfb,
            const exr_chunk_info_t& cinfo,
            int                     fby,
            int                     endScan)
            : Task (group)
            , _outfb (outfb)
            , _ifd (ifd)
            , _fby (fby)
            , _last_fby (endScan)
            , _line (lineg->pop ())
            , _line_group (lineg)
        {
            _line->cinfo = cinfo;
        }

        ~LineBufferTask () override
        {
            _line_group->push (_line);
        }

        void execute () override;

    private:
        void run_decode ();

        const FrameBuffer*    _outfb;
        Data*                 _ifd;
        int                   _fby;
        int                   _last_fby;
        ScanLineProcess*      _line;
        ScanLineProcessGroup* _line_group;
    };
#endif
};

ScanLineInputFile::ScanLineInputFile (InputPartData* part)
    : _ctxt (part->context),
      _data (std::make_shared<Data> (&_ctxt, part->partNumber, part->numThreads))
{
    _data->initialize ();
}

ScanLineInputFile::ScanLineInputFile (
    const char*               filename,
    const ContextInitializer& ctxtinit,
    int                       numThreads)
    : _ctxt (filename, ctxtinit, Context::read_mode_t{})
    , _data (std::make_shared<Data> (&_ctxt, 0, numThreads))
{
    _data->initialize ();
}

ScanLineInputFile::ScanLineInputFile (
        OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is,
        int numThreads)
    : ScanLineInputFile (
        is.fileName (),
        ContextInitializer ()
        .silentHeaderParse (true)
        .strictHeaderValidation (false)
        .setInputStream (&is),
        numThreads)
{
}

ScanLineInputFile::ScanLineInputFile (const char filename[], int numThreads)
    : ScanLineInputFile (
        filename,
        ContextInitializer ()
        .silentHeaderParse (true)
        .strictHeaderValidation (false),
        numThreads)
{
}

const char*
ScanLineInputFile::fileName () const
{
    return _ctxt.fileName ();
}

const Header&
ScanLineInputFile::header () const
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
ScanLineInputFile::version () const
{
    return _ctxt.version ();
}

void
ScanLineInputFile::setFrameBuffer (const FrameBuffer& frameBuffer)
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    _data->fill_list.clear ();
    _data->singleScan.reset();

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
ScanLineInputFile::frameBuffer () const
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    return _data->frameBuffer;
}

bool
ScanLineInputFile::isComplete () const
{
    return _ctxt.chunkTableValid (_data->partNumber);
}

bool
ScanLineInputFile::isOptimizationEnabled () const
{
    // TODO: the core library has a number of special cased patterns,
    // this is all kind of ... not useful? for now, return a pattern
    // similar to legacy version
    return _ctxt.channels (_data->partNumber)->num_channels != 2;
}

void
ScanLineInputFile::readPixels (int scanLine1, int scanLine2)
{
    _data->readPixels (frameBuffer (), scanLine1, scanLine2);
}

void
ScanLineInputFile::readPixels (
    const FrameBuffer& frame, int scanLine1, int scanLine2)
{
    _data->readPixels (frame, scanLine1, scanLine2);
}

////////////////////////////////////////

void
ScanLineInputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}

////////////////////////////////////////

void
ScanLineInputFile::rawPixelData (
    int firstScanLine, const char*& pixelData, int& pixelDataSize)
{
    uint64_t maxsize = 0;
    if (EXR_ERR_SUCCESS !=
        exr_get_chunk_unpacked_size (_ctxt, _data->partNumber, &maxsize))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Unable to query data size of chunk in file '" << fileName ()
                                                           << "'");
    }

    // again, doesn't actually provide any safety given we're handing
    // back a pointer... but will at least prevent two threads
    // allocating at the same time and getting sliced
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (_data->_mx);
#endif
    _data->_pixel_data_scratch.resize (maxsize);

    pixelData     = _data->_pixel_data_scratch.data ();
    pixelDataSize = static_cast<int> (maxsize);

    rawPixelDataToBuffer (
        firstScanLine, _data->_pixel_data_scratch.data (), pixelDataSize);
}

void
ScanLineInputFile::rawPixelDataToBuffer (
    int scanLine, char* pixelData, int& pixelDataSize) const
{
    exr_chunk_info_t cinfo;
    if (EXR_ERR_SUCCESS == exr_read_scanline_chunk_info (
                               _ctxt, _data->partNumber, scanLine, &cinfo))
    {
        if (cinfo.packed_size > static_cast<uint64_t> (pixelDataSize))
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                    << fileName ()
                    << "\". Provided buffer is too small to read raw pixel data:"
                    << pixelDataSize << " bytes.");
        }

        pixelDataSize = static_cast<int> (cinfo.packed_size);

        if (EXR_ERR_SUCCESS !=
            exr_read_chunk (_ctxt, _data->partNumber, &cinfo, pixelData))
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                    << fileName () << "\". Unable to read raw pixel data of "
                    << pixelDataSize << " bytes.");
        }
    }
    else
    {
        if (_ctxt.storage (_data->partNumber) == EXR_STORAGE_TILED)
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                    << fileName ()
                    << "\". Tried to read a raw scanline from a tiled image.");
        }
        else
        {
            THROW (
                IEX_NAMESPACE::ArgExc,
                "Error reading pixel data from image "
                "file \""
                    << fileName ()
                    << "\". Unable to query data block information.");
        }
    }
}

////////////////////////////////////////

void ScanLineInputFile::Data::readPixels (
    const FrameBuffer &fb, int scanLine1, int scanLine2)
{
    exr_attr_box2i_t dw = _ctxt->dataWindow (partNumber);
    exr_chunk_info_t cinfo;
    int32_t          scansperchunk = 1;

    if (EXR_ERR_SUCCESS != exr_get_scanlines_per_chunk (*_ctxt, partNumber, &scansperchunk))
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Error querying scanline counts from image "
            "file \"" << _ctxt->fileName () << "\".");
    }

    if (scanLine2 < scanLine1)
        std::swap (scanLine1, scanLine2);

    if (scanLine1 < dw.min.y || scanLine2 > dw.max.y)
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "Tried to read scan line outside "
            "the image file's data window: "
            << scanLine1 << " - " << scanLine2
            << " vs datawindow "
            << dw.min.y << " - " << dw.max.y);
    }

    // ==================== I/O MERGING FOR SCANLINES ====================
    // Clear any previous prefetch state to avoid cross-call contamination
    // NOTE: Do NOT deallocate g_scanlinePrefetchBuffer.data here because
    // previous decode operations may have stored pointers into it (packed_buffer).
    // The vector will be resized as needed for the next prefetch operation.
    g_scanlinePrefetchBuffer.active = false;
    g_scanlinePrefetchBuffer.offsetMap.clear();

    // Calculate first and last chunk indices correctly
    // Chunk index = (y - minY) / scansperchunk
    int firstChunk = (scanLine1 - dw.min.y) / scansperchunk;
    int lastChunk = (scanLine2 - dw.min.y) / scansperchunk;
    int64_t nchunks = lastChunk - firstChunk + 1;
    
    bool usePrefetch = g_enableIOMerge.load(std::memory_order_relaxed) && nchunks > 1;
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
            
            // Get data window
            exr_attr_box2i_t dw;
            if (EXR_ERR_SUCCESS != exr_get_data_window(*_ctxt, partNumber, &dw)) {
                usePrefetch = false;
            }
            else if (rv != EXR_ERR_SUCCESS || !chunkTable) {
                usePrefetch = false;
            }
            else
            {
                // Calculate chunk indices for the scanline range
                // Scanline chunk index = (y - dw.min.y) / scansperchunk
                struct ScanlineChunkInfo {
                    int startY;
                    int32_t chunkIdx;
                    uint64_t leaderOffset;
                };
                std::vector<ScanlineChunkInfo> chunkInfos;
                chunkInfos.reserve(nchunks);
                
                uint64_t minLeaderOffset = UINT64_MAX;
                uint64_t maxLeaderOffset = 0;
                
                // Iterate through all chunks from firstChunk to lastChunk
                int firstChunkLocal = (scanLine1 - dw.min.y) / scansperchunk;
                int lastChunkLocal = (scanLine2 - dw.min.y) / scansperchunk;
                
                for (int chunkIdx = firstChunkLocal; chunkIdx <= lastChunkLocal; ++chunkIdx) {
                    ScanlineChunkInfo sci;
                    sci.startY = dw.min.y + chunkIdx * scansperchunk;
                    sci.chunkIdx = chunkIdx;
                    
                    if (chunkIdx >= 0 && chunkIdx < chunkCount) {
                        sci.leaderOffset = chunkTable[chunkIdx];
                        if (sci.leaderOffset > 0) {
                            minLeaderOffset = std::min(minLeaderOffset, sci.leaderOffset);
                            maxLeaderOffset = std::max(maxLeaderOffset, sci.leaderOffset);
                            chunkInfos.push_back(sci);
                        }
                    }
                }
                
                if (chunkInfos.empty()) {
                    usePrefetch = false;
                }
                else
                {
                    // Cache channel list; we will compute per-chunk unpacked_size
                    // matching OpenEXRCore's compute_chunk_unpack_size.
                    const exr_attr_chlist_t* chlist = nullptr;
                    bool hasSampling = false;
                    if (EXR_ERR_SUCCESS != exr_get_channels(*_ctxt, partNumber, &chlist) ||
                        !chlist || chlist->num_channels <= 0)
                    {
                        usePrefetch = false;
                    }
                    else
                    {
                        for (int c = 0; c < chlist->num_channels; ++c)
                        {
                            const exr_attr_chlist_entry_t* curc = (chlist->entries + c);
                            if (curc->x_sampling > 1 || curc->y_sampling > 1)
                                hasSampling = true;
                        }
                    }

                    // Get compression type
                    exr_compression_t compType;
                    if (EXR_ERR_SUCCESS != exr_get_compression(*_ctxt, partNumber, &compType)) {
                        usePrefetch = false;
                    }
                    else
                    {
                        // Determine multipart-ness and storage mode to parse leaders correctly.
                        int partCount = 0;
                        bool isMultipart = false;
                        exr_storage_t storageMode = EXR_STORAGE_LAST_TYPE;
                        if (EXR_ERR_SUCCESS != exr_get_count(*_ctxt, &partCount) || partCount <= 0)
                            usePrefetch = false;
                        else
                            isMultipart = (partCount > 1);
                        if (usePrefetch && EXR_ERR_SUCCESS != exr_get_storage(*_ctxt, partNumber, &storageMode))
                            usePrefetch = false;

                        // Leader size for scanline:
                        // - single-part scanline: [y][packed_size] => 8 bytes
                        // - multi-part scanline:  [part][y][packed_size] => 12 bytes
                        // Deep scanline leaders include additional 64-bit values; prefetch for deep
                        // is not implemented here (falls back to standard path).
                        size_t leaderSize = isMultipart ? 12 : 8;
                        if (usePrefetch && storageMode == EXR_STORAGE_DEEP_SCANLINE)
                            usePrefetch = false;
                        
                        // Get the actual max unpacked size from the file
                        uint64_t maxUnpackedSize = 0;
                        exr_get_chunk_unpacked_size(*_ctxt, partNumber, &maxUnpackedSize);
                        
                        // Estimate max data for range calculation
                        int width = dw.max.x - dw.min.x + 1;
                        size_t maxPackedSizeEstimate = (size_t)maxUnpackedSize; // Conservative
                        
                        // Calculate total range to read
                        size_t totalRangeSize = (maxLeaderOffset - minLeaderOffset) + leaderSize + maxPackedSizeEstimate;
                        
                        // Read entire range in one I/O
                        g_scanlinePrefetchBuffer.data.resize(totalRangeSize);
                        
                        std::vector<std::pair<uint64_t, uint64_t>> rangeToRead = {
                            {minLeaderOffset, totalRangeSize}
                        };
                        
                        if (!scanlineReadMultipleRanges(filename, rangeToRead, g_scanlinePrefetchBuffer.data.data())) {
                            usePrefetch = false;
                        }
                        else
                        {
                            allChunks.reserve(nchunks);
                            g_scanlinePrefetchBuffer.offsetMap.clear();
                            g_scanlinePrefetchBuffer.offsetMap.reserve(nchunks);
                            
                            uint64_t actualMaxDataEnd = 0;
                            
                            for (const auto& sci : chunkInfos) {
                                size_t bufferPos = sci.leaderOffset - minLeaderOffset;
                                
                                // Parse leader:
                                // - single-part: [y][packed_size]
                                // - multi-part:  [part][y][packed_size]
                                if (bufferPos + leaderSize > g_scanlinePrefetchBuffer.data.size())
                                {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }

                                // Avoid unaligned int32_t access from byte buffer
                                int32_t leaderVals[3] = {0, 0, 0};
                                std::memcpy(
                                    leaderVals,
                                    g_scanlinePrefetchBuffer.data.data() + bufferPos,
                                    leaderSize);

                                int32_t ldr_part = -1;
                                int32_t ldr_y = 0;
                                int32_t packed_size = 0;
                                if (isMultipart)
                                {
                                    ldr_part   = leaderVals[0];
                                    ldr_y      = leaderVals[1];
                                    packed_size = leaderVals[2];
                                }
                                else
                                {
                                    ldr_y      = leaderVals[0];
                                    packed_size = leaderVals[1];
                                }

                                // Validate leader matches expected chunk
                                if (isMultipart && ldr_part != partNumber)
                                {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }
                                if (ldr_y != sci.startY)
                                {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }
                                
                                // Validate
                                if (packed_size <= 0 || packed_size > (int32_t)(width * scansperchunk * 6)) {
                                    allChunks.clear();
                                    usePrefetch = false;
                                    break;
                                }
                                
                                // Build chunk info
                                exr_chunk_info_t ci;
                                ci.idx = sci.chunkIdx;
                                ci.type = (uint8_t)EXR_STORAGE_SCANLINE;
                                ci.compression = (uint8_t)compType;
                                ci.start_x = dw.min.x;
                                ci.start_y = ldr_y;
                                ci.width = width;
                                // Last scanline chunk may be shorter than scansperchunk
                                int32_t chunkH = scansperchunk;
                                int32_t remaining = (int32_t)dw.max.y - ldr_y + 1;
                                if (remaining < chunkH) chunkH = remaining;
                                if (chunkH < 0) chunkH = 0;
                                ci.height = chunkH;
                                ci.level_x = 0;
                                ci.level_y = 0;
                                ci.packed_size = packed_size;
                                // Compute unpacked_size matching OpenEXRCore's compute_chunk_unpack_size:
                                // - If any channel has line sampling OR this is a partial chunk,
                                //   compute per-channel sampled sizes.
                                // - Otherwise use part->unpacked_size_per_chunk (via maxUnpackedSize).
                                if (hasSampling || chunkH != scansperchunk)
                                {
                                    uint64_t unpacksize = 0;
                                    for (int c = 0; c < chlist->num_channels; ++c)
                                    {
                                        const exr_attr_chlist_entry_t* curc = (chlist->entries + c);
                                        uint64_t chansz =
                                            (uint64_t)((curc->pixel_type == EXR_PIXEL_HALF) ? 2 : 4);
                                        chansz *= (uint64_t) openexr_compute_sampled_width(
                                            width, curc->x_sampling, dw.min.x);
                                        chansz *= (uint64_t) openexr_compute_sampled_height(
                                            chunkH, curc->y_sampling, ldr_y);
                                        unpacksize += chansz;
                                    }
                                    ci.unpacked_size = unpacksize;
                                }
                                else
                                {
                                    ci.unpacked_size = maxUnpackedSize;
                                }
                                ci.data_offset = sci.leaderOffset + leaderSize;
                                ci.sample_count_data_offset = 0;
                                ci.sample_count_table_size = 0;
                                
                                allChunks.push_back(ci);
                                
                                uint64_t dataEnd = ci.data_offset + packed_size;
                                if (dataEnd > actualMaxDataEnd) actualMaxDataEnd = dataEnd;
                                
                                size_t dataBufferPos = ci.data_offset - minLeaderOffset;
                                g_scanlinePrefetchBuffer.offsetMap[ci.data_offset] = {dataBufferPos, (size_t)packed_size};
                            }
                            
                            if (usePrefetch && !allChunks.empty())
                            {
                                // Extend buffer if needed
                                if (actualMaxDataEnd > minLeaderOffset + totalRangeSize) {
                                    size_t extraNeeded = actualMaxDataEnd - (minLeaderOffset + totalRangeSize);
                                    size_t oldSize = g_scanlinePrefetchBuffer.data.size();
                                    g_scanlinePrefetchBuffer.data.resize(oldSize + extraNeeded);
                                    
                                    std::vector<std::pair<uint64_t, uint64_t>> extraRange = {
                                        {minLeaderOffset + totalRangeSize, extraNeeded}
                                    };
                                    if (!scanlineReadMultipleRanges(filename, extraRange,
                                            g_scanlinePrefetchBuffer.data.data() + oldSize)) {
                                        throw IEX_NAMESPACE::IoExc("Failed to prefetch remaining scanline data");
                                    }
                                }
                                
                                g_scanlinePrefetchBuffer.active = true;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fallback if batch read failed
    if (usePrefetch && allChunks.empty()) {
        usePrefetch = false;
    }
    // ==================== END I/O MERGING ====================
    
    // When IOMerge is active, we must use single-threaded processing because
    // g_scanlinePrefetchBuffer is thread_local and won't be visible to worker threads.
    // The I/O merging benefit (fewer pread calls) is still retained.
    bool useMultiThreading = (nchunks > 1 && numThreads > 1 && !usePrefetch);

#if ILMTHREAD_THREADING_ENABLED
    if (useMultiThreading)
    {
        ScanLineProcessGroup sg (numThreads);

        {
            ILMTHREAD_NAMESPACE::TaskGroup tg;

            for (int y = scanLine1; y <= scanLine2; )
            {
                if (EXR_ERR_SUCCESS != exr_read_scanline_chunk_info (*_ctxt, partNumber, y, &cinfo))
                    throw IEX_NAMESPACE::InputExc ("Unable to query scanline information");

                ILMTHREAD_NAMESPACE::ThreadPool::addGlobalTask (
                    new LineBufferTask (&tg, this, &sg, &fb, cinfo, y, scanLine2) );

                y += scansperchunk - (y - cinfo.start_y);
            }
        }

        sg.throw_on_failure ();
    }
    else
#endif
    {
        std::unique_ptr<ScanLineProcess> sp = checkoutScan ();

        if (usePrefetch && !allChunks.empty())
        {
            // Use prefetched chunk info
            for (const auto& chunk : allChunks)
            {
                sp->cinfo = chunk;
                // IMPORTANT: Use max(chunk.start_y, scanLine1) as fbY to match
                // non-IOMerge behavior. This ensures decode_to_ptr calculation
                // in update_pointers() produces correct pointer:
                //   ptr = fbslice->base + fbY * yStride
                //       = (basePtr - origin.y * yStride) + fbY * yStride
                //       = basePtr + (fbY - origin.y) * yStride
                // If origin.y = scanLine1 (from Slice::Make), we need fbY >= scanLine1
                // to avoid writing before the allocated buffer.
                int fbY = std::max(chunk.start_y, scanLine1);
                sp->run_decode (
                    *_ctxt,
                    partNumber,
                    &fb,
                    fbY,
                    scanLine2,
                    fill_list);
            }
        }
        else
        {
            // Fallback to original behavior
            for (int y = scanLine1; y <= scanLine2; )
            {
                if (EXR_ERR_SUCCESS != exr_read_scanline_chunk_info (*_ctxt, partNumber, y, &cinfo))
                    throw IEX_NAMESPACE::InputExc ("Unable to query scanline information");

                // check if we have the same chunk where we can just
                // re-run the unpack (i.e. people reading 1 scan at a time
                // in a multi-scanline chunk)
                if (!sp->first && sp->cinfo.idx == cinfo.idx &&
                    sp->last_decode_err == EXR_ERR_SUCCESS)
                {
                    sp->run_unpack (
                    *_ctxt,
                    partNumber,
                    &fb,
                    y,
                    scanLine2,
                    fill_list);
            }
            else
            {
                sp->cinfo = cinfo;
                sp->run_decode (
                    *_ctxt,
                    partNumber,
                    &fb,
                    y,
                    scanLine2,
                    fill_list);
            }

                y += scansperchunk - (y - cinfo.start_y);
            }
        }

        checkinScan (sp);
    }
    
    // Clean up prefetch state (keep buffer capacity for reuse)
    if (g_scanlinePrefetchBuffer.active) {
        g_scanlinePrefetchBuffer.active = false;
    }
}

////////////////////////////////////////

#if ILMTHREAD_THREADING_ENABLED
void ScanLineInputFile::Data::LineBufferTask::execute ()
{
    try
    {
        _line->run_decode (
            *(_ifd->_ctxt),
            _ifd->partNumber,
            _outfb,
            _fby,
            _last_fby,
            _ifd->fill_list);
    }
    catch (std::exception &e)
    {
        _line_group->record_failure (e.what ());
    }
    catch (...)
    {
        _line_group->record_failure ("Unknown exception");
    }
}
#endif

////////////////////////////////////////

void ScanLineProcess::run_decode (
    exr_const_context_t ctxt,
    int pn,
    const FrameBuffer *outfb,
    int fbY,
    int fbLastY,
    const std::vector<Slice> &filllist)
{
    last_decode_err = EXR_ERR_UNKNOWN;
    // stash the flag off to make sure to clean up in the event
    // of an exception by changing the flag after init...
    bool isfirst = first;
    
    if (first)
    {
        exr_result_t rv = exr_decoding_initialize (ctxt, pn, &cinfo, &decoder);
        if (EXR_ERR_SUCCESS != rv)
        {
            throw IEX_NAMESPACE::IoExc ("Unable to initialize decode pipeline");
        }

        first = false;
    }
    else
    {
        exr_result_t rv = exr_decoding_update (ctxt, pn, &cinfo, &decoder);
        if (EXR_ERR_SUCCESS != rv)
        {
            throw IEX_NAMESPACE::IoExc ("Unable to update decode pipeline");
        }
    }

    update_pointers (outfb, fbY, fbLastY);

    if (isfirst)
    {
        if (EXR_ERR_SUCCESS !=
            exr_decoding_choose_default_routines (ctxt, pn, &decoder))
        {
            throw IEX_NAMESPACE::IoExc ("Unable to choose decoder routines");
        }
        default_read_fn = decoder.read_fn;
    }
    else if (default_read_fn)
    {
        // readPixels() may be called repeatedly and may or may not use IOMerge
        // depending on the requested range. Restore the core-chosen read_fn
        // before optionally overriding it for prefetch.
        decoder.read_fn = default_read_fn;
    }

    // If prefetch buffer is active, use custom read function
    if (g_scanlinePrefetchBuffer.active)
    {
        if (decoder.decompress_fn != nullptr)
        {
            // Compressed: point packed_buffer into prefetch buffer (zero-copy)
            decoder.read_fn = &scanline_prefetched_read_chunk;
        }
        else
        {
            // Uncompressed:
            // - If core selected the "read_uncompressed_direct" fast path
            //   (unpack_and_convert_fn == NULL), we must copy directly into the
            //   user buffers (and byte-swap as needed).
            // - Otherwise, we must present the uncompressed bytes via packed_buffer
            //   so the normal unpack/convert routines can run safely.
            if (decoder.unpack_and_convert_fn == nullptr)
                decoder.read_fn = &scanline_prefetched_read_uncompressed_direct;
            else
                decoder.read_fn = &scanline_prefetched_read_chunk;
        }
    }

    last_decode_err = exr_decoding_run (ctxt, pn, &decoder);
    if (EXR_ERR_SUCCESS != last_decode_err)
        throw IEX_NAMESPACE::IoExc ("Unable to run decoder");

    run_fill (outfb, fbY, filllist);
}

////////////////////////////////////////

void ScanLineProcess::run_unpack (
    exr_const_context_t ctxt,
    int pn,
    const FrameBuffer *outfb,
    int fbY,
    int fbLastY,
    const std::vector<Slice> &filllist)
{
    update_pointers (outfb, fbY, fbLastY);

    /* won't work for deep where we need to re-allocate the number of
     * samples but for normal scanlines is fine to just bypass pipe
     * and run the unpacker */
    if (decoder.chunk.unpacked_size > 0 && decoder.unpack_and_convert_fn)
    {
        last_decode_err = decoder.unpack_and_convert_fn (&decoder);
        if (EXR_ERR_SUCCESS != last_decode_err)
            throw IEX_NAMESPACE::IoExc ("Unable to run decoder");
    }

    run_fill (outfb, fbY, filllist);
}

////////////////////////////////////////

void ScanLineProcess::update_pointers (
    const FrameBuffer *outfb, int fbY, int fbLastY)
{
    decoder.user_line_begin_skip = fbY - cinfo.start_y;
    decoder.user_line_end_ignore = 0;
    int64_t endY = (int64_t)cinfo.start_y + (int64_t)cinfo.height - 1;
    if ((int64_t)fbLastY < endY)
        decoder.user_line_end_ignore = (int32_t)(endY - fbLastY);

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

        curchan.user_bytes_per_element = (fbslice->type == HALF) ? 2 : 4;
        curchan.user_data_type         = (exr_pixel_type_t)fbslice->type;
        curchan.user_pixel_stride      = fbslice->xStride;
        curchan.user_line_stride       = fbslice->yStride;

        ptr  = reinterpret_cast<uint8_t*> (fbslice->base);
        ptr += int64_t (cinfo.start_x / fbslice->xSampling) * int64_t (fbslice->xStride);
        ptr += int64_t (fbY / fbslice->ySampling) * int64_t (fbslice->yStride);

        curchan.decode_to_ptr = ptr;
    }
}

////////////////////////////////////////

void ScanLineProcess::run_fill (
    const FrameBuffer *outfb,
    int fbY,
    const std::vector<Slice> &filllist)
{
    for (auto& s: filllist)
    {
        uint8_t*       ptr;

        ptr  = reinterpret_cast<uint8_t*> (s.base);
        ptr += int64_t (cinfo.start_x / s.xSampling) * int64_t (s.xStride);
        ptr += int64_t (fbY / s.ySampling) * int64_t (s.yStride);

        // TODO: update ImfMisc, lift fill type / value
        int stop = cinfo.start_y + cinfo.height - decoder.user_line_end_ignore;
        for ( int start = fbY; start < stop; ++start )
        {
            if (start % s.ySampling) continue;

            uint8_t* outptr = ptr;
            for ( int sx = cinfo.start_x, ex = cinfo.start_x + cinfo.width;
                  sx < ex; ++sx )
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

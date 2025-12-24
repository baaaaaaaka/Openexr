//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "LustreIO.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdexcept>
#include <cstring>

namespace LustreIO {

MergedFileReader::MergedFileReader(const std::string& filepath)
    : _filepath(filepath), _fd(-1), _fileSize(0)
{
    _fd = ::open(filepath.c_str(), O_RDONLY);
    if (_fd < 0) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    struct stat st;
    if (::fstat(_fd, &st) < 0) {
        ::close(_fd);
        throw std::runtime_error("Failed to stat file: " + filepath);
    }
    _fileSize = st.st_size;
}

MergedFileReader::~MergedFileReader()
{
    if (_fd >= 0) {
        ::close(_fd);
    }
}

size_t MergedFileReader::readRanges(const std::vector<ByteRange>& ranges, uint64_t gap_threshold)
{
    if (ranges.empty()) return 0;
    
    // Make a copy and merge
    std::vector<ByteRange> rangesToMerge = ranges;
    _mergedRanges = mergeRanges(rangesToMerge, gap_threshold);
    
    // Calculate total size needed
    size_t totalSize = 0;
    for (const auto& r : _mergedRanges) {
        totalSize += r.size;
    }
    
    // Allocate buffer
    _buffer.resize(totalSize);
    _bufferOffsets.resize(_mergedRanges.size());
    
    // Read each merged range
    size_t bufferPos = 0;
    for (size_t i = 0; i < _mergedRanges.size(); ++i) {
        const ByteRange& r = _mergedRanges[i];
        _bufferOffsets[i] = bufferPos;
        
        ssize_t bytesRead = ::pread(_fd, _buffer.data() + bufferPos, r.size, r.offset);
        if (bytesRead < 0) {
            throw std::runtime_error("Failed to read from file");
        }
        
        bufferPos += r.size;
    }
    
    return totalSize;
}

const uint8_t* MergedFileReader::getData(uint64_t file_offset, uint64_t size) const
{
    // Find which merged range contains this offset
    for (size_t i = 0; i < _mergedRanges.size(); ++i) {
        const ByteRange& r = _mergedRanges[i];
        if (file_offset >= r.offset && file_offset + size <= r.end()) {
            // Found it
            uint64_t offsetInRange = file_offset - r.offset;
            return _buffer.data() + _bufferOffsets[i] + offsetInRange;
        }
    }
    
    // Not found
    return nullptr;
}

} // namespace LustreIO


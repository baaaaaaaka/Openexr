//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace LustreIO {

// Represents a byte range in the file
struct ByteRange {
    uint64_t offset;
    uint64_t size;
    
    uint64_t end() const { return offset + size; }
    
    bool operator<(const ByteRange& other) const {
        return offset < other.offset;
    }
};

// Merge adjacent or overlapping byte ranges
// gap_threshold: merge ranges if gap between them is less than this
inline std::vector<ByteRange> mergeRanges(
    std::vector<ByteRange>& ranges, 
    uint64_t gap_threshold = 4096)  // 4KB default gap threshold
{
    if (ranges.empty()) return {};
    
    // Sort by offset
    std::sort(ranges.begin(), ranges.end());
    
    std::vector<ByteRange> merged;
    merged.push_back(ranges[0]);
    
    for (size_t i = 1; i < ranges.size(); ++i) {
        ByteRange& last = merged.back();
        const ByteRange& curr = ranges[i];
        
        // Check if can merge (overlapping or gap is small)
        if (curr.offset <= last.end() + gap_threshold) {
            // Extend the last range
            uint64_t new_end = std::max(last.end(), curr.end());
            last.size = new_end - last.offset;
        } else {
            // Start a new range
            merged.push_back(curr);
        }
    }
    
    return merged;
}

// Read multiple byte ranges from a file with minimal I/O calls
// Returns a single buffer containing all the data, with a map of original offsets to buffer positions
class MergedFileReader {
public:
    MergedFileReader(const std::string& filepath);
    ~MergedFileReader();
    
    // Read specified ranges, merging adjacent ones
    // Returns total bytes read
    size_t readRanges(const std::vector<ByteRange>& ranges, uint64_t gap_threshold = 4096);
    
    // Get data for a specific original file offset
    const uint8_t* getData(uint64_t file_offset, uint64_t size) const;
    
    // Get the entire merged buffer
    const std::vector<uint8_t>& getBuffer() const { return _buffer; }
    
    // Get file size
    uint64_t getFileSize() const { return _fileSize; }
    
private:
    std::string _filepath;
    int _fd;
    uint64_t _fileSize;
    
    std::vector<uint8_t> _buffer;
    std::vector<ByteRange> _mergedRanges;
    std::vector<uint64_t> _bufferOffsets;  // Position in _buffer for each merged range
};

} // namespace LustreIO


//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_THREADING_H
#define INCLUDED_IMF_THREADING_H

#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	Threading support for the OpenEXR library
//
//	The OpenEXR library uses threads to perform reading and writing
//	of OpenEXR files in parallel.  The thread that calls the library
//	always performs the actual file IO (this is usually the main
//	application thread) whereas a several worker threads perform
//	data compression and decompression.  The number of worker
//	threads can be any non-negative value (a value of zero reverts
//	to single-threaded operation).  As long as there is at least
//	one worker thread, file IO and compression can potentially be
//	done concurrently through pinelining.  If there are two or more
//	worker threads, then pipelining as well as concurrent compression
//	of multiple blocks can be performed.
//
//	Threading in the EXR library is controllable at two granularities:
//
//	* The functions in this file query and control the total number
//	  of worker threads, which will be created globally for the whole
//	  library.  Regardless of how many input or output files are
//	  opened simultaneously, the library will use at most this number
//	  of worker threads to perform all work.  The default number of
//	  global worker threads is zero (i.e. single-threaded operation;
//	  everything happens in the thread that calls the library).
//
//	* Furthermore, it is possible to set the number of threads that
//	  each input or output file should keep busy.  This number can
//	  be explicitly set for each file.  The default behavior is for
//	  each file to try to occupy all worker threads in the library's
//	  thread pool.
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-----------------------------------------------------------------------------
// Return the number of Imf-global worker threads used for parallel
// compression and decompression of OpenEXR files.
//-----------------------------------------------------------------------------

IMF_EXPORT int globalThreadCount ();

//-----------------------------------------------------------------------------
// Change the number of Imf-global worker threads
//-----------------------------------------------------------------------------

IMF_EXPORT void setGlobalThreadCount (int count);

//-----------------------------------------------------------------------------
// Non-temporal writes control
//
// When enabled, the library will use non-temporal (streaming) store
// instructions for writing decoded pixel data. This bypasses the CPU
// cache, which is beneficial when:
//
// * The output data will not be read again soon (e.g., ML data loaders
//   that immediately transfer data to GPU memory)
// * Processing large amounts of data where cache pollution is a concern
// * The decompression buffer should stay in L2 cache for better performance
//
// Trade-offs:
// * Reduces cache pollution from output writes
// * May be slower if output data is needed immediately after decoding
// * Works best with output buffers that are 32-byte aligned
//
// Default: disabled (false)
//-----------------------------------------------------------------------------

IMF_EXPORT void setNonTemporalWrites (bool enable);

IMF_EXPORT bool nonTemporalWrites ();

//-----------------------------------------------------------------------------
// X-direction cropping for tile decoding (thread-local)
//
// When decoding tiles, setting this allows the decoder to skip pixels at the
// beginning and end of each line, reducing memory bandwidth.
//
// This is useful when reading a crop region that doesn't align with tile
// boundaries - instead of decoding the full tile and copying the needed
// portion, the decoder will only convert the needed pixels.
//
// cropXMin/cropXMax: Absolute pixel coordinates of the crop region
// tileWidth: Full tile width (used internally)
//
// Call clearTileXCrop() after decoding to reset to full-tile mode.
//
// Note: This setting is thread-local, so it can be used safely in
// multi-threaded applications where each thread decodes different regions.
//-----------------------------------------------------------------------------

IMF_EXPORT void setTileXCrop (int cropXMin, int cropXMax, int tileWidth);

IMF_EXPORT void clearTileXCrop ();

//-----------------------------------------------------------------------------
// Y-direction cropping for tile decoding (thread-local)
//
// When decoding tiles, setting this allows the decoder to skip lines at the
// beginning and end of each tile, reducing memory bandwidth.
//
// This is useful when reading a crop region that doesn't align with tile
// boundaries - instead of decoding the full tile and copying the needed
// portion, the decoder will only convert the needed lines.
//
// cropYMin/cropYMax: Absolute pixel coordinates of the crop region
//
// Call clearTileYCrop() after decoding to reset to full-tile mode.
//
// Note: This setting is thread-local, so it can be used safely in
// multi-threaded applications where each thread decodes different regions.
//-----------------------------------------------------------------------------

IMF_EXPORT void setTileYCrop (int cropYMin, int cropYMax);

IMF_EXPORT void clearTileYCrop ();

//-----------------------------------------------------------------------------
// I/O Merging for Lustre/GPFS optimization
//
// When enabled, multiple tile reads are merged into fewer large I/O operations.
// This is beneficial on distributed file systems like Lustre or GPFS where
// each I/O call has significant overhead (e.g., 0.1-0.5ms RTT).
//
// How it works:
// 1. Collects all tile chunk offsets/sizes before reading
// 2. Sorts and merges adjacent byte ranges
// 3. Performs 1-2 large pread() calls instead of 100+ small ones
// 4. Distributes data to individual tile decoders from memory
//
// Trade-offs:
// * Reduces I/O calls from ~100 to 1-3 for typical crop regions
// * May read slightly more data (gap between tiles included)
// * Uses additional memory for prefetch buffer
// * Only works for file-based reads (not memory streams)
//
// Default: disabled (false)
//-----------------------------------------------------------------------------

IMF_EXPORT void setIOMerge (bool enable);

IMF_EXPORT bool isMergeEnabled ();

//-----------------------------------------------------------------------------
// IOMerge mode control
//
// Controls how I/O operations are merged when reading tiled images:
// - "row" (default): One pread per tile row (good balance of I/O count and bandwidth)
// - "single": One pread for entire crop region (minimum I/O, may read extra data)
//
// Can also be set via OPENEXR_IOMERGE_MODE environment variable.
//-----------------------------------------------------------------------------

IMF_EXPORT void setIOMergeMode (const char* mode);

IMF_EXPORT const char* getIOMergeMode ();

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif

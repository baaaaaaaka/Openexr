/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_memory.h"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <stdlib.h>
#endif
#include <inttypes.h>

/**************************************/

/* Cache line alignment for better memory performance */
#define EXR_CACHE_LINE_SIZE 64

static exr_memory_allocation_func_t _glob_alloc_func = NULL;
static exr_memory_free_func_t       _glob_free_func  = NULL;

/**************************************/

void
exr_set_default_memory_routines (
    exr_memory_allocation_func_t alloc_func, exr_memory_free_func_t free_func)
{
    _glob_alloc_func = alloc_func;
    _glob_free_func  = free_func;
}

/**************************************/

void*
internal_exr_alloc (size_t bytes)
{
    if (_glob_alloc_func) return (*_glob_alloc_func) (bytes);
#ifdef _WIN32
    return HeapAlloc (GetProcessHeap (), 0, bytes);
#else
    return malloc (bytes);
#endif
}

/**************************************/

/*
 * Allocate memory aligned to cache line (64 bytes) for better performance.
 * The returned pointer can be freed with standard free().
 */
void*
internal_exr_alloc_cacheline (size_t bytes)
{
    void* ptr = NULL;
    if (_glob_alloc_func) return (*_glob_alloc_func) (bytes);
#ifdef _WIN32
    ptr = _aligned_malloc (bytes, EXR_CACHE_LINE_SIZE);
#elif defined(__APPLE__) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
    if (posix_memalign (&ptr, EXR_CACHE_LINE_SIZE, bytes) != 0)
        ptr = NULL;
#elif __STDC_VERSION__ >= 201112L
    ptr = aligned_alloc (EXR_CACHE_LINE_SIZE, 
                         (bytes + EXR_CACHE_LINE_SIZE - 1) & ~(EXR_CACHE_LINE_SIZE - 1));
#else
    ptr = malloc (bytes);
#endif
    return ptr;
}

/**************************************/

void
internal_exr_free_cacheline (void* ptr)
{
    if (!ptr) return;

    if (_glob_free_func) { (*_glob_free_func) (ptr); }
    else
    {
#ifdef _WIN32
        _aligned_free (ptr);
#else
        free (ptr);
#endif
    }
}

/**************************************/

void*
internal_exr_alloc_aligned (
    void* (*alloc_fn) (size_t), void** tofreeptr, size_t bytes, size_t align)
{
    void* ret;
    if (align == 1 || align > 4096) { align = 0; }

    ret        = alloc_fn (bytes + align);
    *tofreeptr = ret;
    if (ret)
    {
        uintptr_t off = ((uintptr_t) ret) & (align - 1);
        if (off) off = align - off;
        ret = (((uint8_t*) ret) + off);
    }
    return ret;
}

/**************************************/

void
internal_exr_free (void* ptr)
{
    if (!ptr) return;

    if (_glob_free_func) { (*_glob_free_func) (ptr); }
    else
    {
#ifdef _WIN32
        HeapFree (GetProcessHeap (), 0, ptr);
#else
        free (ptr);
#endif
    }
}

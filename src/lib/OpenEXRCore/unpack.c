/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_coding.h"
#include "internal_xdr.h"
#include "internal_cpuid.h"

#include "openexr_attr.h"

#include <string.h>

/**************************************/

/* Prefetch macros for better cache utilization */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <xmmintrin.h>
#    define EXR_PREFETCH_T0(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#    define EXR_PREFETCH_T1(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#    define EXR_PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define EXR_PREFETCH_T0(addr) __builtin_prefetch((const void*)(addr), 0, 3)
#    define EXR_PREFETCH_T1(addr) __builtin_prefetch((const void*)(addr), 0, 2)
#    define EXR_PREFETCH_NTA(addr) __builtin_prefetch((const void*)(addr), 0, 0)
#elif defined(__GNUC__) || defined(__clang__)
#    define EXR_PREFETCH_T0(addr) __builtin_prefetch((const void*)(addr), 0, 3)
#    define EXR_PREFETCH_T1(addr) __builtin_prefetch((const void*)(addr), 0, 2)
#    define EXR_PREFETCH_NTA(addr) __builtin_prefetch((const void*)(addr), 0, 0)
#else
#    define EXR_PREFETCH_T0(addr) ((void)0)
#    define EXR_PREFETCH_T1(addr) ((void)0)
#    define EXR_PREFETCH_NTA(addr) ((void)0)
#endif

/* Prefetch distance in cache lines (64 bytes each) */
#define EXR_PREFETCH_DISTANCE 4

/**************************************/

/* TODO: learn arm neon intrinsics for this */
#if (defined(__x86_64__) || defined(_M_X64))
#    if defined(__AVX__) && (defined(__F16C__) || defined(__GNUC__) || defined(__clang__))
#        define USE_F16C_INTRINSICS
#    elif (defined(__GNUC__) || defined(__clang__))
#        define ENABLE_F16C_TEST
#    endif
#endif

#if defined(USE_F16C_INTRINSICS) || defined(ENABLE_F16C_TEST)
#    if defined(USE_F16C_INTRINSICS)
static inline void
half_to_float_buffer (float* out, const uint16_t* in, int w)
#    elif defined(ENABLE_F16C_TEST)
__attribute__ ((target ("f16c"))) static void
half_to_float_buffer_f16c (float* out, const uint16_t* in, int w)
#    endif
{
    while (w >= 8)
    {
        _mm256_storeu_ps (
            out, _mm256_cvtph_ps (_mm_loadu_si128 ((const __m128i*) in)));
        out += 8;
        in += 8;
        w -= 8;
    }
    // gcc < 9 does not have loadu_si64
#    if defined(__clang__) || (__GNUC__ >= 9)
    switch (w)
    {
        case 7:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            break;
        case 4: _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in))); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
#    else
    while (w > 0)
    {
        *out++ = half_to_float (*in++);
        --w;
    }
#    endif
}

/*
 * Non-temporal (streaming) store version for F16C path.
 * Uses _mm256_stream_ps to bypass cache on output writes.
 * This keeps the decompression buffer in cache while writing output.
 */
#    if defined(USE_F16C_INTRINSICS)
static inline void
half_to_float_buffer_nt (float* out, const uint16_t* in, int w)
#    elif defined(ENABLE_F16C_TEST)
__attribute__ ((target ("f16c,avx"))) static void
half_to_float_buffer_nt_f16c (float* out, const uint16_t* in, int w)
#    endif
{
    /* Handle unaligned prefix (stream requires 32-byte alignment) */
    uintptr_t addr = (uintptr_t) out;
    size_t misalign = (32 - (addr & 31)) & 31;
    size_t prefix_floats = misalign / sizeof(float);
    
    while (prefix_floats > 0 && w > 0)
    {
        *out++ = half_to_float (*in++);
        --w;
        --prefix_floats;
    }
    
    /* Main loop: 8 floats at a time with streaming stores */
    while (w >= 8)
    {
        __m256 vals = _mm256_cvtph_ps (_mm_loadu_si128 ((const __m128i*) in));
        _mm256_stream_ps (out, vals);
        out += 8;
        in += 8;
        w -= 8;
    }
    
    /* Handle remaining floats */
#    if defined(__clang__) || (__GNUC__ >= 9)
    switch (w)
    {
        case 7:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            break;
        case 4: _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in))); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
#    else
    while (w > 0)
    {
        *out++ = half_to_float (*in++);
        --w;
    }
#    endif
}
#endif

#ifndef USE_F16C_INTRINSICS
static inline void
half_to_float4 (float* out, const uint16_t* src)
{
    out[0] = half_to_float (src[0]);
    out[1] = half_to_float (src[1]);
    out[2] = half_to_float (src[2]);
    out[3] = half_to_float (src[3]);
}

static inline void
half_to_float8 (float* out, const uint16_t* src)
{
    half_to_float4 (out, src);
    half_to_float4 (out + 4, src + 4);
}
#else
/* when we explicitly compile against f16, force it in, do not need a chooser */
static inline void
choose_half_to_float_impl (void)
{}
#endif

#ifdef ENABLE_F16C_TEST
static void
half_to_float_buffer_impl (float* out, const uint16_t* in, int w)
{
    while (w >= 8)
    {
        half_to_float8 (out, in);
        out += 8;
        in += 8;
        w -= 8;
    }
    switch (w)
    {
        case 7:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            break;
        case 4: half_to_float4 (out, in); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
}

static void (*half_to_float_buffer) (float*, const uint16_t*, int) =
    &half_to_float_buffer_impl;

/* Non-temporal version function pointer (for runtime selection) */
static void
half_to_float_buffer_nt_impl (float* out, const uint16_t* in, int w)
{
    /* Fallback non-NT implementation for non-F16C path */
    while (w >= 8)
    {
        half_to_float8 (out, in);
        out += 8;
        in += 8;
        w -= 8;
    }
    while (w > 0)
    {
        *out++ = half_to_float (*in++);
        --w;
    }
}

static void (*half_to_float_buffer_nt) (float*, const uint16_t*, int) =
    &half_to_float_buffer_nt_impl;

static inline void
choose_half_to_float_impl (void)
{
    if (has_native_half ())
    {
        half_to_float_buffer = &half_to_float_buffer_f16c;
        half_to_float_buffer_nt = &half_to_float_buffer_nt_f16c;
    }
}

#endif /* ENABLE_F16C_TEST */

#if !(defined(ENABLE_F16C_TEST) || defined(USE_F16C_INTRINSICS))

static inline void
half_to_float_buffer (float* out, const uint16_t* in, int w)
{
#    if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    for (int x = 0; x < w; ++x)
        out[x] = half_to_float (one_to_native16 (in[x]));
#    else
    while (w >= 8)
    {
        half_to_float8 (out, in);
        out += 8;
        in += 8;
        w -= 8;
    }
    switch (w)
    {
        case 7:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            break;
        case 4: half_to_float4 (out, in); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
#    endif
}

/* Non-temporal version for non-F16C path (falls back to regular writes) */
static inline void
half_to_float_buffer_nt (float* out, const uint16_t* in, int w)
{
    /* Without F16C, just use regular implementation */
    /* In the future, could use SSE2 streaming stores with software half conversion */
    half_to_float_buffer (out, in, w);
}

static void
choose_half_to_float_impl (void)
{}

#endif

/**************************************/

/*
 * Non-temporal planar unpack functions.
 * These use streaming stores to bypass cache on output.
 */

static exr_result_t
unpack_half_to_float_3chan_planar_nt (exr_decode_pipeline_t* decode)
{
    /* Same as unpack_half_to_float_3chan_planar but uses NT stores */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h, out_w, x_skip;
    int             linc0, linc1, linc2;

    w      = decode->channels[0].width;
    h      = decode->chunk.height - decode->user_line_end_ignore;
    x_skip = decode->user_pixel_begin_skip;
    out_w  = w - x_skip - decode->user_pixel_end_ignore;
    
    /* If no X crop, use full width */
    if (out_w <= 0 || out_w > w) out_w = w;
    if (x_skip < 0) x_skip = 0;

    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    const size_t line_bytes = (size_t)w * 6;
    srcbuffer += decode->user_line_begin_skip * w * 6;

    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        /* Prefetch next line(s) into L1 cache */
        if (y + EXR_PREFETCH_DISTANCE < h)
        {
            const uint8_t* prefetch_addr = srcbuffer + EXR_PREFETCH_DISTANCE * line_bytes;
            EXR_PREFETCH_T0(prefetch_addr);
            EXR_PREFETCH_T0(prefetch_addr + 64);
            EXR_PREFETCH_T0(prefetch_addr + 128);
        }

        /* Apply X skip to input pointers */
        in0 = (const uint16_t*) srcbuffer + x_skip;
        in1 = (const uint16_t*) srcbuffer + w + x_skip;
        in2 = (const uint16_t*) srcbuffer + w * 2 + x_skip;
        srcbuffer += w * 6;
        
        /* Use non-temporal stores - convert only output width */
        half_to_float_buffer_nt ((float*) out0, in0, out_w);
        half_to_float_buffer_nt ((float*) out1, in1, out_w);
        half_to_float_buffer_nt ((float*) out2, in2, out_w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__SSE__) || defined(_M_AMD64))
    /* Ensure all streaming stores are visible */
    _mm_sfence ();
#endif

    return EXR_ERR_SUCCESS;
}

static exr_result_t
unpack_half_to_float_4chan_planar_nt (exr_decode_pipeline_t* decode)
{
    /* Same as unpack_half_to_float_4chan_planar but uses NT stores */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h, out_w, x_skip;
    int             linc0, linc1, linc2, linc3;

    w      = decode->channels[0].width;
    h      = decode->chunk.height - decode->user_line_end_ignore;
    x_skip = decode->user_pixel_begin_skip;
    out_w  = w - x_skip - decode->user_pixel_end_ignore;
    
    /* If no X crop, use full width */
    if (out_w <= 0 || out_w > w) out_w = w;
    if (x_skip < 0) x_skip = 0;

    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    const size_t line_bytes = (size_t)w * 8;
    srcbuffer += decode->user_line_begin_skip * w * 8;

    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        /* Prefetch next line(s) into L1 cache */
        if (y + EXR_PREFETCH_DISTANCE < h)
        {
            const uint8_t* prefetch_addr = srcbuffer + EXR_PREFETCH_DISTANCE * line_bytes;
            EXR_PREFETCH_T0(prefetch_addr);
            EXR_PREFETCH_T0(prefetch_addr + 64);
            EXR_PREFETCH_T0(prefetch_addr + 128);
            EXR_PREFETCH_T0(prefetch_addr + 192);
        }

        /* Apply X skip to input pointers */
        in0 = (const uint16_t*) srcbuffer + x_skip;
        in1 = (const uint16_t*) srcbuffer + w + x_skip;
        in2 = (const uint16_t*) srcbuffer + w * 2 + x_skip;
        in3 = (const uint16_t*) srcbuffer + w * 3 + x_skip;
        srcbuffer += w * 8;
        
        /* Use non-temporal stores - convert only output width */
        half_to_float_buffer_nt ((float*) out0, in0, out_w);
        half_to_float_buffer_nt ((float*) out1, in1, out_w);
        half_to_float_buffer_nt ((float*) out2, in2, out_w);
        half_to_float_buffer_nt ((float*) out3, in3, out_w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__SSE__) || defined(_M_AMD64))
    _mm_sfence ();
#endif

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        uint16_t* out = (uint16_t*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = one_to_native16 (in0[x]);
            out[1] = one_to_native16 (in1[x]);
            out[2] = one_to_native16 (in2[x]);
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[2].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        uint16_t* out = (uint16_t*) out0;

        in0 = (const uint16_t*) srcbuffer; // B
        in1 = in0 + w;                     // G
        in2 = in1 + w;                     // R

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = one_to_native16 (in2[x]);
            out[1] = one_to_native16 (in1[x]);
            out[2] = one_to_native16 (in0[x]);
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        float* out = (float*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in0[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in2[x]));
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[2].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        float* out = (float*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in2[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in0[x]));
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h;
    int             linc0, linc1, linc2;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    // planar output
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
                            /* specialise to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out0) + x) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out1) + x) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out2) + x) = one_to_native16 (in2[x]);
#else
        memcpy (out0, in0, (size_t) (w) * sizeof (uint16_t));
        memcpy (out1, in1, (size_t) (w) * sizeof (uint16_t));
        memcpy (out2, in2, (size_t) (w) * sizeof (uint16_t));
#endif
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h, out_w, x_skip;
    int             linc0, linc1, linc2;

    w      = decode->channels[0].width;
    h      = decode->chunk.height - decode->user_line_end_ignore;
    x_skip = decode->user_pixel_begin_skip;
    out_w  = w - x_skip - decode->user_pixel_end_ignore;
    
    /* If no X crop, use full width */
    if (out_w <= 0 || out_w > w) out_w = w;
    if (x_skip < 0) x_skip = 0;

    const size_t line_bytes = (size_t)w * 6;

    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    // planar output
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        /* Prefetch next line(s) into L1 cache */
        if (y + EXR_PREFETCH_DISTANCE < h)
        {
            const uint8_t* prefetch_addr = srcbuffer + EXR_PREFETCH_DISTANCE * line_bytes;
            EXR_PREFETCH_T0(prefetch_addr);
            EXR_PREFETCH_T0(prefetch_addr + 64);
            EXR_PREFETCH_T0(prefetch_addr + 128);
        }

        /* Apply X skip to input pointers */
        in0 = (const uint16_t*) srcbuffer + x_skip;
        in1 = (const uint16_t*) srcbuffer + w + x_skip;
        in2 = (const uint16_t*) srcbuffer + w * 2 + x_skip;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), advance by full line
        
        /* Convert only the output width */
        half_to_float_buffer ((float*) out0, in0, out_w);
        half_to_float_buffer ((float*) out1, in1, out_w);
        half_to_float_buffer ((float*) out2, in2, out_w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h;
    int             inc0, inc1, inc2;
    int             linc0, linc1, linc2;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    inc0  = decode->channels[0].user_pixel_stride;
    inc1  = decode->channels[1].user_pixel_stride;
    inc2  = decode->channels[2].user_pixel_stride;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 6;

    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out0 + x * inc0)) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out1 + x * inc1)) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out2 + x * inc2)) = one_to_native16 (in2[x]);
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;
    /* TODO: can do this with sse and do 2 outpixels at once */
    union
    {
        struct
        {
            uint16_t a;
            uint16_t b;
            uint16_t g;
            uint16_t r;
        };
        uint64_t allc;
    } combined;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        uint64_t* outall = (uint64_t*) out0;
        in0              = (const uint16_t*) srcbuffer;
        in1              = in0 + w;
        in2              = in1 + w;
        in3              = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            combined.a = one_to_native16 (in0[x]);
            combined.b = one_to_native16 (in1[x]);
            combined.g = one_to_native16 (in2[x]);
            combined.r = one_to_native16 (in3[x]);
            outall[x]  = combined.allc;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;
    /* TODO: can do this with sse and do 2 outpixels at once */
    union
    {
        struct
        {
            uint16_t r;
            uint16_t g;
            uint16_t b;
            uint16_t a;
        };
        uint64_t allc;
    } combined;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[3].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        uint64_t* outall = (uint64_t*) out0;
        in0              = (const uint16_t*) srcbuffer;
        in1              = in0 + w;
        in2              = in1 + w;
        in3              = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            combined.a = one_to_native16 (in0[x]);
            combined.b = one_to_native16 (in1[x]);
            combined.g = one_to_native16 (in2[x]);
            combined.r = one_to_native16 (in3[x]);
            outall[x]  = combined.allc;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        float* out = (float*) out0;
        in0        = (const uint16_t*) srcbuffer;
        in1        = in0 + w;
        in2        = in1 + w;
        in3        = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in0[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in2[x]));
            out[3] = half_to_float (one_to_native16 (in3[x]));
            out += 4;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[3].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    /* interleaving case, we can do this! */
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        float* out = (float*) out0;
        in0        = (const uint16_t*) srcbuffer;
        in1        = in0 + w;
        in2        = in1 + w;
        in3        = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in3[x]));
            out[1] = half_to_float (one_to_native16 (in2[x]));
            out[2] = half_to_float (one_to_native16 (in1[x]));
            out[3] = half_to_float (one_to_native16 (in0[x]));
            out += 4;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h;
    int             linc0, linc1, linc2, linc3;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    // planar output
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        in3 = in2 + w;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
                            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out0) + x) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out1) + x) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out2) + x) = one_to_native16 (in2[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out3) + x) = one_to_native16 (in3[x]);
#else
        memcpy (out0, in0, (size_t) (w) * sizeof (uint16_t));
        memcpy (out1, in1, (size_t) (w) * sizeof (uint16_t));
        memcpy (out2, in2, (size_t) (w) * sizeof (uint16_t));
        memcpy (out3, in3, (size_t) (w) * sizeof (uint16_t));
#endif
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h, out_w, x_skip;
    int             linc0, linc1, linc2, linc3;

    w      = decode->channels[0].width;
    h      = decode->chunk.height - decode->user_line_end_ignore;
    x_skip = decode->user_pixel_begin_skip;
    out_w  = w - x_skip - decode->user_pixel_end_ignore;
    
    /* If no X crop, use full width */
    if (out_w <= 0 || out_w > w) out_w = w;
    if (x_skip < 0) x_skip = 0;

    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    const size_t line_bytes = (size_t)w * 8;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += decode->user_line_begin_skip * w * 8;

    // planar output
    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        /* Prefetch next line(s) into L1 cache */
        if (y + EXR_PREFETCH_DISTANCE < h)
        {
            const uint8_t* prefetch_addr = srcbuffer + EXR_PREFETCH_DISTANCE * line_bytes;
            EXR_PREFETCH_T0(prefetch_addr);
            EXR_PREFETCH_T0(prefetch_addr + 64);
            EXR_PREFETCH_T0(prefetch_addr + 128);
            EXR_PREFETCH_T0(prefetch_addr + 192);
        }

        /* Apply X skip to input pointers */
        in0 = (const uint16_t*) srcbuffer + x_skip;
        in1 = (const uint16_t*) srcbuffer + w + x_skip;
        in2 = (const uint16_t*) srcbuffer + w * 2 + x_skip;
        in3 = (const uint16_t*) srcbuffer + w * 3 + x_skip;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), advance by full line

        /* Convert only the output width */
        half_to_float_buffer ((float*) out0, in0, out_w);
        half_to_float_buffer ((float*) out1, in1, out_w);
        half_to_float_buffer ((float*) out2, in2, out_w);
        half_to_float_buffer ((float*) out3, in3, out_w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h;
    int             inc0, inc1, inc2, inc3;
    int             linc0, linc1, linc2, linc3;

    w     = decode->channels[0].width;
    h     = decode->chunk.height - decode->user_line_end_ignore;
    inc0  = decode->channels[0].user_pixel_stride;
    inc1  = decode->channels[1].user_pixel_stride;
    inc2  = decode->channels[2].user_pixel_stride;
    inc3  = decode->channels[3].user_pixel_stride;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    /*
     * not actually using y in the loop, so just pre-increment
     * the srcbuffer for any skip
     */
    srcbuffer += w * decode->user_line_begin_skip * 8;

    for (int y = decode->user_line_begin_skip; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        in3 = in2 + w;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out0 + x * inc0)) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out1 + x * inc1)) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out2 + x * inc2)) = one_to_native16 (in2[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out3 + x * inc3)) = one_to_native16 (in3[x]);
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int            w, h, pixincrement;

    h = decode->chunk.height - decode->user_line_end_ignore;
    /*
     * if we have user_line_begin_skip, the user data pointer is at THAT
     * offset but our unpacked data is at y of '0' (well idx * height)
     */
    for (int c = 0; c < decode->channel_count; ++c)
    {
        exr_coding_channel_info_t* decc = (decode->channels + c);
        srcbuffer += decc->width * decode->user_line_begin_skip * 2;
    }
    h -= decode->user_line_begin_skip;

    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata        = decc->decode_to_ptr;
            w            = decc->width;
            pixincrement = decc->user_pixel_stride;
            cdata += (uint64_t) y * (uint64_t) decc->user_line_stride;
            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
            if (pixincrement == 2)
            {
                uint16_t*       tmp = (uint16_t*) cdata;
                const uint16_t* src = (const uint16_t*) srcbuffer;
                uint16_t*       end = tmp + w;

                while (tmp < end)
                    *tmp++ = one_to_native16 (*src++);
            }
            else
            {
                const uint16_t* src = (const uint16_t*) srcbuffer;
                for (int x = 0; x < w; ++x)
                {
                    *((uint16_t*) cdata) = one_to_native16 (*src++);
                    cdata += pixincrement;
                }
            }
#else
            if (pixincrement == 2)
            {
                memcpy (cdata, srcbuffer, (size_t) (w) * 2);
            }
            else
            {
                const uint16_t* src = (const uint16_t*) srcbuffer;
                for (int x = 0; x < w; ++x)
                {
                    *((uint16_t*) cdata) = *src++;
                    cdata += pixincrement;
                }
            }
#endif
            srcbuffer += w * 2;
        }
    }
    return EXR_ERR_SUCCESS;
}

//static exr_result_t unpack_32bit_3chan (exr_decode_pipeline_t* decode);
//static exr_result_t unpack_32bit_4chan (exr_decode_pipeline_t* decode);

static exr_result_t
unpack_32bit (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int64_t        w, h, pixincrement;
    int            chans = decode->channel_count;

    h = (int64_t) decode->chunk.height - decode->user_line_end_ignore;
    /*
     * if we have user_line_begin_skip, the user data pointer is at THAT
     * offset but our unpacked data is at y of '0' (well idx * height)
     */
    for (int c = 0; c < decode->channel_count; ++c)
    {
        exr_coding_channel_info_t* decc = (decode->channels + c);
        srcbuffer += decc->width * decode->user_line_begin_skip * 4;
    }
    h -= decode->user_line_begin_skip;

    for (int64_t y = 0; y < h; ++y)
    {
        for (int c = 0; c < chans; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata        = decc->decode_to_ptr;
            w            = decc->width;
            pixincrement = decc->user_pixel_stride;
            cdata += y * (int64_t) decc->user_line_stride;
            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
            if (pixincrement == 4)
            {
                uint32_t*       tmp = (uint32_t*) cdata;
                const uint32_t* src = (const uint32_t*) srcbuffer;
                uint32_t*       end = tmp + w;

                while (tmp < end)
                    *tmp++ = le32toh (*src++);
            }
            else
            {
                const uint32_t* src = (const uint32_t*) srcbuffer;
                for (int64_t x = 0; x < w; ++x)
                {
                    *((uint32_t*) cdata) = le32toh (*src++);
                    cdata += pixincrement;
                }
            }
#else
            if (pixincrement == 4)
            {
                memcpy (cdata, srcbuffer, (size_t) (w) * 4);
            }
            else
            {
                const uint32_t* src = (const uint32_t*) srcbuffer;
                for (int64_t x = 0; x < w; ++x)
                {
                    *((uint32_t*) cdata) = *src++;
                    cdata += pixincrement;
                }
            }
#endif
            srcbuffer += w * 4;
        }
    }
    return EXR_ERR_SUCCESS;
}

#define UNPACK_HALF_TO_HALF_SAMPLES(samps)                                     \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint16_t*) cdata) = unaligned_load16 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_HALF_TO_FLOAT_SAMPLES(samps)                                    \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint16_t cval = unaligned_load16 (src);                \
                        ++src;                                                 \
                        *((float*) cdata) = half_to_float (cval);              \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_HALF_TO_UINT_SAMPLES(samps)                                     \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint16_t cval = unaligned_load16 (src);                \
                        ++src;                                                 \
                        *((uint32_t*) cdata) = half_to_uint (cval);            \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_FLOAT_TO_HALF_SAMPLES(samps)                                    \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint16_t*) cdata) = float_to_half_int (fint);       \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_FLOAT_TO_FLOAT_SAMPLES(samps)                                   \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint32_t*) cdata) = unaligned_load32 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_FLOAT_TO_UINT_SAMPLES(samps)                                    \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint32_t*) cdata) = float_to_uint_int (fint);       \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_UINT_TO_HALF_SAMPLES(samps)                                     \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint16_t*) cdata) = uint_to_half (fint);            \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_UINT_TO_FLOAT_SAMPLES(samps)                                    \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((float*) cdata) = uint_to_float (fint);              \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_UINT_TO_UINT_SAMPLES(samps)                                     \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint32_t*) cdata) = unaligned_load32 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }

#define UNPACK_SAMPLES(samps)                                                  \
    switch (decc->data_type)                                                   \
    {                                                                          \
        case EXR_PIXEL_HALF:                                                   \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    UNPACK_HALF_TO_HALF_SAMPLES(samps)                         \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    UNPACK_HALF_TO_FLOAT_SAMPLES(samps)                        \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    UNPACK_HALF_TO_UINT_SAMPLES(samps)                         \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        case EXR_PIXEL_FLOAT:                                                  \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    UNPACK_FLOAT_TO_HALF_SAMPLES(samps)                        \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    UNPACK_FLOAT_TO_FLOAT_SAMPLES(samps)                       \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    UNPACK_FLOAT_TO_UINT_SAMPLES(samps)                        \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        case EXR_PIXEL_UINT:                                                   \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    UNPACK_UINT_TO_HALF_SAMPLES(samps)                         \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    UNPACK_UINT_TO_FLOAT_SAMPLES(samps)                        \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    UNPACK_UINT_TO_UINT_SAMPLES(samps)                         \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        default: return EXR_ERR_INVALID_ARGUMENT;                              \
    }

static exr_result_t
generic_unpack (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int            w, h, bpc, ubpc, uls;

    uls = decode->user_line_begin_skip;
    h = decode->chunk.height - decode->user_line_end_ignore;
    /*
     * user data starts at user line begin skip but because of
     * y_samples > 1 case, need to embed the run the loop and skip,
     * incrementing srcbuffer only when there'd be a line...
     */
    for (int y = 0; y < h; ++y)
    {
        int cury = (int)( (int64_t) y +
                          (int64_t) decode->chunk.start_y );

        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata = decc->decode_to_ptr;
            w     = decc->width;
            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_pixel_stride;

            /* avoid a mod operation if we can */
            if (decc->y_samples > 1)
            {
                if ((cury % decc->y_samples) != 0) continue;
                if (y < uls || !cdata)
                {
                    srcbuffer += w * bpc;
                    continue;
                }

                cdata +=
                    ((uint64_t) ((y - uls) / decc->y_samples) *
                     (uint64_t) decc->user_line_stride);
            }
            else
            {
                if (y < uls || !cdata)
                {
                    srcbuffer += w * bpc;
                    continue;
                }

                cdata += ((uint64_t) (y - uls)) * ((uint64_t) decc->user_line_stride);
            }

            UNPACK_SAMPLES (w)
            srcbuffer += w * bpc;
        }
    }
    return EXR_ERR_SUCCESS;
}

#define PREPARE_SAMPLES(sampbuffer, prevsamps, decode)              \
                int32_t samps = sampbuffer[x];                      \
                if (0 == (decode->decode_flags &                    \
                          EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))  \
                {                                                   \
                    int32_t tmp = samps - prevsamps;                \
                    prevsamps   = samps;                            \
                    samps       = tmp;                              \
                }

static exr_result_t
generic_unpack_deep_pointers (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer  = decode->unpacked_buffer;
    const int32_t* sampbuffer = decode->sample_count_table;
    void**         pdata;
    int            w, h, bpc, ubpc, uls;

    w   = decode->chunk.width;
    h   = decode->chunk.height - decode->user_line_end_ignore;
    /* for user line skip, we use y in the loop so account for that */
    uls = decode->user_line_begin_skip;
    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc      = (decode->channels + c);
            int32_t                    prevsamps = 0;
            size_t                     pixstride;
            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_bytes_per_element;
            pdata = (void**) decc->decode_to_ptr;

            if (y < uls || !pdata)
            {
                prevsamps = 0;
                if ((decode->decode_flags &
                     EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    for (int x = 0; x < w; ++x)
                        prevsamps += sampbuffer[x];
                }
                else
                    prevsamps = sampbuffer[w - 1];
                srcbuffer += ((size_t) bpc) * ((size_t) prevsamps);
                continue;
            }

            pdata += ((size_t) y - uls) *
                     (((size_t) decc->user_line_stride) / sizeof (void*));
            pixstride = ((size_t) decc->user_pixel_stride) / sizeof (void*);


            switch (decc->data_type)
            {
                case EXR_PIXEL_HALF:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_HALF_TO_HALF_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_HALF_TO_FLOAT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_HALF_TO_UINT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_FLOAT:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_FLOAT_TO_HALF_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_FLOAT_TO_FLOAT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_FLOAT_TO_UINT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_UINT:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_UINT_TO_HALF_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_UINT_TO_FLOAT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                void*   outpix = *pdata;
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                pdata += pixstride;
                                if (outpix)
                                {
                                    uint8_t* cdata = outpix;
                                    UNPACK_UINT_TO_UINT_SAMPLES(samps)
                                }
                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                default: return EXR_ERR_INVALID_ARGUMENT;
            }
        }
        sampbuffer += w;
    }
    return EXR_ERR_SUCCESS;
}

static exr_result_t
generic_unpack_deep (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer  = decode->unpacked_buffer;
    const int32_t* sampbuffer = decode->sample_count_table;
    uint8_t*       cdata;
    int            w, h, bpc, ubpc, uls;
    size_t         totsamps = 0;

    w = decode->chunk.width;
    h = decode->chunk.height - decode->user_line_end_ignore;

    /* for user line skip, we use y in the loop so account for that */
    uls = decode->user_line_begin_skip;

    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc      = (decode->channels + c);
            int32_t                    prevsamps = 0;

            int incr_tot = (y >= uls && ((c + 1) == decode->channel_count));

            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_bytes_per_element;
            cdata = decc->decode_to_ptr;

            if (!cdata)
            {
                prevsamps = 0;
                if ((decode->decode_flags &
                     EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    for (int x = 0; x < w; ++x)
                        prevsamps += sampbuffer[x];
                }
                else
                    prevsamps = sampbuffer[w - 1];

                srcbuffer += ((size_t) bpc) * ((size_t) prevsamps);

                if (incr_tot) totsamps += (size_t) prevsamps;

                continue;
            }

            cdata += totsamps * ((size_t) ubpc);

            switch (decc->data_type)
            {
                case EXR_PIXEL_HALF:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_HALF_TO_HALF_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_HALF_TO_FLOAT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_HALF_TO_UINT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_FLOAT:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_FLOAT_TO_HALF_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_FLOAT_TO_FLOAT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_FLOAT_TO_UINT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_UINT:
                    switch (decc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_UINT_TO_HALF_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_UINT_TO_FLOAT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            for (int x = 0; x < w; ++x)
                            {
                                PREPARE_SAMPLES (sampbuffer, prevsamps, decode)
                                UNPACK_UINT_TO_UINT_SAMPLES(samps)

                                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                                if (incr_tot) totsamps += (size_t) samps;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                default: return EXR_ERR_INVALID_ARGUMENT;
            }
        }
        sampbuffer += w;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

internal_exr_unpack_fn
internal_exr_match_decode (
    exr_decode_pipeline_t* decode,
    int                    isdeep,
    int                    chanstofill,
    int                    chanstounpack,
    int                    sametype,
    int                    sameouttype,
    int                    samebpc,
    int                    sameoutbpc,
    int                    hassampling,
    int                    hastypechange,
    int                    sameoutinc,
    int                    simpinterleave,
    int                    simpinterleaverev,
    int                    simplineoff)
{
#ifdef EXR_HAS_STD_ATOMICS
    static atomic_int init_cpu_check = 1;
#else
    static int init_cpu_check = 1;
#endif
    if (init_cpu_check)
    {
        choose_half_to_float_impl ();
        init_cpu_check = 0;
    }

    if (isdeep)
    {
        if ((decode->decode_flags & EXR_DECODE_NON_IMAGE_DATA_AS_POINTERS))
            return &generic_unpack_deep_pointers;
        return &generic_unpack_deep;
    }

    /* Check if non-temporal writes are requested */
    int use_nt = (decode->decode_flags & EXR_DECODE_NON_TEMPORAL_WRITES) != 0;

    if (hastypechange > 0)
    {
        /* other optimizations would not be difficult, but this will
         * be the common one (where on encode / pack we want to do the
         * opposite) */
        if (!hassampling &&
            chanstofill == decode->channel_count &&
            sametype == (int) EXR_PIXEL_HALF &&
            sameouttype == (int) EXR_PIXEL_FLOAT)
        {
            if (simpinterleave > 0)
            {
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_interleave;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_interleave;
            }

            if (simpinterleaverev > 0)
            {
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_interleave_rev;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_interleave_rev;
            }

            if (sameoutinc == 4)
            {
                /* Use non-temporal versions if requested */
                if (use_nt)
                {
                    if (decode->channel_count == 4)
                        return &unpack_half_to_float_4chan_planar_nt;
                    if (decode->channel_count == 3)
                        return &unpack_half_to_float_3chan_planar_nt;
                }
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_planar;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_planar;
            }
        }

        return &generic_unpack;
    }

    if (hassampling || chanstofill != decode->channel_count || samebpc <= 0 ||
        sameoutbpc <= 0)
        return &generic_unpack;

    (void) chanstounpack;
    (void) simplineoff;

    if (samebpc == 2)
    {
        if (simpinterleave > 0)
        {
            if (decode->channel_count == 4)
                return &unpack_16bit_4chan_interleave;
            if (decode->channel_count == 3)
                return &unpack_16bit_3chan_interleave;
        }

        if (simpinterleaverev > 0)
        {
            if (decode->channel_count == 4)
                return &unpack_16bit_4chan_interleave_rev;
            if (decode->channel_count == 3)
                return &unpack_16bit_3chan_interleave_rev;
        }

        if (sameoutinc == 2)
        {
            if (decode->channel_count == 4) return &unpack_16bit_4chan_planar;
            if (decode->channel_count == 3) return &unpack_16bit_3chan_planar;
        }

        if (decode->channel_count == 4) return &unpack_16bit_4chan;
        if (decode->channel_count == 3) return &unpack_16bit_3chan;

        return &unpack_16bit;
    }

    if (samebpc == 4)
    {
        //if (decode->channel_count == 4) return &unpack_32bit_4chan;
        //if (decode->channel_count == 3) return &unpack_32bit_3chan;
        return &unpack_32bit;
    }

    return &generic_unpack;
}

/**************************************/

/*
 * External interface for half-to-float conversion.
 * Uses SIMD when available for optimal performance.
 */
void
internal_half_to_float_buffer (float* out, const uint16_t* in, int count)
{
#if defined(USE_F16C_INTRINSICS)
    half_to_float_buffer (out, in, count);
#elif defined(ENABLE_F16C_TEST)
    /* half_to_float_buffer is a function pointer that gets set up
     * at runtime based on CPU capabilities */
    half_to_float_buffer (out, in, count);
#else
    for (int i = 0; i < count; ++i)
        out[i] = half_to_float (in[i]);
#endif
}

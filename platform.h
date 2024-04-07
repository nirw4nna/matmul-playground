#pragma once

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// Once we will have an implementation that works well on AVX2 with FP32 the next step would be
// to add support for more architectures and data types. The good news is that, since we are using
// vector registers, we know based on the supported extensions how many there are.
// This means we can probably get away with hard-coding the values for MR and NR (register blocking factors).
#if !defined(__AVX2__)
#   error "AVX2 is not supported"
#else
#   include <immintrin.h>
#   define f32x8 __m256
#endif

#define INLINE      inline __attribute__((always_inline))
#define NOINLINE    __attribute__((__noinline__))

#define MIN(x, y)  ((x) < (y) ? (x) : (y))

// These are the parameters that control the cache blocking. They have to be computed based on the HW
// on which the kernel is supposed to work. It's also possible to provide sane defaults
// but this requires knowing, at the very least, the architecture of the CPU.
#if !defined(MC)
#   define MC   264
#endif

#if !defined(NC)
#   define NC   2016
#endif

#if !defined(KC)
#   define KC   336
#endif


#if defined(__cplusplus)
extern "C" {
#endif

    using size = ptrdiff_t;
    using u32 = uint32_t;
    using i64 = long long;
    using f32 = float;
    using f64 = double;

    static INLINE f64 now() {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
    }

#if defined(__cplusplus)
}
#endif
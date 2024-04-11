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
#   define MC   128
#endif

#if !defined(NC)
#   define NC   2820
#endif

#if !defined(KC)
#   define KC   384
#endif


#define axpy_8x12(A, B, idx)                                    \
    do {                                                        \
        const f32x8 alpha_p = _mm256_loadu_ps(&A[(idx) * 8]);   \
\
        /* Broadcast beta_0 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 0]);      \
        gamma_0 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_0);   \
\
        /* Broadcast beta_1 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 1]);      \
        gamma_1 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_1);   \
\
        /* Broadcast beta_2 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 2]);      \
        gamma_2 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_2);   \
\
        /* Broadcast beta_3 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 3]);      \
        gamma_3 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_3);   \
\
        /* Broadcast beta_4 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 4]);      \
        gamma_4 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_4);   \
\
        /* Broadcast beta_5 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 5]);      \
        gamma_5 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_5);   \
\
        /* Broadcast beta_6 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 6]);      \
        gamma_6 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_6);   \
\
        /* Broadcast beta_7 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 7]);      \
        gamma_7 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_7);   \
\
        /* Broadcast beta_8 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 8]);      \
        gamma_8 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_8);   \
\
        /* Broadcast beta_9 */                                  \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 9]);      \
        gamma_9 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_9);   \
\
        /* Broadcast beta_10 */                                 \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 10]);     \
        gamma_10 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_10); \
\
        /* Broadcast beta_11 */                                 \
        beta_pj = _mm256_broadcast_ss(&B[(idx) * 12 + 11]);     \
        gamma_11 = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_11); \
    } while(0)


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

    static INLINE void pack_B_KCxNR(const u32 k, const u32 n, const u32 nr,
                                    const f32 *__restrict b, const u32 ldb,
                                    f32 *__restrict b_tilde) noexcept {
        for (u32 p = 0; p < k; ++p) {
            for (u32 j = 0; j < n; ++j) {
                *b_tilde++ = b[j * ldb + p];
            }
            for (u32 j = n; j < nr; ++j) {
                *b_tilde++ = 0.f;
            }
        }
    }

    static INLINE void pack_B(const u32 k, const u32 n, const u32 nr,
                              const f32 *__restrict b, const u32 ldb, 
                              f32 *__restrict b_tilde) noexcept {
        for (u32 j = 0; j < n; j += nr) {

            const u32 jb = MIN(n - j, nr);

            pack_B_KCxNR(k, jb, nr, &b[j * ldb], ldb, b_tilde);
            b_tilde += k * nr;
        }
    }

    static INLINE void pack_A_MRxKC(const u32 k, const u32 m, const u32 mr,
                                    const f32 *__restrict a, const u32 lda,
                                    f32 *__restrict a_tilde) noexcept {
        for (u32 p = 0; p < k; ++p) {
            for (u32 i = 0; i < m; ++i) {
                *a_tilde++ = a[i * lda + p];
            }
            for (u32 i = m; i < mr; ++i) {
                *a_tilde++ = 0.f;
            }
        }
    }

    static INLINE void pack_A(const u32 m, const u32 k, const u32 mr,
                              const f32 *__restrict a, const u32 lda,
                              f32 *__restrict a_tilde) noexcept {
        for (u32 i = 0; i < m; i += mr) {

            const u32 ib = MIN(m - i, mr);

            pack_A_MRxKC(k, ib, mr, &a[i * lda], lda, a_tilde);
            a_tilde += k * mr;
        }

    }

#if defined(__cplusplus)
}
#endif
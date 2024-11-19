// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#include "gemm_8x12_unroll4_parallel1.h"

static INLINE void ukernel_8x12(const u32 k,
                                const f32 *__restrict a,
                                const f32 *__restrict b,
                                f32 *__restrict c, const u32 ldc) noexcept {
    f32x8 gamma_0 = _mm256_loadu_ps(&c[0 * ldc]);
    f32x8 gamma_1 = _mm256_loadu_ps(&c[1 * ldc]);
    f32x8 gamma_2 = _mm256_loadu_ps(&c[2 * ldc]);
    f32x8 gamma_3 = _mm256_loadu_ps(&c[3 * ldc]);
    f32x8 gamma_4 = _mm256_loadu_ps(&c[4 * ldc]);
    f32x8 gamma_5 = _mm256_loadu_ps(&c[5 * ldc]);
    f32x8 gamma_6 = _mm256_loadu_ps(&c[6 * ldc]);
    f32x8 gamma_7 = _mm256_loadu_ps(&c[7 * ldc]);
    f32x8 gamma_8 = _mm256_loadu_ps(&c[8 * ldc]);
    f32x8 gamma_9 = _mm256_loadu_ps(&c[9 * ldc]);
    f32x8 gamma_10 = _mm256_loadu_ps(&c[10 * ldc]);
    f32x8 gamma_11 = _mm256_loadu_ps(&c[11 * ldc]);

    f32x8 beta_pj;

    const u32 pb = (k / 4) * 4;
    for (u32 p = 0; p < pb; p += 4) {
        rank1_8x12(a, b, p + 0);
        rank1_8x12(a, b, p + 1);
        rank1_8x12(a, b, p + 2);
        rank1_8x12(a, b, p + 3);
    }

    for (u32 p = pb; p < k; ++p) {
        rank1_8x12(a, b, p);
    }
    
    _mm256_storeu_ps(&c[0 * ldc], gamma_0);
    _mm256_storeu_ps(&c[1 * ldc], gamma_1);
    _mm256_storeu_ps(&c[2 * ldc], gamma_2);
    _mm256_storeu_ps(&c[3 * ldc], gamma_3);
    _mm256_storeu_ps(&c[4 * ldc], gamma_4);
    _mm256_storeu_ps(&c[5 * ldc], gamma_5);
    _mm256_storeu_ps(&c[6 * ldc], gamma_6);
    _mm256_storeu_ps(&c[7 * ldc], gamma_7);
    _mm256_storeu_ps(&c[8 * ldc], gamma_8);
    _mm256_storeu_ps(&c[9 * ldc], gamma_9);
    _mm256_storeu_ps(&c[10 * ldc], gamma_10);
    _mm256_storeu_ps(&c[11 * ldc], gamma_11);
}

static INLINE void inner_loop(const u32 m, const u32 n, const u32 k,
                              const f32 *__restrict a,
                              const f32 *__restrict b,
                              f32 *__restrict c, const u32 ldc) noexcept {

    for (u32 j = 0; j < n; j += 12) {
        const u32 jb = MIN(n - j, 12);
        #pragma omp parallel for
        for (u32 i = 0; i < m; i += 8) {
            alignas(32) f32 c_tilde[8 * 12];
            const u32 ib = MIN(m - i, 8);
            if (ib == 8 && jb == 12) {
                ukernel_8x12(k, &a[i * k], &b[j * k], &c[j * ldc + i], ldc);
            } else {
                memset(c_tilde, 0, 12 * 8 * sizeof(f32));
                ukernel_8x12(k, &a[i * k], &b[j * k], c_tilde, 8);
                for (u32 jj = 0; jj < jb; ++jj) {
                    for (u32 ii = 0; ii < ib; ++ii) {
                        c[(j + jj) * ldc + (i + ii)] += c_tilde[jj * 8 + ii];
                    }
                }
            }
        }
    }
}

static INLINE void kernel(const u32 m, const u32 n, const u32 k,
                          const f32 *__restrict a, const u32 lda,
                          const f32 *__restrict b, const u32 ldb,
                          f32 *__restrict c, const u32 ldc) noexcept {
    alignas(32) f32 a_tilde[MC * KC];
    alignas(32) f32 b_tilde[KC * NC];

    for (u32 p = 0; p < k; p += KC) {
        const u32 pb = MIN(k - p, KC);

        pack_B(pb, n, 12, &b[p], ldb, b_tilde);

        for (u32 i = 0; i < m; i += MC) {
            const u32 ib = MIN(m - i, MC);

            pack_A(ib, pb, 8, &a[i * lda + p], lda, a_tilde);
            inner_loop(ib, n, pb, a_tilde, b_tilde, &c[i], ldc);
        }
    }
}

f64 gemm_8x12_unroll4_parallel1(const u32 m, const u32 n, const u32 k,
                                const f32 *__restrict a, const u32 lda,
                                const f32 *__restrict b, const u32 ldb,
                                f32 *__restrict c, const u32 ldc) noexcept {
    const f64 start = now();
    
    for (u32 j = 0; j < n; j += NC) {
        const u32 jb = MIN(n - j, NC);
        kernel(m, jb, k, a, lda, &b[j * ldb], ldb, &c[j * ldc], ldc);
    }

    return now() - start;
}

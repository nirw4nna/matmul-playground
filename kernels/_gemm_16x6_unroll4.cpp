// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#include "gemm_16x6_unroll4.h"

static INLINE void ukernel_16x6(const u32 k,
                                const f32 *__restrict a,
                                const f32 *__restrict b,
                                f32 *__restrict c, const u32 ldc) noexcept {
    f32x8 gamma_07_0 = _mm256_loadu_ps(&c[0 * ldc]);
    f32x8 gamma_07_1 = _mm256_loadu_ps(&c[1 * ldc]);
    f32x8 gamma_07_2 = _mm256_loadu_ps(&c[2 * ldc]);
    f32x8 gamma_07_3 = _mm256_loadu_ps(&c[3 * ldc]);
    f32x8 gamma_07_4 = _mm256_loadu_ps(&c[4 * ldc]);
    f32x8 gamma_07_5 = _mm256_loadu_ps(&c[5 * ldc]);
    
    f32x8 gamma_815_0 = _mm256_loadu_ps(&c[0 * ldc + 8]);
    f32x8 gamma_815_1 = _mm256_loadu_ps(&c[1 * ldc + 8]);
    f32x8 gamma_815_2 = _mm256_loadu_ps(&c[2 * ldc + 8]);
    f32x8 gamma_815_3 = _mm256_loadu_ps(&c[3 * ldc + 8]);
    f32x8 gamma_815_4 = _mm256_loadu_ps(&c[4 * ldc + 8]);
    f32x8 gamma_815_5 = _mm256_loadu_ps(&c[5 * ldc + 8]);
    
    f32x8 beta_pj;

    // todo: unroll
    for (u32 p = 0; p < k; ++p) {
        const f32x8 alpha_p_07 = _mm256_loadu_ps(&a[(p + 0) * 8]);
        const f32x8 alpha_p_815 = _mm256_loadu_ps(&a[(p + 1) * 8]);

        _mm_prefetch(&b[p + 12], _MM_HINT_T0);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 0]);
        gamma_07_0 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_0);
        gamma_815_0 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_0);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 1]);
        gamma_07_1 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_1);
        gamma_815_1 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_1);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 2]);
        gamma_07_2 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_2);
        gamma_815_2 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_2);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 3]);
        gamma_07_3 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_3);
        gamma_815_3 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_3);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 4]);
        gamma_07_4 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_4);
        gamma_815_4 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_4);

        beta_pj = _mm256_broadcast_ss(&b[(p) * 6 + 5]);
        gamma_07_5 = _mm256_fmadd_ps(alpha_p_07, beta_pj, gamma_07_5);
        gamma_815_5 = _mm256_fmadd_ps(alpha_p_815, beta_pj, gamma_815_5);
    }
    
    _mm256_storeu_ps(&c[0 * ldc], gamma_07_0);
    _mm256_storeu_ps(&c[0 * ldc + 8], gamma_815_0);
    _mm256_storeu_ps(&c[1 * ldc], gamma_07_1);
    _mm256_storeu_ps(&c[1 * ldc + 8], gamma_815_1);
    _mm256_storeu_ps(&c[2 * ldc], gamma_07_2);
    _mm256_storeu_ps(&c[2 * ldc + 8], gamma_815_2);
    _mm256_storeu_ps(&c[3 * ldc], gamma_07_3);
    _mm256_storeu_ps(&c[3 * ldc + 8], gamma_815_3);
    _mm256_storeu_ps(&c[4 * ldc], gamma_07_4);
    _mm256_storeu_ps(&c[4 * ldc + 8], gamma_815_4);
    _mm256_storeu_ps(&c[5 * ldc], gamma_07_5);
    _mm256_storeu_ps(&c[5 * ldc + 8], gamma_815_5);
}

static INLINE void inner_loop(const u32 m, const u32 n, const u32 k,
                              const f32 *__restrict a,
                              const f32 *__restrict b,
                              f32 *__restrict c, const u32 ldc) noexcept {
    alignas(32) f32 c_tilde[16 * 6];

    for (u32 j = 0; j < n; j += 6) {
        _mm_prefetch(&b[j * k], _MM_HINT_T0);
        const u32 jb = MIN(n - j, 6);
        for (u32 i = 0; i < m; i += 16) {
            _mm_prefetch(&c[(j + 0) * ldc + i], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 1) * ldc + i], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 2) * ldc + i], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 3) * ldc + i], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 4) * ldc + i], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 5) * ldc + i], _MM_HINT_T0);

            const u32 ib = MIN(m - i, 16);
            if (ib == 16 && jb == 6) {
                ukernel_16x6(k, &a[i * k], &b[j * k], &c[j * ldc + i], ldc);
            } else {
                memset(c_tilde, 0, 16 * 6 * sizeof(f32));
                ukernel_16x6(k, &a[i * k], &b[j * k], c_tilde, 16);
                for (u32 jj = 0; jj < jb; ++jj) {
                    for (u32 ii = 0; ii < ib; ++ii) {
                        c[(j + jj) * ldc + (i + ii)] += c_tilde[jj * 16 + ii];
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

        pack_B(pb, n, 6, &b[p], ldb, b_tilde);

        for (u32 i = 0; i < m; i += MC) {
            const u32 ib = MIN(m - i, MC);

            pack_A(ib, pb, 16, &a[i * lda + p], lda, a_tilde);
            inner_loop(ib, n, pb, a_tilde, b_tilde, &c[i], ldc);
        }
    }
}

f64 gemm_16x6_unroll4(const u32 m, const u32 n, const u32 k,
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

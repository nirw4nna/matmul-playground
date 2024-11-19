// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#include "gemm_jpi.h"

f64 gemm_jpi(const u32 m, const u32 n, const u32 k,
             const f32 *__restrict a, const u32 lda,
             const f32 *__restrict b, const u32 ldb,
             f32 *__restrict c, const u32 ldc) noexcept {
    const f64 start = now();
    for (u32 j = 0; j < n; ++j) {
        for (u32 p = 0; p < k; ++p) {
            for (u32 i = 0; i < m; ++i) {
                // Compute Cij += Aip * Bpj
                // Since we are computing C = A.T * B the matrix A must be accessed as if it was stored in row-major.
                c[j * ldc + i] += a[i * lda + p] * b[j * ldb + p];
            }
        }
    }
    return now() - start;                
}

#pragma once

#include "../platform.h"

#if defined(__cplusplus)
extern "C" {
#endif

    // Baseline implementation. Computes C = A.T x B
    extern f64 gemm_jpi(u32 m, u32 n, u32 k,
                        const f32 *__restrict a, u32 lda,
                        const f32 *__restrict b, u32 ldb,
                        f32 *__restrict c, u32 ldc) noexcept;

#if defined(__cplusplus)
}
#endif

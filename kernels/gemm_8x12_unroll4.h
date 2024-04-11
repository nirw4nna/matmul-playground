#pragma once

#include "../platform.h"

#if defined(__cplusplus)
extern "C" {
#endif

    extern f64 gemm_8x12_unroll4(u32 m, u32 n, u32 k,
                                 const f32 *__restrict a, u32 lda,
                                 const f32 *__restrict b, u32 ldb,
                                 f32 *__restrict c, u32 ldc) noexcept;

#if defined(__cplusplus)
}
#endif

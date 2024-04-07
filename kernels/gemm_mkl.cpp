#include "gemm_mkl.h"
#include <mkl_blas.h>

f64 gemm_mkl(u32 m, u32 n, u32 k,
             const f32 *__restrict a, u32 lda,
             const f32 *__restrict b, u32 ldb,
             f32 *__restrict c, u32 ldc) noexcept {
    const f32 alpha = 1.f;
    const f32 beta = 0.f;
    const i64 m_mkl = m, n_mkl = n, k_mkl = k;
    const i64 lda_mkl = lda, ldb_mkl = ldb, ldc_mkl = ldc;

    const f64 start = now();
    SGEMM("T", "N", &m_mkl, &n_mkl, &k_mkl, &alpha, a, &lda_mkl, b, &ldb_mkl, &beta, c, &ldc_mkl);
    return now() - start;
}

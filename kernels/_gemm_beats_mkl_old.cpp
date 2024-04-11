#include "gemm_beats_mkl_old.h"
#include <atomic>

#undef KC
#undef NC
#undef MC

// ==================== https://github.com/jart/matmul/varith.h ==================== //
inline float add(float x, float y) { return x + y; }
inline float sub(float x, float y) { return x - y; }
inline float mul(float x, float y) { return x * y; }

#ifdef __SSE__
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE__

#ifdef __AVX__
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif // __AVX__

#ifdef __AVX512F__
inline __m512 add(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
inline __m512 sub(__m512 x, __m512 y) { return _mm512_sub_ps(x, y); }
inline __m512 mul(__m512 x, __m512 y) { return _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#ifdef __ARM_NEON
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
#endif // __ARM_NEON

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8_t add(float16x8_t x, float16x8_t y) { return vaddq_f16(x, y); }
inline float16x8_t sub(float16x8_t x, float16x8_t y) { return vsubq_f16(x, y); }
inline float16x8_t mul(float16x8_t x, float16x8_t y) { return vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

/**
 * Computes a * b + c.
 *
 * This operation will become fused into a single arithmetic instruction
 * if the hardware has support for this feature, e.g. Intel Haswell+ (c.
 * 2013), AMD Bulldozer+ (c. 2011), etc.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

/**
 * Computes a * b + c with error correction.
 *
 * @see W. Kahan, "Further remarks on reducing truncation errors,"
 *    Communications of the ACM, vol. 8, no. 1, p. 40, Jan. 1965,
 *    doi: 10.1145/363707.363723.
 */
template <typename T, typename U>
inline U madder(T a, T b, U c, U *e) {
    U y = sub(mul(a, b), *e);
    U t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}
// ================================================================================= //

// ===================== https://github.com/jart/matmul/load.h ===================== //
/**
 * Google Brain 16-bit floating point number.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * So be warned that converting between them, destroys several bits.
 *
 * @see IEEE 754-2008
 */
typedef struct {
  uint16_t x;
} ggml_bf16_t;

/**
 * Converts brain16 to float32.
 */
static inline float ggml_bf16_to_fp32(ggml_bf16_t h) {
  union {
    float f;
    uint32_t i;
  } u;
  u.i = (uint32_t)h.x << 16;
  return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This function is binary identical to AMD Zen4 VCVTNEPS2BF16.
 * Subnormals shall be flushed to zero, and NANs will be quiet.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_fp32_to_bf16(float s) {
  ggml_bf16_t h;
  union {
    float f;
    uint32_t i;
  } u;
  u.f = s;
  if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
    h.x = (u.i >> 16) | 64; /* force to quiet */
    return h;
  }
  if (!(u.i & 0x7f800000)) { /* subnormal */
    h.x = (u.i & 0x80000000) >> 16; /* flush to zero */
    return h;
  }
  h.x = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
  return h;
}

template <typename T, typename U> T load(const U *);

template <> inline float load(const float *p) {
    return *p;
}

#if defined(__ARM_NEON)
template <> inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__AVX512F__)
template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <> inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <> inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif

#if defined(__AVX512BF16__) && defined(__AVX512VL__)
template <> inline __m512 load(const ggml_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif
// ================================================================================= //


// ===================== https://github.com/jart/matmul/hsum.h ===================== //
inline float hsum(float x) {
    return x;
}

#ifdef __ARM_NEON
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float hsum(float16x8_t x) {
    float32x4_t t = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t u = vcvt_f32_f16(vget_high_f16(x));
    return vaddvq_f32(vaddq_f32(t, u));
}
#endif

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
#else
    __m128 shuf = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#endif
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}
#endif

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif
// ================================================================================= //


template <int KN, typename T, typename TA, typename TB, typename TC> struct tinyBLAS {
  public:
    tinyBLAS(const TA *A, i64 lda, const TB *B, i64 ldb, TC *C, i64 ldc, int nth)
        : A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), nth(nth) {
    }

    NOINLINE void matmul(i64 m, i64 n, i64 k, int ith) {
        if (!m || !n)
            return;
        zeroify(m, n, ith);
        if (!k)
            return;
        mnpack(0, m, 0, n, k, ith);
    }

  private:
    NOINLINE void zeroify(i64 m, i64 n, int ith) {
        int duty = (n + nth - 1) / nth;
        if (duty < 1)
            duty = 1;
        int start = duty * ith;
        int end = start + duty;
        if (end > n)
            end = n;
        for (int j = start; j < end; ++j)
            memset(C + ldc * j, 0, sizeof(TC) * m);
    }

    NOINLINE void mnpack(i64 m0, i64 m, i64 n0, i64 n, i64 k, int ith) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= (mc = 5) && n - n0 >= (nc = 5)) {
            mp = m0 + (m - m0) / mc * mc;
            np = n0 + (n - n0) / nc * nc;
            kpack<5, 5>(m0, mp, n0, np, 0, k, ith);
        } else {
            mc = 1;
            nc = 1;
            mp = m0 + (m - m0) / mc * mc;
            np = n0 + (n - n0) / nc * nc;
            kpack<1, 1>(m0, mp, n0, np, 0, k, ith);
        }
        mnpack(mp, m, n0, np, k, ith);
        mnpack(m0, mp, np, n, k, ith);
        mnpack(mp, m, np, n, k, ith);
    }

    template <int MC, int NC>
    NOINLINE void kpack(i64 m0, i64 m, i64 n0, i64 n, i64 k0, i64 k, int ith) {
        int kc, kp;
        if (k - k0 <= 0)
            return;
        constexpr int KC = 128;
        if (k - k0 >= (kc = KC) * KN) {
            kp = k0 + (k - k0) / (kc * KN) * (kc * KN);
            gemm<MC, NC, KC>(m0, m, n0, n, k0, kp, ith);
        } else {
            kc = KN;
            kp = k0 + (k - k0) / kc * kc;
            gemm<MC, NC, 1>(m0, m, n0, n, k0, kp, ith);
        }
        kpack<MC, NC>(m0, m, n0, n, kp, k, ith);
    }

    template <int MC, int NC, int KC>
    NOINLINE void gemm(i64 m0, i64 m, i64 n0, i64 n, i64 k0, i64 k, int ith) {
        i64 ytiles = (m - m0) / MC;
        i64 ztiles = (k - k0) / (KC * KN);
        i64 tiles = ytiles * ztiles;
        i64 duty = (tiles + nth - 1) / nth;
        i64 start = duty * ith;
        i64 end = start + duty;
        if (end > tiles)
            end = tiles;
        for (i64 job = start; job < end; ++job) {
            i64 ii = m0 + job / ztiles * MC;
            i64 ll = k0 + job % ztiles * (KC * KN);
            T Ac[KC][MC];
            for (i64 i = 0; i < MC; ++i)
                for (i64 l = 0; l < KC; ++l)
                    Ac[l][i] = load<T>(A + lda * (ii + i) + (ll + KN * l));
            for (i64 jj = n0; jj < n; jj += NC) {
                T Cc[NC][MC] = {0};
                for (i64 l = 0; l < KC; ++l)
                    for (i64 j = 0; j < NC; ++j) {
                        T b = load<T>(B + ldb * (jj + j) + (ll + KN * l));
                        for (i64 i = 0; i < MC; ++i)
                            Cc[j][i] = madd(Ac[l][i], b, Cc[j][i]);
                    }
                TC Ct[NC][MC];
                for (i64 j = 0; j < NC; ++j)
                    for (i64 i = 0; i < MC; ++i)
                        Ct[j][i] = hsum(Cc[j][i]);
                for (i64 j = 0; j < NC; ++j)
                    for (i64 i = 0; i < MC; ++i)
                        C[ldc * (jj + j) + (ii + i)] +=  Ct[j][i];
            }
        }
    }

    const TA *const A;
    const i64 lda;
    const TB *const B;
    const i64 ldb;
    TC *const C;
    const i64 ldc;
    const int nth;
};


static NOINLINE void sgemm(i64 m, i64 n, i64 k,
                           const float *A, i64 lda,
                           const float *B, i64 ldb,
                           float *C, i64 ldc) {
    int nth = 1;
#if defined(__AVX512F__)
    if (!(k % 16)) {
        tinyBLAS<16, __m512, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
        tb.matmul(m, n, k, 0);
    }

#elif defined(__AVX2__)
    if (!(k % 8)) {
        tinyBLAS<8, __m256, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
        tb.matmul(m, n, k, 0);
        return;
    }

#elif defined(__SSE__)
    if (!(k % 4)) {
        tinyBLAS<4, __m128, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
        tb.matmul(m, n, k, 0);
        return;
    }
#endif

    tinyBLAS<1, float, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
    tb.matmul(m, n, k, 0);
}

f64 gemm_beats_mkl_old(u32 m, u32 n, u32 k,
                       const f32 *__restrict a, u32 lda,
                       const f32 *__restrict b, u32 ldb,
                       f32 *__restrict c, u32 ldc) noexcept {
    i64 m_mkl = m, n_mkl = n, k_mkl = k;
    i64 lda_mkl = lda, ldb_mkl = ldb, ldc_mkl = ldc;

    const f64 start = now();
    sgemm(m_mkl, n_mkl, k_mkl, a, lda_mkl, b, ldb_mkl, c, ldc_mkl);
    return now() - start;
}
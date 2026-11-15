// Tachyon CUDA kernels. Compiled at runtime with NVRTC; model dimensions arrive as -D constants so every
// loop bound is a compile-time constant (full unrolling, no runtime index math for shapes).
//
// Required defines: HD NH NKV HID QD KVD FF NL CTX NSLOTS EPS SPLITS EQ
//   EQ: embedding qtype (2 = Q8, 3 = Q4, 1 = F16); matrix kernels take the weight qtype as a template argument
#include <cuda_fp16.h>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

#define GQA (NH / NKV)
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define QT_F16 1
#define QT_Q8 2
#define QT_Q4 3
#define QT_Q3 4
#define EPI_STORE 0
#define EPI_ADD 1
#define EPI_SILU 2

#ifndef KV8
#define KV8 0
#endif

// ------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// streaming 16B load that does not allocate in L1 (weights are touched once per token)
__device__ __forceinline__ uint4 ld_stream(const void* p) {
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// fp16-accumulate variant (2x tensor throughput on GeForce Ampere): d/c are 2 half2 registers
__device__ __forceinline__ void mma16816_h(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    half2 h = __floats2half2_rn(a, b);
    return *(uint32_t*)&h;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Dequantize 32 Q4 values (one 16-byte block) into 16 half2 (element order 0..31).
// byte j holds element j (low nibble) and element j+16 (high nibble).
__device__ __forceinline__ void dequant_q4_block(uint4 w, half2 sc, half2 mn, half2* out) {
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
    const half2 k1024 = __floats2half2_rn(1024.f, 1024.f);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t v = words[i];
        uint32_t t0 = (v & 0x000F000F) | 0x64006400;          // e[4i+0], e[4i+2]
        uint32_t t1 = ((v >> 8) & 0x000F000F) | 0x64006400;   // e[4i+1], e[4i+3]
        uint32_t t2 = ((v >> 4) & 0x000F000F) | 0x64006400;   // e[16+4i+0], e[16+4i+2]
        uint32_t t3 = ((v >> 12) & 0x000F000F) | 0x64006400;  // e[16+4i+1], e[16+4i+3]
        uint32_t p0 = __byte_perm(t0, t1, 0x5410), p1 = __byte_perm(t0, t1, 0x7632);
        uint32_t p2 = __byte_perm(t2, t3, 0x5410), p3 = __byte_perm(t2, t3, 0x7632);
        out[2 * i + 0] = __hfma2(__hsub2(*(half2*)&p0, k1024), sc, mn);
        out[2 * i + 1] = __hfma2(__hsub2(*(half2*)&p1, k1024), sc, mn);
        out[8 + 2 * i + 0] = __hfma2(__hsub2(*(half2*)&p2, k1024), sc, mn);
        out[8 + 2 * i + 1] = __hfma2(__hsub2(*(half2*)&p3, k1024), sc, mn);
    }
}

// dot of one Q4 32-block with 32 int8 activations (dp4a)
__device__ __forceinline__ int dot_q4_blk(const uint4 w, const int4 a0, const int4 a1) {
    int s = 0;
    s = __dp4a((int)(w.x & 0x0F0F0F0F), a0.x, s);
    s = __dp4a((int)((w.x >> 4) & 0x0F0F0F0F), a1.x, s);
    s = __dp4a((int)(w.y & 0x0F0F0F0F), a0.y, s);
    s = __dp4a((int)((w.y >> 4) & 0x0F0F0F0F), a1.y, s);
    s = __dp4a((int)(w.z & 0x0F0F0F0F), a0.z, s);
    s = __dp4a((int)((w.z >> 4) & 0x0F0F0F0F), a1.z, s);
    s = __dp4a((int)(w.w & 0x0F0F0F0F), a0.w, s);
    s = __dp4a((int)((w.w >> 4) & 0x0F0F0F0F), a1.w, s);
    return s;
}
// Q3 block: words {L0, L1, H}; int8x4 vector i (elements 4i..4i+3) = low2 bits of L[i/4] at shift 2*(i%4)
// plus the high bit at bit i of each byte of H.
__device__ __forceinline__ int dot_q3_blk(const uint32_t L0, const uint32_t L1, const uint32_t H, const int4 a0, const int4 a1) {
    const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t L = i < 4 ? L0 : L1;
        const uint32_t v = ((L >> (2 * (i & 3))) & 0x03030303u) | (((H >> i) & 0x01010101u) << 2);
        s = __dp4a((int)v, a[i], s);
    }
    return s;
}
__device__ __forceinline__ int dot_q8_blk(const uint4 w0, const uint4 w1, const int4 a0, const int4 a1) {
    int s = 0;
    s = __dp4a((int)w0.x, a0.x, s); s = __dp4a((int)w0.y, a0.y, s); s = __dp4a((int)w0.z, a0.z, s); s = __dp4a((int)w0.w, a0.w, s);
    s = __dp4a((int)w1.x, a1.x, s); s = __dp4a((int)w1.y, a1.y, s); s = __dp4a((int)w1.z, a1.z, s); s = __dp4a((int)w1.w, a1.w, s);
    return s;
}

// batch descriptor layout in device memory (ints): [tokens | pos | slot | logit_rows]
struct Batch {
    const int* token;
    const int* pos;
    const int* slot;
    const int* lrow;
};

#if KV8
typedef int8_t kv_t;   // 8-bit KV cache: int8 values + one float scale per (token, head)
#else
typedef half kv_t;
#endif
__device__ __forceinline__ size_t kv_sindex(int slot, int layer, int kvh, int pos) {
    return (size_t)((slot * NL + layer) * NKV + kvh) * CTX + pos;
}
__device__ __forceinline__ size_t kv_index(int slot, int layer, int kvh, int pos) {
    return ((size_t)((slot * NL + layer) * NKV + kvh) * CTX + pos) * HD;
}

// ------------------------------------------------------------------------------------------------
// embedding lookup -> fp32 residual stream
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void embed(const uint8_t* __restrict__ E, const uint8_t* __restrict__ ES, const int* __restrict__ tokens, float* __restrict__ x) {
    const int m = blockIdx.x;
    const int tok = tokens[m];
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        float v;
#if EQ == QT_Q4
        const uint8_t* row = E + (size_t)tok * (HID / 2);
        int blk = i / 32, j = i % 32;
        uint8_t b = row[blk * 16 + (j & 15)];
        int q = j < 16 ? (b & 15) : (b >> 4);
        half2 sm = ((const half2*)ES)[(size_t)tok * (HID / 64) + i / 64];
        v = __low2float(sm) * q + __high2float(sm);
#elif EQ == QT_Q8
        v = __half2float(((const half*)ES)[(size_t)tok * (HID / 32) + i / 32]) * (float)((const int8_t*)E)[(size_t)tok * HID + i];
#else
        v = __half2float(((const half*)E)[(size_t)tok * HID + i]);
#endif
        x[(size_t)m * HID + i] = v;
    }
}

// ------------------------------------------------------------------------------------------------
// RMSNorm (+ int8 activation quantization for the GEMV path, or fp16 output for the GEMM path)
// grid: rows; rows may be gathered through `rows` (for logits); block 256
// ------------------------------------------------------------------------------------------------
template <int OUT_F16>
__device__ __forceinline__ void rmsnorm_impl(const float* __restrict__ x, const float* __restrict__ w, int in_row, int out_row,
                                             int8_t* __restrict__ q, float2* __restrict__ ds, half* __restrict__ oh) {
    __shared__ float red[8];
    const float* xr = x + (size_t)in_row * HID;
    float ss = 0;
    for (int i = threadIdx.x; i < HID; i += 256) { float v = xr[i]; ss += v * v; }
    ss = warp_sum(ss);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / HID + EPS);
    for (int blk = warp; blk < HID / 32; blk += 8) {
        int i = blk * 32 + lane;
        float v = xr[i] * scale * w[i];
        if (OUT_F16) {
            oh[(size_t)out_row * HID + i] = __float2half(v);
        } else {
            float amax = warp_max(fabsf(v));
            float d = amax / 127.f;
            int qi = amax > 0 ? __float2int_rn(v / d) : 0;
            int s = warp_sum_i(qi);
            q[(size_t)out_row * HID + i] = (int8_t)qi;
            if (lane == 0) ds[(size_t)out_row * (HID / 32) + blk] = make_float2(d, d * s);
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_q8(const float* __restrict__ x, const float* __restrict__ w, int8_t* __restrict__ q, float2* __restrict__ ds) {
    rmsnorm_impl<0>(x, w, blockIdx.x, blockIdx.x, q, ds, nullptr);
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_q8_rows(const float* __restrict__ x, const float* __restrict__ w, const int* __restrict__ rows, int8_t* __restrict__ q, float2* __restrict__ ds) {
    rmsnorm_impl<0>(x, w, rows[blockIdx.x], blockIdx.x, q, ds, nullptr);
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_f16(const float* __restrict__ x, const float* __restrict__ w, half* __restrict__ o) {
    rmsnorm_impl<1>(x, w, blockIdx.x, blockIdx.x, nullptr, nullptr, o);
}

// ------------------------------------------------------------------------------------------------
// GEMM (prefill): C[M,N] = A[M,K] (fp16) x W[N,K]^T (Q4/Q8 dequantized to fp16 in shared memory)
// Tensor cores (mma.sync m16n8k16, fp32 accumulate). Tile 64(M) x 128(N) x 64(K), 8 warps (2x4), double buffer.
// ------------------------------------------------------------------------------------------------
#define GBM 64
#define GBN 128
#define GBK 64
#define GLD 72  // padded smem row (halves)

template <int QT, int EPI>
__global__ void __launch_bounds__(256) gemm(const half* __restrict__ A, int M, int K, const uint8_t* __restrict__ W, const void* __restrict__ S,
                                            int N, float* __restrict__ Cf, half* __restrict__ Ch, const float* __restrict__ bias) {
    extern __shared__ __align__(16) uint8_t smem[];
    half* As = (half*)smem;                       // [2][GBM][GLD]
    half* Ws = (half*)(smem + 2 * GBM * GLD * 2); // [2][GBN][GLD]
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / 4, wn = warp % 4;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
    const int KT = K / GBK;

    // loader roles
    const int wrow = tid / 2, wpart = tid % 2;  // each thread dequantizes 32 weights of one row
    uint4 wreg0, wreg1;
    half2 wsc;
    auto load_A = [&](int kt, int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int c = tid + i * 256;
            int r = c / 8, col = (c % 8) * 8;
            int gr = m0 + r;
            bool ok = gr < M;
            const half* src = A + (size_t)(ok ? gr : 0) * K + kt * GBK + col;
            cp_async16(As + (buf * GBM + r) * GLD + col, src, ok);
        }
    };
    auto fetch_W = [&](int kt) {
        int gn = n0 + wrow;
        if (gn >= N) gn = N - 1;
        if constexpr (QT == QT_Q4) {
            wreg0 = ld_stream(W + (size_t)gn * (K / 2) + kt * 32 + wpart * 16);
            wsc = ((const half2*)S)[(size_t)gn * (K / 64) + kt];
        } else if constexpr (QT == QT_Q3) {
            const uint32_t* p3 = (const uint32_t*)(W + (size_t)gn * (K * 3 / 8) + (kt * 2 + wpart) * 12);
            wreg0.x = __ldg(p3); wreg0.y = __ldg(p3 + 1); wreg0.z = __ldg(p3 + 2);
            wsc = ((const half2*)S)[(size_t)gn * (K / 64) + kt];
        } else {
            const uint8_t* p = W + (size_t)gn * K + kt * 64 + wpart * 32;
            wreg0 = ld_stream(p); wreg1 = ld_stream(p + 16);
            wsc = __half2half2(((const half*)S)[(size_t)gn * (K / 32) + kt * 2 + wpart]);
        }
    };
    auto store_W = [&](int buf) {
        half2 v[16];
        if constexpr (QT == QT_Q4) {
            dequant_q4_block(wreg0, __half2half2(__low2half(wsc)), __half2half2(__high2half(wsc)), v);
        } else if constexpr (QT == QT_Q3) {
            const half2 sc2 = __half2half2(__low2half(wsc)), mn2 = __half2half2(__high2half(wsc));
            const half2 k1024 = __floats2half2_rn(1024.f, 1024.f);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t L = i < 4 ? wreg0.x : wreg0.y;
                const uint32_t q = ((L >> (2 * (i & 3))) & 0x03030303u) | (((wreg0.z >> i) & 0x01010101u) << 2);  // e[4i..4i+3]
                const uint32_t t0 = (q & 0x00FF00FFu) | 0x64006400u;          // e0, e2
                const uint32_t t1 = ((q >> 8) & 0x00FF00FFu) | 0x64006400u;   // e1, e3
                const uint32_t p0 = __byte_perm(t0, t1, 0x5410), p1 = __byte_perm(t0, t1, 0x7632);
                v[2 * i] = __hfma2(__hsub2(*(const half2*)&p0, k1024), sc2, mn2);
                v[2 * i + 1] = __hfma2(__hsub2(*(const half2*)&p1, k1024), sc2, mn2);
            }
        } else {
            const int8_t* b0 = (const int8_t*)&wreg0;
            const int8_t* b1 = (const int8_t*)&wreg1;
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] = __hmul2(__floats2half2_rn((float)b0[2 * i], (float)b0[2 * i + 1]), wsc);
#pragma unroll
            for (int i = 0; i < 8; i++) v[8 + i] = __hmul2(__floats2half2_rn((float)b1[2 * i], (float)b1[2 * i + 1]), wsc);
        }
        uint4* dst = (uint4*)(Ws + (buf * GBN + wrow) * GLD + wpart * 32);
#pragma unroll
        for (int i = 0; i < 4; i++) dst[i] = *(uint4*)&v[4 * i];
    };

    float c[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) c[i][j][k] = 0.f;

    load_A(0, 0);
    cp_async_commit();
    fetch_W(0);
    store_W(0);
    cp_async_wait_all();
    __syncthreads();

    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) {
            load_A(kt + 1, buf ^ 1);
            cp_async_commit();
            fetch_W(kt + 1);
        }
#ifdef GEMM_F16ACC
        // per-K-tile fp16 accumulation (64 products), then folded into the fp32 accumulators
        uint32_t ch[2][4][2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) ch[i][j][0] = ch[i][j][1] = 0u;
#endif
#pragma unroll
        for (int kk = 0; kk < GBK / 16; kk++) {
            uint32_t a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const half* p = As + (buf * GBM + wm * 32 + mi * 16 + (lane & 15)) * GLD + kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], p);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni += 2) {
                uint32_t b0, b1, b2, b3;
                const half* p = Ws + (buf * GBN + wn * 32 + ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * GLD + kk * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(b0, b1, b2, b3, p);
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
#ifdef GEMM_F16ACC
                    mma16816_h(ch[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1);
                    mma16816_h(ch[mi][ni + 1], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b2, b3);
#else
                    mma16816(c[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1);
                    mma16816(c[mi][ni + 1], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b2, b3);
#endif
                }
            }
        }
#ifdef GEMM_F16ACC
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const float2 lo = __half22float2(*(half2*)&ch[mi][ni][0]), hi = __half22float2(*(half2*)&ch[mi][ni][1]);
                c[mi][ni][0] += lo.x; c[mi][ni][1] += lo.y; c[mi][ni][2] += hi.x; c[mi][ni][3] += hi.y;
            }
#endif
        if (more) {
            store_W(buf ^ 1);
            cp_async_wait_all();
        }
        __syncthreads();
    }
    // epilogue
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = m0 + wm * 32 + mi * 16 + (lane >> 2) + hh * 8;
                const int col = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
                if (r >= M || col >= N) continue;
                float v0 = c[mi][ni][hh * 2], v1 = c[mi][ni][hh * 2 + 1];
                if (EPI == EPI_SILU) {
                    Ch[(size_t)r * (N / 2) + col / 2] = __float2half(silu(v0) * v1);
                } else {
                    if (bias) { v0 += bias[col]; v1 += bias[col + 1]; }
                    float2* dst = (float2*)(Cf + (size_t)r * N + col);
                    if (EPI == EPI_ADD) { float2 o = *dst; o.x += v0; o.y += v1; *dst = o; }
                    else *dst = make_float2(v0, v1);
                }
            }
}

// ------------------------------------------------------------------------------------------------
// QK-norm + RoPE (neox style) + KV-cache write. One warp per (token, head). q output fp32 (decode) or fp16 (prefill)
// grid (M, ceil((NH+NKV)/4)), block 128
// ------------------------------------------------------------------------------------------------
template <int Q_F16>
__device__ __forceinline__ void rope_impl(const float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                                          const float* __restrict__ inv_freq, const int* __restrict__ pos_arr, const int* __restrict__ slot_arr,
                                          int layer, void* __restrict__ qout, kv_t* __restrict__ kc, kv_t* __restrict__ vc,
                                          float* __restrict__ ksc, float* __restrict__ vsc) {
    constexpr int E = HD / 32;
    const int m = blockIdx.x;
    const int head = blockIdx.y * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (head >= NH + NKV) return;
    const bool is_q = head < NH;
    const int kvh = head - NH;
    const float* src = qkv + (size_t)m * (QD + 2 * KVD) + (is_q ? head * HD : QD + kvh * HD);
    float v[E];
#pragma unroll
    for (int e = 0; e < E; e++) v[e] = src[lane + 32 * e];
#ifdef QK_NORM
    {
        float ss = 0;
#pragma unroll
        for (int e = 0; e < E; e++) ss += v[e] * v[e];
        ss = warp_sum(ss);
        const float sc = rsqrtf(ss / HD + EPS);
        const float* nw = is_q ? qn : kn;
#pragma unroll
        for (int e = 0; e < E; e++) v[e] = v[e] * sc * nw[lane + 32 * e];
    }
#endif
    const int pos = pos_arr[m];
#pragma unroll
    for (int e = 0; e < E / 2; e++) {
        const int i = lane + 32 * e;  // pair (i, i + HD/2)
        float sn, cs;
        sincosf((float)pos * inv_freq[i], &sn, &cs);
        float x0 = v[e], x1 = v[e + E / 2];
        v[e] = x0 * cs - x1 * sn;
        v[e + E / 2] = x0 * sn + x1 * cs;
    }
    if (is_q) {
        if (Q_F16) {
            half* q = (half*)qout + (size_t)m * QD + head * HD;
#pragma unroll
            for (int e = 0; e < E; e++) q[lane + 32 * e] = __float2half(v[e]);
        } else {
            float* q = (float*)qout + (size_t)m * QD + head * HD;
#pragma unroll
            for (int e = 0; e < E; e++) q[lane + 32 * e] = v[e];
        }
    } else {
        const size_t base = kv_index(slot_arr[m], layer, kvh, pos);
        const float* vs = qkv + (size_t)m * (QD + 2 * KVD) + QD + KVD + kvh * HD;
#if KV8
        float ka = 0.f, va = 0.f, vv[E];
#pragma unroll
        for (int e = 0; e < E; e++) { vv[e] = vs[lane + 32 * e]; ka = fmaxf(ka, fabsf(v[e])); va = fmaxf(va, fabsf(vv[e])); }
        ka = warp_max(ka); va = warp_max(va);
        const float kd = ka / 127.f, vd = va / 127.f;
        const float ki = ka > 0 ? 127.f / ka : 0.f, vi = va > 0 ? 127.f / va : 0.f;
#pragma unroll
        for (int e = 0; e < E; e++) {
            kc[base + lane + 32 * e] = (int8_t)__float2int_rn(v[e] * ki);
            vc[base + lane + 32 * e] = (int8_t)__float2int_rn(vv[e] * vi);
        }
        if (lane == 0) {
            const size_t si = kv_sindex(slot_arr[m], layer, kvh, pos);
            ksc[si] = kd; vsc[si] = vd;
        }
#else
#pragma unroll
        for (int e = 0; e < E; e++) {
            kc[base + lane + 32 * e] = __float2half(v[e]);
            vc[base + lane + 32 * e] = __float2half(vs[lane + 32 * e]);
        }
#endif
    }
}
extern "C" __global__ void __launch_bounds__(128) rope_decode(const float* qkv, const float* qn, const float* kn, const float* inv_freq,
                                                              const int* pos, const int* slot, int layer, float* qout, kv_t* kc, kv_t* vc,
                                                              float* ksc, float* vsc) {
    rope_impl<0>(qkv, qn, kn, inv_freq, pos, slot, layer, qout, kc, vc, ksc, vsc);
}
extern "C" __global__ void __launch_bounds__(128) rope_prefill(const float* qkv, const float* qn, const float* kn, const float* inv_freq,
                                                               const int* pos, const int* slot, int layer, half* qout, kv_t* kc, kv_t* vc,
                                                               float* ksc, float* vsc) {
    rope_impl<1>(qkv, qn, kn, inv_freq, pos, slot, layer, qout, kc, vc, ksc, vsc);
}

// ------------------------------------------------------------------------------------------------
// Prefill flash attention (FA2-style, tensor cores). Q fp16 [M][QD] for tokens at positions p0..p0+M-1 of one slot.
// grid (ceil(M/64), NH), block 128 (4 warps x 16 query rows). K/V tiles of 64 positions in shared memory.
// ------------------------------------------------------------------------------------------------
#define FKV 64
#define FLD (HD + 8)
extern "C" __global__ void __launch_bounds__(128) flash_prefill(const half* __restrict__ Q, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                                                half* __restrict__ O, int M, int p0, int slot, int layer,
                                                                const float* __restrict__ ksc, const float* __restrict__ vsc) {
    extern __shared__ __align__(16) uint8_t smem[];
    half* Ks = (half*)smem;            // [FKV][FLD]
    half* Vs = Ks + FKV * FLD;         // [FKV][FLD]
    const int h = blockIdx.y, kvh = h / GQA;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int qb = blockIdx.x * 64;
    const int q0 = qb + warp * 16;  // first query row of this warp
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
    // Q fragments
    uint32_t qf[HD / 16][4];
    {
        const int r0 = q0 + (lane >> 2), r1 = r0 + 8;
        const int c = (lane & 3) * 2;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const half* p0r = Q + (size_t)r0 * QD + h * HD + kk * 16 + c;
            const half* p1r = Q + (size_t)r1 * QD + h * HD + kk * 16 + c;
            qf[kk][0] = r0 < M ? *(const uint32_t*)p0r : 0u;
            qf[kk][1] = r1 < M ? *(const uint32_t*)p1r : 0u;
            qf[kk][2] = r0 < M ? *(const uint32_t*)(p0r + 8) : 0u;
            qf[kk][3] = r1 < M ? *(const uint32_t*)(p1r + 8) : 0u;
        }
    }
    float o[HD / 8][4];
#pragma unroll
    for (int i = 0; i < HD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    const int qpos0 = p0 + q0 + (lane >> 2);  // query position of row r0 (r1 = +8)
    const int last_q = min(M, qb + 64) - 1;
    const int n_kv = p0 + last_q + 1;
    const kv_t* kb = kc + kv_index(slot, layer, kvh, 0);
    const kv_t* vb = vc + kv_index(slot, layer, kvh, 0);
#if KV8
    const float* ksb = ksc + kv_sindex(slot, layer, kvh, 0);
    const float* vsb = vsc + kv_sindex(slot, layer, kvh, 0);
#endif
    for (int kv0 = 0; kv0 < n_kv; kv0 += FKV) {
        __syncthreads();
#if KV8
        // dequantize int8 K/V rows (16 values per thread-iteration) into fp16 tiles
        for (int c = threadIdx.x; c < FKV * HD / 16; c += 128) {
            const int r = c / (HD / 16), col = (c % (HD / 16)) * 16;
            const bool ok = kv0 + r < n_kv;
            half2 hk[8], hv[8];
            if (ok) {
                const uint4 rk = *(const uint4*)(kb + (size_t)(kv0 + r) * HD + col);
                const uint4 rv = *(const uint4*)(vb + (size_t)(kv0 + r) * HD + col);
                const float sk = ksb[kv0 + r], sv = vsb[kv0 + r];
                const int8_t* bk = (const int8_t*)&rk;
                const int8_t* bv = (const int8_t*)&rv;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    hk[j] = __floats2half2_rn(bk[2 * j] * sk, bk[2 * j + 1] * sk);
                    hv[j] = __floats2half2_rn(bv[2 * j] * sv, bv[2 * j + 1] * sv);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) hk[j] = hv[j] = __floats2half2_rn(0.f, 0.f);
            }
            uint4* dk = (uint4*)(Ks + r * FLD + col);
            uint4* dv = (uint4*)(Vs + r * FLD + col);
            dk[0] = *(uint4*)&hk[0]; dk[1] = *(uint4*)&hk[4];
            dv[0] = *(uint4*)&hv[0]; dv[1] = *(uint4*)&hv[4];
        }
#else
        // load K/V tile: FKV rows x HD halves = FKV*HD/8 chunks of 16B each
        for (int c = threadIdx.x; c < FKV * HD / 8; c += 128) {
            const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
            const bool ok = kv0 + r < n_kv;
            const size_t off = (size_t)(ok ? kv0 + r : 0) * HD + col;
            cp_async16(Ks + r * FLD + col, kb + off, ok);
            cp_async16(Vs + r * FLD + col, vb + off, ok);
        }
        cp_async_commit();
        cp_async_wait_all();
#endif
        __syncthreads();
        if (kv0 > p0 + q0 + 15) continue;  // whole tile is in the future for this warp
        float s[FKV / 8][4];
#pragma unroll
        for (int i = 0; i < FKV / 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
#pragma unroll
            for (int ni = 0; ni < FKV / 8; ni += 2) {
                uint32_t b0, b1, b2, b3;
                const half* p = Ks + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * FLD + kk * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(b0, b1, b2, b3, p);
                mma16816(s[ni], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                mma16816(s[ni + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
            }
        }
        // scale + causal mask + online softmax
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int ni = 0; ni < FKV / 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int kpos = kv0 + ni * 8 + (lane & 3) * 2 + (e & 1);
                const int qpos = qpos0 + (e >> 1) * 8;
                float v = s[ni][e] * sl2;
                if (kpos > qpos) v = -INFINITY;
                s[ni][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            const float mn = fmaxf(mrow[r], mx[r]);
            alpha[r] = exp2f(mrow[r] - mn);
            mrow[r] = mn;
            lrow[r] *= alpha[r];
        }
#pragma unroll
        for (int ni = 0; ni < FKV / 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float pv = exp2f(s[ni][e] - mrow[e >> 1]);
                s[ni][e] = pv;
                lrow[e >> 1] += pv;
            }
#pragma unroll
        for (int i = 0; i < HD / 8; i++) { o[i][0] *= alpha[0]; o[i][1] *= alpha[0]; o[i][2] *= alpha[1]; o[i][3] *= alpha[1]; }
        // O += P V
#pragma unroll
        for (int kk = 0; kk < FKV / 16; kk++) {
            const uint32_t a0 = pack_half2(s[2 * kk][0], s[2 * kk][1]);
            const uint32_t a1 = pack_half2(s[2 * kk][2], s[2 * kk][3]);
            const uint32_t a2 = pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            const uint32_t a3 = pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dn = 0; dn < HD / 8; dn += 2) {
                uint32_t b0, b1, b2, b3;
                const half* p = Vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FLD + (dn + (lane >> 4)) * 8;
                ldmatrix_x4_trans(b0, b1, b2, b3, p);
                mma16816(o[dn], a0, a1, a2, a3, b0, b1);
                mma16816(o[dn + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 1);
        lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 2);
    }
    const int r0 = q0 + (lane >> 2), r1 = r0 + 8;
    const float il0 = 1.f / lrow[0], il1 = 1.f / lrow[1];
#pragma unroll
    for (int dn = 0; dn < HD / 8; dn++) {
        const int col = h * HD + dn * 8 + (lane & 3) * 2;
        if (r0 < M) *(half2*)(O + (size_t)r0 * QD + col) = __floats2half2_rn(o[dn][0] * il0, o[dn][1] * il0);
        if (r1 < M) *(half2*)(O + (size_t)r1 * QD + col) = __floats2half2_rn(o[dn][2] * il1, o[dn][3] * il1);
    }
}

// ------------------------------------------------------------------------------------------------
// argmax over vocab rows. grid rows, block 1024
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(1024) argmax_rows(const float* __restrict__ logits, int V, int Vlim, int* __restrict__ out) {
    const float* l = logits + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int bi = 0;
    for (int i = threadIdx.x; i < Vlim; i += 1024) { float v = l[i]; if (v > best) { best = v; bi = i; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { sv[warp] = best; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        best = sv[lane]; bi = si[lane];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

// copy KV positions [0,n) between slots. grid (NL*NKV), block 256
extern "C" __global__ void kv_copy(kv_t* kc, kv_t* vc, int src, int dst, int n, float* ksc, float* vsc) {
    const int l = blockIdx.x / NKV, h = blockIdx.x % NKV;
    const size_t s = kv_index(src, l, h, 0), d = kv_index(dst, l, h, 0);
    const size_t cnt = (size_t)n * HD * sizeof(kv_t) / 16;
    for (size_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        ((uint4*)(kc + d))[i] = ((const uint4*)(kc + s))[i];
        ((uint4*)(vc + d))[i] = ((const uint4*)(vc + s))[i];
    }
#if KV8
    const size_t ss = kv_sindex(src, l, h, 0), ds = kv_sindex(dst, l, h, 0);
    for (int i = threadIdx.x; i < n; i += blockDim.x) { ksc[ds + i] = ksc[ss + i]; vsc[ds + i] = vsc[ss + i]; }
#endif
}

// imatrix support: acc[k] += sum_m A[m][k]^2 for an fp16 activation matrix [M][K]
extern "C" __global__ void colsq_accum(const half* __restrict__ A, int M, int K, float* __restrict__ acc) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float s = 0.f;
    for (int m = 0; m < M; m++) { float v = __half2float(A[(size_t)m * K + k]); s += v * v; }
    acc[k] += s;
}

// ================================================================================================
// v2 decode kernels: fused prologues / epilogues, balanced single-wave grids
// ================================================================================================
#define ATILE 128
#define PRO_NONE 0   // int8 activations already in global memory
#define PRO_RMS 1    // RMSNorm(x fp32) * w, quantized to int8 in shared memory by every CTA

// Shared prologue: normalize + quantize M rows of x (optionally gathered through `rows`) into smem.
template <int K, int M>
__device__ __forceinline__ void prologue_rms(const float* __restrict__ x, const float* __restrict__ nw, const int* __restrict__ rows,
                                             int8_t* sq, float2* sds) {
    __shared__ float red[8][M];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float ss[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
        const float* xr = x + (size_t)(rows ? rows[m] : m) * K;
        float a = 0.f;
        for (int i = threadIdx.x; i < K; i += 256) { float v = xr[i]; a += v * v; }
        ss[m] = warp_sum(a);
    }
    if (lane == 0)
#pragma unroll
        for (int m = 0; m < M; m++) red[warp][m] = ss[m];
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; m++) {
        float t = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) t += red[w][m];
        const float scale = rsqrtf(t / K + EPS);
        const float* xr = x + (size_t)(rows ? rows[m] : m) * K;
        for (int blk = warp; blk < K / 32; blk += 8) {
            const int i = blk * 32 + lane;
            const float v = xr[i] * scale * nw[i];
            const float amax = warp_max(fabsf(v));
            const float d = amax / 127.f;
            const int qi = amax > 0 ? __float2int_rn(v / d) : 0;
            const int sm = warp_sum_i(qi);
            sq[m * K + i] = (int8_t)qi;
            if (lane == 0) sds[m * (K / 32) + blk] = make_float2(d, d * sm);
        }
    }
    __syncthreads();
}

// accumulate one row's dot products for M tokens (lane-strided 32-blocks)
template <int QT, int K, int M>
__device__ __forceinline__ void row_dot(const uint8_t* __restrict__ W, const void* __restrict__ S, int row,
                                        const int8_t* xq, const float2* xds, float* acc) {
    constexpr int NB = K / 32;
    constexpr int ITER = (NB + 31) / 32;
    // unroll depth trades loads-in-flight against registers: full for M=1, shallow for big M (avoid spills)
    constexpr int UNR = M == 1 ? ITER : (M == 2 ? 4 : (M <= 4 ? 2 : 1));
    const int lane = threadIdx.x % 32;
#pragma unroll UNR
    for (int it = 0; it < ITER; it++) {
        const int b = it * 32 + lane;
        if (NB % 32 == 0 || b < NB) {
            uint4 w, w1;
            float s, mn = 0.f;
            if constexpr (QT == QT_Q4) {
                w = ld_stream(W + (size_t)row * (K / 2) + b * 16);
                const half2 sm = ((const half2*)S)[(size_t)row * (K / 64) + b / 2];
                s = __low2float(sm); mn = __high2float(sm);
            } else if constexpr (QT == QT_Q3) {
                const uint32_t* p3 = (const uint32_t*)(W + (size_t)row * (K * 3 / 8) + b * 12);
                w.x = __ldg(p3); w.y = __ldg(p3 + 1); w.z = __ldg(p3 + 2);
                const half2 sm = ((const half2*)S)[(size_t)row * (K / 64) + b / 2];
                s = __low2float(sm); mn = __high2float(sm);
            } else {
                w = ld_stream(W + (size_t)row * K + b * 32);
                w1 = ld_stream(W + (size_t)row * K + b * 32 + 16);
                s = __half2float(((const half*)S)[(size_t)row * NB + b]);
            }
#pragma unroll
            for (int m = 0; m < M; m++) {
                const int4* xp = (const int4*)(xq + m * K + b * 32);
                const int4 a0 = xp[0], a1 = xp[1];
                const float2 d = xds[m * NB + b];
                if constexpr (QT == QT_Q4) acc[m] += s * d.x * (float)dot_q4_blk(w, a0, a1) + mn * d.y;
                else if constexpr (QT == QT_Q3) acc[m] += s * d.x * (float)dot_q3_blk(w.x, w.y, w.z, a0, a1) + mn * d.y;
                else acc[m] += s * d.x * (float)dot_q8_blk(w, w1, a0, a1);
            }
        }
    }
}

// y[m][row] (=|+=) W[row] . act[m]. Warp gw handles rows gw, gw+Wtot, ... (balanced, single wave).
template <int QT, int K, int M, int PRO, int EPI>
__global__ void __launch_bounds__(256) gemv2(const uint8_t* __restrict__ W, const void* __restrict__ S, int N, int Wtot,
                                             const float* __restrict__ x, const float* __restrict__ nw, const int* __restrict__ rows,
                                             const int8_t* __restrict__ xq_g, const float2* __restrict__ xds_g,
                                             float* __restrict__ y, const float* __restrict__ bias) {
    extern __shared__ __align__(16) uint8_t dsm[];
    const int8_t* xq = xq_g;
    const float2* xds = xds_g;
    if constexpr (PRO == PRO_RMS) {
        int8_t* sq = (int8_t*)dsm;
        float2* sds = (float2*)(dsm + M * K);
        prologue_rms<K, M>(x, nw, rows, sq, sds);
        xq = sq; xds = sds;
    }
    const int lane = threadIdx.x % 32;
    const int gw = blockIdx.x * 8 + threadIdx.x / 32;
    if (gw >= Wtot) return;
    for (int r1 = gw; r1 < N; r1 += 2 * Wtot) {
        const int r2 = r1 + Wtot;
        const bool two = r2 < N;
        float a1[M], a2[M];
#pragma unroll
        for (int m = 0; m < M; m++) { a1[m] = 0.f; a2[m] = 0.f; }
        row_dot<QT, K, M>(W, S, r1, xq, xds, a1);
        if (two) row_dot<QT, K, M>(W, S, r2, xq, xds, a2);
#pragma unroll
        for (int m = 0; m < M; m++) { a1[m] = warp_sum(a1[m]); a2[m] = warp_sum(a2[m]); }
        if (lane < M) {
            float v1 = 0, v2 = 0;
#pragma unroll
            for (int m = 0; m < M; m++) if (lane == m) { v1 = a1[m]; v2 = a2[m]; }
            if (bias) { v1 += bias[r1]; if (two) v2 += bias[r2]; }
            float* d1 = y + (size_t)lane * N + r1;
            if (EPI == EPI_ADD) *d1 += v1; else *d1 = v1;
            if (two) {
                float* d2 = y + (size_t)lane * N + r2;
                if (EPI == EPI_ADD) *d2 += v2; else *d2 = v2;
            }
        }
    }
}

// gate/up with fused RMSNorm prologue, SiLU*up and int8 quantization of the output.
// Tiles of 64 interleaved rows (= 32 outputs = one activation block); CTA loops over tiles.
template <int QT, int K, int M>
__global__ void __launch_bounds__(256) gemv2_gu(const uint8_t* __restrict__ W, const void* __restrict__ S, int N,
                                                const float* __restrict__ x, const float* __restrict__ nw,
                                                int8_t* __restrict__ oq, float2* __restrict__ ods) {
    extern __shared__ __align__(16) uint8_t dsm[];
    int8_t* sq = (int8_t*)dsm;
    float2* sds = (float2*)(dsm + M * K);
    prologue_rms<K, M>(x, nw, nullptr, sq, sds);
    __shared__ float res[M][32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ntiles = N / 64;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int row0 = tile * 64 + warp * 8;
#pragma unroll (M <= 2 ? 4 : 1)
        for (int r = 0; r < 8; r += 2) {
            float g[M], u[M];
#pragma unroll
            for (int m = 0; m < M; m++) { g[m] = 0.f; u[m] = 0.f; }
            row_dot<QT, K, M>(W, S, row0 + r, sq, sds, g);
            row_dot<QT, K, M>(W, S, row0 + r + 1, sq, sds, u);
#pragma unroll
            for (int m = 0; m < M; m++) {
                const float gs = warp_sum(g[m]), us = warp_sum(u[m]);
                if (lane == 0) res[m][warp * 4 + r / 2] = silu(gs) * us;
            }
        }
        __syncthreads();
        for (int m = warp; m < M; m += 8) {
            const float v = res[m][lane];
            const float amax = warp_max(fabsf(v));
            const float dd = amax / 127.f;
            const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
            const int sm = warp_sum_i(qi);
            oq[(size_t)m * FF + tile * 32 + lane] = (int8_t)qi;
            if (lane == 0) ods[(size_t)m * (FF / 32) + tile] = make_float2(dd, dd * sm);
        }
        __syncthreads();
    }
}

// Decode attention with the split-combine fused in: the last CTA of each (token, kv-head) group to finish
// merges all splits and writes the int8-quantized head outputs for the o-projection.
extern "C" __global__ void __launch_bounds__(128) attn_decode2(const float* __restrict__ q, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                                               const int* __restrict__ pos_arr, const int* __restrict__ slot_arr, int layer,
                                                               float* __restrict__ part_o, float2* __restrict__ part_ml, int* __restrict__ counters,
                                                               int8_t* __restrict__ oq, float2* __restrict__ ods,
                                                               const float* __restrict__ ksc, const float* __restrict__ vsc) {
    const int kvh = blockIdx.x, split = blockIdx.y, m = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int len = pos_arr[m] + 1;
    const int slot = slot_arr[m];
    int chunk = (len + SPLITS - 1) / SPLITS;
    chunk = max(64, (chunk + 63) & ~63);
    const int start = split * chunk;
    const int end = min(len, start + chunk);
    const size_t pidx = ((size_t)m * NH + kvh * GQA) * SPLITS + split;
    __shared__ float qs[GQA][HD];
    __shared__ float ps[GQA][ATILE];
    __shared__ float alpha_s[GQA];
    __shared__ float m_s[GQA], l_s[GQA];
    constexpr int NDO = HD / 8;          // dim octets (16 B of V per load)
    constexpr int PG = 128 / NDO;        // position groups (8 for HD=128)
    __shared__ __align__(16) float red[PG][GQA][HD];
    __shared__ int is_last;
    if (start < len) {
        const float scale = rsqrtf((float)HD) * 1.4426950408889634f;
        for (int i = tid; i < GQA * HD; i += 128) qs[i / HD][i % HD] = q[(size_t)m * QD + (kvh * GQA) * HD + i] * scale;
        if (tid < GQA) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
        const kv_t* kb = kc + kv_index(slot, layer, kvh, 0);
        const kv_t* vb = vc + kv_index(slot, layer, kvh, 0);
#if KV8
        const float* ksb = ksc + kv_sindex(slot, layer, kvh, 0);
        const float* vsb = vsc + kv_sindex(slot, layer, kvh, 0);
#endif
        const int dq = tid % NDO, pg = tid / NDO;
        float acc[GQA][8];
#pragma unroll
        for (int g = 0; g < GQA; g++)
#pragma unroll
            for (int d = 0; d < 8; d++) acc[g][d] = 0.f;
        __syncthreads();
        for (int t0 = start; t0 < end; t0 += ATILE) {
            const int p = t0 + tid;
            if (p < end) {
                const uint4* kr = (const uint4*)(kb + (size_t)p * HD);
                float s[GQA];
#pragma unroll
                for (int g = 0; g < GQA; g++) s[g] = 0.f;
#if KV8
#pragma unroll
                for (int c = 0; c < HD / 16; c++) {
                    const uint4 kv = kr[c];
                    const int8_t* b8 = (const int8_t*)&kv;
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const float f = (float)b8[j];
#pragma unroll
                        for (int g = 0; g < GQA; g++) s[g] += qs[g][c * 16 + j] * f;
                    }
                }
                const float ks = ksb[p];
#pragma unroll
                for (int g = 0; g < GQA; g++) s[g] *= ks;
#else
#pragma unroll
                for (int c = 0; c < HD / 8; c++) {
                    uint4 kv = kr[c];
                    const half2* h = (const half2*)&kv;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float2 f = __half22float2(h[j]);
#pragma unroll
                        for (int g = 0; g < GQA; g++) s[g] += qs[g][c * 8 + 2 * j] * f.x + qs[g][c * 8 + 2 * j + 1] * f.y;
                    }
                }
#endif
#pragma unroll
                for (int g = 0; g < GQA; g++) ps[g][tid] = s[g];
            } else {
#pragma unroll
                for (int g = 0; g < GQA; g++) ps[g][tid] = -INFINITY;
            }
            __syncthreads();
            for (int g = warp; g < GQA; g += 4) {
                float mx = -INFINITY;
                for (int i = lane; i < ATILE; i += 32) mx = fmaxf(mx, ps[g][i]);
                mx = warp_max(mx);
                const float mo = m_s[g];
                const float mn = fmaxf(mo, mx);
                float sum = 0.f;
                for (int i = lane; i < ATILE; i += 32) { float e = exp2f(ps[g][i] - mn); ps[g][i] = e; sum += e; }
                sum = warp_sum(sum);
                if (lane == 0) { const float a = exp2f(mo - mn); alpha_s[g] = a; l_s[g] = l_s[g] * a + sum; m_s[g] = mn; }
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GQA; g++) {
                const float a = alpha_s[g];
#pragma unroll
                for (int d = 0; d < 8; d++) acc[g][d] *= a;
            }
            const int nt = min(ATILE, end - t0);
#pragma unroll 4
            for (int i = pg; i < nt; i += PG) {
                float vv[8];
#if KV8
                const uint2 raw = *(const uint2*)(vb + (size_t)(t0 + i) * HD + dq * 8);
                const int8_t* b8 = (const int8_t*)&raw;
                const float vsc_i = vsb[t0 + i];
#pragma unroll
                for (int j = 0; j < 8; j++) vv[j] = (float)b8[j];
#else
                const uint4 raw = *(const uint4*)(vb + (size_t)(t0 + i) * HD + dq * 8);
                const half2* h = (const half2*)&raw;
                const float vsc_i = 1.f;
#pragma unroll
                for (int j = 0; j < 4; j++) { const float2 f = __half22float2(h[j]); vv[2 * j] = f.x; vv[2 * j + 1] = f.y; }
#endif
#pragma unroll
                for (int g = 0; g < GQA; g++) {
                    const float pr = ps[g][i] * vsc_i;
#pragma unroll
                    for (int d = 0; d < 8; d++) acc[g][d] += pr * vv[d];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int g = 0; g < GQA; g++)
#pragma unroll
            for (int d = 0; d < 8; d++) red[pg][g][dq * 8 + d] = acc[g][d];
        __syncthreads();
        for (int i = tid; i < GQA * HD; i += 128) {
            const int g = i / HD, d = i % HD;
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < PG; k++) sum += red[k][g][d];
            part_o[(pidx + g * SPLITS) * HD + d] = sum;
        }
        if (tid < GQA) part_ml[pidx + tid * SPLITS] = make_float2(m_s[tid], l_s[tid]);
    } else {
        if (tid < GQA) part_ml[pidx + tid * SPLITS] = make_float2(-INFINITY, 0.f);
    }
    // ---- last CTA of this (token, kv head) group combines the splits ----
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int* cnt = counters + m * NKV + kvh;
        const int prev = atomicAdd(cnt, 1);
        is_last = prev == SPLITS - 1;
        if (is_last) *cnt = 0;  // reset for the next layer / step
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    // tid -> (head g, dim d): GQA*HD outputs, 128 threads
    for (int i = tid; i < GQA * HD; i += 128) {
        const int g = i / HD, d = i % HD;
        const size_t base = ((size_t)m * NH + kvh * GQA + g) * SPLITS;
        float mx = -INFINITY;
#pragma unroll
        for (int s2 = 0; s2 < SPLITS; s2++) mx = fmaxf(mx, __ldcg(&part_ml[base + s2]).x);
        float L = 0.f, o = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < SPLITS; s2++) {
            const float2 ml = __ldcg(&part_ml[base + s2]);
            if (ml.y == 0.f) continue;
            const float w = exp2f(ml.x - mx);
            L += w * ml.y;
            o += w * __ldcg(&part_o[(base + s2) * HD + d]);
        }
        const float v = o / L;
        // quantize per 32 consecutive dims (one warp-iteration covers exactly one block since HD % 32 == 0)
        const float amax = warp_max(fabsf(v));
        const float dd = amax / 127.f;
        const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
        const int sm = warp_sum_i(qi);
        const int col = (kvh * GQA + g) * HD + d;
        oq[(size_t)m * QD + col] = (int8_t)qi;
        if (lane == 0) ods[(size_t)m * (QD / 32) + col / 32] = make_float2(dd, dd * sm);
    }
}

// ---- sampling support: compact candidate lists (logit >= rowmax - delta) for host-side top-k/top-p ----
// pass 1: row max (grid rows, block 1024)
extern "C" __global__ void __launch_bounds__(1024) row_max(const float* __restrict__ logits, int V, float* __restrict__ out, int* __restrict__ counts) {
    const float* l = logits + (size_t)blockIdx.x * V;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += 1024) m = fmaxf(m, l[i]);
    m = warp_max(m);
    __shared__ float sm[32];
    if (threadIdx.x % 32 == 0) sm[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = warp_max(sm[threadIdx.x]);
        if (threadIdx.x == 0) { out[blockIdx.x] = m; counts[blockIdx.x] = 0; }
    }
}
// pass 2: grid (blocks, rows)
extern "C" __global__ void cand_filter(const float* __restrict__ logits, int V, const float* __restrict__ rmax, float delta, int cap,
                                       int* __restrict__ counts, float2* __restrict__ out) {
    const int row = blockIdx.y;
    const float* l = logits + (size_t)row * V;
    const float thr = rmax[row] - delta;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const float v = l[i];
        if (v >= thr) {
            const int p = atomicAdd(&counts[row], 1);
            if (p < cap) out[(size_t)row * cap + p] = make_float2(v, __int_as_float(i));
        }
    }
}

// bandwidth probe: stream-read n16 uint4's
extern "C" __global__ void bw_read(const uint4* __restrict__ p, size_t n16, unsigned* __restrict__ out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = ld_stream(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

__device__ __forceinline__ void mma_s8_16832(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ================================================================================================
// gemv_tc: batched (2..8 token) Q4 GEMV on tensor cores with a cp.async pipeline.
//   CTA = 4 warps x 16 rows = 64-row tile; K processed in 256-wide chunks (8 Q4 blocks), 3 stages.
//   Weights: smem [64][128+16] bytes -> ldmatrix -> nibble split -> int8 A fragments (m16n8k32).
//   Activations: smem [8 tokens][256+16] int8 -> ldmatrix -> B fragments.
//   Work unit = (tile, k-part): split-K with atomicAdd for ADD/STORE epilogues (STORE requires zeroed y).
// ================================================================================================
#define TC_ROWS 64
#define TC_KC 256
#define TC_ST 3
#define TC_WLD (TC_KC / 2 + 16)   // 144 bytes per weight row
#define TC_ALD (TC_KC + 16)       // 272 bytes per token row
#define TC_STAGE_BYTES (TC_ROWS * TC_WLD + TC_ROWS * 16 + 8 * TC_ALD + 8 * 8 * 8)

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void cp_async_wait_n1() { asm volatile("cp.async.wait_group %0;" ::"n"(TC_ST - 2) : "memory"); }

template <int QT, int K, int EPI>
__global__ void __launch_bounds__(128) gemv_tc(const uint8_t* __restrict__ W, const half2* __restrict__ S, int N, int M, int nsplit,
                                               const int8_t* __restrict__ xq, const float2* __restrict__ xds,
                                               float* __restrict__ y, const float* __restrict__ bias,
                                               int8_t* __restrict__ oq, float2* __restrict__ ods) {
    constexpr int NCH = K / TC_KC;
    extern __shared__ __align__(16) uint8_t dsm[];
    __shared__ float res[8][32];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane >> 2, t = lane & 3;
    const int t0 = 2 * t, t1 = t0 + 1;
    const bool v0 = t0 < M, v1 = t1 < M;
    const int ntiles = (N + TC_ROWS - 1) / TC_ROWS;
    const int cpp = (NCH + nsplit - 1) / nsplit;
    const int nunits = ntiles * nsplit;
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(dsm);

    for (int unit = blockIdx.x; unit < nunits; unit += gridDim.x) {
        const int tile = unit % ntiles, part = unit / ntiles;
        const int c_begin = part * cpp, c_end = min(NCH, c_begin + cpp);
        const int row_base = tile * TC_ROWS;
        auto load_stage = [&](int ch, int st) {
            uint8_t* sb = dsm + st * TC_STAGE_BYTES;
            uint8_t* sw = sb;
            uint8_t* ss = sb + TC_ROWS * TC_WLD;
            uint8_t* sa = ss + TC_ROWS * 16;
            uint8_t* sd = sa + 8 * TC_ALD;
            if constexpr (QT == QT_Q3) {
                // raw Q3: 64 rows x 96 B = 384 chunks -> 3 per thread (unpacked to nibbles in place later)
#pragma unroll
                for (int i = 0; i < 3; i++) {
                    const int c = tid + i * 128;
                    const int r = c / 6, cc = c % 6;
                    const int gr = row_base + r;
                    const bool ok = gr < N;
                    cp_async16(sw + r * TC_WLD + cc * 16, W + (size_t)(ok ? gr : 0) * (K * 3 / 8) + ch * (TC_KC * 3 / 8) + cc * 16, ok);
                }
            } else {
                // weights: 64 rows x 128 B = 512 chunks -> 4 per thread
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int c = tid + i * 128;
                    const int r = c >> 3, cc = c & 7;
                    const int gr = row_base + r;
                    const bool ok = gr < N;
                    cp_async16(sw + r * TC_WLD + cc * 16, W + (size_t)(ok ? gr : 0) * (K / 2) + ch * (TC_KC / 2) + cc * 16, ok);
                }
            }
            // weight scales: 64 rows x 16 B
            if (tid < 64) {
                const int gr = row_base + tid;
                const bool ok = gr < N;
                cp_async16(ss + tid * 16, S + (size_t)(ok ? gr : 0) * (K / 64) + ch * (TC_KC / 64), ok);
            } else {
                // activations: 8 tokens x 256 B = 128 chunks -> threads 64..127 do 2 each
                const int u = tid - 64;
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int c = u + i * 64;
                    const int tok = c >> 4, cc = c & 15;
                    const bool ok = tok < M;
                    cp_async16(sa + tok * TC_ALD + cc * 16, xq + (size_t)(ok ? tok : 0) * K + ch * TC_KC + cc * 16, ok);
                }
                // activation scales: 8 tokens x 64 B = 32 chunks
                if (u < 32) {
                    const int tok = u >> 2, cc = u & 3;
                    const bool ok = tok < M;
                    cp_async16(sd + tok * 64 + cc * 16, xds + (size_t)(ok ? tok : 0) * (K / 32) + ch * 8 + cc * 2, ok);
                }
            }
        };
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const int nch = c_end - c_begin;
        // prologue
#pragma unroll
        for (int s = 0; s < TC_ST - 1; s++) {
            if (s < nch) load_stage(c_begin + s, s);
            cp_async_commit();
        }
        for (int i = 0; i < nch; i++) {
            cp_async_wait_n1();
            __syncthreads();
            if constexpr (QT == QT_Q3) {
                // unpack this stage's Q3 rows into the Q4 nibble layout, in place, last block first
                if (tid < TC_ROWS) {
                    uint8_t* row = dsm + (i % TC_ST) * TC_STAGE_BYTES + tid * TC_WLD;
#pragma unroll
                    for (int b = 7; b >= 0; b--) {
                        const uint32_t* src = (const uint32_t*)(row + b * 12);
                        const uint32_t L0 = src[0], L1 = src[1], H = src[2];
                        uint32_t o[4];
#pragma unroll
                        for (int k = 0; k < 4; k++) {
                            const uint32_t lo = ((L0 >> (2 * k)) & 0x03030303u) | (((H >> k) & 0x01010101u) << 2);
                            const uint32_t hi = ((L1 >> (2 * k)) & 0x03030303u) | (((H >> (4 + k)) & 0x01010101u) << 2);
                            o[k] = lo | (hi << 4);
                        }
                        *(uint4*)(row + b * 16) = make_uint4(o[0], o[1], o[2], o[3]);
                    }
                }
                __syncthreads();
            }
            {
                const int nx = i + TC_ST - 1;
                if (nx < nch) load_stage(c_begin + nx, nx % TC_ST);
                cp_async_commit();
            }
            const int st = i % TC_ST;
            const unsigned sb = sbase + st * TC_STAGE_BYTES;
            const unsigned sw = sb, sa = sb + TC_ROWS * TC_WLD + TC_ROWS * 16;
            const uint8_t* ssp = dsm + st * TC_STAGE_BYTES + TC_ROWS * TC_WLD;
            const float2* sdp = (const float2*)(dsm + st * TC_STAGE_BYTES + TC_ROWS * TC_WLD + TC_ROWS * 16 + 8 * TC_ALD);
            const int wr = warp * 16;
#pragma unroll
            for (int blk = 0; blk < 8; blk += 2) {
                uint32_t w0, w1, w2, w3, b0, b1, b2, b3;
                ldsm_x4(w0, w1, w2, w3, sw + (wr + (lane & 7) + ((lane >> 3) & 1) * 8) * TC_WLD + (blk + (lane >> 4)) * 16);
                ldsm_x4(b0, b1, b2, b3, sa + (lane & 7) * TC_ALD + blk * 32 + (lane >> 3) * 16);
                int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                mma_s8_16832(c0, w0 & 0x0F0F0F0F, w1 & 0x0F0F0F0F, (w0 >> 4) & 0x0F0F0F0F, (w1 >> 4) & 0x0F0F0F0F, b0, b1);
                mma_s8_16832(c1, w2 & 0x0F0F0F0F, w3 & 0x0F0F0F0F, (w2 >> 4) & 0x0F0F0F0F, (w3 >> 4) & 0x0F0F0F0F, b2, b3);
                // both blocks share the weight group (blk even): half2(scale,min) for rows g and g+8
                const float2 sa2 = __half22float2(*(const half2*)(ssp + (wr + g) * 16 + (blk >> 1) * 4));
                const float2 sb2 = __half22float2(*(const half2*)(ssp + (wr + g + 8) * 16 + (blk >> 1) * 4));
                const float2 da0 = sdp[t0 * 8 + blk], da1 = sdp[t1 * 8 + blk];
                const float2 db0 = sdp[t0 * 8 + blk + 1], db1 = sdp[t1 * 8 + blk + 1];
                acc[0] += sa2.x * (da0.x * (float)c0[0] + db0.x * (float)c1[0]) + sa2.y * (da0.y + db0.y);
                acc[1] += sa2.x * (da1.x * (float)c0[1] + db1.x * (float)c1[1]) + sa2.y * (da1.y + db1.y);
                acc[2] += sb2.x * (da0.x * (float)c0[2] + db0.x * (float)c1[2]) + sb2.y * (da0.y + db0.y);
                acc[3] += sb2.x * (da1.x * (float)c0[3] + db1.x * (float)c1[3]) + sb2.y * (da1.y + db1.y);
            }
        }
        cp_async_wait_all();
        __syncthreads();  // all warps done with smem before the next unit's prologue overwrites it
        const int r0 = row_base + warp * 16 + g, r1 = r0 + 8;
        if constexpr (EPI == EPI_SILU) {
            float u[4];
#pragma unroll
            for (int k = 0; k < 4; k++) u[k] = __shfl_down_sync(0xffffffff, acc[k], 4);
            if ((g & 1) == 0) {
                const int o0 = (warp * 16 + g) / 2, o1 = o0 + 4;
                if (v0) { res[t0][o0] = silu(acc[0]) * u[0]; res[t0][o1] = silu(acc[2]) * u[2]; }
                if (v1) { res[t1][o0] = silu(acc[1]) * u[1]; res[t1][o1] = silu(acc[3]) * u[3]; }
            }
            __syncthreads();
            for (int m = warp; m < M; m += 4) {
                const float v = res[m][lane];
                const float amax = warp_max(fabsf(v));
                const float dd = amax / 127.f;
                const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
                const int sm = warp_sum_i(qi);
                oq[(size_t)m * FF + tile * 32 + lane] = (int8_t)qi;
                if (lane == 0) ods[(size_t)m * (FF / 32) + tile] = make_float2(dd, dd * sm);
            }
            __syncthreads();
        } else {
            const bool add_bias = bias && part == 0;
            if (r0 < N) {
                const float bb0 = add_bias ? bias[r0] : 0.f, bb1 = add_bias ? bias[r1] : 0.f;
                if (nsplit == 1 && EPI == EPI_STORE) {
                    if (v0) { y[(size_t)t0 * N + r0] = acc[0] + bb0; y[(size_t)t0 * N + r1] = acc[2] + bb1; }
                    if (v1) { y[(size_t)t1 * N + r0] = acc[1] + bb0; y[(size_t)t1 * N + r1] = acc[3] + bb1; }
                } else {
                    if (v0) { atomicAdd(&y[(size_t)t0 * N + r0], acc[0] + bb0); atomicAdd(&y[(size_t)t0 * N + r1], acc[2] + bb1); }
                    if (v1) { atomicAdd(&y[(size_t)t1 * N + r0], acc[1] + bb0); atomicAdd(&y[(size_t)t1 * N + r1], acc[3] + bb1); }
                }
            }
        }
    }
}


// Tachyon CUDA kernels. Compiled at runtime with NVRTC; model dimensions arrive as -D constants so every
// loop bound is a compile-time constant (full unrolling, no runtime index math for shapes).
//
// Required defines: HID EPS EQ (embedding qtype: 1 F16, 2 Q8, 3 Q4). Optional: ACT_GELU, KV8.
// Per-layer shapes (head dim, GQA ratio, K of each matmul) are template arguments; everything else is runtime.
#include <cuda_fp16.h>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

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

#ifndef NORM_GROUPS
#define NORM_GROUPS 1
#endif

#ifndef KV8
#define KV8 0
#endif

// TACH_PACKED_SOURCE
__device__ __forceinline__ size_t packed_row_bytes(int qt,int k) {
    return qt==0?(size_t)k*4:qt==1||qt==12?(size_t)k*2:(size_t)(k/packed_group(qt))*packed_bytes(qt);
}
__device__ __forceinline__ float packed_read(int qt,const uint8_t* row,int k) {
    if(qt==0)return ((const float*)row)[k];
    if(qt==1)return __half2float(((const half*)row)[k]);
    if(qt==12)return __uint_as_float((unsigned)((const uint16_t*)row)[k]<<16);
    return packed_value(qt,row,k);
}
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
    return __reduce_add_sync(0xffffffff, v);
}
// Nonnegative IEEE floats have the same ordering as their unsigned bits.
// Zero is the neutral value for an all-zero (or all-NaN) quantization block.
__device__ __forceinline__ float warp_absmax(float v) {
    unsigned magnitude = __float_as_uint(fmaxf(0.f, fabsf(v)));
    return __uint_as_float(__reduce_max_sync(0xffffffff, magnitude));
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
#ifdef ACT_GELU
__device__ __forceinline__ float act_fn(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
#else
__device__ __forceinline__ float act_fn(float x) { return silu(x); }
#endif

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
// per-layer KV cache layout: [slot][kv_head][ctx][head_dim]
__device__ __forceinline__ size_t kv_sindex(int slot, int nkv, int kvh, int ctx, int pos) {
    return ((size_t)slot * nkv + kvh) * ctx + pos;
}

// ------------------------------------------------------------------------------------------------
// embedding lookup -> fp32 residual stream
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void embed(const uint8_t* __restrict__ E, const uint8_t* __restrict__ ES, const int* __restrict__ tokens, float* __restrict__ x, float scale) {
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
#elif EQ >= 5
        v = packed_read(EQ, E + (size_t)tok * packed_row_bytes(EQ,HID),i);
#elif EQ == QT_Q8
        v = __half2float(((const half*)ES)[(size_t)tok * (HID / 32) + i / 32]) * (float)((const int8_t*)E)[(size_t)tok * HID + i];
#else
        v = __half2float(((const half*)E)[(size_t)tok * HID + i]);
#endif
        x[(size_t)m * HID + i] = v * scale;
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
#if NORM_GROUPS > 1
    constexpr int width=HID/NORM_GROUPS;
    for(int g=0;g<NORM_GROUPS;g++){
        float ss=0;for(int i=threadIdx.x;i<width;i+=256){float v=xr[g*width+i];ss+=v*v;}
        ss=warp_sum(ss);if(threadIdx.x%32==0)red[threadIdx.x/32]=ss;__syncthreads();
        float sum=0;for(int i=0;i<8;i++)sum+=red[i];float inv=rsqrtf(sum/width+EPS);
        for(int b=threadIdx.x/32;b<width/32;b+=8){int i=g*width+b*32+threadIdx.x%32;float v=xr[i]*inv*w[i];
            if(OUT_F16)oh[(size_t)out_row*HID+i]=__float2half_rn(v);
            else {float peak=warp_absmax(v),d=peak/127.f;int z=d>0?__float2int_rn(v/d):0;int sumq=warp_sum_i(z);q[(size_t)out_row*HID+i]=z;if(threadIdx.x%32==0)ds[(size_t)out_row*(HID/32)+i/32]=make_float2(d,d*sumq);}}
        __syncthreads();
    }
    return;
#endif
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
            float amax = warp_absmax(v);
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

// Convert four signed bytes to scaled half pairs. In the [1024, 1280)
// interval each mantissa bit is one integer unit. Flipping the sign bit
// shifts int8 to [0, 255], which makes the subtraction exact in half.
__device__ __forceinline__ void dequant_i8x4(uint32_t bytes, half2 scale, half2* v) {
    bytes ^= 0x80808080u;
    const uint32_t lo = __byte_perm(bytes, 0x64646464u, 0x4140);
    const uint32_t hi = __byte_perm(bytes, 0x64646464u, 0x4342);
    const half2 bias = __float2half2_rn(1152.f);
    v[0] = __hmul2(__hsub2(*(const half2*)&lo, bias), scale);
    v[1] = __hmul2(__hsub2(*(const half2*)&hi, bias), scale);
}

// One thread expands one aligned group of 32 for the prefill MMA tile.
// Decode block metadata once and load contiguous words where the format permits.
template <int QT>
__device__ __forceinline__ void dequant_packed32(const uint8_t* row, int k, half2* v) {
    if constexpr (QT == 5) {  // PQ2_0, four successive ternary values per byte
        const uint8_t* block = row + (k / 128) * 34;
        const half2 scale = __half2half2(*(const half*)block);
        const uint8_t* q = block + 2 + (k % 128) / 4;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned b = q[i];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned pair = b >> (4 * j);
                const unsigned h = 0x64006400u | (pair & 3u) | ((pair & 12u) << 14);
                v[2 * i + j] = __hmul2(__hsub2(*(const half2*)&h, __float2half2_rn(1025.f)), scale);
            }
        }
    } else if constexpr (QT == 7 || QT == 8) {  // Q4_0 / Q8_0
        const uint8_t* block = row + (k / 32) * packed_bytes(QT);
        const half2 scale = __half2half2(*(const half*)block);
        const uint16_t* q = (const uint16_t*)(block + 2);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if constexpr (QT == 7) {
                const unsigned b = q[i];
                const unsigned lo = __byte_perm(b & 0x0f0fu, 0x64646464u, 0x4140);
                const unsigned hi = __byte_perm((b >> 4) & 0x0f0fu, 0x64646464u, 0x4140);
                const half2 bias = __float2half2_rn(1032.f);
                v[i] = __hmul2(__hsub2(*(const half2*)&lo, bias), scale);
                v[i + 8] = __hmul2(__hsub2(*(const half2*)&hi, bias), scale);
            } else {
                const unsigned bytes = q[2 * i] | ((unsigned)q[2 * i + 1] << 16);
                dequant_i8x4(bytes, scale, v + 2 * i);
            }
        }
    } else if constexpr (QT == 9 || QT == 10) {  // Q4_K / Q5_K
        const uint8_t* block = row + (k / 256) * packed_bytes(QT);
        const uint8_t* sc = block + 4;
        const int g = k % 256 / 32;
        const int scale = g < 4 ? sc[g] & 63 : (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4);
        const int mn = g < 4 ? sc[g + 4] & 63 : (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
        const float d = packed_half(block) * scale, m = packed_half(block + 2) * mn;
        const uint8_t* q = block + (QT == 9 ? 16 : 48) + (g / 2) * 32;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            int lo = (q[2 * i] >> (4 * (g & 1))) & 15;
            int hi = (q[2 * i + 1] >> (4 * (g & 1))) & 15;
            if constexpr (QT == 10) {
                lo |= ((block[16 + 2 * i] >> g) & 1) << 4;
                hi |= ((block[17 + 2 * i] >> g) & 1) << 4;
            }
            v[i] = __floats2half2_rn(d * lo - m, d * hi - m);
        }
    } else if constexpr (QT == 11) {  // Q6_K
        const uint8_t* block = row + (k / 256) * 210;
        const int h = k % 256 / 128, g = k % 128 / 32;
        const uint8_t* ql = block + h * 64 + (g & 1) * 32;
        const uint8_t* qh = block + 128 + h * 32;
#pragma unroll
        for (int part = 0; part < 2; part++) {
            const float d = packed_half(block + 208) * (signed char)block[192 + 8 * h + 2 * g + part];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int i = part * 16 + j * 2;
                const int a = ((ql[i] >> (4 * (g / 2))) & 15) | (((qh[i] >> (2 * g)) & 3) << 4);
                const int b = ((ql[i + 1] >> (4 * (g / 2))) & 15) | (((qh[i + 1] >> (2 * g)) & 3) << 4);
                v[part * 8 + j] = __floats2half2_rn(d * (a - 32), d * (b - 32));
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++)
            v[i] = __floats2half2_rn(packed_read(QT, row, k + 2 * i), packed_read(QT, row, k + 2 * i + 1));
    }
}

// ------------------------------------------------------------------------------------------------
// GEMM (prefill): C[M,N] = A[M,K] (fp16) x W[N,K]^T (Q4/Q8 dequantized to fp16 in shared memory)
// Tensor cores (mma.sync m16n8k16, fp32 accumulate). Tile 64(M) x BN x 64(K). Each warp computes 32 x 32 outputs.
// ------------------------------------------------------------------------------------------------
#define GBM 64
#define GBK 64
#define GLD 72  // padded smem row (halves)

template <int QT, int EPI, int BN = 128>
__global__ void __launch_bounds__(BN * 2) gemm(const half* __restrict__ A, int M, int K, const uint8_t* __restrict__ W, const void* __restrict__ S,
                                            int N, float* __restrict__ Cf, half* __restrict__ Ch, const float* __restrict__ bias) {
    static_assert(BN == 64 || BN == 128);
    constexpr int threads = BN * 2;
    constexpr int warps_n = BN / 32;
    extern __shared__ __align__(16) uint8_t smem[];
    half* As = (half*)smem;                       // [2][GBM][GLD]
    half* Ws = (half*)(smem + 2 * GBM * GLD * 2); // [2][BN][GLD]
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / warps_n, wn = warp % warps_n;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * BN;
    const int KT = K / GBK;

    // loader roles
    const int wrow = tid / 2, wpart = tid % 2;  // each thread dequantizes 32 weights of one row
    uint4 wreg0, wreg1;
    half2 wsc;
    int packed_kt=0;
    auto stage_async = [&](int kt, int buf) {
#pragma unroll
        for (int c = tid; c < GBM * (GBK / 8); c += threads) {
            int r = c / 8, col = (c % 8) * 8;
            int gr = m0 + r;
            bool ok = gr < M;
            const half* src = A + (size_t)(ok ? gr : 0) * K + kt * GBK + col;
            cp_async16(As + (buf * GBM + r) * GLD + col, src, ok);
        }
        if constexpr (QT == QT_F16) {
#pragma unroll
            for (int c = tid; c < BN * (GBK / 8); c += threads) {
                const int r = c / 8, col = (c % 8) * 8;
                const bool ok = n0 + r < N;
                const half* src = (const half*)W + (size_t)(ok ? n0 + r : 0) * K + kt * GBK + col;
                cp_async16(Ws + (buf * BN + r) * GLD + col, src, ok);
            }
        }
    };
    auto fetch_W = [&](int kt) {
        int gn = n0 + wrow;
        if (gn >= N) gn = N - 1;
        if constexpr (QT >= 5 || QT == 0 || QT == 1) { packed_kt=kt; }
        else if constexpr (QT == QT_Q4) {
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
        if constexpr (QT >= 5 || QT == 0 || QT == 1) {
            const uint8_t* row = W + (size_t)min(n0 + wrow, N - 1) * packed_row_bytes(QT, K);
            dequant_packed32<QT>(row, packed_kt * GBK + wpart * 32, v);
        } else if constexpr (QT == QT_Q4) {
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
            const uint32_t words[8] = {wreg0.x, wreg0.y, wreg0.z, wreg0.w,
                                       wreg1.x, wreg1.y, wreg1.z, wreg1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) dequant_i8x4(words[i], wsc, v + 2 * i);
        }
        uint4* dst = (uint4*)(Ws + (buf * BN + wrow) * GLD + wpart * 32);
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

    stage_async(0, 0);
    cp_async_commit();
    fetch_W(0);
    if constexpr (QT != QT_F16) store_W(0);
    cp_async_wait_all();
    __syncthreads();

    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) {
            stage_async(kt + 1, buf ^ 1);
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
                const half* p = Ws + (buf * BN + wn * 32 + ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * GLD + kk * 16 + ((lane >> 3) & 1) * 8;
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
            if constexpr (QT != QT_F16) store_W(buf ^ 1);
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
                    Ch[(size_t)r * (N / 2) + col / 2] = __float2half(act_fn(v0) * v1);
                } else {
                    if (bias) { v0 += bias[col]; v1 += bias[col + 1]; }
                    float2* dst = (float2*)(Cf + (size_t)r * N + col);
                    if (EPI == EPI_ADD) { float2 o = *dst; o.x += v0; o.y += v1; *dst = o; }
                    else *dst = make_float2(v0, v1);
                }
            }
}

// ------------------------------------------------------------------------------------------------
// Attention input stage, one warp per (token, head): per-head QK-norm, RoPE (pairs (i, i+half); only
// frequencies with inv_freq != 0 rotate), query scaling, optional value norm, KV-cache write.
// q projection rows per head are [q | gate] when the layer has an output gate (qstride = 2*HD).
// grid (M, ceil((nh+nkv)/4)), block 128
// ------------------------------------------------------------------------------------------------
struct RopeArgs {
    const float* qkv; const float* qn; const float* kn; const float* inv_freq;
    const int* pos; const int* slot;
    void* qout; kv_t* kc; kv_t* vc; float* ksc; float* vsc;
    int nrows, nh, nkv, qstride, qrows, kvd, half, qk_norm, v_norm, k_eq_v, own_kv, ctx;
    float qscale, mscale;
};
template <int HD, int Q_F16>
__global__ void __launch_bounds__(128) rope_t(const RopeArgs a) {
    constexpr int E = HD / 32;
    __shared__ float buf[4][HD];
    const int m = blockIdx.x;
    const int w = threadIdx.x / 32;
    const int head = blockIdx.y * 4 + w;
    const int lane = threadIdx.x % 32;
    if (head >= a.nh + (a.own_kv ? a.nkv : 0)) return;
    const bool is_q = head < a.nh;
    const int kvh = head - a.nh;
    const float* row = a.qkv + (size_t)m * a.nrows;
    const float* src = is_q ? row + head * a.qstride : row + a.qrows + kvh * HD;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; e++) v[e] = src[lane + 32 * e];
    if (a.qk_norm == 1) {
        float ss = 0;
#pragma unroll
        for (int e = 0; e < E; e++) ss += v[e] * v[e];
        ss = warp_sum(ss);
        const float sc = rsqrtf(ss / HD + EPS);
        const float* nw = is_q ? a.qn : a.kn;
#pragma unroll
        for (int e = 0; e < E; e++) v[e] = v[e] * sc * nw[lane + 32 * e];
    }
#pragma unroll
    for (int e = 0; e < E; e++) buf[w][lane + 32 * e] = v[e];
    __syncwarp();
    const int pos = a.pos[m];
#pragma unroll
    for (int e = 0; e < E; e++) {
        const int i = lane + 32 * e;
        if (i < 2 * a.half) {
            const int f = i < a.half ? i : i - a.half;
            const float fr = a.inv_freq[f];
            if (fr != 0.f) {
                float sn, cs;
                sincosf((float)pos * fr, &sn, &cs);
                const float x0 = buf[w][f], x1 = buf[w][f + a.half];
                v[e] = (i < a.half ? x0 * cs - x1 * sn : x0 * sn + x1 * cs) * a.mscale;
            }
        }
    }
    if (is_q) {
        if (Q_F16) {
            half* q = (half*)a.qout + (size_t)m * a.nh * HD + head * HD;
#pragma unroll
            for (int e = 0; e < E; e++) q[lane + 32 * e] = __float2half(v[e] * a.qscale);
        } else {
            float* q = (float*)a.qout + (size_t)m * a.nh * HD + head * HD;
#pragma unroll
            for (int e = 0; e < E; e++) q[lane + 32 * e] = v[e] * a.qscale;
        }
        return;
    }
    // values: raw k projection (k_eq_v) or the v projection; optional unweighted RMSNorm
    const float* vs = a.k_eq_v ? src : row + a.qrows + a.kvd + kvh * HD;
    float vv[E];
#pragma unroll
    for (int e = 0; e < E; e++) vv[e] = vs[lane + 32 * e];
    if (a.v_norm) {
        float ss = 0;
#pragma unroll
        for (int e = 0; e < E; e++) ss += vv[e] * vv[e];
        ss = warp_sum(ss);
        const float sc = rsqrtf(ss / HD + EPS);
#pragma unroll
        for (int e = 0; e < E; e++) vv[e] *= sc;
    }
    const int slot = a.slot[m];
    const size_t si = kv_sindex(slot, a.nkv, kvh, a.ctx, pos % a.ctx);  // ctx = ring capacity for sliding layers
    const size_t base = si * HD;
#if KV8
    float ka = 0.f, va = 0.f;
#pragma unroll
    for (int e = 0; e < E; e++) { ka = fmaxf(ka, fabsf(v[e])); va = fmaxf(va, fabsf(vv[e])); }
    ka = warp_max(ka); va = warp_max(va);
    const float ki = ka > 0 ? 127.f / ka : 0.f, vi = va > 0 ? 127.f / va : 0.f;
#pragma unroll
    for (int e = 0; e < E; e++) {
        a.kc[base + lane + 32 * e] = (int8_t)__float2int_rn(v[e] * ki);
        a.vc[base + lane + 32 * e] = (int8_t)__float2int_rn(vv[e] * vi);
    }
    if (lane == 0) { a.ksc[si] = ka / 127.f; a.vsc[si] = va / 127.f; }
#else
#pragma unroll
    for (int e = 0; e < E; e++) {
        a.kc[base + lane + 32 * e] = __float2half(v[e]);
        a.vc[base + lane + 32 * e] = __float2half(vv[e]);
    }
#endif
}

// RMSNorm over a contiguous slice of every row, in place (OLMoE full-width q / k norm). grid M, block 256
extern "C" __global__ void __launch_bounds__(256) norm_slice(float* __restrict__ base, int stride, int offset, int n, const float* __restrict__ w) {
    float* x = base + (size_t)blockIdx.x * stride + offset;
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += 256) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    float t = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) t += red[i];
    const float sc = rsqrtf(t / n + EPS);
    for (int i = threadIdx.x; i < n; i += 256) x[i] = x[i] * sc * w[i];
}

// ------------------------------------------------------------------------------------------------
// Prefill flash attention (FA2-style, tensor cores), HD <= 128. Q fp16 [M][nh*HD] for tokens at positions
// p0..p0+M-1 of one slot; optional sliding window and sigmoid output gate (gate rows in qkv, stride nrows).
// grid (ceil(M/64), nh), block 128 (4 warps x 16 query rows). K/V tiles of 64 positions in shared memory.
// ------------------------------------------------------------------------------------------------
#define FKV 64
template <int HD>
__global__ void __launch_bounds__(128) flash_prefill(const half* __restrict__ Q, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                                     half* __restrict__ O, int M, int p0, int slot, int nh, int nkv, int ctx, int window,
                                                     const float* __restrict__ gate, int nrows, int qstride,
                                                     const float* __restrict__ ksc, const float* __restrict__ vsc) {
    constexpr int FLD = HD + 8;
    extern __shared__ __align__(16) uint8_t smem[];
    half* Ks = (half*)smem;            // [FKV][FLD]
    half* Vs = Ks + FKV * FLD;         // [FKV][FLD]
    const int QD = nh * HD;
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int qb = blockIdx.x * 64;
    const int q0 = qb + warp * 16;
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
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
    const int qpos0 = p0 + q0 + (lane >> 2);
    const int last_q = min(M, qb + 64) - 1;
    const int n_kv = p0 + last_q + 1;
    const int kv_lo = window > 0 ? max(0, p0 + qb - window + 1) & ~(FKV - 1) : 0;
    const kv_t* kb = kc + kv_sindex(slot, nkv, kvh, ctx, 0) * HD;
    const kv_t* vb = vc + kv_sindex(slot, nkv, kvh, ctx, 0) * HD;
#if KV8
    const float* ksb = ksc + kv_sindex(slot, nkv, kvh, ctx, 0);
    const float* vsb = vsc + kv_sindex(slot, nkv, kvh, ctx, 0);
#endif
    for (int kv0 = kv_lo; kv0 < n_kv; kv0 += FKV) {
        __syncthreads();
#if KV8
        for (int c = threadIdx.x; c < FKV * HD / 16; c += 128) {
            const int r = c / (HD / 16), col = (c % (HD / 16)) * 16;
            const bool ok = kv0 + r < n_kv;
            half2 hk[8], hv[8];
            if (ok) {
                const int pr = (kv0 + r) % ctx;
                const uint4 rk = *(const uint4*)(kb + (size_t)pr * HD + col);
                const uint4 rv = *(const uint4*)(vb + (size_t)pr * HD + col);
                const float sk = ksb[pr], sv = vsb[pr];
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
        for (int c = threadIdx.x; c < FKV * HD / 8; c += 128) {
            const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
            const bool ok = kv0 + r < n_kv;
            const size_t off = (size_t)(ok ? (kv0 + r) % ctx : 0) * HD + col;
            cp_async16(Ks + r * FLD + col, kb + off, ok);
            cp_async16(Vs + r * FLD + col, vb + off, ok);
        }
        cp_async_commit();
        cp_async_wait_all();
#endif
        __syncthreads();
        if (kv0 > p0 + q0 + 15) continue;                                    // tile entirely in the future
        if (window > 0 && kv0 + FKV - 1 < p0 + q0 - window + 1) continue;    // tile entirely out of the window
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
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int ni = 0; ni < FKV / 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int kpos = kv0 + ni * 8 + (lane & 3) * 2 + (e & 1);
                const int qpos = qpos0 + (e >> 1) * 8;
                float v = s[ni][e] * sl2;
                if (kpos > qpos || (window > 0 && kpos <= qpos - window)) v = -INFINITY;
                s[ni][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            const float mn = fmaxf(mrow[r], mx[r]);
            alpha[r] = mn == -INFINITY ? 1.f : exp2f(mrow[r] - mn);
            mrow[r] = mn;
            lrow[r] *= alpha[r];
        }
#pragma unroll
        for (int ni = 0; ni < FKV / 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float m_ = mrow[e >> 1];
                const float pv = m_ == -INFINITY ? 0.f : exp2f(s[ni][e] - m_);
                s[ni][e] = pv;
                lrow[e >> 1] += pv;
            }
#pragma unroll
        for (int i = 0; i < HD / 8; i++) { o[i][0] *= alpha[0]; o[i][1] *= alpha[0]; o[i][2] *= alpha[1]; o[i][3] *= alpha[1]; }
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
        const int d = dn * 8 + (lane & 3) * 2;
        const int col = h * HD + d;
        float v00 = o[dn][0] * il0, v01 = o[dn][1] * il0, v10 = o[dn][2] * il1, v11 = o[dn][3] * il1;
        if (gate) {
            if (r0 < M) { const float* g = gate + (size_t)r0 * nrows + h * qstride + HD + d; v00 *= 1.f / (1.f + __expf(-g[0])); v01 *= 1.f / (1.f + __expf(-g[1])); }
            if (r1 < M) { const float* g = gate + (size_t)r1 * nrows + h * qstride + HD + d; v10 *= 1.f / (1.f + __expf(-g[0])); v11 *= 1.f / (1.f + __expf(-g[1])); }
        }
        if (r0 < M) *(half2*)(O + (size_t)r0 * QD + col) = __floats2half2_rn(v00, v01);
        if (r1 < M) *(half2*)(O + (size_t)r1 * QD + col) = __floats2half2_rn(v10, v11);
    }
}

// ------------------------------------------------------------------------------------------------
// Prefill flash attention for wide heads (HD = 256 / 512, Gemma 3 / 4). The head dimension is split into
// 128-wide slices, one per warp: every warp computes the partial S = Q K^T of its slice with tensor cores,
// the partials are summed through shared memory (canonical order, so all slices of a query group see
// bit-identical scores), each warp runs the same online softmax, then accumulates P V for its slice.
// CTA = 4 warps = (4 / NS) query groups of 16 rows x NS slices. One smem tile buffer holds K, then V.
// grid (ceil(M / QR), nh), block 128. Same arguments as flash_prefill.
// ------------------------------------------------------------------------------------------------
#define WKV 32
template <int HD>
__global__ void __launch_bounds__(128) flash_wide(const half* __restrict__ Q, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                                  half* __restrict__ O, int M, int p0, int slot, int nh, int nkv, int ctx, int window,
                                                  const float* __restrict__ gate, int nrows, int qstride,
                                                  const float* __restrict__ ksc, const float* __restrict__ vsc) {
    constexpr int NS = HD / 128, NQ = 4 / NS, QR = 16 * NQ, FLD = HD + 8, NF = WKV / 8;
    extern __shared__ __align__(16) uint8_t smem[];
    half* Ts = (half*)smem;                          // [WKV][FLD]: K tile, then V tile
    float* Sp = (float*)(smem + WKV * FLD * 2);      // [4 warps][NF * 4][32 lanes] partial scores
    const int QD = nh * HD;
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int qg = warp / NS, sl = warp % NS;
    const int qb = blockIdx.x * QR;
    const int q0 = qb + qg * 16;
    const int dc0 = sl * 128;                        // first head dim of this warp's slice
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
    uint32_t qf[8][4];
    {
        const int r0 = q0 + (lane >> 2), r1 = r0 + 8;
        const int c = (lane & 3) * 2;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            const half* p0r = Q + (size_t)r0 * QD + h * HD + dc0 + kk * 16 + c;
            const half* p1r = Q + (size_t)r1 * QD + h * HD + dc0 + kk * 16 + c;
            qf[kk][0] = r0 < M ? *(const uint32_t*)p0r : 0u;
            qf[kk][1] = r1 < M ? *(const uint32_t*)p1r : 0u;
            qf[kk][2] = r0 < M ? *(const uint32_t*)(p0r + 8) : 0u;
            qf[kk][3] = r1 < M ? *(const uint32_t*)(p1r + 8) : 0u;
        }
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    const int qpos0 = p0 + q0 + (lane >> 2);
    const int last_q = min(M, qb + QR) - 1;
    const int n_kv = p0 + last_q + 1;
    const int kv_lo = window > 0 ? max(0, p0 + qb - window + 1) & ~(WKV - 1) : 0;
    const size_t sb = kv_sindex(slot, nkv, kvh, ctx, 0);
    const kv_t* kb = kc + sb * HD;
    const kv_t* vb = vc + sb * HD;
#if KV8
    const float* ksb = ksc + sb;
    const float* vsb = vsc + sb;
#endif
    auto load_tile = [&](const kv_t* base, const float* scb, int kv0) {
#if KV8
        for (int c = threadIdx.x; c < WKV * HD / 16; c += 128) {
            const int r = c / (HD / 16), col = (c % (HD / 16)) * 16;
            half2 hv[8];
            if (kv0 + r < n_kv) {
                const int pr = (kv0 + r) % ctx;
                const uint4 raw = *(const uint4*)(base + (size_t)pr * HD + col);
                const float sc = scb[pr];
                const int8_t* b8 = (const int8_t*)&raw;
#pragma unroll
                for (int j = 0; j < 8; j++) hv[j] = __floats2half2_rn(b8[2 * j] * sc, b8[2 * j + 1] * sc);
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) hv[j] = __floats2half2_rn(0.f, 0.f);
            }
            uint4* d = (uint4*)(Ts + r * FLD + col);
            d[0] = *(uint4*)&hv[0]; d[1] = *(uint4*)&hv[4];
        }
#else
        for (int c = threadIdx.x; c < WKV * HD / 8; c += 128) {
            const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
            const bool ok = kv0 + r < n_kv;
            cp_async16(Ts + r * FLD + col, base + (size_t)(ok ? (kv0 + r) % ctx : 0) * HD + col, ok);
        }
        cp_async_commit();
#endif
    };
    for (int kv0 = kv_lo; kv0 < n_kv; kv0 += WKV) {
#if KV8
        load_tile(kb, ksb, kv0);
#else
        load_tile(kb, nullptr, kv0);
        cp_async_wait_all();
#endif
        __syncthreads();
        // same for all slices of a query group: tile entirely in the future / entirely out of the window
        const bool active = !(kv0 > p0 + q0 + 15) && !(window > 0 && kv0 + WKV - 1 < p0 + q0 - window + 1);
        float s[NF][4];
        if (active) {
#pragma unroll
            for (int i = 0; i < NF; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
#pragma unroll
                for (int ni = 0; ni < NF; ni += 2) {
                    uint32_t b0, b1, b2, b3;
                    const half* p = Ts + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * FLD + dc0 + kk * 16 + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(b0, b1, b2, b3, p);
                    mma16816(s[ni], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                    mma16816(s[ni + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
                }
            }
#pragma unroll
            for (int ni = 0; ni < NF; ni++)
#pragma unroll
                for (int e = 0; e < 4; e++) Sp[(warp * NF * 4 + ni * 4 + e) * 32 + lane] = s[ni][e];
        }
        __syncthreads();   // partial scores visible; nobody reads the K tile any more
#if KV8
        load_tile(vb, vsb, kv0);
#else
        load_tile(vb, nullptr, kv0);
#endif
        float alpha[2] = {1.f, 1.f};
        if (active) {
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int ni = 0; ni < NF; ni++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float v = 0.f;
#pragma unroll
                    for (int k = 0; k < NS; k++) v += Sp[((qg * NS + k) * NF * 4 + ni * 4 + e) * 32 + lane];
                    const int kpos = kv0 + ni * 8 + (lane & 3) * 2 + (e & 1);
                    const int qpos = qpos0 + (e >> 1) * 8;
                    v *= sl2;
                    if (kpos > qpos || (window > 0 && kpos <= qpos - window)) v = -INFINITY;
                    s[ni][e] = v;
                    mx[e >> 1] = fmaxf(mx[e >> 1], v);
                }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
                const float mn = fmaxf(mrow[r], mx[r]);
                alpha[r] = mn == -INFINITY ? 1.f : exp2f(mrow[r] - mn);
                mrow[r] = mn;
                lrow[r] *= alpha[r];
            }
#pragma unroll
            for (int ni = 0; ni < NF; ni++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float m_ = mrow[e >> 1];
                    const float pv = m_ == -INFINITY ? 0.f : exp2f(s[ni][e] - m_);
                    s[ni][e] = pv;
                    lrow[e >> 1] += pv;
                }
#pragma unroll
            for (int i = 0; i < 16; i++) { o[i][0] *= alpha[0]; o[i][1] *= alpha[0]; o[i][2] *= alpha[1]; o[i][3] *= alpha[1]; }
        }
#if !KV8
        cp_async_wait_all();
#endif
        __syncthreads();
        if (active) {
#pragma unroll
            for (int kk = 0; kk < WKV / 16; kk++) {
                const uint32_t a0 = pack_half2(s[2 * kk][0], s[2 * kk][1]);
                const uint32_t a1 = pack_half2(s[2 * kk][2], s[2 * kk][3]);
                const uint32_t a2 = pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                const uint32_t a3 = pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
                for (int dn = 0; dn < 16; dn += 2) {
                    uint32_t b0, b1, b2, b3;
                    const half* p = Ts + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FLD + dc0 + (dn + (lane >> 4)) * 8;
                    ldmatrix_x4_trans(b0, b1, b2, b3, p);
                    mma16816(o[dn], a0, a1, a2, a3, b0, b1);
                    mma16816(o[dn + 1], a0, a1, a2, a3, b2, b3);
                }
            }
        }
        __syncthreads();   // V tile fully consumed before the next K tile overwrites it
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 1);
        lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 2);
    }
    const int r0 = q0 + (lane >> 2), r1 = r0 + 8;
    const float il0 = 1.f / lrow[0], il1 = 1.f / lrow[1];
#pragma unroll
    for (int dn = 0; dn < 16; dn++) {
        const int d = dc0 + dn * 8 + (lane & 3) * 2;
        const int col = h * HD + d;
        float v00 = o[dn][0] * il0, v01 = o[dn][1] * il0, v10 = o[dn][2] * il1, v11 = o[dn][3] * il1;
        if (gate) {
            if (r0 < M) { const float* g = gate + (size_t)r0 * nrows + h * qstride + HD + d; v00 *= 1.f / (1.f + __expf(-g[0])); v01 *= 1.f / (1.f + __expf(-g[1])); }
            if (r1 < M) { const float* g = gate + (size_t)r1 * nrows + h * qstride + HD + d; v10 *= 1.f / (1.f + __expf(-g[0])); v11 *= 1.f / (1.f + __expf(-g[1])); }
        }
        if (r0 < M) *(half2*)(O + (size_t)r0 * QD + col) = __floats2half2_rn(v00, v01);
        if (r1 < M) *(half2*)(O + (size_t)r1 * QD + col) = __floats2half2_rn(v10, v11);
    }
}

// ------------------------------------------------------------------------------------------------
// Greedy selection. Every reduction breaks ties by the lowest token id.
__device__ __forceinline__ int warp_argmax(float& best, int id) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, best, o);
        int other = __shfl_xor_sync(0xffffffff, id, o);
        if (v > best || (v == best && other < id)) { best = v; id = other; }
    }
    return id;
}

extern "C" __global__ void __launch_bounds__(256) argmax_rows(const float* logits, int V, int* ids) {
    const float* row = logits + (size_t)blockIdx.y * V;
    float best = -INFINITY;
    int id = 0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < V; i += gridDim.x * 256) {
        if (row[i] > best) { best = row[i]; id = i; }
    }
    id = warp_argmax(best, id);
    __shared__ float values[8];
    __shared__ int indices[8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (!lane) { values[warp] = best; indices[warp] = id; }
    __syncthreads();
    if (!warp) {
        best = lane < 8 ? values[lane] : -INFINITY;
        id = warp_argmax(best, lane < 8 ? indices[lane] : 0);
        if (!lane) ids[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = id;
    }
}

// One warp completes a large row using at most 32 partial winners.
extern "C" __global__ void argmax_finish(const float* logits, int V, const int* partial, int parts, int* out) {
    const int r = blockIdx.x, lane = threadIdx.x;
    int id = lane < parts ? partial[r * parts + lane] : 0;
    float v = lane < parts ? logits[(size_t)r * V + id] : -INFINITY;
    if (isnan(v)) v = -INFINITY;
    id = warp_argmax(v, id);
    if (!lane) out[r] = id;
}

// copy KV positions [0,n) between slots of one layer. grid nkv, block 256
extern "C" __global__ void kv_copy(kv_t* kc, kv_t* vc, int src, int dst, int n, int nkv, int hd, int ctx, float* ksc, float* vsc) {
    const int h = blockIdx.x;
    const size_t s = kv_sindex(src, nkv, h, ctx, 0) * hd, d = kv_sindex(dst, nkv, h, ctx, 0) * hd;
    const size_t cnt = (size_t)n * hd * sizeof(kv_t) / 16;
    for (size_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        ((uint4*)(kc + d))[i] = ((const uint4*)(kc + s))[i];
        ((uint4*)(vc + d))[i] = ((const uint4*)(vc + s))[i];
    }
#if KV8
    const size_t ss = kv_sindex(src, nkv, h, ctx, 0), ds = kv_sindex(dst, nkv, h, ctx, 0);
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
            const float amax = warp_absmax(v);
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
                if (lane == 0) res[m][warp * 4 + r / 2] = act_fn(gs) * us;
            }
        }
        __syncthreads();
        for (int m = warp; m < M; m += 8) {
            const float v = res[m][lane];
            const float amax = warp_absmax(v);
            const float dd = amax / 127.f;
            const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
            const int sm = warp_sum_i(qi);
            oq[(size_t)m * (N / 2) + tile * 32 + lane] = (int8_t)qi;
            if (lane == 0) ods[(size_t)m * (N / 64) + tile] = make_float2(dd, dd * sm);
        }
        __syncthreads();
    }
}

// Decode attention with the split-combine fused in: the last CTA of each (token, kv-head) group to finish
// merges all splits and writes the head outputs (int8-quantized for the GEMV path, or fp16 for GEMMs).
// Handles any positions/slots per token, sliding windows and sigmoid output gates.
// grid (nkv, SPLITS, M), block 128
struct AttnArgs {
    const float* q; const kv_t* kc; const kv_t* vc; const int* pos; const int* slot;
    float* part_o; float2* part_ml; int* counters;
    int8_t* oq; float2* ods; half* oh;
    const float* gate; const float* ksc; const float* vsc;
    int nh, nkv, ctx, window, nrows, qstride;
};
// G = query heads handled per CTA (a divisor of the GQA ratio, <= 4) to bound shared memory / registers.
// blockIdx.x = kv_head * (gqa / G) + sub-group.
// OUT_TYPE: 0 int8 plus scales, 1 fp16, 2 fp32 (oh points to float storage).
template <int HD, int GQA, int SPLITS, int OUT_TYPE>
__global__ void __launch_bounds__(128) attn_decode(const AttnArgs a) {
    const int gqa_full = a.nh / a.nkv;
    const int ngroups = gqa_full / GQA;
    const int kvh = blockIdx.x / ngroups, sub = blockIdx.x % ngroups, split = blockIdx.y, m = blockIdx.z;
    const int head0 = kvh * gqa_full + sub * GQA;   // first query head of this CTA
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int QD = a.nh * HD;
    const int pos = a.pos[m];
    const int len = pos + 1;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int span = len - lo;
    const int slot = a.slot[m];
    int chunk = (span + SPLITS - 1) / SPLITS;
    chunk = max(64, (chunk + 63) & ~63);
    const int start = lo + split * chunk;
    const int end = min(len, start + chunk);
    const size_t pidx = ((size_t)m * a.nh + head0) * SPLITS + split;
    __shared__ float qs[GQA][HD];
    __shared__ float ps[GQA][ATILE];
    __shared__ float alpha_s[GQA];
    __shared__ float m_s[GQA], l_s[GQA];
    constexpr int NDO = HD / 8;          // dim octets (16 B of V per load)
    constexpr int PG = 128 / NDO;        // position groups
    __shared__ __align__(16) float red[PG][GQA][HD];
    __shared__ int is_last;
    if (start < len) {
        const float scale = rsqrtf((float)HD) * 1.4426950408889634f;
        for (int i = tid; i < GQA * HD; i += 128) qs[i / HD][i % HD] = a.q[(size_t)m * QD + head0 * HD + i] * scale;
        if (tid < GQA) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
        const size_t sb = kv_sindex(slot, a.nkv, kvh, a.ctx, 0);
        const kv_t* kb = a.kc + sb * HD;
        const kv_t* vb = a.vc + sb * HD;
#if KV8
        const float* ksb = a.ksc + sb;
        const float* vsb = a.vsc + sb;
#endif
        const int dq = tid % NDO, pg = tid / NDO;
        float acc[GQA][8];
#pragma unroll
        for (int g = 0; g < GQA; g++)
#pragma unroll
            for (int d = 0; d < 8; d++) acc[g][d] = 0.f;
        __syncthreads();
        for (int t0 = start; t0 < end; t0 += ATILE) {
            // Adjacent lanes read adjacent 16-byte pieces of a key. Grouping
            // lanes by position coalesces the reads even with wide heads.
            constexpr int elems = KV8 ? 16 : 8;
            constexpr int lanes_k = HD >= 128 ? 8 : 4;
            constexpr int positions = 128 / lanes_k;
            const int ring0 = t0 < a.ctx ? t0 : t0 % a.ctx;
            for (int i = tid / lanes_k; i < ATILE; i += positions) {
                float score[GQA] = {};
                int pr = ring0 + i;
                if (pr >= a.ctx) pr -= a.ctx;
                if (t0 + i < end) {
                    const uint4* kr = (const uint4*)(kb + (size_t)pr * HD);
#pragma unroll
                    for (int c = tid % lanes_k; c < HD / elems; c += lanes_k) {
                        const uint4 kv = kr[c];
#if KV8
                        const int8_t* v = (const int8_t*)&kv;
#pragma unroll
                        for (int j = 0; j < elems; j++)
#pragma unroll
                            for (int g = 0; g < GQA; g++) score[g] += qs[g][c * elems + j] * (float)v[j];
#else
                        const half2* v = (const half2*)&kv;
#pragma unroll
                        for (int j = 0; j < elems / 2; j++) {
                            const float2 f = __half22float2(v[j]);
#pragma unroll
                            for (int g = 0; g < GQA; g++)
                                score[g] += qs[g][c * elems + 2 * j] * f.x + qs[g][c * elems + 2 * j + 1] * f.y;
                        }
#endif
                    }
                }
#pragma unroll
                for (int g = 0; g < GQA; g++) {
#pragma unroll
                    for (int d = lanes_k / 2; d; d >>= 1)
                        score[g] += __shfl_down_sync(0xffffffff, score[g], d, lanes_k);
                    if (tid % lanes_k == 0) {
                        float v = -INFINITY;
                        if (t0 + i < end) {
                            v = score[g];
#if KV8
                            v *= ksb[pr];
#endif
                        }
                        ps[g][i] = v;
                    }
                }
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
                if (lane == 0) { const float al = exp2f(mo - mn); alpha_s[g] = al; l_s[g] = l_s[g] * al + sum; m_s[g] = mn; }
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GQA; g++) {
                const float al = alpha_s[g];
#pragma unroll
                for (int d = 0; d < 8; d++) acc[g][d] *= al;
            }
            const int nt = min(ATILE, end - t0);
#pragma unroll 4
            for (int i = pg; pg < PG && i < nt; i += PG) {
                int pr = ring0 + i;
                if (pr >= a.ctx) pr -= a.ctx;
                float vv[8];
#if KV8
                const uint2 raw = *(const uint2*)(vb + (size_t)pr * HD + dq * 8);
                const int8_t* b8 = (const int8_t*)&raw;
                const float vsc_i = vsb[pr];
#pragma unroll
                for (int j = 0; j < 8; j++) vv[j] = (float)b8[j];
#else
                const uint4 raw = *(const uint4*)(vb + (size_t)pr * HD + dq * 8);
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
        if (pg < PG) {
#pragma unroll
            for (int g = 0; g < GQA; g++)
#pragma unroll
                for (int d = 0; d < 8; d++) red[pg][g][dq * 8 + d] = acc[g][d];
        }
        __syncthreads();
        for (int i = tid; i < GQA * HD; i += 128) {
            const int g = i / HD, d = i % HD;
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < PG; k++) sum += red[k][g][d];
            a.part_o[(pidx + g * SPLITS) * HD + d] = sum;
        }
        if (tid < GQA) a.part_ml[pidx + tid * SPLITS] = make_float2(m_s[tid], l_s[tid]);
    } else {
        if (tid < GQA) a.part_ml[pidx + tid * SPLITS] = make_float2(-INFINITY, 0.f);
    }
    if (SPLITS > 1) {
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            int* cnt = a.counters + (size_t)m * (a.nh / GQA) + blockIdx.x;
            const int prev = atomicAdd(cnt, 1);
            is_last = prev == SPLITS - 1;
            if (is_last) *cnt = 0;
        }
        __syncthreads();
        if (!is_last) return;
        __threadfence();
    } else {
        __syncthreads();
    }
    for (int i = tid; i < GQA * HD; i += 128) {
        const int g = i / HD, d = i % HD;
        const size_t base = ((size_t)m * a.nh + head0 + g) * SPLITS;
        float mx = -INFINITY;
#pragma unroll
        for (int s2 = 0; s2 < SPLITS; s2++) mx = fmaxf(mx, __ldcg(&a.part_ml[base + s2]).x);
        float L = 0.f, o = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < SPLITS; s2++) {
            const float2 ml = __ldcg(&a.part_ml[base + s2]);
            if (ml.y == 0.f) continue;
            const float w = exp2f(ml.x - mx);
            L += w * ml.y;
            o += w * __ldcg(&a.part_o[(base + s2) * HD + d]);
        }
        float v = o / L;
        const int head = head0 + g;
        if (a.gate) v *= 1.f / (1.f + __expf(-a.gate[(size_t)m * a.nrows + head * a.qstride + HD + d]));
        const int col = head * HD + d;
        if (OUT_TYPE == 2) {
            ((float*)a.oh)[(size_t)m * QD + col] = v;
        } else if (OUT_TYPE == 1) {
            a.oh[(size_t)m * QD + col] = __float2half(v);
        } else {
            const float amax = warp_absmax(v);
            const float dd = amax / 127.f;
            const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
            const int sm = warp_sum_i(qi);
            a.oq[(size_t)m * QD + col] = (int8_t)qi;
            if (lane == 0) a.ods[(size_t)m * (QD / 32) + col / 32] = make_float2(dd, dd * sm);
        }
    }
}

// ---- sampling support: compact candidate lists (logit >= rowmax - delta) for host-side top-k/top-p ----
// pass 1: 32 partial maxima per row (grid 32 x rows, block 256).
// Large vocabularies need multiple CTAs to use the memory bandwidth of the GPU.
extern "C" __global__ void __launch_bounds__(256) row_max(const float* __restrict__ logits, int V, float* __restrict__ out, int* __restrict__ counts) {
    const int row = blockIdx.y;
    const float* l = logits + (size_t)row * V;
    float m = -INFINITY;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < V; i += 32 * 256) m = fmaxf(m, l[i]);
    m = warp_max(m);
    __shared__ float sm[32];
    if (threadIdx.x % 32 == 0) sm[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = warp_max(threadIdx.x < 8 ? sm[threadIdx.x] : -INFINITY);
        if (threadIdx.x == 0) {
            out[(size_t)row * 32 + blockIdx.x] = m;
            if (blockIdx.x == 0) counts[row] = 0;
        }
    }
}
// pass 2: grid (blocks, rows)
extern "C" __global__ void cand_filter(const float* __restrict__ logits, int V, const float* __restrict__ rmax, float delta, int cap,
                                       int* __restrict__ counts, float2* __restrict__ out) {
    const int row = blockIdx.y;
    const float* l = logits + (size_t)row * V;
    const float thr = warp_max(rmax[(size_t)row * 32 + (threadIdx.x & 31)]) - delta;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const float v = l[i];
        // Reserve one contiguous range per warp instead of contending once per token.
        const unsigned active = __activemask();
        const unsigned mask = __ballot_sync(active, v >= thr);
        if (mask) {
            const int leader = __ffs(mask) - 1, lane = threadIdx.x & 31;
            int base = 0;
            if (lane == leader) base = atomicAdd(&counts[row], __popc(mask));
            base = __shfl_sync(active, base, leader);
            const int p = base + __popc(mask & ((1u << lane) - 1));
            if ((mask & (1u << lane)) && p < cap) out[(size_t)row * cap + p] = make_float2(v, __int_as_float(i));
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
                if (v0) { res[t0][o0] = act_fn(acc[0]) * u[0]; res[t0][o1] = act_fn(acc[2]) * u[2]; }
                if (v1) { res[t1][o0] = act_fn(acc[1]) * u[1]; res[t1][o1] = act_fn(acc[3]) * u[3]; }
            }
            __syncthreads();
            for (int m = warp; m < M; m += 4) {
                const float v = res[m][lane];
                const float amax = warp_absmax(v);
                const float dd = amax / 127.f;
                const int qi = amax > 0 ? __float2int_rn(v / dd) : 0;
                const int sm = warp_sum_i(qi);
                oq[(size_t)m * (N / 2) + tile * 32 + lane] = (int8_t)qi;
                if (lane == 0) ods[(size_t)m * (N / 64) + tile] = make_float2(dd, dd * sm);
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



// ================================================================================================
// Generic helpers for the generalized model graph
// ================================================================================================
// quantize rows of fp32 [M][K] to int8 blocks (+ scale, scale*sum) for GEMV inputs. grid M, block 256
extern "C" __global__ void __launch_bounds__(256) quant_rows(const float* __restrict__ x, int K, int8_t* __restrict__ q, float2* __restrict__ ds) {
    const int m = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int blk = warp; blk < K / 32; blk += 8) {
        const float v = x[(size_t)m * K + blk * 32 + lane];
        const float amax = warp_absmax(v);
        const float d = amax / 127.f;
        const int qi = amax > 0 ? __float2int_rn(v / d) : 0;
        const int sm = warp_sum_i(qi);
        q[(size_t)m * K + blk * 32 + lane] = (int8_t)qi;
        if (lane == 0) ds[(size_t)m * (K / 32) + blk] = make_float2(d, d * sm);
    }
}
extern "C" __global__ void f32_to_f16(const float* __restrict__ x, half* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = __float2half(x[i]);
}

// x = (x + (w ? rmsnorm(y) * w : y)) * scalar        (Gemma sandwich norms, layer scalars). grid M, block 256
extern "C" __global__ void __launch_bounds__(256) norm_add(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w, float scalar) {
    const int m = blockIdx.x;
    float* xr = x + (size_t)m * HID;
    const float* yr = y + (size_t)m * HID;
    float sc = 1.f;
    if (w) {
        __shared__ float red[8];
        float ss = 0;
        for (int i = threadIdx.x; i < HID; i += 256) ss += yr[i] * yr[i];
        ss = warp_sum(ss);
        if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
        __syncthreads();
        float t = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) t += red[i];
        sc = rsqrtf(t / HID + EPS);
    }
    for (int i = threadIdx.x; i < HID; i += 256) xr[i] = (xr[i] + (w ? yr[i] * sc * w[i] : yr[i])) * scalar;
}

extern "C" __global__ void softcap_rows(float* __restrict__ l, size_t n, float cap) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) l[i] = tanhf(l[i] / cap) * cap;
}

// Causal depthwise convolution with per-slot rolling state [slot][K-1][C] (oldest first), tokens processed in
// batch order (one thread per channel walks all tokens, so tokens of one sequence must be in position order).
//  MODE 0 (LFM2 short conv): row = [B | C | x] (3*C wide); y = C * conv(B * x)            -> out [M][C]
//  MODE 1 (delta net):       row[0:C] = SiLU(conv(row[0:C]))                               (in place)
//  MODE 2: MODE 0 with the output quantized to int8 blocks for the next GEMV (oq / ods), C % 32 == 0
#define SNAP_ROWS 8   // batch rows with state snapshots (decode batches with speculative drafts are <= 8 rows)
template <int MODE>
__global__ void causal_conv(float* __restrict__ in, int stride, int C, int K, const float* __restrict__ w, float* __restrict__ state,
                            const int* __restrict__ pos, const int* __restrict__ slot, int M, float* __restrict__ out,
                            int8_t* __restrict__ oq, float2* __restrict__ ods, float* __restrict__ snap) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;   // C % 32 == 0: whole warps leave
    float wk[8];
    for (int k = 0; k < K; k++) wk[k] = w[ch * K + k];
    for (int m = 0; m < M; m++) {
        float* st = state + (size_t)slot[m] * (K - 1) * C;
        if (pos[m] == 0)
            for (int k = 0; k < K - 1; k++) st[(size_t)k * C + ch] = 0.f;
        float* r = in + (size_t)m * stride;
        const float xin = MODE != 1 ? r[ch] * r[2 * C + ch] : r[ch];
        float acc = wk[K - 1] * xin;
        for (int k = 0; k < K - 1; k++) acc += wk[k] * st[(size_t)k * C + ch];
        for (int k = 0; k < K - 2; k++) st[(size_t)k * C + ch] = st[(size_t)(k + 1) * C + ch];
        st[(size_t)(K - 2) * C + ch] = xin;
        // speculative rows: snapshot the state after row m when the next row continues the same sequence
        if (snap && m + 1 < M && m < SNAP_ROWS && slot[m + 1] == slot[m])
            for (int k = 0; k < K - 1; k++) snap[((size_t)m * (K - 1) + k) * C + ch] = st[(size_t)k * C + ch];
        if (MODE == 0) out[(size_t)m * C + ch] = r[C + ch] * acc;
        else if (MODE == 1) r[ch] = acc / (1.f + __expf(-acc));
        else {
            const float v = r[C + ch] * acc;
            const float amax = warp_absmax(v);
            const float d = amax / 127.f;
            const int qi = amax > 0 ? __float2int_rn(v / d) : 0;
            const int sm = warp_sum_i(qi);
            oq[(size_t)m * C + ch] = (int8_t)qi;
            if ((threadIdx.x & 31) == 0) ods[(size_t)m * (C / 32) + ch / 32] = make_float2(d, d * sm);
        }
    }
}

// Gated delta rule (Qwen3.5 linear attention). One CTA per value head, 2*DV threads: thread (j, half) keeps
// rows [half*DK/2, (half+1)*DK/2) of column j of the DK x DV state in registers. Tokens are processed in batch
// order; the state of the token's slot is loaded/stored around runs of the same slot.
// proj row layout: [q (NK*DK) | k (NK*DK) | v (NV*DV) | z (NV*DV) | b (NV) | a (NV)], q/k/v already conv+SiLU'd.
template <int DK, int DV>
__global__ void __launch_bounds__(2 * DV) delta_rule(const float* __restrict__ proj, int R, int NK, int NV, const float* __restrict__ A,
                                                      const float* __restrict__ dt, const float* __restrict__ nw, float* __restrict__ state,
                                                      const int* __restrict__ pos, const int* __restrict__ slot, int M, float* __restrict__ out,
                                                      int8_t* __restrict__ oq, float2* __restrict__ ods, float* __restrict__ snap) {
    constexpr int HK = DK / 2;
    const int h = blockIdx.x;
    const int tid = threadIdx.x, j = tid % DV, half = tid / DV;
    const int kh = h / (NV / NK);
    const int KD = NK * DK, VD = NV * DV, CD = 2 * KD + VD;
    __shared__ float qs[DK], ks[DK], part[2][DV], red[2 * DV / 32];
    __shared__ float sc_s[4];
    float S[HK];
    int cur = -1;
    float* Sg = nullptr;
    for (int m = 0; m < M; m++) {
        const int sl = slot[m];
        if (sl != cur) {
            if (cur >= 0) {
#pragma unroll
                for (int i = 0; i < HK; i++) Sg[(size_t)(half * HK + i) * DV + j] = S[i];
            }
            cur = sl;
            Sg = state + ((size_t)sl * NV + h) * DK * DV;
            if (pos[m] == 0) {
#pragma unroll
                for (int i = 0; i < HK; i++) S[i] = 0.f;
            } else {
#pragma unroll
                for (int i = 0; i < HK; i++) S[i] = Sg[(size_t)(half * HK + i) * DV + j];
            }
        } else if (pos[m] == 0) {
#pragma unroll
            for (int i = 0; i < HK; i++) S[i] = 0.f;
        }
        const float* r = proj + (size_t)m * R;
        // load q, k of this head's key head; l2-normalize (block reduction over DK)
        if (tid < DK) { qs[tid] = r[kh * DK + tid]; ks[tid] = r[KD + kh * DK + tid]; }
        __syncthreads();
        if (tid < 32) {
            float qq = 0, kk = 0;
            for (int i = tid; i < DK; i += 32) { qq += qs[i] * qs[i]; kk += ks[i] * ks[i]; }
            qq = warp_sum(qq); kk = warp_sum(kk);
            if (tid == 0) {
                sc_s[0] = rsqrtf(qq + 1e-6f) * rsqrtf((float)DK);
                sc_s[1] = rsqrtf(kk + 1e-6f);
                const float braw = r[CD + VD + h], araw = r[CD + VD + NV + h];
                sc_s[2] = 1.f / (1.f + __expf(-braw));                 // beta
                const float x = araw + dt[h];
                const float sp = x > 20.f ? x : log1pf(__expf(x));
                sc_s[3] = __expf(A[h] * sp);                            // decay
            }
        }
        __syncthreads();
        const float qsc = sc_s[0], ksc = sc_s[1], beta = sc_s[2], decay = sc_s[3];
        // kv_mem[j] = sum_i S[i][j] * k[i] (after decay)
        float pk = 0.f;
#pragma unroll
        for (int i = 0; i < HK; i++) { S[i] *= decay; pk += S[i] * ks[half * HK + i] * ksc; }
        part[half][j] = pk;
        __syncthreads();
        const float kvm = part[0][j] + part[1][j];
        const float delta = (r[2 * KD + h * DV + j] - kvm) * beta;
        float po = 0.f;
#pragma unroll
        for (int i = 0; i < HK; i++) { S[i] += ks[half * HK + i] * ksc * delta; po += S[i] * qs[half * HK + i] * qsc; }
        __syncthreads();
        part[half][j] = po;
        __syncthreads();
        // gated RMSNorm over the DV outputs: rmsnorm(o) * w * silu(z)
        const float o = part[0][j] + part[1][j];
        float ss = half == 0 ? o * o : 0.f;
        ss = warp_sum(ss);
        if (tid % 32 == 0) red[tid / 32] = ss;
        __syncthreads();
        if (half == 0) {
            float t = 0;
            for (int i = 0; i < DV / 32; i++) t += red[i];
            const float sc = rsqrtf(t / DV + EPS);
            const float z = r[CD + h * DV + j];
            const float v = o * sc * nw[j] * (z / (1.f + __expf(-z)));
            if (oq) {   // int8 blocks for the out-projection GEMV (threads j = 0..DV-1 are whole warps)
                const float amax = warp_absmax(v);
                const float d = amax / 127.f;
                const int qi = amax > 0 ? __float2int_rn(v / d) : 0;
                const int sm = warp_sum_i(qi);
                oq[(size_t)m * VD + h * DV + j] = (int8_t)qi;
                if ((j & 31) == 0) ods[(size_t)m * (VD / 32) + (h * DV + j) / 32] = make_float2(d, d * sm);
            } else {
                out[(size_t)m * VD + h * DV + j] = v;
            }
        }
        if (snap && m + 1 < M && m < SNAP_ROWS && slot[m + 1] == sl) {
            float* sp = snap + ((size_t)m * NV + h) * DK * DV;
#pragma unroll
            for (int i = 0; i < HK; i++) sp[(size_t)(half * HK + i) * DV + j] = S[i];
        }
        __syncthreads();
    }
    if (cur >= 0) {
#pragma unroll
        for (int i = 0; i < HK; i++) Sg[(size_t)(half * HK + i) * DV + j] = S[i];
    }
}

// ================================================================================================
// Prefill versions of the stateful mixers (tokens of ONE slot at consecutive positions)
// ================================================================================================
// Parallel causal depthwise conv: one thread per channel walks a segment of CSEG tokens with the K-1 previous
// inputs in registers (taken from the preceding rows, or from the rolling state for the first tokens).
// Out of place; the rolling state is written afterwards by conv_state. Modes as in causal_conv.
// grid (ceil(C/128), ceil(M/CSEG)), block 128
#define CSEG 32
template <int MODE>
__device__ __forceinline__ float conv_xin(const float* __restrict__ in, int stride, int C, int m, int ch) {
    const float* r = in + (size_t)m * stride;
    return MODE == 0 ? r[ch] * r[2 * C + ch] : r[ch];
}
template <int MODE>
__global__ void __launch_bounds__(128) conv_par(const float* __restrict__ in, int stride, int C, int K, const float* __restrict__ w,
                                                const float* __restrict__ state, int fresh, int M, float* __restrict__ out) {
    const int ch = blockIdx.x * 128 + threadIdx.x;
    if (ch >= C) return;
    const int m0 = blockIdx.y * CSEG, m1 = min(M, m0 + CSEG);
    float wk[8], hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) { wk[k] = k < K ? w[ch * K + k] : 0.f; hist[k] = 0.f; }
    for (int k = 0; k < K - 1; k++) {
        const int mm = m0 - (K - 1) + k;
        hist[k] = mm >= 0 ? conv_xin<MODE>(in, stride, C, mm, ch) : (fresh ? 0.f : state[(size_t)(mm + K - 1) * C + ch]);
    }
    for (int m = m0; m < m1; m++) {
        const float xin = conv_xin<MODE>(in, stride, C, m, ch);
        float acc = wk[K - 1] * xin;
        for (int k = 0; k < K - 1; k++) acc += wk[k] * hist[k];
        for (int k = 0; k < K - 2; k++) hist[k] = hist[k + 1];
        hist[K - 2] = xin;
        if (MODE == 0) out[(size_t)m * C + ch] = in[(size_t)m * stride + C + ch] * acc;
        else out[(size_t)m * C + ch] = acc / (1.f + __expf(-acc));
    }
}
// Rolling state for any batch length, including one-token incremental prefill.
template <int MODE>
__global__ void conv_state(const float* __restrict__ in, int stride, int C, int K, float* __restrict__ state, int M, int fresh) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;
    for (int k = 0; k < K - 1; k++) {
        int row=M-(K-1)+k;
        state[(size_t)k*C+ch]=row>=0?conv_xin<MODE>(in,stride,C,row,ch):fresh?0.f:state[(size_t)(M+k)*C+ch];
    }
}

// ---- chunked gated delta rule (Qwen3.5 prefill) --------------------------------------------------
// The recurrence S_t = d_t S_{t-1} + k_t (x) u_t, u_t = beta_t (v_t - (d_t S_{t-1})^T k_t), o_t = S_t^T q_t is solved
// DC tokens at a time. With c_t = sum_{s<=t} log d_s, G(t,s) = exp(c_t - c_s), A[t][s] = beta_t G(t,s) k_t.k_s (s < t):
//   (I + A) U = beta V - (beta gamma K) S_0            ->  U = V~ - K~ S_0   (V~, K~ by forward substitution)
//   O   = diag(gamma) Q S_0 + tril(G * Q K^T) U
//   S_C = gamma_C S_0 + (G(C,.) K)^T U
// dn_prep (parallel over heads x chunks) computes everything that does not depend on the state: K~, gamma*Q,
// the attention matrix and the decayed keys as fp16 mma operands, V~ in fp32. dn_chunk then walks the chunks with
// three small tensor-core products per chunk. dn_post applies the gated RMSNorm.
#define DC 64
struct DnArgs {
    const float* proj;   // [M][R]: z | b | a behind the conv channels
    const float* qkv;    // [M][CD] conv + SiLU'd q | k | v
    const float* A; const float* dt;
    half* kt;            // [NV][NCH][DC][DK]  K~
    half* gq;            // [NV][NCH][DC][DK]  gamma_t q_t
    half* at;            // [NV][NCH][DC][DC]  attn[t][s]
    half* kh;            // [NV][NCH][DK][DC]  decayed keys, transposed
    float* vt;           // [NV][NCH][DV][DC]  V~ transposed
    float* gam;          // [NV][NCH]          decay over the whole chunk
    int M, R, NK, NV, nch;
};
// grid (nch, NV), block 256 (DK = DV = 128), dynamic smem DN_PREP_SMEM
#define DN_KLD(DK) ((DK) + 4)
template <int DK, int DV>
__global__ void __launch_bounds__(256) dn_prep(const DnArgs a) {
    constexpr int KLD = DN_KLD(DK);
    extern __shared__ __align__(16) uint8_t dsm[];
    half* Kn = (half*)dsm;                               // [DC][KLD] normalized keys
    float* Am = (float*)(dsm + DC * KLD * 2);            // [DC][DC]
    float* beta_s = Am + DC * DC;
    float* cs = beta_s + DC;                             // cumulative log decay (inclusive)
    float* qsc_s = cs + DC;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int c = blockIdx.x, h = blockIdx.y, kh = h / (a.NV / a.NK);
    const int KD = a.NK * DK, VD = a.NV * DV, CD = 2 * KD + VD;
    const int m0 = c * DC;
    const size_t hc = (size_t)h * a.nch + c;
    // 1. per token: l2 norms, beta, log decay; normalized keys -> smem
    for (int t = warp; t < DC; t += 8) {
        const int m = m0 + t;
        const bool ok = m < a.M;
        const float* qr = a.qkv + (size_t)(ok ? m : 0) * CD + kh * DK;
        float kv[DK / 32], qq = 0.f, kk = 0.f;
#pragma unroll
        for (int e = 0; e < DK / 32; e++) {
            const float q = ok ? qr[lane + 32 * e] : 0.f;
            kv[e] = ok ? qr[KD + lane + 32 * e] : 0.f;
            qq += q * q; kk += kv[e] * kv[e];
        }
        qq = warp_sum(qq); kk = warp_sum(kk);
        const float ksc = rsqrtf(kk + 1e-6f);
#pragma unroll
        for (int e = 0; e < DK / 32; e++) Kn[t * KLD + lane + 32 * e] = __float2half(kv[e] * ksc);
        if (lane == 0) {
            qsc_s[t] = rsqrtf(qq + 1e-6f) * rsqrtf((float)DK);
            if (ok) {
                const float* r = a.proj + (size_t)m * a.R;
                beta_s[t] = 1.f / (1.f + __expf(-r[CD + VD + h]));
                const float x = r[CD + VD + a.NV + h] + a.dt[h];
                cs[t] = a.A[h] * (x > 20.f ? x : log1pf(__expf(x)));
            } else { beta_s[t] = 0.f; cs[t] = 0.f; }
        }
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.f;
        for (int t = 0; t < DC; t++) { acc += cs[t]; cs[t] = acc; }
        a.gam[hc] = __expf(acc);
    }
    __syncthreads();
    // 2. Gram matrices: thread (t, group of 16 s): k_t.k_s and q_t.k_s for s <= t
    {
        const int t = tid / 4, s0 = (tid % 4) * 16;
        const int m = m0 + t;
        const bool ok = m < a.M;
        float gk[16], qk[16];
#pragma unroll
        for (int i = 0; i < 16; i++) { gk[i] = 0.f; qk[i] = 0.f; }
        if (s0 <= t) {
            const float2* qr = (const float2*)(a.qkv + (size_t)(ok ? m : 0) * CD + kh * DK);
            const float qs = ok ? qsc_s[t] : 0.f;
            for (int d = 0; d < DK / 2; d++) {
                const float2 q2 = qr[d];
                const float2 kt2 = __half22float2(*(const half2*)(Kn + t * KLD + 2 * d));
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const float2 ks2 = __half22float2(*(const half2*)(Kn + (s0 + i) * KLD + 2 * d));
                    gk[i] += kt2.x * ks2.x + kt2.y * ks2.y;
                    qk[i] += q2.x * ks2.x + q2.y * ks2.y;
                }
            }
#pragma unroll
            for (int i = 0; i < 16; i++) qk[i] *= qs;
        }
        half* arow = a.at + (hc * DC + t) * DC + s0;
        const float ct = cs[t], bt = beta_s[t];
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int s2 = s0 + i;
            const float dec = s2 <= t ? __expf(ct - cs[s2]) : 0.f;
            Am[t * DC + s2] = s2 < t ? bt * dec * gk[i] : 0.f;
            arow[i] = __float2half(dec * qk[i]);
        }
    }
    __syncthreads();
    // 3. forward substitution, one column per thread (registers): V~ columns (tid < DV), K~ columns (tid >= DV)
    float x[DC];
    const int col = tid < DV ? tid : tid - DV;
    if (tid < DV) {
#pragma unroll
        for (int t = 0; t < DC; t++) x[t] = m0 + t < a.M ? beta_s[t] * a.qkv[(size_t)(m0 + t) * CD + 2 * KD + h * DV + col] : 0.f;
    } else {
#pragma unroll
        for (int t = 0; t < DC; t++) x[t] = beta_s[t] * __expf(cs[t]) * __half2float(Kn[t * KLD + col]);
    }
#pragma unroll
    for (int t = 1; t < DC; t++) {
        float v = x[t];
#pragma unroll
        for (int s2 = 0; s2 < t; s2++) v -= Am[t * DC + s2] * x[s2];
        x[t] = v;
    }
    if (tid < DV) {
        float* dst = a.vt + (hc * DV + col) * DC;
#pragma unroll
        for (int t = 0; t < DC; t += 4) *(float4*)(dst + t) = make_float4(x[t], x[t + 1], x[t + 2], x[t + 3]);
    } else {
        half* kt = a.kt + hc * DC * DK;
        half* gq = a.gq + hc * DC * DK;
        half* khp = a.kh + (hc * DK + col) * DC;
        const float cC = cs[DC - 1];
        for (int t = 0; t < DC; t++) {
            kt[t * DK + col] = __float2half(x[t]);
            const float q = m0 + t < a.M ? a.qkv[(size_t)(m0 + t) * CD + kh * DK + col] * qsc_s[t] : 0.f;
            gq[t * DK + col] = __float2half(__expf(cs[t]) * q);
            khp[t] = __float2half(__expf(cC - cs[t]) * __half2float(Kn[t * KLD + col]));
        }
    }
}

// Sequential part: one CTA per value head, warp w owns state columns dv in [16w, 16w+16) as S^T tiles in
// registers (fp32, mma accumulator layout), which double as the A operand of the next products. No communication
// between warps except the shared operand tile in smem. grid NV, block 32 * DV / 16.
template <int DK, int DV>
__global__ void __launch_bounds__(32 * DV / 16) dn_chunk(const DnArgs a, float* __restrict__ state, int fresh, float* __restrict__ out) {
    constexpr int NT = 32 * DV / 16;
    constexpr int LDK = DK + 8, LDC = DC + 8;
    __shared__ __align__(16) half buf[(DC * LDK > DK * LDC ? DC * LDK : DK * LDC)];
    const int h = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int VD = a.NV * DV;
    const int dv0 = warp * 16 + (lane >> 2);        // rows dv0, dv0 + 8
    const int c2 = (lane & 3) * 2;                  // column pair inside an 8-wide tile
    float S[DK / 8][4];
    float* Sg = state + (size_t)h * DK * DV;
#pragma unroll
    for (int j = 0; j < DK / 8; j++)
#pragma unroll
        for (int e = 0; e < 4; e++)
            S[j][e] = fresh ? 0.f : Sg[(size_t)(8 * j + c2 + (e & 1)) * DV + dv0 + 8 * (e >> 1)];
    auto load_rows = [&](const half* src, int rows, int cols) {
        const int cpr = cols / 8;
        for (int i = tid; i < rows * cpr; i += NT) {
            const int r = i / cpr, cc = (i % cpr) * 8;
            cp_async16(buf + r * (cols + 8) + cc, src + (size_t)r * cols + cc, true);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    };
    for (int c = 0; c < a.nch; c++) {
        const size_t hc = (size_t)h * a.nch + c;
        // ---- U^T = V~^T - S^T K~^T ----
        load_rows(a.kt + hc * DC * DK, DC, DK);
        float u[DC / 8][4];
#pragma unroll
        for (int i = 0; i < DC / 8; i++) u[i][0] = u[i][1] = u[i][2] = u[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
            const uint32_t a0 = pack_half2(S[2 * kk][0], S[2 * kk][1]), a1 = pack_half2(S[2 * kk][2], S[2 * kk][3]);
            const uint32_t a2 = pack_half2(S[2 * kk + 1][0], S[2 * kk + 1][1]), a3 = pack_half2(S[2 * kk + 1][2], S[2 * kk + 1][3]);
#pragma unroll
            for (int ni = 0; ni < DC / 8; ni += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, buf + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * LDK + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(u[ni], a0, a1, a2, a3, b0, b1);
                mma16816(u[ni + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        {
            const float* v0 = a.vt + (hc * DV + dv0) * DC + c2;
#pragma unroll
            for (int ni = 0; ni < DC / 8; ni++) {
                const float2 x0 = *(const float2*)(v0 + ni * 8), x1 = *(const float2*)(v0 + 8 * DC + ni * 8);
                u[ni][0] = x0.x - u[ni][0]; u[ni][1] = x0.y - u[ni][1];
                u[ni][2] = x1.x - u[ni][2]; u[ni][3] = x1.y - u[ni][3];
            }
        }
        __syncthreads();
        // ---- O^T = S^T (gamma Q)^T + U^T attn^T ----
        load_rows(a.gq + hc * DC * DK, DC, DK);
        float o[DC / 8][4];
#pragma unroll
        for (int i = 0; i < DC / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
            const uint32_t a0 = pack_half2(S[2 * kk][0], S[2 * kk][1]), a1 = pack_half2(S[2 * kk][2], S[2 * kk][3]);
            const uint32_t a2 = pack_half2(S[2 * kk + 1][0], S[2 * kk + 1][1]), a3 = pack_half2(S[2 * kk + 1][2], S[2 * kk + 1][3]);
#pragma unroll
            for (int ni = 0; ni < DC / 8; ni += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, buf + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * LDK + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(o[ni], a0, a1, a2, a3, b0, b1);
                mma16816(o[ni + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();
        load_rows(a.at + hc * DC * DC, DC, DC);
        uint32_t ua[DC / 16][4];
#pragma unroll
        for (int kk = 0; kk < DC / 16; kk++) {
            ua[kk][0] = pack_half2(u[2 * kk][0], u[2 * kk][1]); ua[kk][1] = pack_half2(u[2 * kk][2], u[2 * kk][3]);
            ua[kk][2] = pack_half2(u[2 * kk + 1][0], u[2 * kk + 1][1]); ua[kk][3] = pack_half2(u[2 * kk + 1][2], u[2 * kk + 1][3]);
        }
#pragma unroll
        for (int kk = 0; kk < DC / 16; kk++) {
#pragma unroll
            for (int ni = 0; ni < DC / 8; ni += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, buf + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * LDC + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(o[ni], ua[kk][0], ua[kk][1], ua[kk][2], ua[kk][3], b0, b1);
                mma16816(o[ni + 1], ua[kk][0], ua[kk][1], ua[kk][2], ua[kk][3], b2, b3);
            }
        }
#pragma unroll
        for (int ni = 0; ni < DC / 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int m = c * DC + ni * 8 + c2 + (e & 1);
                if (m < a.M) out[(size_t)m * VD + h * DV + dv0 + 8 * (e >> 1)] = o[ni][e];
            }
        __syncthreads();
        // ---- S^T = gamma_C S^T + U^T K^ ----
        load_rows(a.kh + hc * DK * DC, DK, DC);
        const float gC = a.gam[hc];
#pragma unroll
        for (int j = 0; j < DK / 8; j++) { S[j][0] *= gC; S[j][1] *= gC; S[j][2] *= gC; S[j][3] *= gC; }
#pragma unroll
        for (int kk = 0; kk < DC / 16; kk++) {
#pragma unroll
            for (int ni = 0; ni < DK / 8; ni += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, buf + (ni * 8 + (lane & 7) + ((lane >> 4) << 3)) * LDC + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(S[ni], ua[kk][0], ua[kk][1], ua[kk][2], ua[kk][3], b0, b1);
                mma16816(S[ni + 1], ua[kk][0], ua[kk][1], ua[kk][2], ua[kk][3], b2, b3);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < DK / 8; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) Sg[(size_t)(8 * j + c2 + (e & 1)) * DV + dv0 + 8 * (e >> 1)] = S[j][e];
}

// gated RMSNorm of the delta-rule outputs: oh = rmsnorm(o) * w * silu(z), fp16. One warp per (token, value head).
// grid (M, ceil(NV / 8)), block 256
extern "C" __global__ void __launch_bounds__(256) dn_post(const float* __restrict__ o, const float* __restrict__ proj, int R, int CD, int NV, int DV,
                                                          const float* __restrict__ nw, half* __restrict__ oh) {
    const int m = blockIdx.x, h = blockIdx.y * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (h >= NV) return;
    const float* orow = o + (size_t)m * NV * DV + h * DV;
    float ss = 0.f;
    for (int j = lane; j < DV; j += 32) ss += orow[j] * orow[j];
    ss = warp_sum(ss);
    const float sc = rsqrtf(ss / DV + EPS);
    const float* z = proj + (size_t)m * R + CD + h * DV;
    for (int j = lane; j < DV; j += 32) oh[(size_t)m * NV * DV + h * DV + j] = __float2half(orow[j] * sc * nw[j] * (z[j] / (1.f + __expf(-z[j]))));
}

// ---- CPU-computed experts: request / response through host-mapped memory (polled, no host callbacks) ----
// header (host mapped): [0] request seq, [1] layer, [2] M, [3] K, [16] done seq (own 64 B line)
// request payload: ids [P] int | weights [P] float | xn [M][H] half ; response: out [M][H] float
// grid 1, block 1024
extern "C" __global__ void __launch_bounds__(1024) moe_post(volatile int* hdr, int* counter, int layer, int M, int K, int H,
                                                            const int* __restrict__ ids, const float* __restrict__ w, const half* __restrict__ xn,
                                                            uint8_t* hbuf) {
    const int P = M * K, tid = threadIdx.x;
    int* hid = (int*)hbuf;
    float* hw = (float*)(hbuf + (size_t)P * 4);
    uint4* hx = (uint4*)(hbuf + (size_t)P * 8);
    for (int i = tid; i < P; i += 1024) { hid[i] = ids[i]; hw[i] = w[i]; }
    const uint4* sx = (const uint4*)xn;
    for (int i = tid; i < M * H / 8; i += 1024) hx[i] = sx[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const int c = *counter + 1;
        *counter = c;
        hdr[1] = layer; hdr[2] = M; hdr[3] = K;
        __threadfence_system();
        hdr[0] = c;
    }
}
// spin until the CPU answered the current request, then copy its output. grid G, block 256
extern "C" __global__ void __launch_bounds__(256) moe_wait(volatile int* hdr, const int* counter, const float* hout, float* __restrict__ out, int n) {
    if (threadIdx.x == 0) {
        const int c = *counter;
        while (hdr[16] != c) __nanosleep(200);
    }
    __syncthreads();
    const volatile float4* src = (const volatile float4*)hout;
    float4* dst = (float4*)out;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n / 4; i += gridDim.x * 256) {
        float4 v; v.x = src[i].x; v.y = src[i].y; v.z = src[i].z; v.w = src[i].w;
        dst[i] = v;
    }
}

// ---- Mixture of experts ----
// Router: one CTA per token. logits = RMSNorm(x) (ffn norm) . W_r (fp32 [E][HID]); softmax or sigmoid scores;
// top-k by score (+ optional selection bias); weights optionally renormalized and scaled.
extern "C" __global__ void __launch_bounds__(256) moe_route(const float* __restrict__ x, const float* __restrict__ nw, const float* __restrict__ Wr,
                                                            int E, int K, int sigmoid_mode, const float* __restrict__ bias, int norm_topk,
                                                            float scale, int* __restrict__ ids, float* __restrict__ wts) {
    const int m = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    __shared__ float xs[HID];
    __shared__ float lg[256];
    __shared__ float red[8];
    const float* xr = x + (size_t)m * HID;
    float ss = 0;
    for (int i = tid; i < HID; i += 256) { float v = xr[i]; xs[i] = v; ss += v * v; }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float t = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) t += red[i];
    const float sc = rsqrtf(t / HID + EPS);
    for (int i = tid; i < HID; i += 256) xs[i] = xs[i] * sc * nw[i];
    __syncthreads();
    for (int e = warp; e < E; e += 8) {
        const float* wr = Wr + (size_t)e * HID;
        float acc = 0;
        for (int i = lane; i < HID; i += 32) acc += wr[i] * xs[i];
        acc = warp_sum(acc);
        if (lane == 0) lg[e] = acc;
    }
    __syncthreads();
    // warp 0: probabilities, then K rounds of warp-wide argmax (E <= 256: 8 experts per lane)
    if (warp == 0) {
        float pv[8], sv[8];
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) { const int e = lane + 32 * i; pv[i] = e < E ? lg[e] : -INFINITY; mx = fmaxf(mx, pv[i]); }
        mx = warp_max(mx);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = lane + 32 * i;
            if (e < E) pv[i] = sigmoid_mode ? 1.f / (1.f + __expf(-pv[i])) : __expf(pv[i] - mx);
            else pv[i] = 0.f;
            sum += pv[i];
        }
        sum = warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = lane + 32 * i;
            if (!sigmoid_mode) pv[i] /= sum;
            sv[i] = e < E ? pv[i] + (bias ? bias[e] : 0.f) : -INFINITY;
        }
        float tot = 0.f, selp[16];
        int sel[16];
        for (int k = 0; k < K; k++) {
            float bv = -INFINITY; int bi = 0x7fffffff; float bp = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int e = lane + 32 * i;
                if (sv[i] > bv) { bv = sv[i]; bi = e; bp = pv[i]; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, o);
                const int oi = __shfl_xor_sync(0xffffffff, bi, o);
                const float op = __shfl_xor_sync(0xffffffff, bp, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bp = op; }
            }
            sel[k] = bi; selp[k] = bp; tot += bp;
            if ((bi & 31) == lane) sv[bi >> 5] = -INFINITY;
        }
        if (lane == 0)
            for (int k = 0; k < K; k++) {
                float w = selp[k];
                if (norm_topk) w /= (tot + (sigmoid_mode ? 1e-6f : 0.f));
                ids[m * K + k] = sel[k];
                wts[m * K + k] = w * scale;
            }
    }
}

// Indirect GEMV over expert matrices: pair p uses expert ids[p] (rows [e*N, (e+1)*N)) and activation row
// p / act_div. y[p][row] = W_e[row] . act. grid (ceil(N/32), P), block 256 (four rows per warp)
template <int QT, int K>
__global__ void __launch_bounds__(256) gemv_ind(const uint8_t* __restrict__ W, const void* __restrict__ S, int N, const int* __restrict__ ids,
                                                int act_div, const int8_t* __restrict__ xq, const float2* __restrict__ xds, float* __restrict__ y) {
    constexpr int R = 4;  // rows per warp (more independent loads in flight)
    const int p = blockIdx.y;
    const int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R;
    const int lane = threadIdx.x % 32;
    if (row0 >= N) return;
    const int e = ids[p];
    const int arow = p / act_div;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r] = 0.f;
        if (row0 + r < N) row_dot<QT, K, 1>(W, S, e * N + row0 + r, xq + (size_t)arow * K, xds + (size_t)arow * (K / 32), &acc[r]);
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[(size_t)p * N + row0 + r] = v;
    }
}

// Grouped expert GEMV for decode batches: CTA (row tile, expert e) collects the (token, expert) pairs routed to e and
// streams each weight row once for all of them (a token selects an expert at most once, so <= 16 pairs).
// y[p][row] = W_e[row] . act[p / act_div]. grid (ceil(N/32), E), block 256 (8 warps x 4 rows)
template <int QT, int K>
__global__ void __launch_bounds__(256) gemv_grp(const uint8_t* __restrict__ W, const void* __restrict__ S, int N, const int* __restrict__ ids,
                                                int P, int act_div, const int8_t* __restrict__ xq, const float2* __restrict__ xds,
                                                float* __restrict__ y) {
    constexpr int R = 4, MP = 16, NB = K / 32, IT = (NB + 31) / 32;
    const int e = blockIdx.y;
    __shared__ int plist[MP];
    __shared__ int pcnt;
    if (threadIdx.x < 32) {
        int c = 0;
        for (int p0 = 0; p0 < P; p0 += 32) {
            const int p = p0 + threadIdx.x;
            const bool hit = p < P && ids[p] == e;
            const unsigned m = __ballot_sync(0xffffffff, hit);
            if (hit) { const int slot = c + __popc(m & ((1u << threadIdx.x) - 1)); if (slot < MP) plist[slot] = p; }
            c += __popc(m);
        }
        if (threadIdx.x == 0) pcnt = min(c, MP);
    }
    __syncthreads();
    const int cnt = pcnt;
    if (cnt == 0) return;
    const int lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    for (int r = 0; r < R; r++) {
        const int row = row0 + r;
        if (row >= N) break;
        const size_t wr = (size_t)e * N + row;
        float acc[MP];
#pragma unroll
        for (int j = 0; j < MP; j++) acc[j] = 0.f;
#pragma unroll
        for (int it = 0; it < IT; it++) {
            const int b = it * 32 + lane;
            if (NB % 32 != 0 && b >= NB) break;
            uint4 w, w1; float sc, mn = 0.f;
            if constexpr (QT == QT_Q4) {
                w = ld_stream(W + wr * (K / 2) + b * 16);
                const half2 sm = ((const half2*)S)[wr * (K / 64) + b / 2]; sc = __low2float(sm); mn = __high2float(sm);
            } else if constexpr (QT == QT_Q3) {
                const uint32_t* p3 = (const uint32_t*)(W + wr * (K * 3 / 8) + b * 12);
                w.x = __ldg(p3); w.y = __ldg(p3 + 1); w.z = __ldg(p3 + 2);
                const half2 sm = ((const half2*)S)[wr * (K / 64) + b / 2]; sc = __low2float(sm); mn = __high2float(sm);
            } else {
                w = ld_stream(W + wr * K + b * 32); w1 = ld_stream(W + wr * K + b * 32 + 16);
                sc = __half2float(((const half*)S)[wr * NB + b]);
            }
#pragma unroll
            for (int j = 0; j < MP; j++) {
                if (j >= cnt) break;
                const int arow = plist[j] / act_div;
                const int4* xp = (const int4*)(xq + (size_t)arow * K + b * 32);
                const int4 a0 = xp[0], a1 = xp[1];
                const float2 d = xds[(size_t)arow * NB + b];
                if constexpr (QT == QT_Q4) acc[j] += sc * d.x * (float)dot_q4_blk(w, a0, a1) + mn * d.y;
                else if constexpr (QT == QT_Q3) acc[j] += sc * d.x * (float)dot_q3_blk(w.x, w.y, w.z, a0, a1) + mn * d.y;
                else acc[j] += sc * d.x * (float)dot_q8_blk(w, w1, a0, a1);
            }
        }
#pragma unroll
        for (int j = 0; j < MP; j++) {
            if (j >= cnt) break;
            const float v = warp_sum(acc[j]);
            if (lane == 0) y[(size_t)plist[j] * N + row] = v;
        }
    }
}

// gate/up pairs [P][2F] (interleaved rows) -> act(g)*u quantized to int8 [P][F]. grid P, block 256
extern "C" __global__ void __launch_bounds__(256) act_quant(const float* __restrict__ gu, int F, int8_t* __restrict__ q, float2* __restrict__ ds) {
    const int p = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* g = gu + (size_t)p * 2 * F;
    for (int blk = warp; blk < F / 32; blk += 8) {
        const int i = blk * 32 + lane;
        const float v = act_fn(g[2 * i]) * g[2 * i + 1];
        const float amax = warp_absmax(v);
        const float d = amax / 127.f;
        const int qi = amax > 0 ? __float2int_rn(v / d) : 0;
        const int sm = warp_sum_i(qi);
        q[(size_t)p * F + i] = (int8_t)qi;
        if (lane == 0) ds[(size_t)p * (F / 32) + blk] = make_float2(d, d * sm);
    }
}

// out[m] = sum_k wts[m][k] * y[m*K+k]   (expert outputs [P][HID] -> [M][HID]). grid M, block 256
extern "C" __global__ void __launch_bounds__(256) moe_combine(const float* __restrict__ y, const float* __restrict__ wts, int K, float* __restrict__ out) {
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < HID; i += 256) {
        float acc = 0;
        for (int k = 0; k < K; k++) acc += wts[m * K + k] * y[((size_t)m * K + k) * HID + i];
        out[(size_t)m * HID + i] = acc;
    }
}

// prefill MoE helpers: gather normalized fp16 rows, scatter-add weighted expert outputs
extern "C" __global__ void gather_rows_f16(const half* __restrict__ src, const int* __restrict__ rows, int K, half* __restrict__ dst) {
    const int i = blockIdx.x;
    const half* s = src + (size_t)rows[i] * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) dst[(size_t)i * K + j] = s[j];
}
extern "C" __global__ void scatter_add_rows(const float* __restrict__ y, const int* __restrict__ rows, const float* __restrict__ w, int K, float* __restrict__ out) {
    const int i = blockIdx.x;
    float* o = out + (size_t)rows[i] * K;
    const float wi = w[i];
    // a token appears once per selected expert, in different blocks: accumulate atomically
    for (int j = threadIdx.x; j < K; j += blockDim.x) atomicAdd(&o[j], wi * y[(size_t)i * K + j]);
}

// ---- Gemma 4 per-layer inputs ----
// generic embedding-row lookup with runtime width / qtype (3 Q4, 2 Q8). grid M, block 256
extern "C" __global__ void embed_rows(const uint8_t* __restrict__ E, const uint8_t* __restrict__ ES, int qt, const int* __restrict__ tokens,
                                      int W, float scale, float* __restrict__ out) {
    const int m = blockIdx.x;
    const size_t tok = tokens[m];
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        float v;
        if (qt == QT_Q4) {
            const uint8_t* row = E + tok * (W / 2);
            const int blk = i / 32, j = i % 32;
            const uint8_t b = row[blk * 16 + (j & 15)];
            const int q = j < 16 ? (b & 15) : (b >> 4);
            const half2 sm = ((const half2*)ES)[tok * (W / 64) + i / 64];
            v = __low2float(sm) * q + __high2float(sm);
        } else {
            v = __half2float(((const half*)ES)[tok * (W / 32) + i / 32]) * (float)((const int8_t*)E)[tok * W + i];
        }
        out[(size_t)m * W + i] = v * scale;
    }
}
// ple[m][l][:] = (rmsnorm(proj[m][l][:] / sqrt(H)) * w + tok[m][l][:] * sqrt(D)) / sqrt(2). grid (M, NL), block D
extern "C" __global__ void ple_combine(const float* __restrict__ proj, const float* __restrict__ tok, const float* __restrict__ w, int D, int NL,
                                       float* __restrict__ out) {
    const int m = blockIdx.x, l = blockIdx.y, i = threadIdx.x;
    const size_t o = (size_t)m * NL * D + (size_t)l * D + i;
    const float v = proj[o] * rsqrtf((float)HID);
    __shared__ float red[32];
    float ss = warp_sum(v * v);
    if (i % 32 == 0) red[i / 32] = ss;
    __syncthreads();
    float t = 0;
    for (int k = 0; k < D / 32; k++) t += red[k];
    const float sc = rsqrtf(t / D + EPS);
    out[o] = (v * sc * w[i] + tok[o] * sqrtf((float)D)) * 0.70710678f;
}
// g[m][i] = act(g[m][i]) * ple[m][l][i]. grid M, block D
extern "C" __global__ void ple_mul(float* __restrict__ g, const float* __restrict__ ple, int l, int D, int W) {
    const int m = blockIdx.x, i = threadIdx.x;
    g[(size_t)m * D + i] = act_fn(g[(size_t)m * D + i]) * ple[(size_t)m * W + (size_t)l * D + i];
}

// Exact floating point activation path for imported block formats. All weight blocks stay packed.
extern "C" __global__ void packed_prepare(const float* x,const float* norm,const int* rows,
        const int8_t* xq,const float2* ds,float* out,int K,int pro) {
    const int m=blockIdx.x,r=rows?rows[m]:m;
    if(!pro){for(int i=threadIdx.x;i<K;i+=blockDim.x)out[(size_t)m*K+i]=float(xq[(size_t)m*K+i])*ds[(size_t)m*(K/32)+i/32].x;return;}
    __shared__ float reduction[8];const int width=K/NORM_GROUPS;
    for(int g=0;g<NORM_GROUPS;g++){
        float sum=0;for(int j=threadIdx.x;j<width;j+=blockDim.x){float v=x[(size_t)r*K+g*width+j];sum+=v*v;}
        sum=warp_sum(sum);if(threadIdx.x%32==0)reduction[threadIdx.x/32]=sum;__syncthreads();
        float inv=0;for(int i=0;i<8;i++)inv+=reduction[i];inv=rsqrtf(inv/width+EPS);
        for(int j=threadIdx.x;j<width;j+=blockDim.x){int i=g*width+j;out[(size_t)m*K+i]=x[(size_t)r*K+i]*inv*(norm?norm[i]:1.f);}
        __syncthreads();
    }
}
// Keep the first five butterfly stages in warp registers. Higher stages
// exchange through shared memory; the operation order matches the scalar FWHT.
__device__ void packed_butterfly(float* v, int K, int B) {
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float a = v[i];
        for (int stride = 1; stride < 32; stride *= 2) {
            float b = __shfl_xor_sync(0xffffffff, a, stride);
            a = (i & stride) ? b - a : a + b;
        }
        v[i] = a;
    }
    __syncthreads();
    for (int stride = 32; stride < B; stride *= 2) {
        for (int pair = threadIdx.x; pair < K / 2; pair += blockDim.x) {
            int i = (pair / stride) * (2 * stride) + pair % stride;
            float a = v[i], b = v[i + stride];
            v[i] = a + b; v[i + stride] = a - b;
        }
        __syncthreads();
    }
}
extern "C" __global__ void packed_fwht(const void* input,void* output,int K,int B,const float* signs,
        int input_half,int output_half,int inverse) {
    extern __shared__ float v[];
    const int off=blockIdx.x*B;const size_t base=(size_t)blockIdx.y*K+off;
    for(int i=threadIdx.x;i<B;i+=blockDim.x){float x=input_half?__half2float(((const half*)input)[base+i]):((const float*)input)[base+i];v[i]=x*(!inverse&&signs?signs[off+i]:1.f);}
    __syncthreads();
    if (B >= 32) packed_butterfly(v, B, B);
    else for(int stride=1;stride<B;stride*=2){
        for(int pair=threadIdx.x;pair<B/2;pair+=blockDim.x){int i=(pair/stride)*(2*stride)+pair%stride;float a=v[i],b=v[i+stride];v[i]=a+b;v[i+stride]=a-b;}
        __syncthreads();
    }
    float scale=rsqrtf(float(B));
    for(int i=threadIdx.x;i<B;i+=blockDim.x){float x=v[i]*scale*(inverse&&signs?signs[off+i]:1.f);if(output_half)((half*)output)[base+i]=__float2half_rn(x);else((float*)output)[base+i]=x;}
}
// One CTA per token combines input preparation and all independent rotations.
// The host limits K so this shared buffer fits without opt-in shared memory.
extern "C" __global__ void packed_prepare_fwht(const float* x,const float* norm,const int* rows,
        const int8_t* xq,const float2* ds,float* out,int K,int pro,int B,const float* signs,int inverse) {
    extern __shared__ float v[];
    __shared__ float reduction[8];
    const int m=blockIdx.x,r=rows?rows[m]:m,width=K/NORM_GROUPS;
    if (pro) {
        for (int g=0;g<NORM_GROUPS;g++) {
            float sum=0;
            for(int j=threadIdx.x;j<width;j+=blockDim.x){float a=x[(size_t)r*K+g*width+j];sum+=a*a;}
            sum=warp_sum(sum);if(threadIdx.x%32==0)reduction[threadIdx.x/32]=sum;__syncthreads();
            float inv=0;for(int i=0;i<8;i++)inv+=reduction[i];inv=rsqrtf(inv/width+EPS);
            for(int j=threadIdx.x;j<width;j+=blockDim.x){int i=g*width+j;v[i]=x[(size_t)r*K+i]*inv*(norm?norm[i]:1.f);}
            __syncthreads();
        }
    } else {
        for(int i=threadIdx.x;i<K;i+=blockDim.x)v[i]=float(xq[(size_t)m*K+i])*ds[(size_t)m*(K/32)+i/32].x;
        __syncthreads();
    }
    if(!inverse&&signs)for(int i=threadIdx.x;i<K;i+=blockDim.x)v[i]*=signs[i];
    __syncthreads();
    packed_butterfly(v,K,B);
    float scale=rsqrtf(float(B));
    for(int i=threadIdx.x;i<K;i+=blockDim.x)out[(size_t)m*K+i]=v[i]*scale*(inverse&&signs?signs[i]:1.f);
}
// Fuse gate activation with the following down-projection rotation.
extern "C" __global__ void packed_act_fwht(const float* gu, float* out, int K, int B,
                                          const float* signs, int inverse) {
    extern __shared__ float v[];
    const int off = blockIdx.x * B;
    const size_t base = (size_t)blockIdx.y * K + off;
    for (int i=threadIdx.x;i<B;i+=blockDim.x) {
        float a=act_fn(gu[2*(base+i)])*gu[2*(base+i)+1];
        v[i]=a*(!inverse&&signs?signs[off+i]:1.f);
    }
    __syncthreads();
    packed_butterfly(v,B,B);
    float scale=rsqrtf(float(B));
    for(int i=threadIdx.x;i<B;i+=blockDim.x)out[base+i]=v[i]*scale*(inverse&&signs?signs[off+i]:1.f);
}
template<int QT>
__global__ void packed_mv(const uint8_t* weights,int unused_qt,const void* input,int input_half,
        int K,int N,float* output,int stride,int offset,int add,const float* bias) {
    constexpr int qt=QT;int row=blockIdx.x*8+threadIdx.x/32,m=blockIdx.y,lane=threadIdx.x%32;if(row>=N)return;
    const uint8_t* w=weights+(size_t)row*packed_row_bytes(qt,K);float acc=0;
    for(int k=lane;k<K;k+=32){float x=input_half?__half2float(((const half*)input)[(size_t)m*K+k]):((const float*)input)[(size_t)m*K+k];acc=fmaf(packed_read(qt,w,k),x,acc);}
    acc=warp_sum(acc);if(lane==0){if(bias)acc+=bias[row];float* y=output+(size_t)m*stride+offset+row;if(add)*y+=acc;else*y=acc;}
}
extern "C" __global__ void packed_gate(const float* input,half* output,int F) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<F;i+=gridDim.x*blockDim.x)output[i]=__float2half_rn(act_fn(input[2*i])*input[2*i+1]);
}

// ---------------------------------------------------------------------------------------------------------------
// Fast GEMV for packed block formats (decode and small verify batches). One warp per output row, MB tokens per pass.
// Weights are decoded a block at a time into registers (scale applied once per block), activations are fp32 and are
// never requantized, so results match the reference decode exactly up to summation order.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned ld_u16x2(const uint8_t* p) {
    return (unsigned)*(const unsigned short*)p | ((unsigned)*(const unsigned short*)(p + 2) << 16);
}
__device__ __forceinline__ float4 ld_x4(const float* p) { return *(const float4*)p; }
// int -> float without the (slow, 1/8 rate) I2F instruction: for 0 <= v < 2^23, (2^23 | v) as float is 2^23 + v exactly
__device__ __forceinline__ float u2f(unsigned v, float bias) { return __uint_as_float(v | 0x4B000000u) - bias; }

// R output rows per warp share each activation load (the activation vector re-read per row was the bottleneck).
// w0 = first row, rb = row stride in bytes, nr = valid rows (<= R; surplus rows alias the last valid one and are not stored)
template <int QT, int MB, int R>
__device__ __forceinline__ void pdot(const uint8_t* __restrict__ w0, size_t rb, int nr, const float* __restrict__ X, int K, int M, int lane, float (&acc)[R][MB]) {
#define PROW(r) (w0 + (size_t)min(r, nr - 1) * rb)
    if constexpr (QT == 7) {                       // Q4_0: 18-byte blocks of 32, 4 lanes per block
        const int nb = K / 32, t = lane & 3;
        #pragma unroll 2
        for (int b = lane >> 2; b < nb; b += 8) {
            float4 a[MB], c[MB];
#pragma unroll
            for (int m = 0; m < MB; m++) { const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 32; a[m] = ld_x4(x + 4 * t); c[m] = ld_x4(x + 16 + 4 * t); }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint8_t* p = PROW(r) + (size_t)b * 18;
                const float d = packed_half(p);
                const unsigned v = ld_u16x2(p + 2 + 4 * t);
                float lo[4], hi[4];
#pragma unroll
                for (int j = 0; j < 4; j++) { lo[j] = u2f((v >> (8 * j)) & 15u, 8388616.f); hi[j] = u2f((v >> (8 * j + 4)) & 15u, 8388616.f); }
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    const float s = lo[0] * a[m].x + lo[1] * a[m].y + lo[2] * a[m].z + lo[3] * a[m].w + hi[0] * c[m].x + hi[1] * c[m].y + hi[2] * c[m].z + hi[3] * c[m].w;
                    acc[r][m] = fmaf(d, s, acc[r][m]);
                }
            }
        }
    } else if constexpr (QT == 8) {                // Q8_0: 34-byte blocks of 32, 4 lanes per block
        const int nb = K / 32, t = lane & 3;
        #pragma unroll 2
        for (int b = lane >> 2; b < nb; b += 8) {
            float4 a[MB], c[MB];
#pragma unroll
            for (int m = 0; m < MB; m++) { const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 32 + 8 * t; a[m] = ld_x4(x); c[m] = ld_x4(x + 4); }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint8_t* p = PROW(r) + (size_t)b * 34;
                const float d = packed_half(p);
                const unsigned v0 = ld_u16x2(p + 2 + 8 * t), v1 = ld_u16x2(p + 6 + 8 * t);
                float q[8];
#pragma unroll
                for (int j = 0; j < 4; j++) { q[j] = u2f(((v0 >> (8 * j)) & 255u) ^ 128u, 8388736.f); q[4 + j] = u2f(((v1 >> (8 * j)) & 255u) ^ 128u, 8388736.f); }
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    const float s = q[0] * a[m].x + q[1] * a[m].y + q[2] * a[m].z + q[3] * a[m].w + q[4] * c[m].x + q[5] * c[m].y + q[6] * c[m].z + q[7] * c[m].w;
                    acc[r][m] = fmaf(d, s, acc[r][m]);
                }
            }
        }
    } else if constexpr (QT == 5) {                // PQ2_0: 34-byte blocks of 128 (2-bit codes, value = code-1), 8 lanes per block
        const int nb = K / 128, t = lane & 7;
        #pragma unroll 2
        for (int b = lane >> 3; b < nb; b += 4) {
            float4 xv[MB][4];
#pragma unroll
            for (int m = 0; m < MB; m++) {
                const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 128 + 16 * t;
#pragma unroll
                for (int j = 0; j < 4; j++) xv[m][j] = ld_x4(x + 4 * j);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint8_t* p = PROW(r) + (size_t)b * 34;
                const float d = packed_half(p);
                const unsigned v0 = ld_u16x2(p + 2 + 4 * t);
                float q[16];
#pragma unroll
                for (int j = 0; j < 16; j++) q[j] = u2f((v0 >> (2 * j)) & 3u, 8388609.f);
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    float s = 0.f;
#pragma unroll
                    for (int j = 0; j < 4; j++) s += q[4 * j] * xv[m][j].x + q[4 * j + 1] * xv[m][j].y + q[4 * j + 2] * xv[m][j].z + q[4 * j + 3] * xv[m][j].w;
                    acc[r][m] = fmaf(d, s, acc[r][m]);
                }
            }
        }
    } else if constexpr (QT == 6) {                // PTQ1_0: 28-byte blocks of 128 (26 base-3 bytes + fp16 scale), one block per pass
        const int nb = K / 128;
        const int base = lane < 16 ? lane : lane < 24 ? 80 + (lane - 16) : 120 + (lane - 24);
        const int step = lane < 16 ? 16 : lane < 24 ? 8 : 2;
        const int cnt = lane < 24 ? 5 : 4;
        for (int b = 0; b < nb; b++) {
            float xv[MB][5];
#pragma unroll
            for (int m = 0; m < MB; m++) {
                const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 128 + base;
#pragma unroll
                for (int k = 0; k < 5; k++) xv[m][k] = (lane < 26 && k < cnt) ? x[step * k] : 0.f;
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint8_t* p = PROW(r) + (size_t)b * 28;
                const float d = packed_half(p + 26);
                const int byte = p[lane < 26 ? lane : 0];
                float q[5];
#pragma unroll
                for (int k = 0; k < 5; k++) {
                    const int f = k == 0 ? 1 : k == 1 ? 3 : k == 2 ? 9 : k == 3 ? 27 : 81;
                    q[k] = u2f((((byte * f) & 255) * 3 >> 8), 8388609.f);
                }
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    float s = 0.f;
#pragma unroll
                    for (int k = 0; k < 5; k++) s = fmaf(q[k], xv[m][k], s);   // inactive lanes / k >= cnt have x = 0
                    acc[r][m] = fmaf(d, s, acc[r][m]);
                }
            }
        }
    } else if constexpr (QT == 9 || QT == 10) {    // Q4_K / Q5_K: 256-element super-blocks, one per pass, 8 elements per lane
        const int nb = K / 256, c = lane >> 3, j = (lane & 7) * 4;
        constexpr int BS = QT == 9 ? 144 : 176;
        for (int b = 0; b < nb; b++) {
            float4 a[MB], h[MB];
#pragma unroll
            for (int m = 0; m < MB; m++) { const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 256 + 64 * c + j; a[m] = ld_x4(x); h[m] = ld_x4(x + 32); }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint8_t* p = PROW(r) + (size_t)b * BS;
                const float dd = packed_half(p), dm = packed_half(p + 2);
                uint8_t sb[12];
#pragma unroll
                for (int i = 0; i < 3; i++) { const unsigned u = *(const unsigned*)(p + 4 + 4 * i); sb[4 * i] = u & 255; sb[4 * i + 1] = (u >> 8) & 255; sb[4 * i + 2] = (u >> 16) & 255; sb[4 * i + 3] = u >> 24; }
                float sc0, mn0, sc1, mn1;
                {
                    const int g0 = 2 * c, g1 = 2 * c + 1;
                    if (c < 2) { sc0 = sb[g0] & 63; mn0 = sb[g0 + 4] & 63; sc1 = sb[g1] & 63; mn1 = sb[g1 + 4] & 63; }
                    else {
                        sc0 = (sb[g0 + 4] & 15) | ((sb[g0 - 4] >> 6) << 4); mn0 = (sb[g0 + 4] >> 4) | ((sb[g0] >> 6) << 4);
                        sc1 = (sb[g1 + 4] & 15) | ((sb[g1 - 4] >> 6) << 4); mn1 = (sb[g1 + 4] >> 4) | ((sb[g1] >> 6) << 4);
                    }
                }
                const unsigned qs = *(const unsigned*)(p + (QT == 9 ? 16 : 48) + c * 32 + j);
                float q0[4], q1[4];
#pragma unroll
                for (int e = 0; e < 4; e++) { q0[e] = u2f((qs >> (8 * e)) & 15u, 8388608.f); q1[e] = u2f((qs >> (8 * e + 4)) & 15u, 8388608.f); }
                if constexpr (QT == 10) {
                    const unsigned qh = *(const unsigned*)(p + 16 + j);
#pragma unroll
                    for (int e = 0; e < 4; e++) { q0[e] += (float)(((qh >> (8 * e + 2 * c)) & 1) << 4); q1[e] += (float)(((qh >> (8 * e + 2 * c + 1)) & 1) << 4); }
                }
                const float d0 = dd * sc0, m0 = dm * mn0, d1 = dd * sc1, m1 = dm * mn1;
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    const float sq0 = q0[0] * a[m].x + q0[1] * a[m].y + q0[2] * a[m].z + q0[3] * a[m].w, sx0 = a[m].x + a[m].y + a[m].z + a[m].w;
                    const float sq1 = q1[0] * h[m].x + q1[1] * h[m].y + q1[2] * h[m].z + q1[3] * h[m].w, sx1 = h[m].x + h[m].y + h[m].z + h[m].w;
                    acc[r][m] += d0 * sq0 - m0 * sx0 + d1 * sq1 - m1 * sx1;
                }
            }
        }
    } else if constexpr (QT == 11) {               // Q6_K: 210-byte blocks of 256, one per pass, 8 elements per lane
        const int nb = K / 256;
        for (int b = 0; b < nb; b++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                float xv[MB][4];
#pragma unroll
                for (int m = 0; m < MB; m++) { const float* x = X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + (size_t)b * 256 + h * 128 + lane; xv[m][0] = x[0]; xv[m][1] = x[32]; xv[m][2] = x[64]; xv[m][3] = x[96]; }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const uint8_t* p = PROW(r) + (size_t)b * 210;
                    const float d = packed_half(p + 208);
                    const int a0 = p[h * 64 + lane], a1 = p[h * 64 + 32 + lane], qh = p[128 + h * 32 + lane];
                    const int q0 = ((a0 & 15) | ((qh & 3) << 4)) - 32, q1 = ((a1 & 15) | (((qh >> 2) & 3) << 4)) - 32;
                    const int q2 = ((a0 >> 4) | (((qh >> 4) & 3) << 4)) - 32, q3 = ((a1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                    const int sb = 192 + h * 8 + (lane >> 4);
                    const float c0 = d * (float)(signed char)p[sb] * q0, c1 = d * (float)(signed char)p[sb + 2] * q1;
                    const float c2 = d * (float)(signed char)p[sb + 4] * q2, c3 = d * (float)(signed char)p[sb + 6] * q3;
#pragma unroll
                    for (int m = 0; m < MB; m++) acc[r][m] += c0 * xv[m][0] + c1 * xv[m][1] + c2 * xv[m][2] + c3 * xv[m][3];
                }
            }
        }
    } else if constexpr (QT == 0) {                // F32
        for (int k = lane * 4; k < K; k += 128) {
            float4 a[MB];
#pragma unroll
            for (int m = 0; m < MB; m++) a[m] = ld_x4(X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + k);
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float4 wv = *(const float4*)(PROW(r) + (size_t)k * 4);
#pragma unroll
                for (int m = 0; m < MB; m++) acc[r][m] += wv.x * a[m].x + wv.y * a[m].y + wv.z * a[m].z + wv.w * a[m].w;
            }
        }
    } else {                                       // F16 (1) / BF16 (12)
        for (int k = lane * 4; k < K; k += 128) {
            float4 a[MB];
#pragma unroll
            for (int m = 0; m < MB; m++) a[m] = ld_x4(X + (size_t)(MB == 1 ? 0 : min(m, M - 1)) * K + k);
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint2 u = *(const uint2*)(PROW(r) + (size_t)k * 2);
                float wv[4];
                if constexpr (QT == 1) {
                    const float2 f0 = __half22float2(*(const half2*)&u.x), f1 = __half22float2(*(const half2*)&u.y);
                    wv[0] = f0.x; wv[1] = f0.y; wv[2] = f1.x; wv[3] = f1.y;
                } else {
                    wv[0] = __uint_as_float(u.x << 16); wv[1] = __uint_as_float(u.x & 0xffff0000u);
                    wv[2] = __uint_as_float(u.y << 16); wv[3] = __uint_as_float(u.y & 0xffff0000u);
                }
#pragma unroll
                for (int m = 0; m < MB; m++) acc[r][m] += wv[0] * a[m].x + wv[1] * a[m].y + wv[2] * a[m].z + wv[3] * a[m].w;
            }
        }
    }
#undef PROW
}

// y[m][row] (op)= W[row] . X[m]; grid ceil(N / (8 R)), block 256 (8 warps x R rows), X rows contiguous (stride K)
template <int QT, int MB, int R>
__global__ void __launch_bounds__(256) pmv2(const uint8_t* __restrict__ W, const float* __restrict__ X, int K, int N, int M,
                                            float* __restrict__ Y, int stride, int offset, int add, const float* __restrict__ bias) {
    const int lane = threadIdx.x & 31, row0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
    if (row0 >= N) return;
    const int nr = min(R, N - row0);
    float acc[R][MB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MB; m++) acc[r][m] = 0.f;
    const size_t rb = packed_row_bytes(QT, K);
    pdot<QT, MB, R>(W + (size_t)row0 * rb, rb, nr, X, K, M, lane, acc);
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < MB; m++) {
            const float v = warp_sum(acc[r][m]);
            if (lane == 0 && m < M && r < nr) { const int row = row0 + r; float* y = Y + (size_t)m * stride + offset + row; const float t = v + (bias ? bias[row] : 0.f); if (add) *y += t; else *y = t; }
        }
}

// Expert-indexed variant for MoE decode: pair p uses expert ids[p] (rows [e*N, (e+1)*N)) and activation row p / act_div.
// grid (ceil(N / (8 R)), P)
template <int QT, int R>
__global__ void __launch_bounds__(256) pmv_ind(const uint8_t* __restrict__ W, const float* __restrict__ X, int K, int N,
                                               const int* __restrict__ ids, int act_div, float* __restrict__ Y) {
    const int lane = threadIdx.x & 31, row0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R, p = blockIdx.y;
    if (row0 >= N) return;
    const int nr = min(R, N - row0);
    float acc[R][1];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r][0] = 0.f;
    const size_t rb = packed_row_bytes(QT, K);
    pdot<QT, 1, R>(W + ((size_t)ids[p] * N + row0) * rb, rb, nr, X + (size_t)(p / act_div) * K, K, 1, lane, acc);
#pragma unroll
    for (int r = 0; r < R; r++) {
        const float v = warp_sum(acc[r][0]);
        if (lane == 0 && r < nr) Y[(size_t)p * N + row0 + r] = v;
    }
}

// gate/up pairs [P][2F] (interleaved) -> act(g)*u in fp32 [P][F]. grid P, block 256
extern "C" __global__ void __launch_bounds__(256) packed_act_rows(const float* __restrict__ gu, int F, float* __restrict__ out) {
    const float* g = gu + (size_t)blockIdx.x * 2 * F;
    for (int i = threadIdx.x; i < F; i += 256) out[(size_t)blockIdx.x * F + i] = act_fn(g[2 * i]) * g[2 * i + 1];
}

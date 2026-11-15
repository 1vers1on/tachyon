// Tachyon - CPU backend (AVX-512 VNNI on Zen4 / Ice Lake+, with AVX2 fallbacks)
//
// Weights: Q4 (asymmetric 4-bit, group 64) or Q8; activations are quantized on the fly to int8 with one
// scale per 64 values, so the inner loop is u8 x s8 VPDPBUSD (4-bit weight nibbles are unsigned 0..15).
#include "backend.h"
#include "threadpool.h"
#include "quant.h"
#include <sys/mman.h>
#include <algorithm>
#include <memory>

#if !defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__)
#error "CPU backend currently requires AVX-512 (F/BW/VL/VNNI). Build with -march=native on a supported CPU."
#endif

namespace {

constexpr int CPU_ABLK = 64;  // activation quant block on CPU (== Q4 group)

struct QMat {
    std::string name;
    QType t = QType::F32;
    int rows = 0, cols = 0;
    const uint8_t* d = nullptr;
    const uint8_t* s = nullptr;
};

struct Layer {
    const float *attn_norm, *ffn_norm, *q_norm = nullptr, *k_norm = nullptr, *bqkv = nullptr;
    QMat wqkv, wo, wgu, wdown;
};

struct QAct {
    int M = 0, K = 0;
    int8_t* q = nullptr;   // [M][K]
    float* d = nullptr;    // [M][K/64]
    float* s = nullptr;    // [M][K/64]  d * sum(q)
    std::vector<int8_t> qbuf;
    std::vector<float> dbuf, sbuf;
    const float* x = nullptr;  // original float activations (for F16/F32 weights)
    void resize(int m, int k) {
        M = m; K = k;
        if ((int64_t)qbuf.size() < (int64_t)m * k) { qbuf.resize((size_t)m * k + 64); }
        if ((int64_t)dbuf.size() < (int64_t)m * k / CPU_ABLK) { dbuf.resize((size_t)m * k / CPU_ABLK + 16); sbuf.resize(dbuf.size()); }
        q = qbuf.data(); d = dbuf.data(); s = sbuf.data();
    }
};

inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// quantize one row of K floats to int8 blocks of 64
void quantize_act_row(const float* x, int K, int8_t* q, float* d, float* s) {
    for (int g = 0; g < K / CPU_ABLK; g++) {
        const float* xp = x + g * CPU_ABLK;
        __m512 v0 = _mm512_loadu_ps(xp), v1 = _mm512_loadu_ps(xp + 16), v2 = _mm512_loadu_ps(xp + 32), v3 = _mm512_loadu_ps(xp + 48);
        __m512 m = _mm512_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)), _mm512_max_ps(_mm512_abs_ps(v2), _mm512_abs_ps(v3)));
        float amax = _mm512_reduce_max_ps(m);
        float dd = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0.f;
        __m512 vi = _mm512_set1_ps(inv);
        __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vi)), i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vi));
        __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(v2, vi)), i3 = _mm512_cvtps_epi32(_mm512_mul_ps(v3, vi));
        int isum = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(i0, i1), _mm512_add_epi32(i2, i3)));
        _mm_storeu_si128((__m128i*)(q + g * 64 + 0), _mm512_cvtsepi32_epi8(i0));
        _mm_storeu_si128((__m128i*)(q + g * 64 + 16), _mm512_cvtsepi32_epi8(i1));
        _mm_storeu_si128((__m128i*)(q + g * 64 + 32), _mm512_cvtsepi32_epi8(i2));
        _mm_storeu_si128((__m128i*)(q + g * 64 + 48), _mm512_cvtsepi32_epi8(i3));
        d[g] = dd;
        s[g] = dd * isum;
    }
}

// ---- Q4 x int8 dot products for T tokens against one weight row ----
template <int T>
inline void dot_q4(const uint8_t* wq, const uint16_t* ws, int K, const QAct& a, int m0, float* out) {
    const int G = K / 64;
    __m512 acc[T];
    float macc[T];
    for (int t = 0; t < T; t++) { acc[t] = _mm512_setzero_ps(); macc[t] = 0; }
    const __m256i mask = _mm256_set1_epi8(0x0F);
    alignas(64) float sc[2 * 1024];
    // convert half2 (scale,min) to float pairs
    int g2 = 0;
    for (; g2 + 16 <= 2 * G; g2 += 16) _mm512_store_ps(sc + g2, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(ws + g2))));
    for (; g2 < 2 * G; g2++) sc[g2] = h2f(ws[g2]);
    for (int g = 0; g < G; g++) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(wq + g * 32));
        __m256i lo = _mm256_and_si256(v, mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
        __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        w = _mm512_shuffle_i64x2(w, w, _MM_SHUFFLE(3, 1, 2, 0));
        const float sw = sc[2 * g], mw = sc[2 * g + 1];
        for (int t = 0; t < T; t++) {
            const int m = m0 + t;
            __m512i x = _mm512_loadu_si512((const void*)(a.q + (size_t)m * K + g * 64));
            __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_set1_ps(sw * a.d[(size_t)m * G + g]), acc[t]);
            macc[t] += mw * a.s[(size_t)m * G + g];
        }
    }
    for (int t = 0; t < T; t++) out[t] = hsum(acc[t]) + macc[t];
}

template <int T>
inline void dot_q8(const uint8_t* wq, const uint16_t* ws, int K, const QAct& a, int m0, float* out) {
    const int G = K / 32, GA = K / 64;
    __m256 acc[T];
    for (int t = 0; t < T; t++) acc[t] = _mm256_setzero_ps();
    for (int g = 0; g < G; g++) {
        __m256i w = _mm256_loadu_si256((const __m256i*)(wq + g * 32));
        __m256i aw = _mm256_abs_epi8(w);
        const float sw = h2f(ws[g]);
        for (int t = 0; t < T; t++) {
            const int m = m0 + t;
            __m256i x = _mm256_loadu_si256((const __m256i*)(a.q + (size_t)m * K + g * 32));
            __m256i sx = _mm256_sign_epi8(x, w);
            __m256i is = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sx);
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is), _mm256_set1_ps(sw * a.d[(size_t)m * GA + g / 2]), acc[t]);
        }
    }
    for (int t = 0; t < T; t++) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        out[t] = _mm_cvtss_f32(s);
    }
}

inline float dot_f16(const uint16_t* w, const float* x, int K) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < K; i += 16)
        acc = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w + i))), _mm512_loadu_ps(x + i), acc);
    return hsum(acc);
}
inline float dot_f32(const float* w, const float* x, int K) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < K; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), acc);
    return hsum(acc);
}

template <int T>
inline void dot_rows(const QMat& W, int r, const QAct& a, int m0, float* out) {
    const int K = W.cols;
    switch (W.t) {
        case QType::Q4:
            dot_q4<T>(W.d + (size_t)r * K / 2, (const uint16_t*)W.s + (size_t)r * (K / 64) * 2, K, a, m0, out);
            break;
        case QType::Q8:
            dot_q8<T>(W.d + (size_t)r * K, (const uint16_t*)W.s + (size_t)r * (K / 32), K, a, m0, out);
            break;
        case QType::F16:
            for (int t = 0; t < T; t++) out[t] = dot_f16((const uint16_t*)W.d + (size_t)r * K, a.x + (size_t)(m0 + t) * K, K);
            break;
        case QType::F32:
            for (int t = 0; t < T; t++) out[t] = dot_f32((const float*)W.d + (size_t)r * K, a.x + (size_t)(m0 + t) * K, K);
            break;
        case QType::Q3:  // converted to Q4 at load time
            die("unexpected Q3 matrix in CPU kernel");
    }
}

// y[m][r] = W[r] . x[m]   for rows [r0,r1), all M tokens
void gemm_rows(const QMat& W, const QAct& a, float* y, int ldy, int r0, int r1) {
    const int M = a.M;
    float tmp[8];
    for (int r = r0; r < r1; r++) {
        int m = 0;
        for (; m + 8 <= M; m += 8) { dot_rows<8>(W, r, a, m, tmp); for (int t = 0; t < 8; t++) y[(size_t)(m + t) * ldy + r] = tmp[t]; }
        switch (M - m) {
            case 0: break;
            case 1: dot_rows<1>(W, r, a, m, tmp); break;
            case 2: dot_rows<2>(W, r, a, m, tmp); break;
            case 3: dot_rows<3>(W, r, a, m, tmp); break;
            case 4: dot_rows<4>(W, r, a, m, tmp); break;
            case 5: dot_rows<5>(W, r, a, m, tmp); break;
            case 6: dot_rows<6>(W, r, a, m, tmp); break;
            case 7: dot_rows<7>(W, r, a, m, tmp); break;
        }
        for (int t = 0; m + t < M; t++) y[(size_t)(m + t) * ldy + r] = tmp[t];
    }
}

void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { __m512 v = _mm512_loadu_ps(x + i); acc = _mm512_fmadd_ps(v, v, acc); }
    float ss = hsum(acc) / n;
    __m512 sc = _mm512_set1_ps(1.f / sqrtf(ss + eps));
    for (int i = 0; i < n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), sc), _mm512_loadu_ps(w + i)));
}

inline float silu(float x) { return x / (1.f + expf(-x)); }

class CpuBackend : public Backend {
public:
    const char* name() const override { return "cpu"; }
    ~CpuBackend() override {
        if (wmem_) munmap(wmem_, wmem_size_);
    }

    void init(const ModelFile& mf, const BackendOptions& o) override {
        opt = o;
        c_ = mf.config;
        int nt = o.threads > 0 ? o.threads : std::max(1u, std::thread::hardware_concurrency() / 2);
        pool_ = std::make_unique<ThreadPool>(nt);
        load_weights(mf);
        // KV cache: [slot][layer][kv_head][pos][head_dim] fp16
        kv_stride_pos_ = c_.head_dim;
        kv_stride_head_ = (size_t)opt.ctx * c_.head_dim;
        kv_stride_layer_ = kv_stride_head_ * c_.n_kv_heads;
        kv_stride_slot_ = kv_stride_layer_ * c_.n_layers;
        size_t kv_elems = kv_stride_slot_ * opt.n_slots;
        kcache_.assign(kv_elems, 0);
        vcache_.assign(kv_elems, 0);
        // rope
        int half = c_.head_dim / 2;
        inv_freq_.resize(half);
        for (int i = 0; i < half; i++) {
            float f = powf(c_.rope_theta, -2.f * i / c_.head_dim);
            if (c_.rope_scaling == "llama3") {
                float wl = 2 * M_PI / f;
                float lo_wl = c_.rope_orig_ctx / c_.rope_low_freq, hi_wl = c_.rope_orig_ctx / c_.rope_high_freq;
                if (wl > lo_wl) f = f / c_.rope_factor;
                else if (wl >= hi_wl) {
                    float sm = (c_.rope_orig_ctx / wl - c_.rope_low_freq) / (c_.rope_high_freq - c_.rope_low_freq);
                    f = (1 - sm) * f / c_.rope_factor + sm * f;
                }
            } else if (c_.rope_scaling == "linear") f /= c_.rope_factor;
            inv_freq_[i] = f;
        }
        logi("cpu backend: %d threads, KV cache %.1f MB (%d slots x %d ctx)", nt, kv_elems * 4 / 1e6, opt.n_slots, opt.ctx);
    }

    int vocab() const override { return c_.vocab; }

    void forward(const std::vector<BatchToken>& batch) override {
        const int M = (int)batch.size();
        const int H = c_.hidden, qd = c_.q_dim(), kvd = c_.kv_dim(), F = c_.ffn, hd = c_.head_dim;
        const int nqkv = qd + 2 * kvd;
        ensure_buffers(M);
        // embeddings
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++) {
                const QMat& E = embd_;
                int tok = batch[m].token;
                size_t ds = qtype_data_bytes(E.t, 1, H), ss = qtype_scale_bytes(E.t, 1, H);
                dequantize_row(E.t, E.d + (size_t)tok * ds, E.s ? E.s + (size_t)tok * ss : nullptr, H, x_.data() + (size_t)m * H);
            }
        });
        const float scale = 1.f / sqrtf((float)hd);
        const int nl_dbg = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl_dbg; l++) {
            const Layer& L = layers_[l];
            // attn norm + quantize
            norm_quant(M, H, L.attn_norm, act_h_);
            matmul(L.wqkv, act_h_, qkv_.data(), nqkv);
            // bias, qk-norm, rope, kv-cache write
            pool_->parallel_for(M, [&](int64_t a, int64_t b) {
                for (int64_t m = a; m < b; m++) {
                    float* row = qkv_.data() + (size_t)m * nqkv;
                    if (L.bqkv) for (int i = 0; i < nqkv; i++) row[i] += L.bqkv[i];
                    float* q = row;
                    float* k = row + qd;
                    float* v = row + qd + kvd;
                    if (L.q_norm) {
                        for (int h = 0; h < c_.n_heads; h++) rmsnorm(q + h * hd, L.q_norm, q + h * hd, hd, c_.rms_eps);
                        for (int h = 0; h < c_.n_kv_heads; h++) rmsnorm(k + h * hd, L.k_norm, k + h * hd, hd, c_.rms_eps);
                    }
                    int pos = batch[m].pos;
                    float cs[256], sn[256];
                    for (int i = 0; i < hd / 2; i++) { float th = pos * inv_freq_[i]; cs[i] = cosf(th); sn[i] = sinf(th); }
                    auto rope = [&](float* vv) {
                        for (int i = 0; i < hd / 2; i++) {
                            float x0 = vv[i], x1 = vv[i + hd / 2];
                            vv[i] = x0 * cs[i] - x1 * sn[i];
                            vv[i + hd / 2] = x0 * sn[i] + x1 * cs[i];
                        }
                    };
                    for (int h = 0; h < c_.n_heads; h++) rope(q + h * hd);
                    for (int h = 0; h < c_.n_kv_heads; h++) {
                        rope(k + h * hd);
                        uint16_t* kc = kv_ptr(kcache_, batch[m].slot, l, h, pos);
                        uint16_t* vc = kv_ptr(vcache_, batch[m].slot, l, h, pos);
                        for (int i = 0; i < hd; i += 16) {
                            _mm256_storeu_si256((__m256i*)(kc + i), _mm512_cvtps_ph(_mm512_loadu_ps(k + h * hd + i), _MM_FROUND_TO_NEAREST_INT));
                            _mm256_storeu_si256((__m256i*)(vc + i), _mm512_cvtps_ph(_mm512_loadu_ps(v + h * hd + i), _MM_FROUND_TO_NEAREST_INT));
                        }
                    }
                }
            });
            // attention: one task per (token, head)
            const int group = c_.n_heads / c_.n_kv_heads;
            pool_->parallel_for((int64_t)M * c_.n_heads, [&](int64_t a, int64_t b) {
                std::vector<float> sc(opt.ctx);
                for (int64_t task = a; task < b; task++) {
                    int m = (int)(task / c_.n_heads), h = (int)(task % c_.n_heads);
                    int kvh = h / group;
                    const float* q = qkv_.data() + (size_t)m * nqkv + h * hd;
                    int n = batch[m].pos + 1;
                    const uint16_t* kb = kv_ptr(kcache_, batch[m].slot, l, kvh, 0);
                    const uint16_t* vb = kv_ptr(vcache_, batch[m].slot, l, kvh, 0);
                    float mx = -INFINITY;
                    for (int t = 0; t < n; t++) {
                        float s = dot_f16(kb + (size_t)t * hd, q, hd) * scale;
                        sc[t] = s;
                        mx = std::max(mx, s);
                    }
                    float sum = 0;
                    for (int t = 0; t < n; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                    float inv = 1.f / sum;
                    float* o = attn_.data() + (size_t)m * qd + h * hd;
                    __m512 acc[16];
                    int nv = hd / 16;
                    for (int i = 0; i < nv; i++) acc[i] = _mm512_setzero_ps();
                    for (int t = 0; t < n; t++) {
                        __m512 p = _mm512_set1_ps(sc[t] * inv);
                        const uint16_t* vr = vb + (size_t)t * hd;
                        for (int i = 0; i < nv; i++)
                            acc[i] = _mm512_fmadd_ps(p, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vr + i * 16))), acc[i]);
                    }
                    for (int i = 0; i < nv; i++) _mm512_storeu_ps(o + i * 16, acc[i]);
                }
            });
            // output projection + residual
            quant_rows(M, qd, attn_.data(), act_a_);
            matmul(L.wo, act_a_, tmp_.data(), H);
            add_into(x_.data(), tmp_.data(), (size_t)M * H);
            // ffn
            norm_quant(M, H, L.ffn_norm, act_h_);
            matmul(L.wgu, act_h_, gu_.data(), 2 * F);
            pool_->parallel_for(M, [&](int64_t a, int64_t b) {
                for (int64_t m = a; m < b; m++) {
                    const float* g = gu_.data() + (size_t)m * 2 * F;
                    float* o = ff_.data() + (size_t)m * F;
                    for (int i = 0; i < F; i++) o[i] = silu(g[2 * i]) * g[2 * i + 1];
                }
            });
            quant_rows(M, F, ff_.data(), act_f_);
            matmul(L.wdown, act_f_, tmp_.data(), H);
            add_into(x_.data(), tmp_.data(), (size_t)M * H);
        }
        // final norm + lm head, only for rows that want logits
        std::vector<int> rows;
        for (int m = 0; m < M; m++) if (batch[m].logits) rows.push_back(m);
        n_logits_ = (int)rows.size();
        if (rows.empty()) return;
        std::vector<float> xs((size_t)rows.size() * H);
        for (size_t i = 0; i < rows.size(); i++) rmsnorm(x_.data() + (size_t)rows[i] * H, out_norm_, xs.data() + i * H, H, c_.rms_eps);
        act_o_.resize((int)rows.size(), H);
        act_o_.x = xs.data();
        for (size_t i = 0; i < rows.size(); i++)
            quantize_act_row(xs.data() + i * H, H, act_o_.q + i * H, act_o_.d + i * (H / CPU_ABLK), act_o_.s + i * (H / CPU_ABLK));
        logits_.resize((size_t)rows.size() * c_.vocab);
        matmul(out_, act_o_, logits_.data(), c_.vocab);
    }

    void get_logits(int i, float* dst) override { memcpy(dst, logits_.data() + (size_t)i * c_.vocab, c_.vocab * 4); }

    void argmax_all(int* out) override {
        for (int i = 0; i < n_logits_; i++) {
            const float* l = logits_.data() + (size_t)i * c_.vocab;
            out[i] = (int)(std::max_element(l, l + c_.vocab) - l);
        }
    }

    void copy_kv(int src, int dst, int n) override {
        if (src == dst || n <= 0) return;
        for (int l = 0; l < c_.n_layers; l++)
            for (int h = 0; h < c_.n_kv_heads; h++) {
                memcpy(kv_ptr(kcache_, dst, l, h, 0), kv_ptr(kcache_, src, l, h, 0), (size_t)n * c_.head_dim * 2);
                memcpy(kv_ptr(vcache_, dst, l, h, 0), kv_ptr(vcache_, src, l, h, 0), (size_t)n * c_.head_dim * 2);
            }
    }

private:
    uint16_t* kv_ptr(std::vector<uint16_t>& c, int slot, int l, int h, int pos) {
        return c.data() + slot * kv_stride_slot_ + l * kv_stride_layer_ + h * kv_stride_head_ + (size_t)pos * kv_stride_pos_;
    }

    void ensure_buffers(int M) {
        const int H = c_.hidden, qd = c_.q_dim(), kvd = c_.kv_dim(), F = c_.ffn;
        if (M <= cap_) return;
        cap_ = M;
        x_.resize((size_t)M * H);
        hn_.resize((size_t)M * H);
        qkv_.resize((size_t)M * (qd + 2 * kvd));
        attn_.resize((size_t)M * qd);
        tmp_.resize((size_t)M * H);
        gu_.resize((size_t)M * 2 * F);
        ff_.resize((size_t)M * F);
    }

    void norm_quant(int M, int H, const float* w, QAct& a) {
        a.resize(M, H);
        a.x = hn_.data();
        pool_->parallel_for(M, [&](int64_t s, int64_t e) {
            for (int64_t m = s; m < e; m++) {
                rmsnorm(x_.data() + (size_t)m * H, w, hn_.data() + (size_t)m * H, H, c_.rms_eps);
                quantize_act_row(hn_.data() + (size_t)m * H, H, a.q + (size_t)m * H, a.d + (size_t)m * (H / CPU_ABLK), a.s + (size_t)m * (H / CPU_ABLK));
            }
        });
    }
    void quant_rows(int M, int K, const float* x, QAct& a) {
        a.resize(M, K);
        a.x = x;
        pool_->parallel_for(M, [&](int64_t s, int64_t e) {
            for (int64_t m = s; m < e; m++)
                quantize_act_row(x + (size_t)m * K, K, a.q + (size_t)m * K, a.d + (size_t)m * (K / CPU_ABLK), a.s + (size_t)m * (K / CPU_ABLK));
        });
    }
    void matmul(const QMat& W, const QAct& a, float* y, int ldy) {
        if (g_imatrix && a.x) g_imatrix->add(W.name, a.x, a.M, a.K);
        int64_t chunk = std::max<int64_t>(8, W.rows / (pool_->size() * 4));
        // dynamic-ish: interleaved static split for balance
        pool_->parallel_for(W.rows, [&](int64_t r0, int64_t r1) { gemm_rows(W, a, y, ldy, (int)r0, (int)r1); }, 16);
        (void)chunk;
    }
    void add_into(float* x, const float* y, size_t n) {
        for (size_t i = 0; i < n; i += 16) {
            if (i + 16 <= n) _mm512_storeu_ps(x + i, _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
            else for (size_t j = i; j < n; j++) x[j] += y[j];
        }
    }

    // Copy weights into anonymous (transparent-hugepage) memory: fewer TLB misses than the file mmap.
    void load_weights(const ModelFile& mf) {
        size_t total = 0;
        for (auto& kv : mf.tensors) total += align_up(kv.second.data_bytes, 64) + align_up(kv.second.scale_bytes, 64);
        wmem_size_ = align_up(total, 2 << 20);
        wmem_ = mmap(nullptr, wmem_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (wmem_ == MAP_FAILED) die("out of memory allocating %zu bytes for weights", wmem_size_);
        madvise(wmem_, wmem_size_, MADV_HUGEPAGE);
        uint8_t* p = (uint8_t*)wmem_;
        std::vector<std::pair<const TensorInfo*, std::pair<uint8_t*, uint8_t*>>> jobs;
        for (auto& kv : mf.tensors) {
            const TensorInfo& t = kv.second;
            uint8_t* d = p; p += align_up(t.data_bytes, 64);
            uint8_t* s = nullptr;
            if (t.scale_bytes) { s = p; p += align_up(t.scale_bytes, 64); }
            jobs.push_back({&t, {d, s}});
            ptrs_[t.name] = {d, s};
        }
        pool_->parallel_for((int64_t)jobs.size(), [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; i++) {
                memcpy(jobs[i].second.first, jobs[i].first->data, jobs[i].first->data_bytes);
                if (jobs[i].second.second) memcpy(jobs[i].second.second, jobs[i].first->scales, jobs[i].first->scale_bytes);
            }
        });
        auto mat = [&](const std::string& n) {
            const TensorInfo* t = mf.tensor(n);
            QMat q; q.name = n; q.t = t->type; q.rows = (int)t->rows; q.cols = (int)t->cols;
            q.d = ptrs_[n].first; q.s = ptrs_[n].second;
            if (q.t == QType::Q3) {
                // CPU runs Q3 through the Q4 kernels (values 0..7 fit in nibbles; same scale format)
                auto* buf = new std::vector<uint8_t>((size_t)q.rows * q.cols / 2);
                for (int r = 0; r < q.rows; r++) {
                    const uint8_t* src = q.d + (size_t)r * q.cols * 3 / 8;
                    uint8_t* dst = buf->data() + (size_t)r * q.cols / 2;
                    for (int b = 0; b < q.cols / 32; b++) {
                        const uint32_t* blk = (const uint32_t*)(src + b * 12);
                        uint8_t v[32];
                        for (int e = 0; e < 32; e++) {
                            int w = e / 16, k = (e % 16) / 4, bb = e % 4;
                            v[e] = ((blk[w] >> (8 * bb + 2 * k)) & 3) | (((blk[2] >> (8 * bb + e / 4)) & 1) << 2);
                        }
                        for (int j = 0; j < 16; j++) dst[b * 16 + j] = v[j] | (v[j + 16] << 4);
                    }
                }
                q.t = QType::Q4; q.d = buf->data();
            }
            if (getenv("TACH_DEBUG_DEQUANT") && (q.t == QType::Q4 || q.t == QType::Q8)) {
                // debug: run with dequantized f16 weights (isolates quantization error from kernel bugs)
                auto* buf = new std::vector<uint16_t>((size_t)q.rows * q.cols);
                std::vector<float> row(q.cols);
                for (int r = 0; r < q.rows; r++) {
                    dequantize_row(q.t, q.d + qtype_data_bytes(q.t, 1, q.cols) * r, q.s + qtype_scale_bytes(q.t, 1, q.cols) * r, q.cols, row.data());
                    for (int k = 0; k < q.cols; k++) (*buf)[(size_t)r * q.cols + k] = f2h(row[k]);
                }
                q.t = QType::F16; q.d = (const uint8_t*)buf->data(); q.s = nullptr;
            }
            return q;
        };
        auto vec = [&](const std::string& n, bool req = true) -> const float* {
            if (!mf.tensor(n, req)) return nullptr;
            return (const float*)ptrs_[n].first;
        };
        embd_ = mat("tok_embd");
        out_ = c_.tied_embed ? embd_ : mat("output");
        out_norm_ = vec("output_norm");
        layers_.resize(c_.n_layers);
        for (int l = 0; l < c_.n_layers; l++) {
            std::string p = "blk." + std::to_string(l) + ".";
            Layer& L = layers_[l];
            L.attn_norm = vec(p + "attn_norm");
            L.ffn_norm = vec(p + "ffn_norm");
            L.q_norm = vec(p + "q_norm", false);
            L.k_norm = vec(p + "k_norm", false);
            L.bqkv = vec(p + "bqkv", false);
            L.wqkv = mat(p + "wqkv");
            L.wo = mat(p + "wo");
            L.wgu = mat(p + "wgu");
            L.wdown = mat(p + "wdown");
        }
        logi("cpu backend: loaded %.2f GB of weights", total / 1e9);
    }

    ModelConfig c_;
    std::unique_ptr<ThreadPool> pool_;
    void* wmem_ = nullptr;
    size_t wmem_size_ = 0;
    std::map<std::string, std::pair<uint8_t*, uint8_t*>> ptrs_;
    QMat embd_, out_;
    const float* out_norm_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<uint16_t> kcache_, vcache_;
    size_t kv_stride_pos_, kv_stride_head_, kv_stride_layer_, kv_stride_slot_;
    std::vector<float> inv_freq_;
    int cap_ = 0;
    std::vector<float> x_, hn_, qkv_, attn_, tmp_, gu_, ff_, logits_;
    QAct act_h_, act_a_, act_f_, act_o_;
    int n_logits_ = 0;
};

}  // namespace

Backend* create_cpu_backend() { return new CpuBackend(); }

// ---- kernel self test (tachyon selftest) ----
#include <random>
void cpu_selftest() {
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    for (QType qt : {QType::Q4, QType::Q8}) {
        int K = 1024, N = 4, M = 3;
        std::vector<float> W((size_t)N * K), X((size_t)M * K);
        for (auto& v : W) v = nd(rng) * 0.02f;
        for (auto& v : X) v = nd(rng);
        std::vector<uint8_t> d, s;
        quantize_matrix(qt, N, K, [&](int64_t r, float* b) { memcpy(b, &W[r * K], K * 4); }, d, s, 1);
        QMat q; q.t = qt; q.rows = N; q.cols = K; q.d = d.data(); q.s = s.data();
        QAct a; a.resize(M, K); a.x = X.data();
        for (int m = 0; m < M; m++) quantize_act_row(&X[m * K], K, a.q + m * K, a.d + m * (K / 64), a.s + m * (K / 64));
        std::vector<float> y((size_t)M * N);
        gemm_rows(q, a, y.data(), N, 0, N);
        for (int m = 0; m < M; m++)
            for (int r = 0; r < N; r++) {
                std::vector<float> wr(K);
                dequantize_row(qt, d.data() + qtype_data_bytes(qt, 1, K) * r, s.data() + qtype_scale_bytes(qt, 1, K) * r, K, wr.data());
                double ref = 0;
                for (int k = 0; k < K; k++) ref += wr[k] * X[m * K + k];
                printf("%s m=%d r=%d kernel=%.5f ref=%.5f\n", qtype_name(qt), m, r, y[m * N + r], ref);
            }
    }
}

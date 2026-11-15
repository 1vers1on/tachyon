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

// out = x / rms(x) * w   (w may be null: unweighted). In-place allowed.
void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) { __m512 v = _mm512_loadu_ps(x + i); acc = _mm512_fmadd_ps(v, v, acc); }
    float ss = hsum(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    const float sc = 1.f / sqrtf(ss / n + eps);
    for (int j = 0; j < n; j++) out[j] = x[j] * sc * (w ? w[j] : 1.f);
}

inline float silu(float x) { return x / (1.f + expf(-x)); }

inline float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }

// rows [r0, r0+n) of a quantized matrix as its own matrix
inline QMat sub_rows(const QMat& W, int r0, int n) {
    QMat q = W;
    q.rows = n;
    q.d = W.d + qtype_data_bytes(W.t, r0, W.cols);
    if (W.s) q.s = W.s + qtype_scale_bytes(W.t, r0, W.cols);
    return q;
}

struct CLayer {
    LayerCfg c;
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *post_attn_norm = nullptr, *post_ffn_norm = nullptr;
    const float *q_norm = nullptr, *k_norm = nullptr, *bqkv = nullptr;
    QMat wqkv, wo, wgu, wdown;
    QMat conv_in, conv_out; const float* conv_w = nullptr;
    QMat dn_in, dn_out; const float *dn_conv = nullptr, *dn_A = nullptr, *dn_dt = nullptr, *dn_norm = nullptr;
    QMat router, wgu_e, wdown_e; const float* expert_bias = nullptr;
    QMat ple_gate, ple_out; const float* ple_post_norm = nullptr;
    std::vector<float> inv_freq;
    int kv = -1;          // index of the KV buffer used (own or shared)
    int kv_cap = 0;       // positions stored (ring for sliding-window layers)
    int state = -1;       // index of the recurrent state buffer
};

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
        size_t kv_bytes = 0, st_bytes = 0;
        for (int l = 0; l < c_.n_layers; l++) {
            CLayer& L = layers_[l];
            const LayerCfg& lc = L.c;
            if (lc.is_attn()) {
                if (lc.kv_src >= 0) { L.kv = layers_[lc.kv_src].kv; L.kv_cap = layers_[lc.kv_src].kv_cap; }
                else {
                    L.kv = (int)kc_.size();
                    L.kv_cap = lc.window > 0 ? std::min(opt.ctx, lc.window + opt.max_batch) : opt.ctx;
                    size_t n = (size_t)opt.n_slots * lc.n_kv_heads * L.kv_cap * lc.head_dim;
                    kc_.emplace_back(n, 0); vc_.emplace_back(n, 0);
                    kv_bytes += n * 4;
                }
                // rope frequencies: pairs (i, i + rope_half); only the first rope_rot rotate
                L.inv_freq.assign(lc.rope_half, 0.f);
                const int rd = lc.rope_den > 0 ? lc.rope_den : c_.arch == "qwen3_5" ? 2 * lc.rope_half : lc.head_dim;  // exponent denominator
                for (int i = 0; i < lc.rope_rot; i++) {
                    float f = powf(lc.rope_theta, -2.f * i / rd);
                    if (c_.rope_scaling == "llama3") {
                        float wl = 2 * M_PI / f;
                        float lo_wl = c_.rope_orig_ctx / c_.rope_low_freq, hi_wl = c_.rope_orig_ctx / c_.rope_high_freq;
                        if (wl > lo_wl) f = f / c_.rope_factor;
                        else if (wl >= hi_wl) {
                            float sm = (c_.rope_orig_ctx / wl - c_.rope_low_freq) / (c_.rope_high_freq - c_.rope_low_freq);
                            f = (1 - sm) * f / c_.rope_factor + sm * f;
                        }
                    } else if (c_.rope_scaling == "linear") f /= c_.rope_factor;
                    else if (c_.rope_scaling == "longrope") {
                        const auto& fac = opt.ctx > c_.rope_orig_ctx ? c_.rope_long : c_.rope_short;
                        if (i < (int)fac.size()) f /= fac[i];
                    }
                    L.inv_freq[i] = f;
                }
            } else if (lc.type == "conv") {
                L.state = (int)state_.size();
                state_.emplace_back((size_t)opt.n_slots * (lc.conv_k - 1) * c_.hidden, 0.f);
                st_bytes += state_.back().size() * 4;
            } else if (lc.type == "deltanet") {
                L.state = (int)state_.size();
                state_.emplace_back((size_t)opt.n_slots * (lc.dn_conv - 1) * lc.dn_conv_dim(), 0.f);
                state_.emplace_back((size_t)opt.n_slots * lc.dn_nv * lc.dn_dk * lc.dn_dv, 0.f);
                st_bytes += (state_[state_.size() - 2].size() + state_.back().size()) * 4;
            }
        }
        logi("cpu backend: %d threads, KV cache %.1f MB, recurrent state %.1f MB (%d slots x %d ctx)", nt, kv_bytes / 2 / 1e6,
             st_bytes / 1e6, opt.n_slots, opt.ctx);
    }

    int vocab() const override { return c_.vocab; }

    void forward(const std::vector<BatchToken>& batch) override {
        const int M = (int)batch.size();
        const int H = c_.hidden;
        ensure_buffers(M);
        for (auto& t : batch)
            if (t.pos < 0 || t.pos >= opt.ctx || t.slot < 0 || t.slot >= opt.n_slots) die("token position %d / slot %d out of range (ctx %d, slots %d)", t.pos, t.slot, opt.ctx, opt.n_slots);
        // embeddings
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++) {
                const QMat& E = embd_;
                int tok = batch[m].token;
                size_t ds = qtype_data_bytes(E.t, 1, H), ss = qtype_scale_bytes(E.t, 1, H);
                float* x = x_.data() + (size_t)m * H;
                dequantize_row(E.t, E.d + (size_t)tok * ds, E.s ? E.s + (size_t)tok * ss : nullptr, H, x);
                if (c_.embed_scale != 1.f) for (int i = 0; i < H; i++) x[i] *= c_.embed_scale;
            }
        });
        // a new sequence (position 0) starts from zero recurrent state
        for (int m = 0; m < M; m++) if (batch[m].pos == 0) reset_state(batch[m].slot);
        if (c_.ple_dim) ple_inputs(batch);
        const int nl = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl; l++) {
            CLayer& L = layers_[l];
            norm_quant(M, H, L.attn_norm, act_h_);
            if (L.c.is_attn()) attention(L, l, batch);
            else if (L.c.type == "conv") short_conv(L, batch);
            else delta_net(L, batch);
            residual_add(M, L.post_attn_norm, 1.f);
            norm_quant(M, H, L.ffn_norm, act_h_);
            if (L.c.n_experts) moe(L, M);
            else dense_ffn(L, M);
            if (c_.ple_dim) {
                residual_add(M, L.post_ffn_norm, 1.f);
                ple_layer(L, l, M);
                residual_add(M, L.ple_post_norm, L.c.layer_scalar);
            } else {
                residual_add(M, L.post_ffn_norm, L.c.layer_scalar);
            }
            if (getenv("TACH_DUMP_X")) {  // debug: residual after each layer -> $TACH_DUMP_X.<layer> (fp32 [M][H]); tools/hf_compare.py
                float m0 = 0, m1 = 0;
                for (int i = 0; i < H; i++) m0 = std::max(m0, fabsf(x_[i]));
                for (size_t i = H; i < (size_t)M * H; i++) m1 = std::max(m1, fabsf(x_[i]));
                fprintf(stderr, "[dump] layer %d out: row0 %.1f rest %.1f\n", l, m0, m1);
                std::string fn = std::string(getenv("TACH_DUMP_X")) + "." + std::to_string(l);
                if (FILE* f = fopen(fn.c_str(), "wb")) { fwrite(x_.data(), 4, (size_t)M * H, f); fclose(f); }
            }
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
        if (c_.final_softcap > 0) {
            const float cap = c_.final_softcap;
            for (auto& v : logits_) v = tanhf(v / cap) * cap;
        }
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
        for (int l = 0; l < c_.n_layers; l++) {
            const CLayer& L = layers_[l];
            if (!L.c.is_attn() || L.c.kv_src >= 0) continue;
            const int nn = std::min(n, L.kv_cap);
            for (int h = 0; h < L.c.n_kv_heads; h++) {
                memcpy(kv_ptr(L, kc_, dst, h, 0), kv_ptr(L, kc_, src, h, 0), (size_t)nn * L.c.head_dim * 2);
                memcpy(kv_ptr(L, vc_, dst, h, 0), kv_ptr(L, vc_, src, h, 0), (size_t)nn * L.c.head_dim * 2);
            }
        }
    }

private:
    uint16_t* kv_ptr(const CLayer& L, std::vector<std::vector<uint16_t>>& c, int slot, int h, int pos) {
        const LayerCfg& lc = L.c;
        return c[L.kv].data() + (((size_t)slot * lc.n_kv_heads + h) * L.kv_cap + pos % L.kv_cap) * lc.head_dim;
    }

    void reset_state(int slot) {
        for (auto& L : layers_) {
            if (L.state < 0) continue;
            if (L.c.type == "conv") {
                size_t n = (size_t)(L.c.conv_k - 1) * c_.hidden;
                std::fill(state_[L.state].begin() + slot * n, state_[L.state].begin() + (slot + 1) * n, 0.f);
            } else {
                size_t n0 = (size_t)(L.c.dn_conv - 1) * L.c.dn_conv_dim();
                size_t n1 = (size_t)L.c.dn_nv * L.c.dn_dk * L.c.dn_dv;
                std::fill(state_[L.state].begin() + slot * n0, state_[L.state].begin() + (slot + 1) * n0, 0.f);
                std::fill(state_[L.state + 1].begin() + slot * n1, state_[L.state + 1].begin() + (slot + 1) * n1, 0.f);
            }
        }
    }

    void ensure_buffers(int M) {
        if (M <= cap_) return;
        cap_ = M;
        const int H = c_.hidden;
        size_t maxw = H;
        for (auto& L : layers_) {
            const LayerCfg& lc = L.c;
            maxw = std::max<size_t>(maxw, (size_t)lc.qkv_rows() + lc.q_dim());
            maxw = std::max<size_t>(maxw, lc.q_dim());
            maxw = std::max<size_t>(maxw, 3 * (size_t)H);
            maxw = std::max<size_t>(maxw, lc.dn_in_rows());
            maxw = std::max<size_t>(maxw, 2 * (size_t)std::max(lc.ffn, lc.expert_ffn));
        }
        x_.resize((size_t)M * H);
        hn_.resize((size_t)M * H);
        big_.resize((size_t)M * maxw);
        mid_.resize((size_t)M * maxw);
        tmp_.resize((size_t)M * H);
    }

    // y = mixer output in tmp_ ([M][H]); x += post_norm ? rmsnorm(y) : y; x *= scalar
    void residual_add(int M, const float* post_norm, float scalar) {
        const int H = c_.hidden;
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            std::vector<float> t(H);
            for (int64_t m = a; m < b; m++) {
                float* x = x_.data() + (size_t)m * H;
                const float* y = tmp_.data() + (size_t)m * H;
                if (post_norm) { rmsnorm(y, post_norm, t.data(), H, c_.rms_eps); y = t.data(); }
                for (int i = 0; i < H; i++) x[i] = (x[i] + y[i]) * scalar;
            }
        });
    }

    void attention(CLayer& L, int l, const std::vector<BatchToken>& batch) {
        const LayerCfg& lc = L.c;
        const int M = (int)batch.size(), H = c_.hidden, hd = lc.head_dim, qd = lc.q_dim(), kvd = lc.kv_dim();
        const int nrows = lc.qkv_rows();
        const int qstride = lc.attn_gate ? 2 * hd : hd;  // q (and gate) per head in the projection
        const int qrows = qd * (lc.attn_gate ? 2 : 1);
        float* qkv = big_.data();
        matmul(L.wqkv, act_h_, qkv, nrows);
        const float scale = lc.attn_scale > 0 ? lc.attn_scale : 1.f / sqrtf((float)hd);
        const bool own_kv = lc.kv_src < 0;
        float* qbuf = mid_.data();  // [M][qd] rotated, scaled queries
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            std::vector<float> cs(lc.rope_half), sn(lc.rope_half), kbuf(kvd), vbuf(kvd);
            for (int64_t m = a; m < b; m++) {
                float* row = qkv + (size_t)m * nrows;
                if (L.bqkv) for (int i = 0; i < nrows; i++) row[i] += L.bqkv[i];
                const int pos = batch[m].pos;
                for (int i = 0; i < lc.rope_half; i++) {
                    float th = pos * L.inv_freq[i];
                    const float ms = L.inv_freq[i] != 0.f ? c_.rope_mscale : 1.f;   // longrope attention factor
                    cs[i] = cosf(th) * ms; sn[i] = sinf(th) * ms;
                }
                auto rope = [&](float* v) {
                    for (int i = 0; i < lc.rope_half; i++) {
                        float x0 = v[i], x1 = v[i + lc.rope_half];
                        v[i] = x0 * cs[i] - x1 * sn[i];
                        v[i + lc.rope_half] = x0 * sn[i] + x1 * cs[i];
                    }
                };
                float* q = qbuf + (size_t)m * qd;
                for (int h = 0; h < lc.n_heads; h++) memcpy(q + h * hd, row + h * qstride, hd * 4);
                if (lc.qk_norm == 2) rmsnorm(q, L.q_norm, q, qd, c_.rms_eps);
                for (int h = 0; h < lc.n_heads; h++) {
                    if (lc.qk_norm == 1) rmsnorm(q + h * hd, L.q_norm, q + h * hd, hd, c_.rms_eps);
                    rope(q + h * hd);
                    for (int i = 0; i < hd; i++) q[h * hd + i] *= scale;
                }
                if (!own_kv) continue;
                const float* kraw = row + qrows;
                memcpy(kbuf.data(), kraw, kvd * 4);
                memcpy(vbuf.data(), lc.k_eq_v ? kraw : kraw + kvd, kvd * 4);
                if (lc.qk_norm == 2) rmsnorm(kbuf.data(), L.k_norm, kbuf.data(), kvd, c_.rms_eps);
                for (int h = 0; h < lc.n_kv_heads; h++) {
                    float* k = kbuf.data() + h * hd;
                    float* v = vbuf.data() + h * hd;
                    if (lc.qk_norm == 1) rmsnorm(k, L.k_norm, k, hd, c_.rms_eps);
                    rope(k);
                    if (lc.v_norm) rmsnorm(v, nullptr, v, hd, c_.rms_eps);
                    uint16_t* kc = kv_ptr(L, kc_, batch[m].slot, h, pos);
                    uint16_t* vc = kv_ptr(L, vc_, batch[m].slot, h, pos);
                    for (int i = 0; i < hd; i++) { kc[i] = f2h(k[i]); vc[i] = f2h(v[i]); }
                }
            }
        });
        // attention: one task per (token, head)
        const int group = lc.n_heads / lc.n_kv_heads;
        float* out = big_.data() + (size_t)M * nrows;  // [M][qd] (after the qkv rows)
        pool_->parallel_for((int64_t)M * lc.n_heads, [&](int64_t a, int64_t b) {
            std::vector<float> sc(opt.ctx), kf(hd);
            for (int64_t task = a; task < b; task++) {
                int m = (int)(task / lc.n_heads), h = (int)(task % lc.n_heads);
                int kvh = h / group;
                const float* q = qbuf + (size_t)m * qd + h * hd;
                const int pos = batch[m].pos;
                const int t0 = lc.window > 0 ? std::max(0, pos - lc.window + 1) : 0;
                const uint16_t* kb = kv_ptr(L, kc_, batch[m].slot, kvh, 0);
                const uint16_t* vb = kv_ptr(L, vc_, batch[m].slot, kvh, 0);
                float mx = -INFINITY;
                for (int t = t0; t <= pos; t++) {
                    float s = 0;
                    const uint16_t* kr = kb + (size_t)(t % L.kv_cap) * hd;
                    for (int i = 0; i < hd; i++) s += q[i] * h2f(kr[i]);
                    sc[t] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0;
                for (int t = t0; t <= pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                float inv = 1.f / sum;
                float* o = out + (size_t)m * qd + h * hd;
                for (int i = 0; i < hd; i++) o[i] = 0;
                for (int t = t0; t <= pos; t++) {
                    float p = sc[t] * inv;
                    const uint16_t* vr = vb + (size_t)(t % L.kv_cap) * hd;
                    for (int i = 0; i < hd; i++) o[i] += p * h2f(vr[i]);
                }
                if (lc.attn_gate) {
                    const float* g = qkv + (size_t)m * nrows + h * qstride + hd;
                    for (int i = 0; i < hd; i++) o[i] *= 1.f / (1.f + expf(-g[i]));
                }
            }
        });
        (void)l;
        quant_rows(M, qd, out, act_a_);
        matmul(L.wo, act_a_, tmp_.data(), H);
    }

    // LFM2 gated short convolution: [B; C; x] = W_in h; y = W_out (C * conv(B * x))
    void short_conv(CLayer& L, const std::vector<BatchToken>& batch) {
        const int M = (int)batch.size(), H = c_.hidden, K = L.c.conv_k;
        float* bcx = big_.data();
        matmul(L.conv_in, act_h_, bcx, 3 * H);
        float* y = mid_.data();
        std::vector<float>& st = state_[L.state];
        pool_->parallel_for(H, [&](int64_t a, int64_t b) {
            for (int64_t ch = a; ch < b; ch++) {
                const float* w = L.conv_w + ch * K;
                for (int m = 0; m < M; m++) {  // tokens in order (each slot's tokens are consecutive positions)
                    float* s = st.data() + (size_t)batch[m].slot * (K - 1) * H;  // [K-1][H], oldest first
                    const float* r = bcx + (size_t)m * 3 * H;
                    float bx = r[ch] * r[2 * H + ch];
                    float acc = w[K - 1] * bx;
                    for (int k = 0; k < K - 1; k++) acc += w[k] * s[(size_t)k * H + ch];
                    for (int k = 0; k < K - 2; k++) s[(size_t)k * H + ch] = s[(size_t)(k + 1) * H + ch];
                    s[(size_t)(K - 2) * H + ch] = bx;
                    y[(size_t)m * H + ch] = r[H + ch] * acc;
                }
            }
        });
        quant_rows(M, H, y, act_a_);
        matmul(L.conv_out, act_a_, tmp_.data(), H);
    }

    // Qwen3.5 gated delta net (linear attention with a per-head dk x dv state)
    void delta_net(CLayer& L, const std::vector<BatchToken>& batch) {
        const LayerCfg& lc = L.c;
        const int M = (int)batch.size(), H = c_.hidden;
        const int CD = lc.dn_conv_dim(), KD = lc.dn_key_dim(), VD = lc.dn_val_dim(), NV = lc.dn_nv, NK = lc.dn_nk;
        const int DK = lc.dn_dk, DV = lc.dn_dv, KC = lc.dn_conv, R = lc.dn_in_rows();
        float* proj = big_.data();
        matmul(L.dn_in, act_h_, proj, R);
        // causal depthwise conv + SiLU over the qkv channels (in place)
        std::vector<float>& cst = state_[L.state];
        pool_->parallel_for(CD, [&](int64_t a, int64_t b) {
            for (int64_t ch = a; ch < b; ch++) {
                const float* w = L.dn_conv + ch * KC;
                for (int m = 0; m < M; m++) {
                    float* s = cst.data() + (size_t)batch[m].slot * (KC - 1) * CD;
                    float* r = proj + (size_t)m * R;
                    float xin = r[ch];
                    float acc = w[KC - 1] * xin;
                    for (int k = 0; k < KC - 1; k++) acc += w[k] * s[(size_t)k * CD + ch];
                    for (int k = 0; k < KC - 2; k++) s[(size_t)k * CD + ch] = s[(size_t)(k + 1) * CD + ch];
                    s[(size_t)(KC - 2) * CD + ch] = xin;
                    r[ch] = acc / (1.f + expf(-acc));
                }
            }
        });
        // recurrence per value head
        std::vector<float>& S = state_[L.state + 1];
        float* out = mid_.data();
        const int rep = NV / NK;
        pool_->parallel_for(NV, [&](int64_t a, int64_t b) {
            std::vector<float> q(DK), k(DK), kvm(DV), d(DV);
            for (int64_t h = a; h < b; h++) {
                const int kh = (int)h / rep;
                for (int m = 0; m < M; m++) {
                    float* Sh = S.data() + (((size_t)batch[m].slot * NV + h) * DK) * DV;
                    const float* r = proj + (size_t)m * R;
                    const float* qr = r + kh * DK;
                    const float* kr = r + KD + kh * DK;
                    const float* vr = r + 2 * KD + h * DV;
                    float qn = 0, kn = 0;
                    for (int i = 0; i < DK; i++) { qn += qr[i] * qr[i]; kn += kr[i] * kr[i]; }
                    qn = 1.f / sqrtf(qn + 1e-6f); kn = 1.f / sqrtf(kn + 1e-6f);
                    const float qs = qn / sqrtf((float)DK);
                    for (int i = 0; i < DK; i++) { q[i] = qr[i] * qs; k[i] = kr[i] * kn; }
                    const float braw = r[CD + VD + h], araw = r[CD + VD + NV + h];
                    const float beta = 1.f / (1.f + expf(-braw));
                    const float x = araw + L.dn_dt[h];
                    const float sp = x > 20.f ? x : log1pf(expf(x));
                    const float decay = expf(L.dn_A[h] * sp);
                    for (int j = 0; j < DV; j++) kvm[j] = 0;
                    for (int i = 0; i < DK; i++) {
                        float* row = Sh + (size_t)i * DV;
                        for (int j = 0; j < DV; j++) { row[j] *= decay; kvm[j] += row[j] * k[i]; }
                    }
                    for (int j = 0; j < DV; j++) d[j] = (vr[j] - kvm[j]) * beta;
                    float* o = out + (size_t)m * VD + h * DV;
                    for (int j = 0; j < DV; j++) o[j] = 0;
                    for (int i = 0; i < DK; i++) {
                        float* row = Sh + (size_t)i * DV;
                        for (int j = 0; j < DV; j++) { row[j] += k[i] * d[j]; o[j] += row[j] * q[i]; }
                    }
                    // gated RMSNorm: rmsnorm(o) * w * silu(z)
                    const float* z = r + CD + h * DV;
                    float ss = 0;
                    for (int j = 0; j < DV; j++) ss += o[j] * o[j];
                    const float sc = 1.f / sqrtf(ss / DV + c_.rms_eps);
                    for (int j = 0; j < DV; j++) o[j] = o[j] * sc * L.dn_norm[j] * (z[j] / (1.f + expf(-z[j])));
                }
            }
        });
        quant_rows(M, VD, out, act_a_);
        matmul(L.dn_out, act_a_, tmp_.data(), H);
    }

    float act(float g) const { return c_.act == "gelu" ? gelu_tanh(g) : silu(g); }

    // Gemma 4 per-layer inputs: ple[m][l][:] = (rmsnorm(P x / sqrt(H)) + E_ple[tok] * sqrt(D)) / sqrt(2)
    void ple_inputs(const std::vector<BatchToken>& batch) {
        const int M = (int)batch.size(), H = c_.hidden, D = c_.ple_dim, NL = c_.n_layers, W = NL * D;
        ple_.resize((size_t)M * W);
        QAct xa;
        quant_rows_into(M, H, x_.data(), xa);
        std::vector<float> proj((size_t)M * W);
        matmul(ple_proj_, xa, proj.data(), W);
        const float ps = 1.f / sqrtf((float)H), ts = sqrtf((float)D), is2 = 1.f / sqrtf(2.f);
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            std::vector<float> tok(W);
            for (int64_t m = a; m < b; m++) {
                const QMat& E = ple_embd_;
                size_t ds = qtype_data_bytes(E.t, 1, W), ss = qtype_scale_bytes(E.t, 1, W);
                dequantize_row(E.t, E.d + (size_t)batch[m].token * ds, E.s ? E.s + (size_t)batch[m].token * ss : nullptr, W, tok.data());
                for (int l = 0; l < NL; l++) {
                    float* pr = proj.data() + (size_t)m * W + l * D;
                    for (int i = 0; i < D; i++) pr[i] *= ps;
                    rmsnorm(pr, ple_proj_norm_, pr, D, c_.rms_eps);
                    float* o = ple_.data() + (size_t)m * W + l * D;
                    for (int i = 0; i < D; i++) o[i] = (pr[i] + tok[l * D + i] * ts) * is2;
                }
            }
        });
    }
    // per layer: tmp = W_out (act(W_gate x) * ple[l])   (residual + post norm applied by the caller)
    void ple_layer(CLayer& L, int l, int M) {
        const int H = c_.hidden, D = c_.ple_dim, W = c_.n_layers * D;
        QAct xa;
        quant_rows_into(M, H, x_.data(), xa);
        std::vector<float> g((size_t)M * D);
        matmul(L.ple_gate, xa, g.data(), D);
        for (int m = 0; m < M; m++)
            for (int i = 0; i < D; i++) g[(size_t)m * D + i] = act(g[(size_t)m * D + i]) * ple_[(size_t)m * W + l * D + i];
        QAct ga;
        quant_rows_into(M, D, g.data(), ga);
        matmul(L.ple_out, ga, tmp_.data(), H);
    }

    void dense_ffn(CLayer& L, int M) {
        const int H = c_.hidden, F = L.c.ffn;
        float* gu = big_.data();
        matmul(L.wgu, act_h_, gu, 2 * F);
        float* ff = mid_.data();
        pool_->parallel_for(M, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++) {
                const float* g = gu + (size_t)m * 2 * F;
                float* o = ff + (size_t)m * F;
                for (int i = 0; i < F; i++) o[i] = act(g[2 * i]) * g[2 * i + 1];
            }
        });
        quant_rows(M, F, ff, act_f_);
        matmul(L.wdown, act_f_, tmp_.data(), H);
    }

    // Mixture of experts: route each token, then run every used expert on its gathered tokens.
    void moe(CLayer& L, int M) {
        const LayerCfg& lc = L.c;
        const int H = c_.hidden, E = lc.n_experts, K = lc.top_k, F = lc.expert_ffn;
        std::vector<float> logits((size_t)M * E);
        matmul(L.router, act_h_, logits.data(), E);
        std::vector<int> sel((size_t)M * K);
        std::vector<float> wsel((size_t)M * K);
        for (int m = 0; m < M; m++) {
            float* lg = logits.data() + (size_t)m * E;
            std::vector<float> p(E), score(E);
            if (lc.router == "softmax") {
                float mx = *std::max_element(lg, lg + E), sum = 0;
                for (int e = 0; e < E; e++) { p[e] = expf(lg[e] - mx); sum += p[e]; }
                for (int e = 0; e < E; e++) p[e] /= sum;
            } else {
                for (int e = 0; e < E; e++) p[e] = 1.f / (1.f + expf(-lg[e]));
            }
            for (int e = 0; e < E; e++) score[e] = p[e] + (L.expert_bias ? L.expert_bias[e] : 0.f);
            std::vector<int> idx(E);
            for (int e = 0; e < E; e++) idx[e] = e;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return score[a] > score[b]; });
            float tot = 0;
            for (int k = 0; k < K; k++) tot += p[idx[k]];
            for (int k = 0; k < K; k++) {
                float w = p[idx[k]];
                if (lc.norm_topk) w /= (tot + (lc.router == "sigmoid" ? 1e-6f : 0.f));
                sel[(size_t)m * K + k] = idx[k];
                wsel[(size_t)m * K + k] = w * lc.routed_scale;
            }
        }
        std::fill(tmp_.begin(), tmp_.begin() + (size_t)M * H, 0.f);
        for (int e = 0; e < E; e++) {
            std::vector<std::pair<int, float>> toks;
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++) if (sel[(size_t)m * K + k] == e) toks.push_back({m, wsel[(size_t)m * K + k]});
            if (toks.empty()) continue;
            const int n = (int)toks.size();
            QAct ga;
            ga.resize(n, H);
            std::vector<float> gx((size_t)n * H);
            for (int i = 0; i < n; i++) {
                int m = toks[i].first;
                memcpy(gx.data() + (size_t)i * H, hn_.data() + (size_t)m * H, H * 4);
                memcpy(ga.q + (size_t)i * H, act_h_.q + (size_t)m * H, H);
                memcpy(ga.d + (size_t)i * (H / CPU_ABLK), act_h_.d + (size_t)m * (H / CPU_ABLK), (H / CPU_ABLK) * 4);
                memcpy(ga.s + (size_t)i * (H / CPU_ABLK), act_h_.s + (size_t)m * (H / CPU_ABLK), (H / CPU_ABLK) * 4);
            }
            ga.x = gx.data();
            std::vector<float> gu((size_t)n * 2 * F), ff((size_t)n * F), dn((size_t)n * H);
            matmul(sub_rows(L.wgu_e, e * 2 * F, 2 * F), ga, gu.data(), 2 * F);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < F; j++) ff[(size_t)i * F + j] = act(gu[(size_t)i * 2 * F + 2 * j]) * gu[(size_t)i * 2 * F + 2 * j + 1];
            QAct fa;
            quant_rows_into(n, F, ff.data(), fa);
            matmul(sub_rows(L.wdown_e, e * H, H), fa, dn.data(), H);
            for (int i = 0; i < n; i++) {
                float* y = tmp_.data() + (size_t)toks[i].first * H;
                const float w = toks[i].second;
                for (int j = 0; j < H; j++) y[j] += w * dn[(size_t)i * H + j];
            }
        }
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
    void quant_rows(int M, int K, const float* x, QAct& a) { quant_rows_into(M, K, x, a); }
    void quant_rows_into(int M, int K, const float* x, QAct& a) {
        a.resize(M, K);
        a.x = x;
        pool_->parallel_for(M, [&](int64_t s, int64_t e) {
            for (int64_t m = s; m < e; m++)
                quantize_act_row(x + (size_t)m * K, K, a.q + (size_t)m * K, a.d + (size_t)m * (K / CPU_ABLK), a.s + (size_t)m * (K / CPU_ABLK));
        });
    }
    void matmul(const QMat& W, const QAct& a, float* y, int ldy) {
        if (g_imatrix && a.x) g_imatrix->add(W.name, a.x, a.M, a.K);
        pool_->parallel_for(W.rows, [&](int64_t r0, int64_t r1) { gemm_rows(W, a, y, ldy, (int)r0, (int)r1); }, 16);
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
        if (c_.ple_dim) {
            ple_embd_ = mat("ple_embd");
            ple_proj_ = mat("ple_proj");
            ple_proj_norm_ = vec("ple_proj_norm");
        }
        layers_.resize(c_.n_layers);
        for (int l = 0; l < c_.n_layers; l++) {
            std::string p = "blk." + std::to_string(l) + ".";
            CLayer& L = layers_[l];
            L.c = c_.layers[l];
            L.attn_norm = vec(p + "attn_norm");
            L.ffn_norm = vec(p + "ffn_norm");
            L.post_attn_norm = vec(p + "post_attn_norm", false);
            L.post_ffn_norm = vec(p + "post_ffn_norm", false);
            if (L.c.is_attn()) {
                L.q_norm = vec(p + "q_norm", false);
                L.k_norm = vec(p + "k_norm", false);
                if (!L.k_norm && L.c.kv_src >= 0) L.k_norm = layers_[L.c.kv_src].k_norm;
                L.bqkv = vec(p + "bqkv", false);
                L.wqkv = mat(p + "wqkv");
                L.wo = mat(p + "wo");
            } else if (L.c.type == "conv") {
                L.conv_in = mat(p + "conv_in"); L.conv_out = mat(p + "conv_out"); L.conv_w = vec(p + "conv_w");
            } else if (L.c.type == "deltanet") {
                L.dn_in = mat(p + "dn_in"); L.dn_out = mat(p + "dn_out");
                L.dn_conv = vec(p + "dn_conv"); L.dn_A = vec(p + "dn_A"); L.dn_dt = vec(p + "dn_dt"); L.dn_norm = vec(p + "dn_norm");
            }
            if (c_.ple_dim) {
                L.ple_gate = mat(p + "ple_gate");
                L.ple_out = mat(p + "ple_out");
                L.ple_post_norm = vec(p + "ple_post_norm");
            }
            if (L.c.n_experts) {
                L.router = mat(p + "router");
                L.expert_bias = vec(p + "expert_bias", false);
                L.wgu_e = mat(p + "wgu_e");
                L.wdown_e = mat(p + "wdown_e");
            } else if (L.c.ffn) {
                L.wgu = mat(p + "wgu");
                L.wdown = mat(p + "wdown");
            }
        }
        logi("cpu backend: loaded %.2f GB of weights", total / 1e9);
    }

    ModelConfig c_;
    std::unique_ptr<ThreadPool> pool_;
    void* wmem_ = nullptr;
    size_t wmem_size_ = 0;
    std::map<std::string, std::pair<uint8_t*, uint8_t*>> ptrs_;
    QMat embd_, out_, ple_embd_, ple_proj_;
    const float* out_norm_ = nullptr;
    const float* ple_proj_norm_ = nullptr;
    std::vector<float> ple_;
    std::vector<CLayer> layers_;
    std::vector<std::vector<uint16_t>> kc_, vc_;
    std::vector<std::vector<float>> state_;
    int cap_ = 0;
    std::vector<float> x_, hn_, big_, mid_, tmp_, logits_;
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

// ================================================================================================
// CpuMoe: experts in system RAM for the CUDA backend (see cpu_moe.h)
// ================================================================================================
#include "cpu_moe.h"
#include <fcntl.h>
#include <unistd.h>

struct CpuMoe::Impl {
    struct L { QMat gu, dn; int E = 0, F = 0, H = 0; bool gelu = false; std::string pre; };
    std::unique_ptr<ThreadPool> pool;
    std::vector<L> layers;
    std::vector<std::pair<void*, size_t>> maps;
    size_t total = 0;
    // scratch
    QAct xa, ha;
    std::vector<float> xf, gu, h, dn;
    std::vector<int> order, start, cnt;

    // copy (and widen Q3 -> Q4) one tensor into hugepage-backed anonymous memory
    QMat load(const TensorInfo& t) {
        QMat q; q.name = t.name; q.rows = (int)t.rows; q.cols = (int)t.cols;
        const bool q3 = t.type == QType::Q3;
        q.t = q3 ? QType::Q4 : t.type;
        const size_t db = q3 ? (size_t)q.rows * q.cols / 2 : t.data_bytes, sb = t.scale_bytes;
        const size_t n = align_up(db, 64) + align_up(sb, 64);
        void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) die("out of memory for CPU experts (%s, %.1f GB so far)", t.name.c_str(), total / 1e9);
        madvise(m, n, MADV_HUGEPAGE);
        maps.push_back({m, n});
        total += n;
        uint8_t* d = (uint8_t*)m;
        uint8_t* s = d + align_up(db, 64);
        pool->parallel_for(q.rows, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; r++) {
                if (!q3) { memcpy(d + (size_t)r * (db / q.rows), t.data + (size_t)r * (db / q.rows), db / q.rows); continue; }
                const uint8_t* src = t.data + (size_t)r * q.cols * 3 / 8;
                uint8_t* dst = d + (size_t)r * q.cols / 2;
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
        }, 256);
        memcpy(s, t.scales, sb);
        q.d = d; q.s = s;
        return q;
    }
};

CpuMoe::CpuMoe(int threads) : p_(new Impl) {
    p_->pool = std::make_unique<ThreadPool>(threads > 0 ? threads : std::max(1u, std::thread::hardware_concurrency() / 2));
}
CpuMoe::~CpuMoe() { for (auto& m : p_->maps) munmap(m.first, m.second); }
size_t CpuMoe::bytes() const { return p_->total; }
std::vector<std::pair<void*, size_t>> CpuMoe::regions() const { return p_->maps; }
CpuMoe::HostMat CpuMoe::mat(int li, int which) const {
    const QMat& q = which == 0 ? p_->layers[li].gu : p_->layers[li].dn;
    return {q.t, q.rows, q.cols, q.d, q.s, qtype_data_bytes(q.t, q.rows, q.cols), qtype_scale_bytes(q.t, q.rows, q.cols)};
}

int CpuMoe::add_layer(const TensorInfo& gu, const TensorInfo& dn, int E, bool gelu, const std::string& pre) {
    Impl::L l;
    l.gu = p_->load(gu); l.dn = p_->load(dn);
    l.E = E; l.H = (int)gu.cols; l.F = (int)dn.cols; l.gelu = gelu; l.pre = pre;
    if (l.gu.rows != E * 2 * l.F || l.dn.rows != E * l.H) die("CpuMoe: unexpected expert shapes for %s", pre.c_str());
    p_->layers.push_back(l);
    return (int)p_->layers.size() - 1;
}

void CpuMoe::forward(int li, int M, int K, const uint16_t* xn, const int* ids, const float* w, float* out) {
    static const bool prof = getenv("TACH_CMOE_PROF") != nullptr;
    static double tsum[6]; static int ncall = 0;
    double tp = prof ? now_sec() : 0;
    auto mark = [&](int i) { if (!prof) return; double t = now_sec(); tsum[i] += t - tp; tp = t; };
    struct Rep { ~Rep() { if (prof && ncall) fprintf(stderr, "[cmoe] %d calls, avg ms: group+quant %.3f gu %.3f act %.3f down %.3f combine %.3f gap %.3f\n", ncall,
        1e3 * tsum[0] / ncall, 1e3 * tsum[1] / ncall, 1e3 * tsum[2] / ncall, 1e3 * tsum[3] / ncall, 1e3 * tsum[4] / ncall, 1e3 * tsum[5] / ncall); } };
    static Rep rep;
    static double tlast = 0;
    if (prof) { if (tlast > 0) tsum[5] += tp - tlast; ncall++; }
    Impl& I = *p_;
    const Impl::L& L = I.layers[li];
    const int H = L.H, F = L.F, E = L.E, P = M * K;
    // 1. group (token, expert) pairs by expert
    I.cnt.assign(E, 0);
    for (int p = 0; p < P; p++) I.cnt[ids[p]]++;
    I.start.assign(E + 1, 0);
    for (int e = 0; e < E; e++) I.start[e + 1] = I.start[e] + I.cnt[e];
    I.order.assign(P, 0);
    {
        std::vector<int> fill(I.start.begin(), I.start.end() - 1);
        for (int p = 0; p < P; p++) I.order[fill[ids[p]]++] = p;
    }
    // 2. gathered, quantized activations (row i = pair order[i])
    I.xf.resize((size_t)P * H);
    I.xa.resize(P, H);
    I.pool->parallel_for(P, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; i++) {
            const int m = I.order[i] / K;
            float* x = I.xf.data() + (size_t)i * H;
            for (int j = 0; j < H; j++) x[j] = h2f(xn[(size_t)m * H + j]);
            quantize_act_row(x, H, I.xa.q + (size_t)i * H, I.xa.d + (size_t)i * (H / CPU_ABLK), I.xa.s + (size_t)i * (H / CPU_ABLK));
        }
    }, 4);
    mark(0);
    if (g_imatrix) g_imatrix->add(L.pre + "wgu_e", I.xf.data(), P, H);
    auto view = [&](const QAct& a, int K2, int off, int n) {
        QAct v; v.M = n; v.K = K2;
        v.q = a.q + (size_t)off * K2; v.d = a.d + (size_t)off * (K2 / CPU_ABLK); v.s = a.s + (size_t)off * (K2 / CPU_ABLK);
        return v;
    };
    // tasks: (expert, row chunk) over all used experts in one dispatch
    std::vector<int> used;
    for (int e = 0; e < E; e++) if (I.cnt[e]) used.push_back(e);
    constexpr int CH = 32;
    // 3. gate/up for every pair, act(g) * u
    I.gu.resize((size_t)P * 2 * F);
    {
        const int nch = (2 * F + CH - 1) / CH;
        I.pool->parallel_for((int64_t)used.size() * nch, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; t++) {
                const int e = used[t / nch], c = (int)(t % nch);
                const QAct av = view(I.xa, H, I.start[e], I.cnt[e]);
                gemm_rows(sub_rows(L.gu, e * 2 * F, 2 * F), av, I.gu.data() + (size_t)I.start[e] * 2 * F, 2 * F, c * CH, std::min(2 * F, (c + 1) * CH));
            }
        });
    }
    mark(1);
    I.h.resize((size_t)P * F);
    I.ha.resize(P, F);
    I.pool->parallel_for(P, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; i++) {
            const float* g = I.gu.data() + (size_t)i * 2 * F;
            float* hh = I.h.data() + (size_t)i * F;
            for (int j = 0; j < F; j++) hh[j] = (L.gelu ? gelu_tanh(g[2 * j]) : silu(g[2 * j])) * g[2 * j + 1];
            quantize_act_row(hh, F, I.ha.q + (size_t)i * F, I.ha.d + (size_t)i * (F / CPU_ABLK), I.ha.s + (size_t)i * (F / CPU_ABLK));
        }
    }, 4);
    if (g_imatrix) g_imatrix->add(L.pre + "wdown_e", I.h.data(), P, F);
    mark(2);
    // 4. down projections
    I.dn.resize((size_t)P * H);
    {
        const int nch = (H + CH - 1) / CH;
        I.pool->parallel_for((int64_t)used.size() * nch, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; t++) {
                const int e = used[t / nch], c = (int)(t % nch);
                const QAct av = view(I.ha, F, I.start[e], I.cnt[e]);
                gemm_rows(sub_rows(L.dn, e * H, H), av, I.dn.data() + (size_t)I.start[e] * H, H, c * CH, std::min(H, (c + 1) * CH));
            }
        });
    }
    mark(3);
    // 5. weighted combine per token
    std::vector<int> pos(P);
    for (int i = 0; i < P; i++) pos[I.order[i]] = i;
    I.pool->parallel_for(M, [&](int64_t a, int64_t b) {
        for (int64_t m = a; m < b; m++) {
            float* o = out + (size_t)m * H;
            for (int j = 0; j < H; j++) o[j] = 0.f;
            for (int k = 0; k < K; k++) {
                const float wk = w[m * K + k];
                const float* d = I.dn.data() + (size_t)pos[m * K + k] * H;
                for (int j = 0; j < H; j++) o[j] += wk * d[j];
            }
        }
    }, 4);
    mark(4);
    if (prof) tlast = now_sec();
}

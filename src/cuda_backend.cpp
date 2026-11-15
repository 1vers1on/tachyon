// Tachyon - CUDA backend (driver API + NVRTC runtime compilation, CUDA graphs for decode)
//
// The model is a list of per-layer descriptors (LayerCfg). Mixers: attention (any head dim / GQA / window /
// gate / partial RoPE), LFM2 short conv, Qwen3.5 gated delta net. FFN: dense gated or mixture of experts.
// Shapes that drive register / shared-memory layout are template arguments; everything else is runtime.
#include "backend.h"
#include "quant.h"
#include <cuda.h>
#include <nvrtc.h>
#include <map>
#include <set>
#include <memory>
#include <sys/stat.h>
#include <algorithm>

namespace {

static const char kKernelSource[] = {
#embed "kernels/kernels.cu"
    , 0};

#define CU(x) do { CUresult _e = (x); if (_e != CUDA_SUCCESS) { const char* s = nullptr; cuGetErrorString(_e, &s); die("CUDA error %s at %s:%d: %s", s ? s : "?", __FILE__, __LINE__, #x); } } while (0)
#define NVRTC(x) do { nvrtcResult _e = (x); if (_e != NVRTC_SUCCESS) die("NVRTC error %s at %s:%d", nvrtcGetErrorString(_e), __FILE__, __LINE__); } while (0)

constexpr int MAX_DEC = 16;     // max tokens on the GEMV (decode) path
constexpr int MAX_LOGITS = 64;  // max logits rows per forward
constexpr int SPLITS = 16;      // attention KV splits for decode

struct DMat {
    QType t = QType::Q4;
    int rows = 0, cols = 0;
    CUdeviceptr d = 0, s = 0;
    bool valid() const { return d != 0; }
};
struct DLayer {
    LayerCfg c;
    CUdeviceptr attn_norm = 0, ffn_norm = 0, post_attn_norm = 0, post_ffn_norm = 0, q_norm = 0, k_norm = 0, bqkv = 0, inv_freq = 0;
    DMat wqkv, wo, wgu, wdown, conv_in, conv_out, dn_in, dn_out, wgu_e, wdown_e;
    CUdeviceptr conv_w = 0, dn_conv = 0, dn_A = 0, dn_dt = 0, dn_norm = 0, router = 0, expert_bias = 0;
    CUdeviceptr kc = 0, vc = 0, ksc = 0, vsc = 0;   // KV cache (shared with kv_src for Gemma 4)
    int kv_cap = 0;                                 // positions stored (ring of window + batch for sliding layers)
    CUdeviceptr st0 = 0, st1 = 0;                   // recurrent state (conv / delta net)
    DMat ple_gate, ple_out; CUdeviceptr ple_post_norm = 0;
    int in_rows() const { return c.is_attn() ? c.qkv_rows() : c.type == "conv" ? 3 * conv_in.cols : c.dn_in_rows(); }
};

// must match the device-side structs in kernels.cu
struct RopeArgs {
    CUdeviceptr qkv, qn, kn, inv_freq, pos, slot, qout, kc, vc, ksc, vsc;
    int nrows, nh, nkv, qstride, qrows, kvd, half, qk_norm, v_norm, k_eq_v, own_kv, ctx;
    float qscale;
};
struct AttnArgs {
    CUdeviceptr q, kc, vc, pos, slot, part_o, part_ml, counters, oq, ods, oh, gate, ksc, vsc;
    int nh, nkv, ctx, window, nrows, qstride;
};

struct DnArgs {
    CUdeviceptr proj, qkv, A, dt, kt, gq, at, kh, vt, gam;
    int M, R, NK, NV, nch;
};
constexpr int DN_CHUNK = 64;    // tokens per chunk of the chunked delta rule (DC in kernels.cu)

struct dim3_ { unsigned x = 1, y = 1, z = 1; };
inline dim3_ D3(unsigned x, unsigned y = 1, unsigned z = 1) { dim3_ d; d.x = x; d.y = y; d.z = z; return d; }

static int qt_code(QType t) { return t == QType::Q4 ? 3 : t == QType::Q3 ? 4 : t == QType::Q8 ? 2 : 1; }

class CudaBackend : public Backend {
public:
    const char* name() const override { return "cuda"; }
    ~CudaBackend() override {
        if (ctx_) {
            cuCtxSynchronize();
            for (auto& g : graphs_) if (g.second) cuGraphExecDestroy(g.second);
            if (module_) cuModuleUnload(module_);
            for (auto p : allocs_) cuMemFree(p);
            if (h_batch_) cuMemFreeHost(h_batch_);
            if (h_out_) cuMemFreeHost(h_out_);
            if (h_moe_) cuMemFreeHost(h_moe_);
            cuDevicePrimaryCtxRelease(dev_);
        }
    }

    bool probe() {
        if (cuInit(0) != CUDA_SUCCESS) return false;
        int n = 0;
        if (cuDeviceGetCount(&n) != CUDA_SUCCESS || n == 0) return false;
        return true;
    }

    void init(const ModelFile& mf, const BackendOptions& o) override {
        opt = o;
        c_ = mf.config;
        CU(cuDeviceGet(&dev_, o.device));
        CU(cuDevicePrimaryCtxRetain(&ctx_, dev_));
        CU(cuCtxSetCurrent(ctx_));
        CU(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
        char name[256];
        cuDeviceGetName(name, sizeof name, dev_);
        CU(cuDeviceGetAttribute(&cc_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_));
        CU(cuDeviceGetAttribute(&cc_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev_));
        CU(cuDeviceGetAttribute(&n_sm_, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_));
        if (cc_major_ < 8) die("CUDA backend needs compute capability >= 8.0 (got %d.%d)", cc_major_, cc_minor_);
        for (auto& l : c_.layers) {
            if (l.is_attn()) {
                if (l.head_dim % 32 || l.head_dim > 512) die("head_dim %d not supported on CUDA", l.head_dim);
                if (l.n_heads % l.n_kv_heads) die("bad GQA config");
            }
        }
        size_t free_b, total_b;
        cuMemGetInfo(&free_b, &total_b);
        logi("cuda: %s (sm_%d%d, %d SMs, %.1f/%.1f GB free)", name, cc_major_, cc_minor_, n_sm_, free_b / 1e9, total_b / 1e9);
        upload_weights(mf);
        alloc_caches();
        alloc_activations();
        compile_kernels();
        CU(cuStreamSynchronize(stream_));
        if (getenv("TACH_KBENCH")) {
            int M = atoi(getenv("TACH_KBENCH")); if (M < 1) M = 1;
            int pos = getenv("TACH_KBENCH_POS") ? atoi(getenv("TACH_KBENCH_POS")) : 512;
            kbench(M, pos);
            exit(0);
        }
    }

    int vocab() const override { return c_.vocab; }

    void bind() {
        thread_local CUcontext bound = nullptr;
        if (bound != ctx_) { CU(cuCtxSetCurrent(ctx_)); bound = ctx_; }
    }

    void forward(const std::vector<BatchToken>& batch) override {
        bind();
        const int M = (int)batch.size();
        if (M == 0) return;
        if (M > opt.max_batch) die("batch too large (%d > %d)", M, opt.max_batch);
        for (auto& t : batch)
            if (t.pos < 0 || t.pos >= opt.ctx || t.slot < 0 || t.slot >= opt.n_slots) die("token position %d / slot %d out of range (ctx %d, slots %d)", t.pos, t.slot, opt.ctx, opt.n_slots);
        lrows_.clear();
        for (int i = 0; i < M; i++) if (batch[i].logits) lrows_.push_back(i);
        n_logits_ = (int)lrows_.size();
        if (n_logits_ > MAX_LOGITS) die("too many logits rows (%d)", n_logits_);
        // upload batch descriptor (pinned staging ring -> device). A staging buffer is reused only after the
        // async copy that last read it has completed.
        stage_ = (stage_ + 1) % NSTAGE;
        CU(cuEventSynchronize(stage_ev_[stage_]));
        int* hb = h_batch_ + (size_t)stage_ * 4 * cap_;
        for (int i = 0; i < M; i++) {
            hb[i] = batch[i].token;
            hb[cap_ + i] = batch[i].pos;
            hb[2 * cap_ + i] = batch[i].slot;
        }
        for (int i = 0; i < n_logits_; i++) hb[3 * cap_ + i] = lrows_[i];
        CU(cuMemcpyHtoDAsync(d_batch_, hb, 4 * cap_ * sizeof(int), stream_));
        CU(cuEventRecord(stage_ev_[stage_], stream_));
        if (prof_ && ++prof_count_ % prof_every_ == 0) prof_report();
        if (M <= MAX_DEC) {
            const bool graphable = opt.use_graphs && n_logits_ == M && !prof_;
            if (graphable) {
                auto it = graphs_.find(M);
                if (it == graphs_.end()) {
                    CUgraph g;
                    CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_RELAXED));
                    decode_pass(M);
                    CU(cuStreamEndCapture(stream_, &g));
                    CUgraphExec ge;
                    CU(cuGraphInstantiate(&ge, g, 0));
                    cuGraphDestroy(g);
                    graphs_[M] = ge;
                    it = graphs_.find(M);
                }
                CU(cuGraphLaunch(it->second, stream_));
            } else {
                decode_pass(M);
            }
        } else {
            int slot = batch[0].slot, p0 = batch[0].pos;
            for (int i = 1; i < M; i++)
                if (batch[i].slot != slot || batch[i].pos != p0 + i) die("prefill batch must be one slot with consecutive positions");
            prefill_pass(M, slot, p0);
        }
    }

    void get_logits(int i, float* dst) override {
        bind();
        CU(cuMemcpyDtoHAsync(h_out_, logits_ + (size_t)i * c_.vocab * 4, (size_t)c_.vocab * 4, stream_));
        CU(cuStreamSynchronize(stream_));
        memcpy(dst, h_out_, (size_t)c_.vocab * 4);
    }

    void argmax_all(int* out) override {
        bind();
        launch(f_argmax_, D3(n_logits_), D3(1024), 0, logits_, c_.vocab, logit_rows(), d_argmax_);
        CU(cuMemcpyDtoHAsync(h_out_, d_argmax_, n_logits_ * 4, stream_));
        CU(cuStreamSynchronize(stream_));
        memcpy(out, h_out_, n_logits_ * 4);
    }

    bool candidates(int n, float delta, int cap, std::vector<std::vector<std::pair<float, int>>>& out) override {
        bind();
        cap = std::min(cap, CAND_CAP);
        launch(f_rowmax_, D3(n), D3(1024), 0, logits_, c_.vocab, d_rmax_, d_ccount_);
        launch(f_cand_, D3(32, n), D3(256), 0, logits_, c_.vocab, d_rmax_, delta, cap, d_ccount_, d_cand_);
        int* hc = (int*)h_out_;
        struct F2 { float x, y; };
        F2* hv = (F2*)(h_out_ + 64);
        CU(cuMemcpyDtoHAsync(hc, d_ccount_, n * 4, stream_));
        CU(cuMemcpyDtoHAsync(hv, d_cand_, (size_t)n * cap * 8, stream_));
        CU(cuStreamSynchronize(stream_));
        out.assign(n, {});
        for (int r = 0; r < n; r++) {
            int k = std::min(hc[r], cap);
            out[r].resize(k);
            for (int i = 0; i < k; i++) {
                F2 v = hv[(size_t)r * cap + i];
                int id; memcpy(&id, &v.y, 4);
                out[r][i] = {v.x, id};
            }
        }
        return true;
    }

    void copy_kv(int src, int dst, int n) override {
        bind();
        if (src == dst || n <= 0) return;
        for (auto& L : layers_) {
            if (!L.c.is_attn() || L.c.kv_src >= 0) continue;
            launch(f_kvcopy_, D3(L.c.n_kv_heads), D3(256), 0, L.kc, L.vc, src, dst, std::min(n, L.kv_cap), L.c.n_kv_heads, L.c.head_dim, L.kv_cap, L.ksc, L.vsc);
        }
    }

    void imat_flush() override {
        if (!g_imatrix) return;
        bind();
        CU(cuStreamSynchronize(stream_));
        for (auto& kv : imat_) {
            std::vector<float> h(kv.second.second);
            CU(cuMemcpyDtoH(h.data(), kv.second.first, h.size() * 4));
            std::lock_guard<std::mutex> lk(g_imatrix->mu);
            auto& e = g_imatrix->acc[kv.first];
            if (e.first.empty()) e.first.assign(h.size(), 0.0);
            for (size_t i = 0; i < h.size(); i++) e.first[i] += h[i];
            e.second += imat_rows_;
            CU(cuMemsetD8(kv.second.first, 0, h.size() * 4));
        }
        imat_rows_ = 0;
    }

private:
    // ---------------------------------------------------------------------------------------------
    CUdeviceptr alloc(size_t n) {
        CUdeviceptr p;
        CU(cuMemAlloc(&p, std::max<size_t>(n, 256)));
        allocs_.push_back(p);
        return p;
    }

    template <class... T>
    void launch(CUfunction f, dim3_ g, dim3_ b, unsigned smem, T... args) {
        void* p[] = {(void*)&args...};
        if (!kskip_.empty()) {  // TACH_KBENCH_SKIP: drop kernels whose name contains one of the substrings (timing ablation)
            if (knames_.empty()) for (auto& kv : funcs_) knames_[kv.second] = kv.first;
            auto it = knames_.find(f);
            const std::string n = it != knames_.end() ? it->second : std::string();
            for (auto& k : kskip_) if (!n.empty() && n.find(k) != std::string::npos) { kskipped_++; return; }
            klaunched_++;
        }
        if (prof_) {
            CUevent e0, e1;
            cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
            cuEventRecord(e0, stream_);
            CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, stream_, p, nullptr));
            cuEventRecord(e1, stream_);
            prof_events_.push_back({f, {e0, e1}});
            return;
        }
        CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, stream_, p, nullptr));
    }
    void prof_report() {
        cuStreamSynchronize(stream_);
        std::map<CUfunction, std::pair<double, int>> agg;
        double tot = 0;
        for (auto& e : prof_events_) {
            float ms; cuEventElapsedTime(&ms, e.second.first, e.second.second);
            agg[e.first].first += ms; agg[e.first].second++; tot += ms;
            cuEventDestroy(e.second.first); cuEventDestroy(e.second.second);
        }
        prof_events_.clear();
        std::map<CUfunction, std::string> names;
        for (auto& kv : funcs_) names[kv.second] = kv.first;
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : agg) v.push_back({kv.second.first, names[kv.first] + " x" + std::to_string(kv.second.second)});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[profile] total kernel time %.3f ms\n", tot);
        for (auto& x : v) fprintf(stderr, "  %8.3f ms %5.1f%%  %s\n", x.first, 100 * x.first / tot, x.second.c_str());
    }
    bool prof_ = getenv("TACH_PROFILE") != nullptr;
    std::vector<std::string> kskip_;
    std::map<CUfunction, std::string> knames_;
    int kskipped_ = 0, klaunched_ = 0;
    int prof_count_ = 0;
    int prof_every_ = prof_ && atoi(getenv("TACH_PROFILE")) > 1 ? atoi(getenv("TACH_PROFILE")) : 50;   // TACH_PROFILE=N: report every N forwards
    std::vector<std::pair<CUfunction, std::pair<CUevent, CUevent>>> prof_events_;

    // Timing of one full decode step (graph replay) + memory bandwidth probe.
    void kbench(int M, int pos) {
        std::vector<BatchToken> b(M);
        for (int i = 0; i < M; i++) b[i] = {100 + i, 0, pos + i, true};
        forward(b);
        if (const char* sk = getenv("TACH_KBENCH_SKIP")) {   // ';'-separated substrings (kernel names contain commas)
            std::string all = sk;
            for (size_t st = 0; st <= all.size();) {
                size_t e = all.find(';', st); if (e == std::string::npos) e = all.size();
                if (e > st) kskip_.push_back(all.substr(st, e - st));
                st = e + 1;
            }
        }
        CUgraph g; CUgraphExec ge;
        CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_RELAXED));
        decode_pass(M);
        CU(cuStreamEndCapture(stream_, &g));
        CU(cuGraphInstantiate(&ge, g, 0));
        if (!kskip_.empty()) fprintf(stderr, "[kbench] skipped %d kernels, kept %d\n", kskipped_, klaunched_);
        for (int i = 0; i < 5; i++) CU(cuGraphLaunch(ge, stream_));
        CU(cuStreamSynchronize(stream_));
        CUevent e0, e1; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
        cuEventRecord(e0, stream_);
        const int R = 100;
        for (int i = 0; i < R; i++) CU(cuGraphLaunch(ge, stream_));
        cuEventRecord(e1, stream_);
        CU(cuEventSynchronize(e1));
        float ms; cuEventElapsedTime(&ms, e0, e1);
        ms /= R;
        fprintf(stderr, "[kbench] M=%d pos=%d full decode step: %.3f ms (%.1f tok/s)\n", M, pos, ms, M * 1000.f / ms);
        fprintf(stderr, "  weights %.0f MB -> effective %.0f GB/s\n", weight_bytes_ / 1e6, weight_bytes_ / 1e6 / ms);
    }

    // ---------------------------------------------------------------------------------------------
    // weights / caches / buffers
    // ---------------------------------------------------------------------------------------------
    void upload_weights(const ModelFile& mf) {
        size_t total = 0;
        for (auto& kv : mf.tensors) total += align_up(kv.second.data_bytes, 256) + align_up(kv.second.scale_bytes, 256);
        CUdeviceptr base = alloc(total);
        size_t chunk = 64 << 20;
        void* pinned;
        CU(cuMemAllocHost(&pinned, chunk));
        size_t off = 0;
        auto up = [&](const uint8_t* src, size_t n) -> CUdeviceptr {
            CUdeviceptr dst = base + off;
            for (size_t done = 0; done < n; done += chunk) {
                size_t k = std::min(chunk, n - done);
                CU(cuStreamSynchronize(stream_));
                memcpy(pinned, src + done, k);
                CU(cuMemcpyHtoDAsync(dst + done, pinned, k, stream_));
            }
            off += align_up(n, 256);
            return dst;
        };
        std::map<std::string, std::pair<CUdeviceptr, CUdeviceptr>> ptr;
        for (auto& kv : mf.tensors) {
            const TensorInfo& t = kv.second;
            CUdeviceptr d = up(t.data, t.data_bytes);
            CUdeviceptr s = t.scale_bytes ? up(t.scales, t.scale_bytes) : 0;
            ptr[t.name] = {d, s};
        }
        CU(cuStreamSynchronize(stream_));
        cuMemFreeHost(pinned);
        weight_bytes_ = total;
        auto mat = [&](const std::string& n, bool req = true) {
            DMat m;
            const TensorInfo* t = mf.tensor(n, req);
            if (!t) return m;
            if (t->type == QType::F16 || t->type == QType::F32) die("tensor %s: f16/f32 matrices are not supported on CUDA; convert with -q q4|q8", n.c_str());
            m.t = t->type; m.rows = (int)t->rows; m.cols = (int)t->cols;
            m.d = ptr[n].first; m.s = ptr[n].second;
            return m;
        };
        auto vec = [&](const std::string& n, bool req = true) -> CUdeviceptr {
            if (!mf.tensor(n, req)) return 0;
            return ptr[n].first;
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
            DLayer& L = layers_[l];
            L.c = c_.layers[l];
            L.attn_norm = vec(p + "attn_norm");
            L.ffn_norm = vec(p + "ffn_norm");
            L.post_attn_norm = vec(p + "post_attn_norm", false);
            L.post_ffn_norm = vec(p + "post_ffn_norm", false);
            if (L.c.is_attn()) {
                L.q_norm = vec(p + "q_norm", false);
                L.k_norm = vec(p + "k_norm", false);
                L.bqkv = vec(p + "bqkv", false);
                L.wqkv = mat(p + "wqkv");
                L.wo = mat(p + "wo");
            } else if (L.c.type == "conv") {
                L.conv_in = mat(p + "conv_in"); L.conv_out = mat(p + "conv_out"); L.conv_w = vec(p + "conv_w");
            } else {
                L.dn_in = mat(p + "dn_in"); L.dn_out = mat(p + "dn_out");
                L.dn_conv = vec(p + "dn_conv"); L.dn_A = vec(p + "dn_A"); L.dn_dt = vec(p + "dn_dt"); L.dn_norm = vec(p + "dn_norm");
            }
            if (c_.ple_dim) {
                L.ple_gate = mat(p + "ple_gate");
                L.ple_out = mat(p + "ple_out");
                L.ple_post_norm = vec(p + "ple_post_norm");
            }
            if (L.c.n_experts) {
                L.router = vec(p + "router");
                L.expert_bias = vec(p + "expert_bias", false);
                L.wgu_e = mat(p + "wgu_e");
                L.wdown_e = mat(p + "wdown_e");
            } else if (L.c.ffn) {
                L.wgu = mat(p + "wgu");
                L.wdown = mat(p + "wdown");
            }
        }
        logi("cuda: uploaded %.2f GB of weights", total / 1e9);
    }

    void alloc_caches() {
        // KV caches per attention layer (shared when kv_src >= 0), recurrent states per stateful layer
        size_t free_b, total_b;
        cuMemGetInfo(&free_b, &total_b);
        const size_t kv_elt = opt.kv_int8 ? 1 : 2;
        size_t per_tok = 0, state_bytes = 0;
        size_t ring_bytes = 0;  // sliding-window layers: fixed-size ring, independent of ctx
        for (auto& L : layers_) {
            const LayerCfg& lc = L.c;
            const size_t tokb = (size_t)lc.n_kv_heads * (lc.head_dim * kv_elt + (opt.kv_int8 ? 4 : 0)) * 2;
            if (lc.is_attn() && lc.kv_src < 0) {
                if (lc.window > 0) ring_bytes += tokb * (lc.window + opt.max_batch);
                else per_tok += tokb;
            }
            if (lc.type == "conv") state_bytes += (size_t)(lc.conv_k - 1) * c_.hidden * 4;
            if (lc.type == "deltanet") state_bytes += ((size_t)(lc.dn_conv - 1) * lc.dn_conv_dim() + (size_t)lc.dn_nv * lc.dn_dk * lc.dn_dv) * 4;
        }
        size_t act_bytes = 256ull << 20;
        size_t reserve = (512ull << 20) + opt.reserve_bytes;
        size_t avail = free_b > reserve + act_bytes ? free_b - reserve - act_bytes : 0;
        const size_t st_total = (state_bytes + ring_bytes) * opt.n_slots;
        avail = avail > st_total ? avail - st_total : 0;
        const size_t budget_tok = per_tok + opt.extra_kv_per_tok;
        if (budget_tok > 0) {
            size_t want = budget_tok * opt.ctx * opt.n_slots;
            if (want > avail) {
                int ctx2 = (int)(avail / (budget_tok * opt.n_slots)) / 256 * 256;
                if (ctx2 < 512) { opt.n_slots = 1; ctx2 = (int)(avail / budget_tok) / 256 * 256; }
                if (ctx2 < 256) die("not enough GPU memory for KV cache");
                logi("cuda: reducing context to %d tokens/slot to fit KV cache in VRAM", ctx2);
                opt.ctx = ctx2;
            }
        }
        for (int l = 0; l < c_.n_layers; l++) {
            DLayer& L = layers_[l];
            const LayerCfg& lc = L.c;
            if (lc.is_attn()) {
                if (lc.kv_src >= 0) {
                    const DLayer& S = layers_[lc.kv_src];
                    L.kc = S.kc; L.vc = S.vc; L.ksc = S.ksc; L.vsc = S.vsc; L.kv_cap = S.kv_cap;
                } else {
                    L.kv_cap = lc.window > 0 ? std::min(opt.ctx, lc.window + opt.max_batch) : opt.ctx;
                    size_t n = (size_t)opt.n_slots * lc.n_kv_heads * L.kv_cap * lc.head_dim;
                    L.kc = alloc(n * kv_elt); L.vc = alloc(n * kv_elt);
                    CU(cuMemsetD8Async(L.kc, 0, n * kv_elt, stream_));
                    CU(cuMemsetD8Async(L.vc, 0, n * kv_elt, stream_));
                    if (opt.kv_int8) {
                        size_t ns = n / lc.head_dim;
                        L.ksc = alloc(ns * 4); L.vsc = alloc(ns * 4);
                        CU(cuMemsetD8Async(L.ksc, 0, ns * 4, stream_));
                        CU(cuMemsetD8Async(L.vsc, 0, ns * 4, stream_));
                    }
                }
                // rope table: pairs (i, i + rope_half); first rope_rot frequencies rotate
                std::vector<float> inv(std::max(1, lc.rope_half), 0.f);
                const int rd = c_.arch == "qwen3_5" ? 2 * lc.rope_half : lc.head_dim;
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
                    inv[i] = f;
                }
                L.inv_freq = alloc(inv.size() * 4);
                CU(cuMemcpyHtoD(L.inv_freq, inv.data(), inv.size() * 4));
            } else if (lc.type == "conv") {
                size_t n = (size_t)opt.n_slots * (lc.conv_k - 1) * c_.hidden * 4;
                L.st0 = alloc(n);
                CU(cuMemsetD8Async(L.st0, 0, n, stream_));
            } else {
                size_t n0 = (size_t)opt.n_slots * (lc.dn_conv - 1) * lc.dn_conv_dim() * 4;
                size_t n1 = (size_t)opt.n_slots * lc.dn_nv * lc.dn_dk * lc.dn_dv * 4;
                L.st0 = alloc(n0); L.st1 = alloc(n1);
                CU(cuMemsetD8Async(L.st0, 0, n0, stream_));
                CU(cuMemsetD8Async(L.st1, 0, n1, stream_));
            }
        }
        logi("cuda: KV cache %.2f GB (%d slots x %d ctx, %s)%s", per_tok * opt.ctx * opt.n_slots / 1e9, opt.n_slots, opt.ctx,
             opt.kv_int8 ? "int8" : "fp16", st_total ? (", recurrent state " + std::to_string(st_total >> 20) + " MB").c_str() : "");
    }

    void alloc_activations() {
        const int H = c_.hidden;
        cap_ = std::max(opt.max_batch, MAX_LOGITS);
        size_t maxK = H, maxRows = H, maxQ = 0, maxF = 0, maxP = 1;
        for (auto& L : layers_) {
            const LayerCfg& lc = L.c;
            maxRows = std::max<size_t>(maxRows, L.in_rows());
            maxQ = std::max<size_t>(maxQ, lc.q_dim());
            maxK = std::max<size_t>({maxK, (size_t)lc.q_dim(), (size_t)lc.ffn, (size_t)lc.expert_ffn, (size_t)lc.dn_val_dim()});
            maxF = std::max<size_t>({maxF, (size_t)lc.ffn, (size_t)lc.expert_ffn});
            if (lc.n_experts) { maxP = std::max<size_t>(maxP, (size_t)lc.top_k); has_moe_ = true; }
            maxHD_ = std::max(maxHD_, lc.head_dim);
            maxNH_ = std::max(maxNH_, lc.n_heads);
            maxNKV_ = std::max(maxNKV_, lc.n_kv_heads);
        }
        maxRows = std::max(maxRows, 2 * maxF);
        const size_t P = (size_t)cap_ * maxP;     // MoE (token, expert) pairs
        x_ = alloc((size_t)cap_ * H * 4);
        proj_ = alloc(std::max<size_t>((size_t)std::max<size_t>(cap_, MAX_DEC * maxP) * maxRows, P * H) * 4);
        tmp_ = alloc((size_t)std::max<size_t>(cap_, MAX_DEC * maxP) * std::max<size_t>(H, maxK) * 4);
        qf_ = alloc((size_t)cap_ * std::max<size_t>(maxQ, 32) * 4);
        qh_ = alloc((size_t)cap_ * std::max<size_t>(maxQ, 32) * 2);
        xq_ = alloc((size_t)std::max<size_t>({(size_t)MAX_DEC, (size_t)MAX_LOGITS}) * maxK + 64);
        xds_ = alloc((size_t)std::max<size_t>({(size_t)MAX_DEC, (size_t)MAX_LOGITS}) * (maxK / 32 + 1) * 8);
        fq_ = alloc((size_t)MAX_DEC * maxP * maxK + 64);
        fds_ = alloc((size_t)MAX_DEC * maxP * (maxK / 32 + 1) * 8);
        tcq_ = alloc((size_t)8 * maxK + 64);
        tcds_ = alloc((size_t)8 * (maxK / 32 + 1) * 8);
        ha_ = alloc(P * std::max<size_t>(maxK, H) * 2);
        hb_ = alloc(P * std::max<size_t>(maxK, H) * 2);
        const size_t parts = std::max<size_t>((size_t)MAX_DEC * SPLITS, (size_t)cap_) * std::max(1, maxNH_);
        part_o_ = alloc(parts * std::max(1, maxHD_) * 4);
        part_ml_ = alloc(parts * 8);
        counters_ = alloc((size_t)std::max(MAX_DEC, cap_) * std::max(1, maxNH_) * 4);
        CU(cuMemsetD8(counters_, 0, (size_t)std::max(MAX_DEC, cap_) * std::max(1, maxNH_) * 4));
        logits_ = alloc((size_t)MAX_LOGITS * c_.vocab * 4);
        d_argmax_ = alloc(MAX_LOGITS * 4);
        d_rmax_ = alloc(MAX_LOGITS * 4);
        d_ccount_ = alloc(MAX_LOGITS * 4);
        d_cand_ = alloc((size_t)MAX_DEC * CAND_CAP * 8);
        d_batch_ = alloc(4 * cap_ * sizeof(int));
        if (c_.ple_dim) {
            const size_t W = (size_t)c_.n_layers * c_.ple_dim;
            ple_buf_ = alloc((size_t)cap_ * W * 4);
            ple_tmp_ = alloc((size_t)cap_ * W * 4);
            ple_tmp2_ = alloc((size_t)cap_ * W * 4);
        }
        size_t maxCD = 0, maxNV = 0;
        for (auto& L : layers_) if (dn_chunk_ok(L.c)) { maxCD = std::max<size_t>(maxCD, L.c.dn_conv_dim()); maxNV = std::max<size_t>(maxNV, L.c.dn_nv); }
        if (maxCD) {
            const size_t hc = maxNV * ((cap_ + DN_CHUNK - 1) / DN_CHUNK);   // (head, chunk) pairs, dk = dv = 128
            dn_cv_ = alloc((size_t)cap_ * maxCD * 4);
            dn_kt_ = alloc(hc * DN_CHUNK * 128 * 2);
            dn_gq_ = alloc(hc * DN_CHUNK * 128 * 2);
            dn_kh_ = alloc(hc * DN_CHUNK * 128 * 2);
            dn_at_ = alloc(hc * DN_CHUNK * DN_CHUNK * 2);
            dn_vt_ = alloc(hc * DN_CHUNK * 128 * 4);
            dn_gam_ = alloc(hc * 4);
        }
        if (has_moe_) {
            moe_ids_ = alloc(P * 4);
            moe_w_ = alloc(P * 4);
            moe_rows_ = alloc(P * 4);
            moe_rw_ = alloc(P * 4);
            CU(cuMemAllocHost((void**)&h_moe_, P * 16 + 4096));
        }
        CU(cuMemAllocHost((void**)&h_batch_, NSTAGE * 4 * cap_ * sizeof(int)));
        for (int i = 0; i < NSTAGE; i++) {
            CU(cuEventCreate(&stage_ev_[i], CU_EVENT_DISABLE_TIMING));
            CU(cuEventRecord(stage_ev_[i], stream_));
        }
        CU(cuMemAllocHost((void**)&h_out_, std::max((size_t)c_.vocab * 4, (size_t)MAX_DEC * CAND_CAP * 8) + 4096));
    }

    // ---------------------------------------------------------------------------------------------
    // Kernel compilation (NVRTC) with an on-disk cubin cache
    // ---------------------------------------------------------------------------------------------
    static std::string I(int v) { return std::to_string(v); }
    std::string gemv2_name(QType t, int K, int M, int pro, int epi) { return "gemv2<" + I(qt_code(t)) + "," + I(K) + "," + I(M) + "," + I(pro) + "," + I(epi) + ">"; }
    std::string gu2_name(QType t, int K, int M) { return "gemv2_gu<" + I(qt_code(t)) + "," + I(K) + "," + I(M) + ">"; }
    std::string tc_name(QType t, int K, int epi) { return "gemv_tc<" + I(qt_code(t)) + "," + I(K) + "," + I(epi) + ">"; }
    std::string gemm_name(QType t, int epi) { return "gemm<" + I(qt_code(t)) + "," + I(epi) + ">"; }
    std::string ind_name(QType t, int K) { return "gemv_ind<" + I(qt_code(t)) + "," + I(K) + ">"; }
    std::string rope_name(int hd, int f16) { return "rope_t<" + I(hd) + "," + I(f16) + ">"; }
    std::string attn_name(int hd, int g, int splits, int f16) { return "attn_decode<" + I(hd) + "," + I(g) + "," + I(splits) + "," + I(f16) + ">"; }
    // query heads per attention CTA: largest divisor of the GQA ratio that is <= 4
    static int attn_group(const LayerCfg& lc) {
        const int gqa = lc.n_heads / lc.n_kv_heads;
        for (int g = 4; g >= 1; g--) if (gqa % g == 0) return g;
        return 1;
    }
    std::string flash_name(int hd) { return (hd > 128 ? "flash_wide<" : "flash_prefill<") + I(hd) + ">"; }
    // tensor-core prefill attention: FA2-style kernel up to 128 dims, sliced-head kernel for 256 / 512
    static bool flash_ok(int hd) { return hd <= 128 || hd == 256 || hd == 512; }
    static int flash_smem(int hd) { return hd > 128 ? 32 * (hd + 8) * 2 + 4 * 32 * 16 * 4 : 2 * 64 * (hd + 8) * 2; }
    static int flash_qrows(int hd) { return hd > 128 ? 16 * (4 / (hd / 128)) : 64; }
    std::string dn_name(int dk, int dv) { return "delta_rule<" + I(dk) + "," + I(dv) + ">"; }
    // chunked (tensor-core) delta rule for prefill; TACH_NO_DN_CHUNK=1 / TACH_NO_CONV_PAR=1 select the sequential kernels
    static bool dn_chunk_ok(const LayerCfg& lc) {
        static const bool off = getenv("TACH_NO_DN_CHUNK") != nullptr;
        return !off && lc.type == "deltanet" && lc.dn_dk == 128 && lc.dn_dv == 128 && lc.dn_nv % lc.dn_nk == 0 && lc.dn_conv <= 8;
    }
    static bool conv_par_ok(const LayerCfg& lc) {
        static const bool off = getenv("TACH_NO_CONV_PAR") != nullptr;
        return !off && lc.type == "conv" && lc.conv_k >= 2 && lc.conv_k <= 8;
    }
    static constexpr int DN_PREP_SMEM = DN_CHUNK * (128 + 4) * 2 + DN_CHUNK * DN_CHUNK * 4 + 3 * DN_CHUNK * 4;
    static bool tc_ok(QType t) { return t == QType::Q4 || t == QType::Q3; }

    void compile_kernels() {
        const int H = c_.hidden;
        std::vector<std::string> opts = {
            "--gpu-architecture=sm_" + std::to_string(cc_major_) + std::to_string(cc_minor_),
            "-std=c++17", "-default-device", "--extra-device-vectorization",
            getenv("TACH_PTXAS_V") ? "--ptxas-options=-v" : "-DTACH_NOP",
            "-I/usr/local/cuda/include",
            "-DHID=" + std::to_string(H), "-DEPS=" + std::to_string(c_.rms_eps) + "f", "-DEQ=" + std::to_string(qt_code(embd_.t)),
        };
        if (c_.act == "gelu") opts.push_back("-DACT_GELU=1");
        if (opt.kv_int8) opts.push_back("-DKV8=1");
        if (getenv("TACH_GEMM_F16ACC")) opts.push_back("-DGEMM_F16ACC=1");
        std::set<std::string> names;
        auto add_in = [&](const DMat& W, int epi) {  // RMS-prologue GEMV + GEMM + tc
            for (int M = 1; M <= 8; M++) names.insert(gemv2_name(W.t, W.cols, M, 1, epi));
            names.insert(gemm_name(W.t, epi));
            if (tc_ok(W.t) && W.cols % 256 == 0) names.insert(tc_name(W.t, W.cols, epi));
        };
        auto add_out = [&](const DMat& W) {  // int8-activation GEMV (store or add) + GEMM + tc
            for (int M = 1; M <= 8; M++) { names.insert(gemv2_name(W.t, W.cols, M, 0, 1)); names.insert(gemv2_name(W.t, W.cols, M, 0, 0)); }
            names.insert(gemm_name(W.t, 1)); names.insert(gemm_name(W.t, 0));
            if (tc_ok(W.t) && W.cols % 256 == 0) { names.insert(tc_name(W.t, W.cols, 1)); names.insert(tc_name(W.t, W.cols, 0)); }
        };
        for (auto& L : layers_) {
            const LayerCfg& lc = L.c;
            if (lc.is_attn()) {
                add_in(L.wqkv, 0);
                add_out(L.wo);
                names.insert(rope_name(lc.head_dim, 0));
                names.insert(rope_name(lc.head_dim, 1));
                const int g = attn_group(lc);
                names.insert(attn_name(lc.head_dim, g, SPLITS, 0));
                if (flash_ok(lc.head_dim) && !no_flash_wide(lc.head_dim)) names.insert(flash_name(lc.head_dim));
                else names.insert(attn_name(lc.head_dim, g, 1, 1));
            } else if (lc.type == "conv") {
                add_in(L.conv_in, 0);
                add_out(L.conv_out);
            } else {
                add_in(L.dn_in, 0);
                add_out(L.dn_out);
                names.insert(dn_name(lc.dn_dk, lc.dn_dv));
                if (dn_chunk_ok(lc)) {
                    names.insert("dn_prep<128,128>"); names.insert("dn_chunk<128,128>");
                    names.insert("conv_par<1>"); names.insert("conv_state<1>");
                }
            }
            if (conv_par_ok(lc)) { names.insert("conv_par<0>"); names.insert("conv_state<0>"); }
            if (lc.n_experts) {
                names.insert(ind_name(L.wgu_e.t, L.wgu_e.cols));
                names.insert(ind_name(L.wdown_e.t, L.wdown_e.cols));
                names.insert(gemm_name(L.wgu_e.t, 2));
                names.insert(gemm_name(L.wdown_e.t, 0));
            } else if (lc.ffn) {
                for (int M = 1; M <= 8; M++) names.insert(gu2_name(L.wgu.t, L.wgu.cols, M));
                names.insert(gemm_name(L.wgu.t, 2));
                if (tc_ok(L.wgu.t) && L.wgu.cols % 256 == 0) names.insert(tc_name(L.wgu.t, L.wgu.cols, 2));
                add_out(L.wdown);
            }
        }
        if (c_.ple_dim) {
            add_out(ple_proj_);
            for (auto& L : layers_) { add_out(L.ple_gate); add_out(L.ple_out); }
        }
        names.insert("causal_conv<0>");
        names.insert("causal_conv<1>");
        for (int M = 1; M <= 8; M++) names.insert(gemv2_name(out_.t, out_.cols, M, 1, 0));
        if (tc_ok(out_.t) && out_.cols % 256 == 0) names.insert(tc_name(out_.t, out_.cols, 0));
        std::string key = std::string(kKernelSource);
        for (auto& o : opts) key += "|" + o;
        for (auto& n : names) key += "|" + n;
        int nv_major = 0, nv_minor = 0;
        nvrtcVersion(&nv_major, &nv_minor);
        key += "|nvrtc" + std::to_string(nv_major) + "." + std::to_string(nv_minor);
        char hash[32];
        snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a64(key.data(), key.size()));
        std::string home = getenv("HOME") ? getenv("HOME") : "/tmp";
        std::string dir = home + "/.cache/tachyon";
        mkdir((home + "/.cache").c_str(), 0755);
        mkdir(dir.c_str(), 0755);
        std::string cubin_path = dir + "/kernels-" + hash + ".cubin", names_path = dir + "/kernels-" + hash + ".names";
        std::string cubin;
        std::map<std::string, std::string> lowered;
        if (file_exists(cubin_path) && file_exists(names_path)) {
            cubin = read_file(cubin_path);
            Json nj = Json::parse(read_file(names_path));
            for (auto& kv : nj.obj) lowered[kv.first] = kv.second.str;
            logd("cuda: loaded cached kernels %s", cubin_path.c_str());
        } else {
            double t0 = now_sec();
            nvrtcProgram prog;
            NVRTC(nvrtcCreateProgram(&prog, kKernelSource, "kernels.cu", 0, nullptr, nullptr));
            for (auto& n : names) NVRTC(nvrtcAddNameExpression(prog, n.c_str()));
            std::vector<const char*> co;
            for (auto& o : opts) co.push_back(o.c_str());
            nvrtcResult r = nvrtcCompileProgram(prog, (int)co.size(), co.data());
            size_t logn;
            nvrtcGetProgramLogSize(prog, &logn);
            if (logn > 1) {
                std::string log(logn, '\0');
                nvrtcGetProgramLog(prog, &log[0]);
                if (r != NVRTC_SUCCESS || g_log_level >= 2) fprintf(stderr, "%s\n", log.c_str());
            }
            if (r != NVRTC_SUCCESS) die("kernel compilation failed");
            size_t n;
            NVRTC(nvrtcGetCUBINSize(prog, &n));
            cubin.resize(n);
            NVRTC(nvrtcGetCUBIN(prog, &cubin[0]));
            Json nj = Json::object();
            for (auto& nm : names) {
                const char* low;
                NVRTC(nvrtcGetLoweredName(prog, nm.c_str(), &low));
                lowered[nm] = low;
                nj.set(nm, std::string(low));
            }
            nvrtcDestroyProgram(&prog);
            FILE* f = fopen(cubin_path.c_str(), "wb");
            if (f) { fwrite(cubin.data(), 1, cubin.size(), f); fclose(f); }
            f = fopen(names_path.c_str(), "wb");
            if (f) { std::string s = nj.dump(); fwrite(s.data(), 1, s.size(), f); fclose(f); }
            logi("cuda: compiled %zu kernel instantiations in %.1fs (cached)", names.size(), now_sec() - t0);
        }
        CU(cuModuleLoadData(&module_, cubin.data()));
        auto fn = [&](const std::string& n) {
            CUfunction f;
            auto it = lowered.find(n);
            CU(cuModuleGetFunction(&f, module_, it != lowered.end() ? it->second.c_str() : n.c_str()));
            return f;
        };
        int max_pro_smem = 8 * H + 8 * (H / 32) * 8;
        for (auto& n : names) {
            CUfunction f = fn(n);
            funcs_[n] = f;
            if (n.rfind("gemm<", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gemm_smem()));
            if (n.rfind("gemv_tc", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, TC_SMEM));
            if (n.rfind("gemv2", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_pro_smem));
            if (n.rfind("dn_prep", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DN_PREP_SMEM));
            if (n.rfind("flash_", 0) == 0)
                CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, flash_smem(atoi(n.c_str() + n.find('<') + 1))));
        }
        f_embed_ = fn("embed");
        f_rms_q8_ = fn("rmsnorm_q8");
        f_rms_q8_rows_ = fn("rmsnorm_q8_rows");
        f_rms_f16_ = fn("rmsnorm_f16");
        f_argmax_ = fn("argmax_rows");
        f_kvcopy_ = fn("kv_copy");
        f_colsq_ = fn("colsq_accum");
        f_rowmax_ = fn("row_max");
        f_cand_ = fn("cand_filter");
        f_quant_ = fn("quant_rows");
        f_tof16_ = fn("f32_to_f16");
        f_normadd_ = fn("norm_add");
        f_softcap_ = fn("softcap_rows");
        f_normslice_ = fn("norm_slice");
        f_route_ = fn("moe_route");
        f_actq_ = fn("act_quant");
        f_combine_ = fn("moe_combine");
        f_gather_ = fn("gather_rows_f16");
        f_scatter_ = fn("scatter_add_rows");
        f_embrows_ = fn("embed_rows");
        f_plecomb_ = fn("ple_combine");
        f_plemul_ = fn("ple_mul");
        f_dnpost_ = fn("dn_post");
    }
    static constexpr int TC_SMEM = 3 * (64 * 144 + 64 * 16 + 8 * 272 + 512);
    int gemm_smem() const { return 2 * 64 * 72 * 2 + 2 * 128 * 72 * 2; }

    CUfunction F(const std::string& n) {
        auto it = funcs_.find(n);
        if (it == funcs_.end()) die("kernel %s not compiled", n.c_str());
        return it->second;
    }

    // ---------------------------------------------------------------------------------------------
    // building blocks
    // ---------------------------------------------------------------------------------------------
    CUdeviceptr tok_ptr() const { return d_batch_; }
    CUdeviceptr pos_ptr() const { return d_batch_ + (size_t)cap_ * 4; }
    CUdeviceptr slot_ptr() const { return d_batch_ + (size_t)2 * cap_ * 4; }
    CUdeviceptr lrow_ptr() const { return d_batch_ + (size_t)3 * cap_ * 4; }

    int occupancy(CUfunction f, int smem, int threads = 256) {
        auto key = std::make_pair(f, smem * 2 + (threads == 128));
        auto it = occ_.find(key);
        if (it != occ_.end()) return it->second;
        int nb = 1;
        CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, threads, smem));
        if (nb < 1) nb = 1;
        occ_[key] = nb;
        return nb;
    }
    bool use_tc(const DMat& W, int M, int epi) const {
        if (no_mma_ || M < 2 || !tc_ok(W.t) || W.cols % 256) return false;
        return epi == 2 ? W.rows % 64 == 0 : W.rows % 16 == 0;
    }
    bool no_mma_ = getenv("TACH_NO_MMA") != nullptr;
    // debug: TACH_NO_FLASH_WIDE=1 runs wide heads through the scalar per-token attention (reference path)
    static bool no_flash_wide(int hd) { static const bool off = getenv("TACH_NO_FLASH_WIDE") != nullptr; return off && hd > 128; }

    void gemv_tc(const DMat& W, int epi, int M, CUdeviceptr xq, CUdeviceptr xds, CUdeviceptr y, CUdeviceptr bias,
                 CUdeviceptr oq = 0, CUdeviceptr ods = 0) {
        const int K = W.cols, N = W.rows;
        CUfunction f = F(tc_name(W.t, K, epi));
        const int tiles = (N + 63) / 64;
        const int slots = occupancy(f, TC_SMEM, 128) * n_sm_;
        int nsplit = 1;
        if (epi != 2) {
            const int nch = K / 256;
            while (tiles * nsplit < 2 * slots && nsplit * 2 <= nch) nsplit *= 2;
            if (epi == 0 && nsplit > 1) CU(cuMemsetD32Async(y, 0, (size_t)M * N, stream_));
        }
        const int units = tiles * nsplit;
        launch(f, D3(std::min(units, slots)), D3(128), TC_SMEM, W.d, W.s, N, M, nsplit, xq, xds, y, bias, oq, ods);
    }

    // y (op)= W . act for M <= 16 tokens. pro=1: act = RMSNorm(x rows [optionally gathered]) * nw, computed in the
    // kernel prologue; pro=0: int8 activations xq/xds. epi 0 store, 1 add.
    void gemv(const DMat& W, int pro, int epi, int M, CUdeviceptr x, CUdeviceptr nw, CUdeviceptr rows,
              CUdeviceptr xq, CUdeviceptr xds, CUdeviceptr y, CUdeviceptr bias = 0) {
        const int K = W.cols, N = W.rows;
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = std::min(8, M - m0);
            if (use_tc(W, mm, epi)) {
                CUdeviceptr q = xq ? xq + (size_t)m0 * K : 0, d = xds ? xds + (size_t)m0 * (K / 32) * 8 : 0;
                if (pro == 1) {
                    if (rows) launch(f_rms_q8_rows_, D3(mm), D3(256), 0, x, nw, (CUdeviceptr)(rows + (size_t)m0 * 4), tcq_, tcds_);
                    else launch(f_rms_q8_, D3(mm), D3(256), 0, (CUdeviceptr)(x + (size_t)m0 * K * 4), nw, tcq_, tcds_);
                    q = tcq_; d = tcds_;
                }
                gemv_tc(W, epi, mm, q, d, (CUdeviceptr)(y + (size_t)m0 * N * 4), bias);
                continue;
            }
            const int smem = pro == 1 ? mm * K + mm * (K / 32) * 8 : 0;
            CUfunction f = F(gemv2_name(W.t, K, mm, pro, epi));
            const int wmax = occupancy(f, smem) * 8 * n_sm_;
            const int rpw = (N + wmax - 1) / wmax;
            const int wtot = (N + rpw - 1) / rpw;
            if (getenv("TACH_GEMV_LOG")) {
                static std::set<std::pair<CUfunction, int>> seen;
                if (seen.insert({f, N}).second) {
                    int regs = 0; cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
                    fprintf(stderr, "[gemv] q%d K=%d N=%d M=%d pro=%d epi=%d: regs %d occ %d blk/SM, rows/warp %d, grid %d\n", qt_code(W.t), K, N, mm, pro, epi, regs, occupancy(f, smem), rpw, (wtot + 7) / 8);
                }
            }
            launch(f, D3((wtot + 7) / 8), D3(256), smem, W.d, W.s, N, wtot,
                   (CUdeviceptr)(x ? (rows ? x : x + (size_t)m0 * K * 4) : 0), nw, (CUdeviceptr)(rows ? rows + (size_t)m0 * 4 : 0),
                   (CUdeviceptr)(xq ? xq + (size_t)m0 * K : 0), (CUdeviceptr)(xds ? xds + (size_t)m0 * (K / 32) * 8 : 0),
                   (CUdeviceptr)(y + (size_t)m0 * N * 4), bias);
        }
    }

    // gate/up GEMV with RMS prologue, act(gate)*up, int8 output (decode)
    void gemv_gu(const DMat& W, int M, CUdeviceptr x, CUdeviceptr nw, CUdeviceptr oq, CUdeviceptr ods) {
        const int H = c_.hidden, F2 = W.rows, F_ = F2 / 2;
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = std::min(8, M - m0);
            if (use_tc(W, mm, 2)) {
                launch(f_rms_q8_, D3(mm), D3(256), 0, (CUdeviceptr)(x + (size_t)m0 * H * 4), nw, tcq_, tcds_);
                gemv_tc(W, 2, mm, tcq_, tcds_, 0, 0, (CUdeviceptr)(oq + (size_t)m0 * F_), (CUdeviceptr)(ods + (size_t)m0 * (F_ / 32) * 8));
                continue;
            }
            CUfunction f = F(gu2_name(W.t, H, mm));
            const int smem = mm * H + mm * (H / 32) * 8;
            const int tiles = F2 / 64;
            const int maxc = occupancy(f, smem) * n_sm_;
            const int tpc = (tiles + maxc - 1) / maxc;
            launch(f, D3((tiles + tpc - 1) / tpc), D3(256), smem, W.d, W.s, F2, (CUdeviceptr)(x + (size_t)m0 * H * 4), nw,
                   (CUdeviceptr)(oq + (size_t)m0 * F_), (CUdeviceptr)(ods + (size_t)m0 * (F_ / 32) * 8));
        }
    }

    void gemm(const DMat& W, CUdeviceptr A, int M, CUdeviceptr Cf, CUdeviceptr Ch, int epi, CUdeviceptr bias = 0, int row0 = 0, int nrows = -1) {
        const int N = nrows < 0 ? W.rows : nrows;
        CUdeviceptr d = W.d + qtype_data_bytes(W.t, row0, W.cols), s = W.s + qtype_scale_bytes(W.t, row0, W.cols);
        launch(F(gemm_name(W.t, epi)), D3((N + 127) / 128, (M + 63) / 64), D3(256), gemm_smem(), A, M, W.cols, d, s, N, Cf, Ch, bias);
    }

    // mixer output y [M][H] (fp32) -> residual: x = (x + post_norm(y)) * scalar
    void residual(int M, CUdeviceptr y, CUdeviceptr post_norm, float scalar) {
        launch(f_normadd_, D3(M), D3(256), 0, x_, y, post_norm, scalar);
    }
    // does the mixer / FFN output need the separate residual kernel (sandwich norm or scalar)?
    static bool direct_add(const DLayer& L, bool ffn) {
        return (ffn ? L.post_ffn_norm : L.post_attn_norm) == 0 && (!ffn || L.c.layer_scalar == 1.f);
    }

    RopeArgs rope_args(const DLayer& L, int M, CUdeviceptr qout) {
        const LayerCfg& lc = L.c;
        RopeArgs a{};
        a.qkv = proj_; a.qn = L.q_norm; a.kn = L.k_norm; a.inv_freq = L.inv_freq; a.pos = pos_ptr(); a.slot = slot_ptr();
        a.qout = qout; a.kc = L.kc; a.vc = L.vc; a.ksc = L.ksc; a.vsc = L.vsc;
        a.nrows = lc.qkv_rows(); a.nh = lc.n_heads; a.nkv = lc.n_kv_heads; a.qstride = lc.attn_gate ? 2 * lc.head_dim : lc.head_dim;
        a.qrows = lc.q_dim() * (lc.attn_gate ? 2 : 1); a.kvd = lc.kv_dim(); a.half = lc.rope_half;
        a.qk_norm = lc.qk_norm == 1 ? 1 : 0; a.v_norm = lc.v_norm; a.k_eq_v = lc.k_eq_v; a.own_kv = lc.kv_src < 0; a.ctx = L.kv_cap;
        const float scale = lc.attn_scale > 0 ? lc.attn_scale : 1.f / sqrtf((float)lc.head_dim);
        a.qscale = scale * sqrtf((float)lc.head_dim);  // attention kernels apply 1/sqrt(head_dim)
        (void)M;
        return a;
    }
    void full_qk_norm(const DLayer& L, int M) {
        const LayerCfg& lc = L.c;
        if (lc.qk_norm != 2) return;
        launch(f_normslice_, D3(M), D3(256), 0, proj_, lc.qkv_rows(), 0, lc.q_dim(), L.q_norm);
        launch(f_normslice_, D3(M), D3(256), 0, proj_, lc.qkv_rows(), lc.q_dim(), lc.kv_dim(), L.k_norm);
    }
    AttnArgs attn_args(const DLayer& L, CUdeviceptr q, CUdeviceptr oq, CUdeviceptr ods, CUdeviceptr oh) {
        const LayerCfg& lc = L.c;
        AttnArgs a{};
        a.q = q; a.kc = L.kc; a.vc = L.vc; a.pos = pos_ptr(); a.slot = slot_ptr();
        a.part_o = part_o_; a.part_ml = part_ml_; a.counters = counters_;
        a.oq = oq; a.ods = ods; a.oh = oh;
        a.gate = lc.attn_gate ? proj_ : 0; a.ksc = L.ksc; a.vsc = L.vsc;
        a.nh = lc.n_heads; a.nkv = lc.n_kv_heads; a.ctx = L.kv_cap; a.window = lc.window;
        a.nrows = lc.qkv_rows(); a.qstride = lc.attn_gate ? 2 * lc.head_dim : lc.head_dim;
        return a;
    }

    // Gemma 4 per-layer inputs (after the embedding): ple_buf_ [M][NL*D]
    void ple_inputs(int M, bool prefill) {
        const int H = c_.hidden, D = c_.ple_dim, NL = c_.n_layers, W = NL * D;
        if (prefill) {
            launch(f_tof16_, D3(256), D3(256), 0, x_, ha_, (size_t)M * H);
            if (g_imatrix) imat_acc("ple_proj", ha_, M, H);
            gemm(ple_proj_, ha_, M, ple_tmp_, 0, 0);
        } else {
            launch(f_quant_, D3(M), D3(256), 0, x_, H, xq_, xds_);
            gemv(ple_proj_, 0, 0, M, 0, 0, 0, xq_, xds_, ple_tmp_);
        }
        launch(f_embrows_, D3(M), D3(256), 0, ple_embd_.d, ple_embd_.s, qt_code(ple_embd_.t), tok_ptr(), W, 1.f, ple_tmp2_);
        launch(f_plecomb_, D3(M, NL), D3(D), 0, ple_tmp_, ple_tmp2_, ple_proj_norm_, D, NL, ple_buf_);
    }
    // per layer, after the FFN residual: x = (x + norm(W_out(act(W_gate x) * ple_l))) * layer_scalar
    void ple_layer(const DLayer& L, int l, int M, bool prefill) {
        const int H = c_.hidden, D = c_.ple_dim, W = c_.n_layers * D;
        if (prefill) {
            launch(f_tof16_, D3(256), D3(256), 0, x_, ha_, (size_t)M * H);
            const std::string bp = "blk." + std::to_string(l) + ".";
            if (g_imatrix) imat_acc(bp + "ple_gate", ha_, M, H);
            gemm(L.ple_gate, ha_, M, proj_, 0, 0);
            launch(f_plemul_, D3(M), D3(D), 0, proj_, ple_buf_, l, D, W);
            launch(f_tof16_, D3(64), D3(256), 0, proj_, hb_, (size_t)M * D);
            if (g_imatrix) imat_acc(bp + "ple_out", hb_, M, D);
            gemm(L.ple_out, hb_, M, tmp_, 0, 0);
        } else {
            launch(f_quant_, D3(M), D3(256), 0, x_, H, xq_, xds_);
            gemv(L.ple_gate, 0, 0, M, 0, 0, 0, xq_, xds_, proj_);
            launch(f_plemul_, D3(M), D3(D), 0, proj_, ple_buf_, l, D, W);
            launch(f_quant_, D3(M), D3(256), 0, proj_, D, xq_, xds_);
            gemv(L.ple_out, 0, 0, M, 0, 0, 0, xq_, xds_, tmp_);
        }
        residual(M, tmp_, L.ple_post_norm, L.c.layer_scalar);
    }

    // ---------------------------------------------------------------------------------------------
    // decode pass (M <= 16 tokens, any slots / positions): captured as one CUDA graph per M
    // ---------------------------------------------------------------------------------------------
    void decode_pass(int M) {
        const int H = c_.hidden;
        launch(f_embed_, D3(M), D3(256), 0, embd_.d, embd_.s, tok_ptr(), x_, c_.embed_scale);
        if (c_.ple_dim) ple_inputs(M, false);
        const int nl = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl; l++) {
            const DLayer& L = layers_[l];
            const LayerCfg& lc = L.c;
            const bool dadd = direct_add(L, false);
            // ---- mixer ----
            if (lc.is_attn()) {
                gemv(L.wqkv, 1, 0, M, x_, L.attn_norm, 0, 0, 0, proj_, L.bqkv);
                full_qk_norm(L, M);
                RopeArgs ra = rope_args(L, M, qf_);
                launch(F(rope_name(lc.head_dim, 0)), D3(M, (lc.n_heads + lc.n_kv_heads + 3) / 4), D3(128), 0, ra);
                AttnArgs aa = attn_args(L, qf_, xq_, xds_, 0);
                launch(F(attn_name(lc.head_dim, attn_group(lc), SPLITS, 0)), D3(lc.n_heads / attn_group(lc), SPLITS, M), D3(128), 0, aa);
                gemv(L.wo, 0, dadd ? 1 : 0, M, 0, 0, 0, xq_, xds_, dadd ? x_ : tmp_);
            } else if (lc.type == "conv") {
                gemv(L.conv_in, 1, 0, M, x_, L.attn_norm, 0, 0, 0, proj_);
                launch(F("causal_conv<0>"), D3((H + 127) / 128), D3(128), 0, proj_, 3 * H, H, lc.conv_k, L.conv_w, L.st0,
                       pos_ptr(), slot_ptr(), M, tmp_);
                launch(f_quant_, D3(M), D3(256), 0, tmp_, H, xq_, xds_);
                gemv(L.conv_out, 0, dadd ? 1 : 0, M, 0, 0, 0, xq_, xds_, dadd ? x_ : tmp_);
            } else {
                const int R = lc.dn_in_rows(), CD = lc.dn_conv_dim(), VD = lc.dn_val_dim();
                gemv(L.dn_in, 1, 0, M, x_, L.attn_norm, 0, 0, 0, proj_);
                launch(F("causal_conv<1>"), D3((CD + 127) / 128), D3(128), 0, proj_, R, CD, lc.dn_conv, L.dn_conv, L.st0,
                       pos_ptr(), slot_ptr(), M, (CUdeviceptr)0);
                launch(F(dn_name(lc.dn_dk, lc.dn_dv)), D3(lc.dn_nv), D3(2 * lc.dn_dv), 0, proj_, R, lc.dn_nk, lc.dn_nv, L.dn_A, L.dn_dt,
                       L.dn_norm, L.st1, pos_ptr(), slot_ptr(), M, tmp_);
                launch(f_quant_, D3(M), D3(256), 0, tmp_, VD, xq_, xds_);
                gemv(L.dn_out, 0, dadd ? 1 : 0, M, 0, 0, 0, xq_, xds_, dadd ? x_ : tmp_);
            }
            if (!dadd) residual(M, tmp_, L.post_attn_norm, 1.f);
            // ---- feed forward ----
            const bool fadd = direct_add(L, true) && !c_.ple_dim;
            const float fscalar = c_.ple_dim ? 1.f : lc.layer_scalar;
            if (lc.n_experts) {
                moe_decode(L, M);
            } else if (lc.ffn) {
                gemv_gu(L.wgu, M, x_, L.ffn_norm, fq_, fds_);
                gemv(L.wdown, 0, fadd ? 1 : 0, M, 0, 0, 0, fq_, fds_, fadd ? x_ : tmp_);
                if (!fadd) residual(M, tmp_, L.post_ffn_norm, fscalar);
            }
            if (c_.ple_dim) ple_layer(L, l, M, false);
        }
        logits_pass();
    }

    void moe_decode(const DLayer& L, int M) {
        const LayerCfg& lc = L.c;
        const int H = c_.hidden, F_ = lc.expert_ffn, K = lc.top_k, P = M * K;
        launch(f_route_, D3(M), D3(256), 0, x_, L.ffn_norm, L.router, lc.n_experts, K, lc.router == "sigmoid" ? 1 : 0,
               L.expert_bias, lc.norm_topk ? 1 : 0, lc.routed_scale, moe_ids_, moe_w_);
        launch(f_rms_q8_, D3(M), D3(256), 0, x_, L.ffn_norm, xq_, xds_);
        launch(F(ind_name(L.wgu_e.t, H)), D3((2 * F_ + 31) / 32, P), D3(256), 0, L.wgu_e.d, L.wgu_e.s, 2 * F_, moe_ids_, K, xq_, xds_, proj_);
        launch(f_actq_, D3(P), D3(256), 0, proj_, F_, fq_, fds_);
        launch(F(ind_name(L.wdown_e.t, F_)), D3((H + 31) / 32, P), D3(256), 0, L.wdown_e.d, L.wdown_e.s, H, moe_ids_, 1, fq_, fds_, proj_);
        launch(f_combine_, D3(M), D3(256), 0, proj_, moe_w_, K, tmp_);
        residual(M, tmp_, L.post_ffn_norm, lc.layer_scalar);
    }

    int logit_rows() const { return opt.vocab_limit > 0 && opt.vocab_limit < out_.rows ? opt.vocab_limit / 64 * 64 : out_.rows; }
    void logits_pass() {
        if (n_logits_ == 0) return;
        DMat head = out_;
        head.rows = logit_rows();
        gemv(head, 1, 0, n_logits_, x_, out_norm_, lrow_ptr(), 0, 0, logits_);
        if (c_.final_softcap > 0)
            launch(f_softcap_, D3(std::min<size_t>(1024, ((size_t)n_logits_ * c_.vocab + 255) / 256)), D3(256), 0, logits_,
                   (size_t)n_logits_ * c_.vocab, c_.final_softcap);
    }

    // ---------------------------------------------------------------------------------------------
    // prefill pass (M > 16 tokens of one slot, consecutive positions)
    // ---------------------------------------------------------------------------------------------
    std::map<std::string, std::pair<CUdeviceptr, int>> imat_;
    int64_t imat_rows_ = 0;
    void imat_acc(const std::string& name, CUdeviceptr A, int M, int K) {
        if (!g_imatrix) return;
        auto it = imat_.find(name);
        if (it == imat_.end()) {
            CUdeviceptr p = alloc((size_t)K * 4);
            CU(cuMemsetD8Async(p, 0, (size_t)K * 4, stream_));
            it = imat_.emplace(name, std::make_pair(p, K)).first;
        }
        launch(f_colsq_, D3((K + 255) / 256), D3(256), 0, A, M, K, it->second.first);
    }

    void prefill_pass(int M, int slot, int p0) {
        const int H = c_.hidden;
        launch(f_embed_, D3(M), D3(256), 0, embd_.d, embd_.s, tok_ptr(), x_, c_.embed_scale);
        if (c_.ple_dim) ple_inputs(M, true);
        if (g_imatrix) imat_rows_ += M;
        const int nl = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl; l++) {
            const DLayer& L = layers_[l];
            const LayerCfg& lc = L.c;
            const std::string bp = "blk." + std::to_string(l) + ".";
            const bool dadd = direct_add(L, false);
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, L.attn_norm, ha_);
            if (lc.is_attn()) {
                if (g_imatrix) imat_acc(bp + "wqkv", ha_, M, H);
                gemm(L.wqkv, ha_, M, proj_, 0, 0, L.bqkv);
                full_qk_norm(L, M);
                const int qd = lc.q_dim();
                if (flash_ok(lc.head_dim) && !no_flash_wide(lc.head_dim)) {
                    RopeArgs ra = rope_args(L, M, qh_);
                    launch(F(rope_name(lc.head_dim, 1)), D3(M, (lc.n_heads + lc.n_kv_heads + 3) / 4), D3(128), 0, ra);
                    const int qr = flash_qrows(lc.head_dim);
                    launch(F(flash_name(lc.head_dim)), D3((M + qr - 1) / qr, lc.n_heads), D3(128), flash_smem(lc.head_dim),
                           qh_, L.kc, L.vc, hb_, M, p0, slot, lc.n_heads, lc.n_kv_heads, L.kv_cap, lc.window,
                           (CUdeviceptr)(lc.attn_gate ? proj_ : 0), lc.qkv_rows(), lc.attn_gate ? 2 * lc.head_dim : lc.head_dim, L.ksc, L.vsc);
                } else {
                    // large heads: decode-style attention for every prompt token (one split each), fp16 output
                    RopeArgs ra = rope_args(L, M, qf_);
                    launch(F(rope_name(lc.head_dim, 0)), D3(M, (lc.n_heads + lc.n_kv_heads + 3) / 4), D3(128), 0, ra);
                    AttnArgs aa = attn_args(L, qf_, 0, 0, hb_);
                    launch(F(attn_name(lc.head_dim, attn_group(lc), 1, 1)), D3(lc.n_heads / attn_group(lc), 1, M), D3(128), 0, aa);
                }
                if (g_imatrix) imat_acc(bp + "wo", hb_, M, qd);
                gemm(L.wo, hb_, M, dadd ? x_ : tmp_, 0, dadd ? 1 : 0);
            } else if (lc.type == "conv") {
                if (g_imatrix) imat_acc(bp + "conv_in", ha_, M, H);
                gemm(L.conv_in, ha_, M, proj_, 0, 0);
                if (conv_par_ok(lc)) {
                    const CUdeviceptr st = L.st0 + (size_t)slot * (lc.conv_k - 1) * H * 4;
                    launch(F("conv_par<0>"), D3((H + 127) / 128, (M + 31) / 32), D3(128), 0, proj_, 3 * H, H, lc.conv_k, L.conv_w, st, p0 == 0 ? 1 : 0, M, tmp_);
                    launch(F("conv_state<0>"), D3((H + 255) / 256), D3(256), 0, proj_, 3 * H, H, lc.conv_k, st, M);
                } else {
                    launch(F("causal_conv<0>"), D3((H + 127) / 128), D3(128), 0, proj_, 3 * H, H, lc.conv_k, L.conv_w, L.st0,
                           pos_ptr(), slot_ptr(), M, tmp_);
                }
                launch(f_tof16_, D3(256), D3(256), 0, tmp_, hb_, (size_t)M * H);
                if (g_imatrix) imat_acc(bp + "conv_out", hb_, M, H);
                gemm(L.conv_out, hb_, M, dadd ? x_ : tmp_, 0, dadd ? 1 : 0);
            } else {
                const int R = lc.dn_in_rows(), CD = lc.dn_conv_dim(), VD = lc.dn_val_dim();
                if (g_imatrix) imat_acc(bp + "dn_in", ha_, M, H);
                gemm(L.dn_in, ha_, M, proj_, 0, 0);
                if (dn_chunk_ok(lc)) {
                    const int NV = lc.dn_nv, DK = lc.dn_dk, DV = lc.dn_dv, nch = (M + DN_CHUNK - 1) / DN_CHUNK;
                    const CUdeviceptr cst = L.st0 + (size_t)slot * (lc.dn_conv - 1) * CD * 4;
                    launch(F("conv_par<1>"), D3((CD + 127) / 128, (M + 31) / 32), D3(128), 0, proj_, R, CD, lc.dn_conv, L.dn_conv, cst, p0 == 0 ? 1 : 0, M, dn_cv_);
                    launch(F("conv_state<1>"), D3((CD + 255) / 256), D3(256), 0, proj_, R, CD, lc.dn_conv, cst, M);
                    DnArgs da{proj_, dn_cv_, L.dn_A, L.dn_dt, dn_kt_, dn_gq_, dn_at_, dn_kh_, dn_vt_, dn_gam_, M, R, lc.dn_nk, NV, nch};
                    launch(F("dn_prep<128,128>"), D3(nch, NV), D3(256), DN_PREP_SMEM, da);
                    launch(F("dn_chunk<128,128>"), D3(NV), D3(32 * DV / 16), 0, da, (CUdeviceptr)(L.st1 + (size_t)slot * NV * DK * DV * 4), p0 == 0 ? 1 : 0, tmp_);
                    launch(f_dnpost_, D3(M, (NV + 7) / 8), D3(256), 0, tmp_, proj_, R, CD, NV, DV, L.dn_norm, hb_);
                } else {
                    launch(F("causal_conv<1>"), D3((CD + 127) / 128), D3(128), 0, proj_, R, CD, lc.dn_conv, L.dn_conv, L.st0,
                           pos_ptr(), slot_ptr(), M, (CUdeviceptr)0);
                    launch(F(dn_name(lc.dn_dk, lc.dn_dv)), D3(lc.dn_nv), D3(2 * lc.dn_dv), 0, proj_, R, lc.dn_nk, lc.dn_nv, L.dn_A, L.dn_dt,
                           L.dn_norm, L.st1, pos_ptr(), slot_ptr(), M, tmp_);
                    launch(f_tof16_, D3(256), D3(256), 0, tmp_, hb_, (size_t)M * VD);
                }
                if (g_imatrix) imat_acc(bp + "dn_out", hb_, M, VD);
                gemm(L.dn_out, hb_, M, dadd ? x_ : tmp_, 0, dadd ? 1 : 0);
            }
            if (!dadd) residual(M, tmp_, L.post_attn_norm, 1.f);
            // ---- feed forward ----
            const bool fadd = direct_add(L, true) && !c_.ple_dim;
            const float fscalar = c_.ple_dim ? 1.f : lc.layer_scalar;
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, L.ffn_norm, ha_);
            if (lc.n_experts) {
                moe_prefill(L, M, bp);
            } else if (lc.ffn) {
                if (g_imatrix) imat_acc(bp + "wgu", ha_, M, H);
                gemm(L.wgu, ha_, M, 0, hb_, 2);
                if (g_imatrix) imat_acc(bp + "wdown", hb_, M, lc.ffn);
                gemm(L.wdown, hb_, M, fadd ? x_ : tmp_, 0, fadd ? 1 : 0);
                if (!fadd) residual(M, tmp_, L.post_ffn_norm, fscalar);
            }
            if (c_.ple_dim) ple_layer(L, l, M, true);
        }
        if (g_imatrix) {
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, out_norm_, ha_);
            imat_acc(c_.tied_embed ? "tok_embd" : "output", ha_, M, H);
        }
        logits_pass();
    }

    // MoE prefill: route on the GPU, group (token, expert) pairs by expert on the host, run one GEMM per expert.
    void moe_prefill(const DLayer& L, int M, const std::string& bp) {
        const LayerCfg& lc = L.c;
        const int H = c_.hidden, F_ = lc.expert_ffn, K = lc.top_k, E = lc.n_experts;
        launch(f_route_, D3(M), D3(256), 0, x_, L.ffn_norm, L.router, E, K, lc.router == "sigmoid" ? 1 : 0,
               L.expert_bias, lc.norm_topk ? 1 : 0, lc.routed_scale, moe_ids_, moe_w_);
        int* hid = (int*)h_moe_;
        float* hw = (float*)(h_moe_ + (size_t)M * K * 4);
        CU(cuMemcpyDtoHAsync(hid, moe_ids_, (size_t)M * K * 4, stream_));
        CU(cuMemcpyDtoHAsync(hw, moe_w_, (size_t)M * K * 4, stream_));
        CU(cuStreamSynchronize(stream_));
        std::vector<std::vector<std::pair<int, float>>> per(E);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) per[hid[m * K + k]].push_back({m, hw[m * K + k]});
        int* hrows = (int*)(h_moe_ + (size_t)M * K * 8);
        float* hrw = (float*)(h_moe_ + (size_t)M * K * 12);
        std::vector<std::pair<int, int>> spans(E);
        int off = 0;
        for (int e = 0; e < E; e++) {
            spans[e] = {off, (int)per[e].size()};
            for (auto& t : per[e]) { hrows[off] = t.first; hrw[off] = t.second; off++; }
        }
        CU(cuMemcpyHtoDAsync(moe_rows_, hrows, (size_t)off * 4, stream_));
        CU(cuMemcpyHtoDAsync(moe_rw_, hrw, (size_t)off * 4, stream_));
        // gathered normalized activations [pairs][H] (fp16), expert outputs accumulate into tmp_ (zeroed)
        launch(f_gather_, D3(off), D3(256), 0, ha_, moe_rows_, H, hb_);
        CU(cuMemsetD32Async(tmp_, 0, (size_t)M * H, stream_));
        const size_t gu_bytes = (size_t)off * F_ * 2;
        CUdeviceptr act = ha_ + 0;  // reuse ha_ after gather: [pairs][F] fp16 (ha_ is sized for pairs x max K)
        (void)gu_bytes;
        if (g_imatrix) imat_acc(bp + "wgu_e", hb_, off, H);   // all routed (token, expert) rows; experts share columns
        for (int e = 0; e < E; e++) {
            const int n = spans[e].second;
            if (!n) continue;
            const int o = spans[e].first;
            gemm(L.wgu_e, hb_ + (size_t)o * H * 2, n, 0, act + (size_t)o * F_ * 2, 2, 0, e * 2 * F_, 2 * F_);
        }
        for (int e = 0; e < E; e++) {
            const int n = spans[e].second;
            if (!n) continue;
            const int o = spans[e].first;
            gemm(L.wdown_e, act + (size_t)o * F_ * 2, n, proj_ + (size_t)o * H * 4, 0, 0, 0, e * H, H);
        }
        if (g_imatrix) imat_acc(bp + "wdown_e", act, off, F_);
        launch(f_scatter_, D3(off), D3(256), 0, proj_, moe_rows_, moe_rw_, H, tmp_);
        residual(M, tmp_, L.post_ffn_norm, lc.layer_scalar);
    }

    ModelConfig c_;
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    CUstream stream_ = nullptr;
    CUmodule module_ = nullptr;
    int cc_major_ = 0, cc_minor_ = 0, n_sm_ = 0;
    std::vector<CUdeviceptr> allocs_;
    size_t weight_bytes_ = 0;
    DMat embd_, out_;
    CUdeviceptr out_norm_ = 0;
    std::vector<DLayer> layers_;
    int cap_ = 0, maxHD_ = 0, maxNH_ = 0, maxNKV_ = 0;
    bool has_moe_ = false;
    CUdeviceptr x_ = 0, proj_ = 0, tmp_ = 0, qf_ = 0, qh_ = 0, xq_ = 0, xds_ = 0, fq_ = 0, fds_ = 0, tcq_ = 0, tcds_ = 0, ha_ = 0, hb_ = 0;
    CUdeviceptr part_o_ = 0, part_ml_ = 0, counters_ = 0, logits_ = 0, d_argmax_ = 0, d_batch_ = 0;
    CUdeviceptr moe_ids_ = 0, moe_w_ = 0, moe_rows_ = 0, moe_rw_ = 0;
    static constexpr int CAND_CAP = 1024;
    CUdeviceptr d_rmax_ = 0, d_ccount_ = 0, d_cand_ = 0;
    std::map<std::pair<CUfunction, int>, int> occ_;
    int* h_batch_ = nullptr;
    uint8_t* h_moe_ = nullptr;
    static constexpr int NSTAGE = 4;
    CUevent stage_ev_[NSTAGE] = {};
    int stage_ = 0;
    float* h_out_ = nullptr;
    std::map<std::string, CUfunction> funcs_;
    CUfunction f_embed_, f_rms_q8_, f_rms_q8_rows_, f_rms_f16_, f_argmax_, f_kvcopy_, f_colsq_, f_rowmax_, f_cand_;
    CUfunction f_embrows_, f_plecomb_, f_plemul_, f_dnpost_;
    CUdeviceptr dn_cv_ = 0, dn_kt_ = 0, dn_gq_ = 0, dn_kh_ = 0, dn_at_ = 0, dn_vt_ = 0, dn_gam_ = 0;
    DMat ple_embd_, ple_proj_;
    CUdeviceptr ple_proj_norm_ = 0, ple_buf_ = 0, ple_tmp_ = 0, ple_tmp2_ = 0;
    CUfunction f_quant_, f_tof16_, f_normadd_, f_softcap_, f_normslice_, f_route_, f_actq_, f_combine_, f_gather_, f_scatter_;
    std::map<int, CUgraphExec> graphs_;
    std::vector<int> lrows_;
    int n_logits_ = 0;
};

}  // namespace

Backend* create_cuda_backend() {
    auto* b = new CudaBackend();
    if (!b->probe()) { delete b; return nullptr; }
    return b;
}

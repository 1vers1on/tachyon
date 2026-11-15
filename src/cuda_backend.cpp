// Tachyon - CUDA backend (driver API + NVRTC runtime compilation, CUDA graphs for decode)
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
};
struct DLayer {
    CUdeviceptr attn_norm = 0, ffn_norm = 0, q_norm = 0, k_norm = 0, bqkv = 0;
    DMat wqkv, wo, wgu, wdown;
};

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
        if (c_.head_dim != 64 && c_.head_dim != 128) die("head_dim %d not supported on CUDA", c_.head_dim);
        if (c_.n_heads % c_.n_kv_heads) die("bad GQA config");
        size_t free_b, total_b;
        cuMemGetInfo(&free_b, &total_b);
        logi("cuda: %s (sm_%d%d, %d SMs, %.1f/%.1f GB free)", name, cc_major_, cc_minor_, n_sm_, free_b / 1e9, total_b / 1e9);
        upload_weights(mf);
        // fit KV cache into remaining memory
        cuMemGetInfo(&free_b, &total_b);
        const size_t kv_elt = opt.kv_int8 ? 1 : 2;
        size_t per_tok = (size_t)c_.n_layers * c_.n_kv_heads * (c_.head_dim * kv_elt + (opt.kv_int8 ? 4 : 0)) * 2;
        size_t act_bytes = 64ull << 20;
        size_t reserve = (512ull << 20) + opt.reserve_bytes;
        size_t avail = free_b > reserve + act_bytes ? free_b - reserve - act_bytes : 0;
        const size_t budget_tok = per_tok + opt.extra_kv_per_tok;
        size_t want = budget_tok * opt.ctx * opt.n_slots;
        if (want > avail) {
            int ctx2 = (int)(avail / (budget_tok * opt.n_slots)) / 256 * 256;
            if (ctx2 < 512) { opt.n_slots = 1; ctx2 = (int)(avail / budget_tok) / 256 * 256; }
            if (ctx2 < 256) die("not enough GPU memory for KV cache");
            logi("cuda: reducing context to %d tokens/slot to fit KV cache in VRAM", ctx2);
            opt.ctx = ctx2;
        }
        size_t kv_elems = (size_t)opt.n_slots * c_.n_layers * c_.n_kv_heads * opt.ctx * c_.head_dim;
        kc_ = alloc(kv_elems * kv_elt);
        vc_ = alloc(kv_elems * kv_elt);
        CU(cuMemsetD8Async(kc_, 0, kv_elems * kv_elt, stream_));
        CU(cuMemsetD8Async(vc_, 0, kv_elems * kv_elt, stream_));
        if (opt.kv_int8) {
            size_t ns = kv_elems / c_.head_dim;
            ksc_ = alloc(ns * 4);
            vsc_ = alloc(ns * 4);
            CU(cuMemsetD8Async(ksc_, 0, ns * 4, stream_));
            CU(cuMemsetD8Async(vsc_, 0, ns * 4, stream_));
        }
        logi("cuda: KV cache %.2f GB (%d slots x %d ctx, %s)", per_tok * opt.ctx * opt.n_slots / 1e9, opt.n_slots, opt.ctx, opt.kv_int8 ? "int8" : "fp16");
        alloc_activations();
        compile_kernels();
        // rope frequencies
        int half = c_.head_dim / 2;
        std::vector<float> inv(half);
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
            inv[i] = f;
        }
        inv_freq_ = alloc(half * 4);
        CU(cuMemcpyHtoD(inv_freq_, inv.data(), half * 4));
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
        // logits rows
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
        argmax_valid_ = false;
        logits_on_host_ = -1;
        if (prof_ && ++prof_count_ % 50 == 0) prof_report();
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
        launch(f_kvcopy_, D3(c_.n_layers * c_.n_kv_heads), D3(256), 0, kc_, vc_, src, dst, n, ksc_, vsc_);
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
        names[f_embed_] = "embed"; names[f_rms_q8_] = "rmsnorm_q8"; names[f_rms_q8_rows_] = "rmsnorm_q8_rows"; names[f_rms_f16_] = "rmsnorm_f16";
        names[f_rope_dec_] = "rope_decode"; names[f_rope_pre_] = "rope_prefill"; names[f_attn_dec2_] = "attn_decode2";
        names[f_flash_] = "flash_prefill"; names[f_argmax_] = "argmax";
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : agg) v.push_back({kv.second.first, names[kv.first] + " x" + std::to_string(kv.second.second)});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[profile] total kernel time %.3f ms\n", tot);
        for (auto& x : v) fprintf(stderr, "  %8.3f ms %5.1f%%  %s\n", x.first, 100 * x.first / tot, x.second.c_str());
    }
    bool prof_ = getenv("TACH_PROFILE") != nullptr;
    int skip_ = 0;
public:
    // Ablation timing: replay the decode graph with one kernel class removed; the delta is that class's cost.
    void kbench(int M, int pos) {
        std::vector<BatchToken> b(M);
        for (int i = 0; i < M; i++) b[i] = {100 + i, 0, pos + i, true};
        auto time_graph = [&](int skip) {
            skip_ = skip;
            forward(b);  // upload batch
            CUgraph g; CUgraphExec ge;
            CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_RELAXED));
            decode_pass(M);
            CU(cuStreamEndCapture(stream_, &g));
            CU(cuGraphInstantiate(&ge, g, 0));
            for (int i = 0; i < 5; i++) cuGraphLaunch(ge, stream_);
            CUevent e0, e1; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
            cuEventRecord(e0, stream_);
            const int R = 100;
            for (int i = 0; i < R; i++) cuGraphLaunch(ge, stream_);
            cuEventRecord(e1, stream_);
            cuEventSynchronize(e1);
            float ms; cuEventElapsedTime(&ms, e0, e1);
            cuGraphExecDestroy(ge); cuGraphDestroy(g);
            skip_ = 0;
            return ms / R;
        };
        {
            CUfunction fb = fn_raw("bw_read");
            size_t bytes = std::min<size_t>(weight_bytes_, 1ull << 30) & ~(size_t)15;
            CUdeviceptr src = layers_[0].wqkv.d;  // weights region is one contiguous allocation
            for (int g : {n_sm_ * 4, n_sm_ * 8, n_sm_ * 16}) {
                for (int i = 0; i < 2; i++) launch(fb, D3(g), D3(256), 0, src, bytes / 16, d_argmax_);
                CUevent e0, e1; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
                cuEventRecord(e0, stream_);
                for (int i = 0; i < 10; i++) launch(fb, D3(g), D3(256), 0, src, bytes / 16, d_argmax_);
                cuEventRecord(e1, stream_); cuEventSynchronize(e1);
                float ms; cuEventElapsedTime(&ms, e0, e1);
                fprintf(stderr, "[kbench] streaming read %zu MB, grid %d: %.1f GB/s\n", bytes >> 20, g, bytes * 10 / (ms * 1e6));
            }
        }
        float full = time_graph(0);
        fprintf(stderr, "[kbench] M=%d pos=%d full decode step: %.3f ms (%.1f tok/s)\n", M, pos, full, M * 1000.f / full);
        const char* names[] = {"(unused)", "gemv qkv+norm", "rope", "attention", "(unused)", "gemv o", "gemv gate/up", "gemv down", "lm head"};
        float sum = 0;
        for (int i = 0; i < 9; i++) {
            float t = time_graph(1 << i);
            fprintf(stderr, "  %-14s %7.3f ms  (%.1f us/layer)\n", names[i], full - t, (full - t) * 1000 / c_.n_layers);
            sum += full - t;
        }
        fprintf(stderr, "  %-14s %7.3f ms\n", "other/embed", full - sum);
        // weight bandwidth estimate
        fprintf(stderr, "  weights %.0f MB -> effective %.0f GB/s\n", weight_bytes_ / 1e6, weight_bytes_ / 1e6 / full);
    }
private:
    int prof_count_ = 0;
    std::vector<std::pair<CUfunction, std::pair<CUevent, CUevent>>> prof_events_;

    void upload_weights(const ModelFile& mf) {
        size_t total = 0;
        for (auto& kv : mf.tensors) total += align_up(kv.second.data_bytes, 256) + align_up(kv.second.scale_bytes, 256);
        CUdeviceptr base = alloc(total);
        // stage through pinned memory in chunks
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
            if ((t.type == QType::F16) && t.rows > 1) die("f16 weight matrices are not supported on CUDA; convert with -q q4 or q8");
            CUdeviceptr d = up(t.data, t.data_bytes);
            CUdeviceptr s = t.scale_bytes ? up(t.scales, t.scale_bytes) : 0;
            ptr[t.name] = {d, s};
        }
        CU(cuStreamSynchronize(stream_));
        cuMemFreeHost(pinned);
        weight_bytes_ = total;
        auto mat = [&](const std::string& n) {
            const TensorInfo* t = mf.tensor(n);
            DMat m; m.t = t->type; m.rows = (int)t->rows; m.cols = (int)t->cols;
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
        layers_.resize(c_.n_layers);
        for (int l = 0; l < c_.n_layers; l++) {
            std::string p = "blk." + std::to_string(l) + ".";
            DLayer& L = layers_[l];
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
        logi("cuda: uploaded %.2f GB of weights", total / 1e9);
    }

    void alloc_activations() {
        const int H = c_.hidden, qd = c_.q_dim(), kvd = c_.kv_dim(), F = c_.ffn;
        cap_ = std::max(opt.max_batch, MAX_LOGITS);
        const int maxK = std::max({H, qd, F});
        x_ = alloc((size_t)cap_ * H * 4);
        qkv_ = alloc((size_t)cap_ * (qd + 2 * kvd) * 4);
        qf_ = alloc((size_t)MAX_DEC * qd * 4);
        qh_ = alloc((size_t)cap_ * qd * 2);
        xq_ = alloc((size_t)std::max(MAX_DEC, MAX_LOGITS) * maxK);
        xds_ = alloc((size_t)std::max(MAX_DEC, MAX_LOGITS) * (maxK / 32) * 8);
        fq_ = alloc((size_t)MAX_DEC * maxK);
        fds_ = alloc((size_t)MAX_DEC * (maxK / 32) * 8);
        ha_ = alloc((size_t)cap_ * maxK * 2);
        hb_ = alloc((size_t)cap_ * maxK * 2);
        part_o_ = alloc((size_t)MAX_DEC * c_.n_heads * SPLITS * c_.head_dim * 4);
        part_ml_ = alloc((size_t)MAX_DEC * c_.n_heads * SPLITS * 8);
        logits_ = alloc((size_t)MAX_LOGITS * c_.vocab * 4);
        counters_ = alloc((size_t)MAX_DEC * c_.n_kv_heads * 4);
        tcq_ = alloc((size_t)8 * maxK);
        tcds_ = alloc((size_t)8 * (maxK / 32) * 8);
        d_rmax_ = alloc(MAX_LOGITS * 4);
        d_ccount_ = alloc(MAX_LOGITS * 4);
        d_cand_ = alloc((size_t)MAX_DEC * CAND_CAP * 8);
        CU(cuMemsetD8(counters_, 0, (size_t)MAX_DEC * c_.n_kv_heads * 4));
        d_argmax_ = alloc(MAX_LOGITS * 4);
        d_batch_ = alloc(4 * cap_ * sizeof(int));
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
    std::string gemv2_name(QType t, int K, int M, int pro, int epi) {
        return "gemv2<" + std::to_string(qt_code(t)) + "," + std::to_string(K) + "," + std::to_string(M) + "," + std::to_string(pro) + "," + std::to_string(epi) + ">";
    }
    std::string gu2_name(QType t, int K, int M) {
        return "gemv2_gu<" + std::to_string(qt_code(t)) + "," + std::to_string(K) + "," + std::to_string(M) + ">";
    }
    bool use_mma(const DMat& W, int M, int epi) const {
        if (no_mma_ || M < 2 || (W.t != QType::Q4 && W.t != QType::Q3) || W.cols % 256) return false;
        return epi == 2 ? W.rows % 64 == 0 : W.rows % 16 == 0;
    }
    std::string tc_name(QType t, int K, int epi) { return "gemv_tc<" + std::to_string(qt_code(t)) + "," + std::to_string(K) + "," + std::to_string(epi) + ">"; }
    static constexpr int TC_SMEM = 3 * (64 * 144 + 64 * 16 + 8 * 272 + 512);
    // tensor-core GEMV for 2..8 tokens; x (if given) is RMS-normalized + quantized first into xq_/xds_
    void gemv_tc(const DMat& W, int epi, int M, CUdeviceptr xq, CUdeviceptr xds, CUdeviceptr y, CUdeviceptr bias,
                 CUdeviceptr oq = 0, CUdeviceptr ods = 0) {
        const int K = W.cols, N = W.rows;
        CUfunction f = F(tc_name(W.t, K, epi));
        const int tiles = (N + 63) / 64;
        const int slots = occupancy128(f, TC_SMEM) * n_sm_;
        int nsplit = 1;
        if (epi != 2) {
            // split K until there are enough work units to fill every CTA slot ~2x
            const int nch = K / 256;
            while (tiles * nsplit < 2 * slots && nsplit * 2 <= nch) nsplit *= 2;
            if (nsplit > 1 || epi == 1) {} 
            if (epi == 0 && nsplit > 1) CU(cuMemsetD32Async(y, 0, (size_t)M * N, stream_));
        }
        const int units = tiles * nsplit;
        launch(f, D3(std::min(units, slots)), D3(128), TC_SMEM, W.d, W.s, N, M, nsplit, xq, xds, y, bias, oq, ods);
    }
    int occupancy128(CUfunction f, int smem) {
        auto key = std::make_pair(f, smem + 1);
        auto it = occ_.find(key);
        if (it != occ_.end()) return it->second;
        int nb = 1;
        CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 128, smem));
        if (nb < 1) nb = 1;
        occ_[key] = nb;
        return nb;
    }
    bool no_mma_ = getenv("TACH_NO_MMA") != nullptr;
    std::string gemm_name(QType t, int epi) { return "gemm<" + std::to_string(qt_code(t)) + "," + std::to_string(epi) + ">"; }

    void compile_kernels() {
        const int H = c_.hidden, qd = c_.q_dim();
        std::vector<std::string> opts = {
            "--gpu-architecture=sm_" + std::to_string(cc_major_) + std::to_string(cc_minor_),
            "-std=c++17", "-default-device", "--extra-device-vectorization",
            getenv("TACH_PTXAS_V") ? "--ptxas-options=-v" : "-DTACH_NOP",
            "-I/usr/local/cuda/include",
            "-DHD=" + std::to_string(c_.head_dim), "-DNH=" + std::to_string(c_.n_heads), "-DNKV=" + std::to_string(c_.n_kv_heads),
            "-DHID=" + std::to_string(H), "-DQD=" + std::to_string(qd), "-DKVD=" + std::to_string(c_.kv_dim()),
            "-DFF=" + std::to_string(c_.ffn), "-DNL=" + std::to_string(c_.n_layers), "-DCTX=" + std::to_string(opt.ctx),
            "-DNSLOTS=" + std::to_string(opt.n_slots), "-DEPS=" + std::to_string(c_.rms_eps) + "f",
            "-DSPLITS=" + std::to_string(SPLITS), "-DEQ=" + std::to_string(qt_code(embd_.t)),
        };
        if (c_.qk_norm) opts.push_back("-DQK_NORM=1");
        if (opt.kv_int8) opts.push_back("-DKV8=1");
        if (getenv("TACH_GEMM_F16ACC")) opts.push_back("-DGEMM_F16ACC=1");
        // template instantiations needed by this model
        std::set<std::string> names;
        for (auto& L : layers_) {
            for (int M = 1; M <= 8; M++) {
                names.insert(gemv2_name(L.wqkv.t, L.wqkv.cols, M, 1, 0));
                names.insert(gemv2_name(L.wo.t, L.wo.cols, M, 0, 1));
                names.insert(gemv2_name(L.wdown.t, L.wdown.cols, M, 0, 1));
                names.insert(gu2_name(L.wgu.t, L.wgu.cols, M));
            }
            names.insert(gemm_name(L.wqkv.t, 0));
            names.insert(gemm_name(L.wo.t, 1));
            names.insert(gemm_name(L.wdown.t, 1));
            names.insert(gemm_name(L.wgu.t, 2));
        }
        for (int M = 1; M <= 8; M++) names.insert(gemv2_name(out_.t, out_.cols, M, 1, 0));
        auto tc_ok = [](QType t) { return t == QType::Q4 || t == QType::Q3; };
        for (auto& L : layers_) {
            if (tc_ok(L.wqkv.t)) names.insert(tc_name(L.wqkv.t, L.wqkv.cols, 0));
            if (tc_ok(L.wo.t)) names.insert(tc_name(L.wo.t, L.wo.cols, 1));
            if (tc_ok(L.wdown.t)) names.insert(tc_name(L.wdown.t, L.wdown.cols, 1));
            if (tc_ok(L.wgu.t)) names.insert(tc_name(L.wgu.t, L.wgu.cols, 2));
        }
        if (tc_ok(out_.t)) names.insert(tc_name(out_.t, out_.cols, 0));
        std::string key = std::string(kKernelSource);
        for (auto& o : opts) key += "|" + o;
        for (auto& n : names) key += "|" + n;
        int nv_major = 0, nv_minor = 0;
        nvrtcVersion(&nv_major, &nv_minor);
        key += "|nvrtc" + std::to_string(nv_major) + "." + std::to_string(nv_minor);
        char hash[32];
        snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a64(key.data(), key.size()));
        std::string dir = std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache/tachyon";
        mkdir((std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache").c_str(), 0755);
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
        for (auto& n : names) {
            CUfunction f = fn(n);
            funcs_[n] = f;
            if (n.rfind("gemm<", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gemm_smem()));
            if (n.rfind("gemv_tc", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, TC_SMEM));
            if (n.rfind("gemv2", 0) == 0) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8 * c_.hidden + 8 * (c_.hidden / 32) * 8));
        }
        f_embed_ = fn("embed");
        f_rms_q8_ = fn("rmsnorm_q8");
        f_rms_q8_rows_ = fn("rmsnorm_q8_rows");
        f_rms_f16_ = fn("rmsnorm_f16");
        f_rope_dec_ = fn("rope_decode");
        f_rope_pre_ = fn("rope_prefill");
        f_flash_ = fn("flash_prefill");
        f_argmax_ = fn("argmax_rows");
        f_kvcopy_ = fn("kv_copy");
        f_colsq_ = fn("colsq_accum");
        f_attn_dec2_ = fn("attn_decode2");
        f_rowmax_ = fn("row_max");
        f_cand_ = fn("cand_filter");
        CU(cuFuncSetAttribute(f_flash_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, flash_smem()));
    }
    CUfunction fn_raw(const char* n) { CUfunction f; CU(cuModuleGetFunction(&f, module_, n)); return f; }
    int gemm_smem() const { return 2 * 64 * 72 * 2 + 2 * 128 * 72 * 2; }
    int flash_smem() const { return 2 * 64 * (c_.head_dim + 8) * 2; }

    CUfunction F(const std::string& n) {
        auto it = funcs_.find(n);
        if (it == funcs_.end()) die("kernel %s not compiled", n.c_str());
        return it->second;
    }

    // ---------------------------------------------------------------------------------------------
    // forward passes
    // ---------------------------------------------------------------------------------------------
    CUdeviceptr tok_ptr() const { return d_batch_; }
    CUdeviceptr pos_ptr() const { return d_batch_ + (size_t)cap_ * 4; }
    CUdeviceptr slot_ptr() const { return d_batch_ + (size_t)2 * cap_ * 4; }
    CUdeviceptr lrow_ptr() const { return d_batch_ + (size_t)3 * cap_ * 4; }

    int occupancy(CUfunction f, int smem) {
        auto key = std::make_pair(f, smem);
        auto it = occ_.find(key);
        if (it != occ_.end()) return it->second;
        int nb = 1;
        CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256, smem));
        if (nb < 1) nb = 1;
        occ_[key] = nb;
        return nb;
    }
    // Balanced single-wave GEMV: every warp gets the same number of rows.
    void gemv2(const DMat& W, int pro, int epi, int M, CUdeviceptr x, CUdeviceptr nw, CUdeviceptr rows,
               CUdeviceptr xq, CUdeviceptr xds, CUdeviceptr y, CUdeviceptr bias = 0) {
        const int K = W.cols, N = W.rows;
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = std::min(8, M - m0);
            if (use_mma(W, mm, epi)) {
                CUdeviceptr q = xq ? xq + (size_t)m0 * K : 0, d = xds ? xds + (size_t)m0 * (K / 32) * 8 : 0;
                if (pro == 1) {  // normalize + quantize the (gathered) rows into scratch first
                    if (rows) launch(f_rms_q8_rows_, D3(mm), D3(256), 0, x, nw, (CUdeviceptr)(rows + (size_t)m0 * 4), tcq_, tcds_);
                    else launch(f_rms_q8_, D3(mm), D3(256), 0, (CUdeviceptr)(x + (size_t)m0 * K * 4), nw, tcq_, tcds_);
                    q = tcq_; d = tcds_;
                }
                gemv_tc(W, epi, mm, q, d, (CUdeviceptr)(y + (size_t)m0 * N * 4), bias);
                continue;
            }
            CUfunction f = F(gemv2_name(W.t, K, mm, pro, epi));
            const int smem = pro == 1 ? mm * K + mm * (K / 32) * 8 : 0;
            const int wmax = occupancy(f, smem) * 8 * n_sm_;
            const int rpw = (N + wmax - 1) / wmax;
            const int wtot = (N + rpw - 1) / rpw;
            launch(f, D3((wtot + 7) / 8), D3(256), smem, W.d, W.s, N, wtot,
                   (CUdeviceptr)(x ? x + (size_t)m0 * K * 4 : 0), nw, (CUdeviceptr)(rows ? rows + (size_t)m0 * 4 : 0),
                   (CUdeviceptr)(xq ? xq + (size_t)m0 * K : 0), (CUdeviceptr)(xds ? xds + (size_t)m0 * (K / 32) * 8 : 0),
                   (CUdeviceptr)(y + (size_t)m0 * N * 4), bias);
        }
    }

    void decode_pass(int M) {
        const int H = c_.hidden, F_ = c_.ffn;
        launch(f_embed_, D3(M), D3(256), 0, embd_.d, embd_.s, tok_ptr(), x_);
        const int nl = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl; l++) {
            const DLayer& L = layers_[l];
            if (!(skip_ & 2)) gemv2(L.wqkv, 1, 0, M, x_, L.attn_norm, 0, 0, 0, qkv_, L.bqkv);
            if (!(skip_ & 4)) launch(f_rope_dec_, D3(M, (c_.n_heads + c_.n_kv_heads + 3) / 4), D3(128), 0, qkv_, L.q_norm, L.k_norm, inv_freq_,
                                     pos_ptr(), slot_ptr(), l, qf_, kc_, vc_, ksc_, vsc_);
            if (!(skip_ & 8)) launch(f_attn_dec2_, D3(c_.n_kv_heads, SPLITS, M), D3(128), 0, qf_, kc_, vc_, pos_ptr(), slot_ptr(), l,
                                     part_o_, part_ml_, counters_, xq_, xds_, ksc_, vsc_);
            if (!(skip_ & 32)) gemv2(L.wo, 0, 1, M, 0, 0, 0, xq_, xds_, x_);
            if (!(skip_ & 64)) {
                for (int m0 = 0; m0 < M; m0 += 8) {
                    const int mm = std::min(8, M - m0);
                    if (use_mma(L.wgu, mm, 2)) {
                        launch(f_rms_q8_, D3(mm), D3(256), 0, (CUdeviceptr)(x_ + (size_t)m0 * H * 4), L.ffn_norm, tcq_, tcds_);
                        gemv_tc(L.wgu, 2, mm, tcq_, tcds_, 0, 0, (CUdeviceptr)(fq_ + (size_t)m0 * F_), (CUdeviceptr)(fds_ + (size_t)m0 * (F_ / 32) * 8));
                        continue;
                    }
                    CUfunction f = F(gu2_name(L.wgu.t, H, mm));
                    const int smem = mm * H + mm * (H / 32) * 8;
                    const int tiles = 2 * F_ / 64;
                    const int maxc = occupancy(f, smem) * n_sm_;
                    const int tpc = (tiles + maxc - 1) / maxc;
                    launch(f, D3((tiles + tpc - 1) / tpc), D3(256), smem, L.wgu.d, L.wgu.s, 2 * F_,
                           (CUdeviceptr)(x_ + (size_t)m0 * H * 4), L.ffn_norm,
                           (CUdeviceptr)(fq_ + (size_t)m0 * F_), (CUdeviceptr)(fds_ + (size_t)m0 * (F_ / 32) * 8));
                }
            }
            if (!(skip_ & 128)) gemv2(L.wdown, 0, 1, M, 0, 0, 0, fq_, fds_, x_);
        }
        if (!(skip_ & 256)) logits_pass();
    }

    void logits_pass() {
        if (n_logits_ == 0) return;
        DMat head = out_;
        if (opt.vocab_limit > 0 && opt.vocab_limit < head.rows) head.rows = opt.vocab_limit / 64 * 64;
        gemv2(head, 1, 0, n_logits_, x_, out_norm_, lrow_ptr(), 0, 0, logits_);
    }
    int logit_rows() const { return opt.vocab_limit > 0 && opt.vocab_limit < out_.rows ? opt.vocab_limit / 64 * 64 : out_.rows; }

    void gemm(const DMat& W, CUdeviceptr A, int M, CUdeviceptr Cf, CUdeviceptr Ch, int epi, CUdeviceptr bias = 0) {
        launch(F(gemm_name(W.t, epi)), D3((W.rows + 127) / 128, (M + 63) / 64), D3(256), gemm_smem(), A, M, W.cols, W.d, W.s, W.rows, Cf, Ch, bias);
    }

    // importance-matrix statistics (GPU): per-layer column sums of x^2 for each matmul input
    std::map<std::string, std::pair<CUdeviceptr, int>> imat_;  // name -> (acc buffer, K)
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
public:
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
    void prefill_pass(int M, int slot, int p0) {
        launch(f_embed_, D3(M), D3(256), 0, embd_.d, embd_.s, tok_ptr(), x_);
        if (g_imatrix) imat_rows_ += M;
        const int nl_dbg = getenv("TACH_DEBUG_NL") ? atoi(getenv("TACH_DEBUG_NL")) : c_.n_layers;
        for (int l = 0; l < nl_dbg; l++) {
            const DLayer& L = layers_[l];
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, L.attn_norm, ha_);
            const std::string bp = "blk." + std::to_string(l) + ".";
            if (g_imatrix) imat_acc(bp + "wqkv", ha_, M, c_.hidden);
            gemm(L.wqkv, ha_, M, qkv_, 0, 0, L.bqkv);
            launch(f_rope_pre_, D3(M, (c_.n_heads + c_.n_kv_heads + 3) / 4), D3(128), 0, qkv_, L.q_norm, L.k_norm, inv_freq_,
                   pos_ptr(), slot_ptr(), l, qh_, kc_, vc_, ksc_, vsc_);
            launch(f_flash_, D3((M + 63) / 64, c_.n_heads), D3(128), flash_smem(), qh_, kc_, vc_, hb_, M, p0, slot, l, ksc_, vsc_);
            if (g_imatrix) imat_acc(bp + "wo", hb_, M, c_.q_dim());
            gemm(L.wo, hb_, M, x_, 0, 1);
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, L.ffn_norm, ha_);
            if (g_imatrix) imat_acc(bp + "wgu", ha_, M, c_.hidden);
            gemm(L.wgu, ha_, M, 0, hb_, 2);
            if (g_imatrix) imat_acc(bp + "wdown", hb_, M, c_.ffn);
            gemm(L.wdown, hb_, M, x_, 0, 1);
        }
        if (g_imatrix) {
            launch(f_rms_f16_, D3(M), D3(256), 0, x_, out_norm_, ha_);
            imat_acc(c_.tied_embed ? "tok_embd" : "output", ha_, M, c_.hidden);
        }
        logits_pass();
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
    CUdeviceptr kc_ = 0, vc_ = 0, ksc_ = 0, vsc_ = 0, inv_freq_ = 0;
    int cap_ = 0;
    CUdeviceptr counters_ = 0, tcq_ = 0, tcds_ = 0;
    std::map<std::pair<CUfunction, int>, int> occ_;
    CUdeviceptr x_, qkv_, qf_, qh_, xq_, xds_, fq_, fds_, ha_, hb_, part_o_, part_ml_, logits_, d_argmax_, d_batch_;
    int* h_batch_ = nullptr;
    static constexpr int NSTAGE = 4;
    CUevent stage_ev_[NSTAGE] = {};
    int stage_ = 0;
    float* h_out_ = nullptr;
    std::map<std::string, CUfunction> funcs_;
    CUfunction f_embed_, f_rms_q8_, f_rms_q8_rows_, f_rms_f16_, f_rope_dec_, f_rope_pre_, f_flash_, f_argmax_, f_kvcopy_, f_colsq_, f_attn_dec2_, f_rowmax_, f_cand_;
    static constexpr int CAND_CAP = 1024;
    CUdeviceptr d_rmax_ = 0, d_ccount_ = 0, d_cand_ = 0;
    std::map<int, CUgraphExec> graphs_;
    std::vector<int> lrows_;
    int n_logits_ = 0;
    bool argmax_valid_ = false;
    int logits_on_host_ = -1;
};

}  // namespace

Backend* create_cuda_backend() {
    auto* b = new CudaBackend();
    if (!b->probe()) { delete b; return nullptr; }
    return b;
}

// Tachyon - command line entry point
#include "common.h"
#include "engine.h"
#include "server.h"
#include "quant.h"
#include "safetensors.h"
#include <thread>
#include <algorithm>

int g_log_level = 1;
int convert_main(int argc, char** argv);

static void usage() {
    fprintf(stderr,
        "Tachyon %s - fast LLM inference server (CUDA + CPU)\n\n"
        "usage:\n"
        "  tachyon convert <hf_model_dir> <out.tach> [-q q4|q8|f16] [--embed-quant q4|q8|f16]\n"
        "                  [--calib text.txt]  (importance-weighted quantization from calibration text)\n"
        "                  [--imatrix file] [--tensor-quant wdown=q8,blk.0.=q8]\n"
        "  tachyon serve   -m model.tach [--host 127.0.0.1] [--port 8080] [options]\n"
        "  tachyon run     -m model.tach -p \"prompt\" [--chat] [-n 256] [options]\n"
        "  tachyon bench   -m model.tach [-n 256] [--prompt-len 512] [options]\n"
        "  tachyon tokenize -m model.tach -p \"text\"\n\n"
        "options:\n"
        "  --device auto|cuda|cpu   backend (default auto)\n"
        "  -c, --ctx N              context length per slot (default 8192)\n"
        "  --slots N                concurrent sequences / KV slots (default 4 on GPU, 1 on CPU)\n"
        "  -t, --threads N          CPU threads (default: physical cores)\n"
        "  --spec N                 n-gram speculative draft tokens (default 4 on GPU, 0 on CPU; 0 = off)\n"
        "  --draft model.tach       small draft model for speculative decoding (same tokenizer), --draft-k N\n"
        "  --kv8                    8-bit KV cache (CUDA, experimental)\n"
        "  --no-graphs              disable CUDA graphs\n"
        "  --temp T --top-k K --top-p P --min-p P --seed S   sampling (run/bench)\n"
        "  -v / -q                  verbose / quiet logging\n", TACH_VERSION);
}

struct Args {
    std::string model, prompt, host = "127.0.0.1", device = "auto", out;
    int port = 8080, n = 256, ctx = 8192, slots = -1, threads = 0, spec = -1, prompt_len = 512;
    bool chat = false, kv8 = false, graphs = true, no_think = false, think = false;
    std::string draft;
    int draft_k = 4, draft_vocab = 0;
    SamplingParams sp;
    bool temp_set = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    a.sp.temperature = 0;
    for (int i = 0; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for %s", s.c_str()); return argv[++i]; };
        if (s == "-m" || s == "--model") a.model = next();
        else if (s == "-p" || s == "--prompt") a.prompt = next();
        else if (s == "-f" || s == "--file") a.prompt = read_file(next());
        else if (s == "--host") a.host = next();
        else if (s == "-o" || s == "--output") a.out = next();
        else if (s == "--port") a.port = atoi(next().c_str());
        else if (s == "-n" || s == "--n-predict") a.n = atoi(next().c_str());
        else if (s == "-c" || s == "--ctx") a.ctx = atoi(next().c_str());
        else if (s == "--slots") a.slots = atoi(next().c_str());
        else if (s == "-t" || s == "--threads") a.threads = atoi(next().c_str());
        else if (s == "--spec") a.spec = atoi(next().c_str());
        else if (s == "--draft") a.draft = next();
        else if (s == "--draft-k") a.draft_k = atoi(next().c_str());
        else if (s == "--draft-vocab") a.draft_vocab = atoi(next().c_str());
        else if (s == "--prompt-len") a.prompt_len = atoi(next().c_str());
        else if (s == "--device") a.device = next();
        else if (s == "--chat") a.chat = true;
        else if (s == "--no-think") a.no_think = true;
        else if (s == "--think") a.think = true;
        else if (s == "--kv8") a.kv8 = true;
        else if (s == "--no-graphs") a.graphs = false;
        else if (s == "--temp") { a.sp.temperature = atof(next().c_str()); a.temp_set = true; }
        else if (s == "--top-k") a.sp.top_k = atoi(next().c_str());
        else if (s == "--top-p") a.sp.top_p = atof(next().c_str());
        else if (s == "--min-p") a.sp.min_p = atof(next().c_str());
        else if (s == "--seed") a.sp.seed = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--repeat-penalty") a.sp.repetition_penalty = atof(next().c_str());
        else if (s == "-v") g_log_level = 2;
        else if (s == "-q") g_log_level = 0;
        else if (s == "-h" || s == "--help") { usage(); exit(0); }
        else die("unknown argument %s (see tachyon --help)", s.c_str());
    }
    if (a.model.empty()) die("missing -m model.tach");
    return a;
}

static void load_engine(Engine& e, const Args& a) {
    EngineOptions o;
    o.device = a.device;
    o.backend.ctx = a.ctx;
    o.backend.threads = a.threads;
    o.backend.kv_int8 = a.kv8;
    o.backend.use_graphs = a.graphs;
    o.spec_draft = a.spec;
    o.draft_model = a.draft;
    o.draft_k = a.draft_k;
    o.draft_vocab = a.draft_vocab;
    o.backend.n_slots = a.slots > 0 ? a.slots : (a.device == "cpu" ? 1 : 4);
    e.load(a.model, o);
}

static void print_stats(const GenRequest& r) {
    double pp = r.t_prefill_done - r.t_submit;
    double gen = r.t_done - r.t_first;
    fprintf(stderr, "\n[stats] prompt %d tok (%d cached) in %.3fs = %.1f tok/s | gen %d tok in %.3fs = %.1f tok/s | ttft %.3fs",
            r.n_prompt, r.n_cached, pp, (r.n_prompt - r.n_cached) / std::max(pp, 1e-9), r.n_gen, gen,
            (r.n_gen - 1) / std::max(gen, 1e-9), r.t_first - r.t_submit);
    if (r.n_draft) fprintf(stderr, " | spec accepted %d/%d (%.0f%%)", r.n_draft_accepted, r.n_draft, 100.0 * r.n_draft_accepted / r.n_draft);
    fprintf(stderr, " | finish=%s\n", r.finish_reason.c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    if (cmd == "convert") {
        // --calib <text>: build a temporary Q8 model, collect an importance matrix on it, then convert with it.
        std::vector<char*> rest;
        std::string calib, device = "auto";
        int calib_windows = 40;
        for (int i = 2; i < argc; i++) {
            std::string s = argv[i];
            if (s == "--calib" && i + 1 < argc) calib = argv[++i];
            else if (s == "--calib-windows" && i + 1 < argc) calib_windows = atoi(argv[++i]);
            else if (s == "--device" && i + 1 < argc) device = argv[++i];
            else rest.push_back(argv[i]);
        }
        if (calib.empty()) return convert_main((int)rest.size(), rest.data());
        if (rest.size() < 2) die("usage: tachyon convert <hf_dir> <out.tach> --calib <text file> [...]");
        std::string out = rest[1];
        std::string tmp = out + ".q8tmp", imat = out + ".imatrix";
        {
            // calibration model: Q8 when it fits comfortably in VRAM, else plain Q4 (statistics barely change)
            size_t ckpt = 0;
            {
                SafeTensors st; st.open_dir(rest[0]);
                for (auto& n : st.names()) ckpt += st.get(n)->nbytes;
            }
            const char* cq = ckpt > (12ull << 30) ? "q4" : "q8";
            logi("calibration model: %s (checkpoint %.1f GB)", cq, ckpt / 1e9);
            std::vector<char*> a8 = {rest[0], (char*)tmp.c_str(), (char*)"-q", (char*)cq};
            convert_main((int)a8.size(), a8.data());
        }
        {
            static IMatrix im;
            g_imatrix = &im;
            Engine e;
            EngineOptions o;
            o.device = device;
            o.backend.ctx = 1024; o.backend.n_slots = 1;
            e.load(tmp, o);
            auto ids = e.tok.encode(read_file(calib), false);
            int win = 512, nwin = 0;
            for (size_t st = 0; st + win <= ids.size() && nwin < calib_windows; st += win, nwin++)
                for (int c = 0; c < win; c += 128) {
                    std::vector<BatchToken> b(128);
                    for (int i = 0; i < 128; i++) b[i] = {ids[st + c + i], 0, c + i, i == 127};
                    e.be->forward(b);
                }
            e.be->imat_flush();
            im.save(imat);
            g_imatrix = nullptr;
            logi("importance matrix from %d x %d calibration tokens -> %s", nwin, win, imat.c_str());
        }
        remove(tmp.c_str());
        std::vector<char*> a4 = rest;
        a4.push_back((char*)"--imatrix");
        a4.push_back((char*)imat.c_str());
        return convert_main((int)a4.size(), a4.data());
    }
    if (cmd == "selftest") { void cpu_selftest(); cpu_selftest(); return 0; }
    Args a = parse_args(argc - 2, argv + 2);
    if (cmd == "tokenize") {
        ModelFile mf; mf.open(a.model);
        Tokenizer t; t.load(mf);
        auto ids = t.encode(a.prompt);
        for (int id : ids) printf("%d ", id);
        printf("\n");
        return 0;
    }
    if (cmd == "compare") {
        // debug: run the same batch on CPU and CUDA and compare logits per row
        ModelFile mf; mf.open(a.model);
        Tokenizer t; t.load(mf);
        auto ids = t.encode(a.prompt);
        BackendOptions bo; bo.n_slots = a.slots > 0 ? a.slots : 1; bo.ctx = a.ctx; bo.max_batch = 512;
        bool cpu2 = getenv("TACH_COMPARE_CPU2") != nullptr;  // second backend = CPU with float activations
        std::unique_ptr<Backend> cpu(create_cpu_backend()), gpu(cpu2 ? create_cpu_backend() : create_cuda_backend());
        cpu->init(mf, bo);
        if (cpu2) setenv("TACH_DEBUG_DEQUANT", "1", 1);
        gpu->init(mf, bo);
        int chunk = a.n;  // tokens per forward
        std::vector<float> l1(cpu->vocab()), l2(cpu->vocab());
        for (size_t st = 0; st < ids.size(); st += chunk) {
            int n = std::min<int>(chunk, ids.size() - st);
            std::vector<BatchToken> b(n);
            for (int i = 0; i < n; i++) b[i] = {ids[st + i], 0, (int)(st + i), i < 64};
            cpu->forward(b); gpu->forward(b);
            for (int i = 0; i < std::min(n, 64); i++) {
                cpu->get_logits(i, l1.data()); gpu->get_logits(i, l2.data());
                double d = 0, nrm = 0; int a1 = 0, a2 = 0;
                for (int k = 0; k < cpu->vocab(); k++) { d += (l1[k]-l2[k])*(l1[k]-l2[k]); nrm += l1[k]*l1[k]; if (l1[k] > l1[a1]) a1 = k; if (l2[k] > l2[a2]) a2 = k; }
                size_t P = st + i;
                static double nll1 = 0, nll2 = 0; static int nn = 0;
                if (P + 1 < ids.size()) {
                    auto lse = [&](std::vector<float>& l) { float mx = *std::max_element(l.begin(), l.end()); double se = 0; for (float v : l) se += exp(v - mx); return mx + log(se); };
                    nll1 += lse(l1) - l1[ids[P + 1]]; nll2 += lse(l2) - l2[ids[P + 1]]; nn++;
                    if ((P + 1) % 256 == 0) printf("ppl so far: A %.4f  B %.4f\n", exp(nll1 / nn), exp(nll2 / nn));
                }
                static double sd = 0; static int agree = 0, cntp = 0;
                sd += sqrt(d / nrm); agree += a1 == a2; cntp++;
                if ((P + 1) % 64 == 0) { printf("pos %4zu..%4zu mean rel_diff %.4f argmax agree %d/%d\n", P - 63, P, sd / cntp, agree, cntp); sd = 0; agree = cntp = 0; }
            }
        }
        return 0;
    }
    Engine e;
    load_engine(e, a);
    if (cmd == "run") {
        auto r = std::make_shared<GenRequest>();
        std::string text = a.prompt;
        if (a.chat) text = e.tok.render_chat({{"user", a.prompt, ""}}, true, a.no_think ? false : (a.think || e.tok.think_default()));
        r->prompt = e.tok.encode(text);
        r->sp = a.sp;
        r->max_tokens = a.n;
        r->on_text = [](const std::string& d) { fwrite(d.data(), 1, d.size(), stdout); fflush(stdout); return true; };
        e.run_sync(r);
        printf("\n");
        print_stats(*r);
        return 0;
    }
    if (cmd == "bench") {
        // prefill benchmark with synthetic prompt, then decode benchmark
        std::vector<int> prompt;
        auto base = e.tok.encode("The quick brown fox jumps over the lazy dog. Tachyon benchmarks prompt processing speed. ");
        while ((int)prompt.size() < a.prompt_len) prompt.insert(prompt.end(), base.begin(), base.end());
        prompt.resize(a.prompt_len);
        for (int rep = 0; rep < 2; rep++) {
            auto r = std::make_shared<GenRequest>();
            if (rep == 1) prompt[0] ^= 1;  // defeat prefix cache for second run
            r->prompt = prompt;
            r->sp = a.sp;
            r->max_tokens = a.n;
            r->ignore_eos = true;
            e.run_sync(r);
            if (rep == 1 || g_log_level >= 2) print_stats(*r);
        }
        return 0;
    }
    if (cmd == "imatrix") {
        // collect per-channel activation statistics on calibration text (CPU backend)
        static IMatrix im;
        g_imatrix = &im;  // must be set before the first forward
        auto ids = e.tok.encode(a.prompt, false);
        int win = 512, nwin = 0;
        double t0 = now_sec();
        for (size_t st = 0; st + win <= ids.size() && nwin < a.n; st += win, nwin++) {
            for (int c = 0; c < win; c += 128) {
                std::vector<BatchToken> b(128);
                for (int i = 0; i < 128; i++) b[i] = {ids[st + c + i], 0, c + i, i == 127};
                e.be->forward(b);
            }
            fprintf(stderr, "\r[tachyon] imatrix window %d", nwin + 1);
        }
        e.be->imat_flush();
        std::string out = a.out.empty() ? "imatrix.bin" : a.out;
        im.save(out);
        fprintf(stderr, "\n[tachyon] wrote %s (%d windows of %d tokens, %.1fs)\n", out.c_str(), nwin, win, now_sec() - t0);
        return 0;
    }
    if (cmd == "ppl") {
        // perplexity over a text file: independent windows of `ctx_win` tokens, scored in chunks of 64
        auto ids = e.tok.encode(a.prompt, false);
        // models that expect <bos> (Gemma) get it at position 0 of every window (replacing that token)
        const int bos = e.tok.add_bos() ? e.tok.bos() : -1;
        if (bos >= 0 && !ids.empty() && ids[0] == bos) ids.erase(ids.begin());
        int win = std::min(a.prompt_len > 0 ? a.prompt_len : 512, e.opt.backend.ctx);
        int V = e.be->vocab();
        std::vector<float> lg(V);
        double nll = 0;
        int64_t cnt = 0;
        double t0 = now_sec();
        int nwin = 0;
        for (size_t st = 0; st + win <= ids.size() && nwin < a.n; st += win, nwin++) {
            const int pchunk = getenv("TACH_PPL_CHUNK") ? atoi(getenv("TACH_PPL_CHUNK")) : 64;
            // chunks above 64 tokens (big prefill batches): score a rotating subset of at most 64 rows per forward
            const int stride = (pchunk + 63) / 64;
            for (int c = 0; c < win; c += pchunk) {
                int n = std::min(pchunk, win - c);
                std::vector<BatchToken> b(n);
                std::vector<int> rows;
                for (int i = 0; i < n; i++) {
                    const bool want = i % stride == (nwin + c / pchunk) % stride;
                    b[i] = {c + i == 0 && bos >= 0 ? bos : ids[st + c + i], 0, c + i, want};
                    if (want) rows.push_back(i);
                }
                e.be->forward(b);
                for (size_t ri = 0; ri < rows.size(); ri++) {
                    int p = c + rows[ri];
                    if (p + 1 >= win || p < win / 4) continue;  // score positions with some context
                    e.be->get_logits((int)ri, lg.data());
                    float mx = *std::max_element(lg.begin(), lg.end());
                    double se = 0;
                    for (float v : lg) se += exp(v - mx);
                    nll += -(lg[ids[st + p + 1]] - mx - log(se));
                    cnt++;
                }
            }
        }
        printf("perplexity: %.4f over %lld tokens (%d windows of %d) in %.1fs\n", exp(nll / cnt), (long long)cnt, nwin, win, now_sec() - t0);
        return 0;
    }
    if (cmd == "serve") {
        e.start();
        return server_main(e, a.host, a.port);
    }
    usage();
    return 1;
}

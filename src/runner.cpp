// Tachyon - model runner: named models with automatic download + conversion, and a model-manager server that
// starts one `tachyon serve` process per model on demand and routes API requests to it (ollama-style).
//
//   tachyon pull qwen3:8b            download from Hugging Face, convert (Q4 + importance matrix), register
//   tachyon run qwen3:8b ["prompt"]  chat (starts the background server and pulls the model when needed)
//   tachyon serve                    model manager on :28657 (OpenAI + Anthropic APIs, any local model by name)
//   tachyon list | ps | rm | stop | show | import | catalog
#include "common.h"
#include "json.h"
#include "http.h"
#include "model_file.h"
#include "ollama.h"
#include "https.h"
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <signal.h>
#include <glob.h>
#include <ftw.h>
#include <map>
#include <set>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <future>
#include <deque>

int convert_calibrated(const std::string& hf_dir, const std::string& out, const std::vector<std::string>& extra,
                       const std::string& calib_text, const std::string& device, int calib_windows);
int convert_main(int argc, char** argv);

static const unsigned char kCalibBytes[] = {
#embed "../data/calib.txt"
    , 0};
static const char* const kCalibText = (const char*)kCalibBytes;

namespace {

// ------------------------------------------------------------------------------------------------ catalog

struct CatalogEntry {
    const char* name;    // family:tag
    const char* repo;    // Hugging Face repo
    const char* quant;   // default quantization
    const char* about;
    bool is_default;     // the family's default tag (`qwen3` = `qwen3:8b`)
};

const CatalogEntry kCatalog[] = {
    {"qwen3:0.6b", "Qwen/Qwen3-0.6B", "q4", "Qwen3 0.6B, hybrid thinking, tools", false},
    {"qwen3:1.7b", "Qwen/Qwen3-1.7B", "q4", "Qwen3 1.7B, hybrid thinking, tools", false},
    {"qwen3:4b", "Qwen/Qwen3-4B-Instruct-2507", "q4", "Qwen3 4B Instruct 2507, tools", false},
    {"qwen3:4b-thinking", "Qwen/Qwen3-4B-Thinking-2507", "q4", "Qwen3 4B Thinking 2507, tools", false},
    {"qwen3:8b", "Qwen/Qwen3-8B", "q4", "Qwen3 8B, hybrid thinking, tools", true},
    {"qwen3:14b", "Qwen/Qwen3-14B", "q4", "Qwen3 14B, hybrid thinking, tools", false},
    {"qwen3:32b", "Qwen/Qwen3-32B", "q4", "Qwen3 32B, hybrid thinking, tools (large: ~20 GB VRAM)", false},
    {"qwen3:30b-a3b", "Qwen/Qwen3-30B-A3B-Instruct-2507", "q4", "Qwen3 30B-A3B MoE Instruct 2507 (experts partly in RAM), tools", false},
    {"qwen3:30b-a3b-thinking", "Qwen/Qwen3-30B-A3B-Thinking-2507", "q4", "Qwen3 30B-A3B MoE Thinking 2507 (experts partly in RAM), tools", false},
    {"qwen3.5:0.8b", "Qwen/Qwen3.5-0.8B", "q8", "Qwen3.5 0.8B (Gated DeltaNet hybrid), tools", false},
    {"qwen3.5:2b", "Qwen/Qwen3.5-2B", "q4", "Qwen3.5 2B, tools", false},
    {"qwen3.5:4b", "Qwen/Qwen3.5-4B", "q4", "Qwen3.5 4B, tools", false},
    {"qwen3.5:9b", "Qwen/Qwen3.5-9B", "q4", "Qwen3.5 9B, tools", true},
    {"qwen2.5:0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "q8", "Qwen2.5 0.5B Instruct, tools", false},
    {"qwen2.5:1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "q4", "Qwen2.5 1.5B Instruct, tools", false},
    {"qwen2.5:3b", "Qwen/Qwen2.5-3B-Instruct", "q4", "Qwen2.5 3B Instruct, tools", false},
    {"qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct", "q4", "Qwen2.5 7B Instruct, tools", true},
    {"qwen2.5:14b", "Qwen/Qwen2.5-14B-Instruct", "q4", "Qwen2.5 14B Instruct, tools", false},
    {"qwen2.5:32b", "Qwen/Qwen2.5-32B-Instruct", "q4", "Qwen2.5 32B Instruct, tools (large: ~20 GB VRAM)", false},
    {"qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct", "q4", "Qwen2.5 Coder 7B Instruct, tools", true},
    {"qwen2.5-coder:14b", "Qwen/Qwen2.5-Coder-14B-Instruct", "q4", "Qwen2.5 Coder 14B Instruct, tools", false},
    {"qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct", "q4", "Qwen2.5 Coder 32B Instruct, tools (large: ~20 GB VRAM)", false},
    {"gemma3:270m", "google/gemma-3-270m-it", "q8", "Gemma 3 270M instruct", true},
    {"gemma:2b", "google/gemma-2b-it", "q4", "Gemma 2B instruct (gated)", true},
    {"gemma:7b", "google/gemma-7b-it", "q4", "Gemma 7B instruct (gated)", false},
    {"gemma4:e2b", "google/gemma-4-E2B-it", "q4", "Gemma 4 E2B instruct, thinking, tools", false},
    {"gemma4:e4b", "google/gemma-4-E4B-it", "q4", "Gemma 4 E4B instruct, thinking, tools", true},
    {"gemma4:12b", "google/gemma-4-12B-it", "q4", "Gemma 4 12B instruct, thinking, tools", false},
    {"lfm2.5:350m", "LiquidAI/LFM2.5-350M", "q4", "LFM2.5 350M (short-conv hybrid), tools", false},
    {"lfm2.5:1.2b", "LiquidAI/LFM2.5-1.2B-Instruct", "q4", "LFM2.5 1.2B Instruct, tools", true},
    {"lfm2.5:1.2b-thinking", "LiquidAI/LFM2.5-1.2B-Thinking", "q4", "LFM2.5 1.2B Thinking, tools", false},
    {"lfm2.5:2.6b", "LiquidAI/LFM2.5-2.6B", "q4", "LFM2.5 2.6B, tools", false},
    {"lfm2.5:8b-a1b", "LiquidAI/LFM2.5-8B-A1B", "q4", "LFM2.5 8B-A1B MoE, tools", false},
    {"olmoe:1b-7b", "allenai/OLMoE-1B-7B-0125-Instruct", "q4", "OLMoE 1B-7B MoE instruct", true},
    {"mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3", "q4", "Mistral 7B Instruct v0.3, tools", true},
    {"mixtral:8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1", "q4", "Mixtral 8x7B MoE instruct (large; needs system RAM)", true},
    {"mixtral:8x22b", "mistralai/Mixtral-8x22B-Instruct-v0.1", "q4", "Mixtral 8x22B MoE instruct (large; needs system RAM)", false},
    {"deepseek-r1:1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "q8", "DeepSeek-R1 distill (Qwen 1.5B), reasoning", false},
    {"deepseek-r1:7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "q4", "DeepSeek-R1 distill (Qwen 7B), reasoning", true},
    {"smollm3:3b", "HuggingFaceTB/SmolLM3-3B", "q4", "SmolLM3 3B, thinking", true},
    {"granite3.3:2b", "ibm-granite/granite-3.3-2b-instruct", "q4", "Granite 3.3 2B instruct", true},
    {"granite3.3:8b", "ibm-granite/granite-3.3-8b-instruct", "q4", "Granite 3.3 8B instruct", false},
    {"phi4-mini", "microsoft/Phi-4-mini-instruct", "q4", "Phi-4-mini instruct (3.8B)", true},
};

std::string lower(std::string s) { for (auto& c : s) c = tolower((unsigned char)c); return s; }
bool ends_with(const std::string& s, const std::string& x) { return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0; }

// a model reference resolved to what to download
struct ModelRef {
    std::string name;    // canonical local name (qwen3:8b, qwen3:8b-q8, hf.co/org/repo)
    std::string repo;    // Hugging Face repo id
    std::string quant;   // q4 | q8 | q3
    bool known = false;  // catalog entry or explicit Hugging Face repo
};

ModelRef parse_ref(std::string s) {
    ModelRef r;
    s = lower(s);
    std::string quant;
    for (const char* q : {"-q4", "-q8", "-q3"})   // quantization suffix: qwen3:8b-q8
        if (ends_with(s, q)) { quant = q + 1; s = s.substr(0, s.size() - 3); break; }
    if (s.rfind("hf.co/", 0) == 0 || s.rfind("huggingface.co/", 0) == 0 || (s.find('/') != std::string::npos && s.find(':') == std::string::npos)) {
        std::string repo = s.substr(s.rfind("hf.co/", 0) == 0 ? 6 : s.rfind("huggingface.co/", 0) == 0 ? 15 : 0);
        r.repo = repo;   // case is restored from the Hub listing at download time
        r.quant = quant.empty() ? "q4" : quant;
        r.name = "hf.co/" + repo + (quant.empty() ? "" : "-" + quant);
        r.known = true;
        return r;
    }
    std::string base = s;
    if (base.find(':') == std::string::npos) {
        for (auto& c : kCatalog) if (c.is_default && std::string(c.name).rfind(base + ":", 0) == 0) { base = c.name; break; }
    }
    if (ends_with(base, ":latest")) base = base.substr(0, base.size() - 7);
    for (auto& c : kCatalog)
        if (base == c.name) {
            r.repo = c.repo;
            r.quant = quant.empty() ? c.quant : quant;
            r.name = base + (quant.empty() || quant == c.quant ? "" : "-" + quant);
            r.known = true;
            return r;
        }
    r.name = s + (quant.empty() ? "" : "-" + quant);
    r.quant = quant.empty() ? "q4" : quant;
    return r;
}

// ------------------------------------------------------------------------------------------------ local store

std::string home_dir() {
    const char* h = getenv("TACHYON_HOME");
    if (h && *h) return h;
    const char* u = getenv("HOME");
    return std::string(u ? u : ".") + "/.tachyon";
}
std::string models_dir() {
    const char* m = getenv("TACHYON_MODELS");
    return m && *m ? std::string(m) : home_dir() + "/models";
}
void mkdirs(const std::string& p) {
    for (size_t i = 1; i <= p.size(); i++)
        if (i == p.size() || p[i] == '/') mkdir(p.substr(0, i).c_str(), 0755);
}
std::string slug(const std::string& name) {
    std::string s;
    for (char c : name) s += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '-';
    return s;
}
bool is_file(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode); }
bool is_dir(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }
size_t file_size(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 ? st.st_size : 0; }
std::string human(double b) {
    char s[32];
    if (b >= 1e9) snprintf(s, sizeof s, "%.1f GB", b / 1e9);
    else if (b >= 1e6) snprintf(s, sizeof s, "%.0f MB", b / 1e6);
    else snprintf(s, sizeof s, "%.0f KB", b / 1e3);
    return s;
}
std::string ago(double sec) {
    char s[32];
    if (sec < 120) snprintf(s, sizeof s, "%.0f seconds", sec);
    else if (sec < 7200) snprintf(s, sizeof s, "%.0f minutes", sec / 60);
    else if (sec < 172800) snprintf(s, sizeof s, "%.0f hours", sec / 3600);
    else snprintf(s, sizeof s, "%.0f days", sec / 86400);
    return s;
}
std::string iso_now(time_t t = time(nullptr)) {
    char b[64];
    struct tm tmv;
    gmtime_r(&t, &tmv);
    strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return b;
}

struct LocalModel {
    std::string name, path, manifest, source, quant, created;
    size_t size = 0;
    Json args;   // extra `tachyon serve` arguments for this model
};

std::vector<LocalModel> list_local() {
    std::vector<LocalModel> out;
    DIR* d = opendir(models_dir().c_str());
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        std::string f = e->d_name;
        if (!ends_with(f, ".json")) continue;
        std::string mp = models_dir() + "/" + f;
        std::string err;
        Json j = Json::parse(read_file(mp), &err);
        if (!err.empty() || !j.is_obj()) continue;
        LocalModel m;
        m.manifest = mp;
        m.name = j.get_str("name", "");
        m.path = j.get_str("path", "");
        if (!m.path.empty() && m.path[0] != '/') m.path = models_dir() + "/" + m.path;
        m.source = j.get_str("source", "");
        m.quant = j.get_str("quant", "");
        m.created = j.get_str("created", "");
        m.args = j["args"];
        m.size = file_size(m.path);
        if (!m.name.empty() && is_file(m.path)) out.push_back(m);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.name < b.name; });
    return out;
}

bool find_local(const std::string& ref, LocalModel& out) {
    if (ends_with(ref, ".tach") && is_file(ref)) {
        out = LocalModel();
        out.name = ref;
        out.path = ref;
        out.size = file_size(ref);
        return true;
    }
    auto all = list_local();
    std::string want = parse_ref(ref).name;
    for (auto& m : all) if (m.name == want || m.name == lower(ref)) { out = m; return true; }
    return false;
}

void write_manifest(const std::string& name, const std::string& path, const std::string& source, const std::string& quant) {
    mkdirs(models_dir());
    Json j = Json::object();
    j.set("name", name);
    std::string rel = path;
    if (rel.rfind(models_dir() + "/", 0) == 0) rel = rel.substr(models_dir().size() + 1);
    j.set("path", rel);
    j.set("source", source);
    j.set("quant", quant);
    j.set("created", iso_now());
    std::string mp = models_dir() + "/" + slug(name) + ".json";
    FILE* f = fopen((mp + ".tmp").c_str(), "wb");
    if (!f) die("cannot write %s", mp.c_str());
    std::string s = j.dump(2) + "\n";
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
    rename((mp + ".tmp").c_str(), mp.c_str());
}

// ------------------------------------------------------------------------------------------------ processes

std::string self_exe() {
    char b[4096];
    ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n <= 0) return "tachyon";
    b[n] = 0;
    // the binary was replaced (rebuilt / upgraded) while this process runs: use the new file at the same path
    std::string p = b;
    if (ends_with(p, " (deleted)") && is_file(p.substr(0, p.size() - 10))) p.resize(p.size() - 10);
    return p;
}

// run a program (no shell) with stdout and stderr redirected to the file `log`; returns its exit status
int run_prog(const std::vector<std::string>& argv, const std::string& log) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) _exit(127);
        dup2(fd, 1); dup2(fd, 2); close(fd);
        std::vector<char*> av;
        for (auto& a : argv) av.push_back((char*)a.c_str());
        av.push_back(nullptr);
        execv(av[0], av.data());
        _exit(127);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

// rm -rf
void remove_tree(const std::string& dir) {
    nftw(dir.c_str(), [](const char* p, const struct stat*, int, struct FTW*) { return remove(p); }, 16, FTW_DEPTH | FTW_PHYS);
}

// ------------------------------------------------------------------------------------------------ Hugging Face

std::string hf_home() {
    const char* h = getenv("HF_HOME");
    if (h && *h) return h;
    const char* u = getenv("HOME");
    return std::string(u ? u : ".") + "/.cache/huggingface";
}

std::string hf_token() {
    for (const char* v : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}) { const char* t = getenv(v); if (t && *t) return t; }
    std::string p = hf_home() + "/token";
    if (is_file(p)) {
        std::string t = read_file(p);
        while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
        return t;
    }
    return "";
}

HttpGetOptions hf_opts() {
    HttpGetOptions o;
    std::string t = hf_token();
    if (!t.empty()) o.auth = "Bearer " + t;   // sent to huggingface.co only, not to the CDN it redirects to
    return o;
}

bool want_file(const std::string& f) {
    if (f.find('/') != std::string::npos) return false;
    static const char* names[] = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                                  "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"};
    for (auto n : names) if (f == n) return true;
    return ends_with(f, ".safetensors");
}

// existing download in the Hugging Face cache (huggingface-cli / transformers) with weights + tokenizer
std::string hf_cache_dir(const std::string& repo) {
    std::string base = hf_home() + "/hub/models--";
    for (char c : repo) base += c == '/' ? std::string("--") : std::string(1, c);
    glob_t g{};
    std::string found;
    if (glob((base + "/snapshots/*/config.json").c_str(), 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc && found.empty(); i++) {
            std::string d = g.gl_pathv[i];
            d = d.substr(0, d.size() - 12);
            glob_t g2{};
            bool st = glob((d + "/*.safetensors").c_str(), 0, nullptr, &g2) == 0;
            globfree(&g2);
            if (st && is_file(d + "/tokenizer.json")) found = d;
        }
    }
    globfree(&g);
    return found;
}

struct HfFile { std::string path; size_t size = 0; };

bool hf_list(const std::string& repo, std::vector<HfFile>& files, std::string& canonical, std::string& err) {
    std::string out;
    HttpGetResult r;
    for (int attempt = 0; attempt < 3; attempt++) {
        r = http_get_string("https://huggingface.co/api/models/" + repo, hf_opts(), out);
        if (r.status > 0 && r.status < 500) break;
        sleep(1 + attempt);
    }
    if (r.status < 0) { err = "cannot reach huggingface.co: " + r.error; return false; }
    int code = r.status;
    if (code == 401 || code == 404) { err = "model " + repo + " not found on Hugging Face (or private: set HF_TOKEN)"; return false; }
    if (code != 200) { err = "Hugging Face API returned HTTP " + std::to_string(code); return false; }
    Json j = Json::parse(out);
    canonical = j.get_str("id", repo);
    if (j["gated"].is_str() || (j["gated"].is_bool() && j["gated"].b)) {
        if (hf_token().empty()) { err = canonical + " is gated: accept its license on https://huggingface.co/" + canonical + " and set HF_TOKEN"; return false; }
    }
    // sizes come from the tree listing
    r = http_get_string("https://huggingface.co/api/models/" + canonical + "/tree/main", hf_opts(), out);
    if (r.status != 200) { err = "cannot list files of " + canonical + " (gated? accept the license and set HF_TOKEN)"; return false; }
    Json t = Json::parse(out);
    for (auto& f : t.arr) {
        if (f.get_str("type", "") != "file") continue;
        HfFile hf;
        hf.path = f.get_str("path", "");
        hf.size = (size_t)(f["lfs"].is_obj() ? f["lfs"].get_num("size", 0) : f.get_num("size", 0));
        if (want_file(hf.path)) files.push_back(hf);
    }
    bool st = false, tk = false;
    for (auto& f : files) { st |= ends_with(f.path, ".safetensors"); tk |= f.path == "tokenizer.json"; }
    if (!st) { err = canonical + " has no .safetensors weights"; return false; }
    if (!tk) { err = canonical + " has no tokenizer.json"; return false; }
    return true;
}

// one file to dst + ".part" (resumed when present), with retries; renamed to dst when complete
bool hf_fetch(const std::string& url, const std::string& dst, size_t expect, std::string& err) {
    const std::string part = dst + ".part";
    const std::string label = dst.substr(dst.rfind('/') + 1);   // file name for messages (the whole of dst if it has no directory)
    bool ok = false;
    for (int attempt = 0; attempt < 6 && !ok; attempt++) {
        if (attempt) { logw("retrying %s (%s)", label.c_str(), err.c_str()); sleep(2 * attempt); }
        HttpGetOptions o = hf_opts();
        o.range_from = (int64_t)file_size(part);
        if (expect && (size_t)o.range_from == expect) { ok = true; break; }   // complete from an earlier run
        if (expect && (size_t)o.range_from > expect) { unlink(part.c_str()); o.range_from = 0; }
        FILE* f = nullptr;
        std::string werr;
        size_t have = 0, start = 0;
        double t0 = now_sec(), last = 0;
        auto progress = [&](bool final) {
            double t = now_sec();
            if (!f || (!final && t - last < 0.25)) return;
            double rate = (have - start) / std::max(t - t0, 1e-3);
            char m[256];
            if (expect) snprintf(m, sizeof m, "downloading %s: %.1f%%  %s / %s  %s/s", label.c_str(), 100.0 * have / expect, human((double)have).c_str(), human((double)expect).c_str(), human(rate).c_str());
            else snprintf(m, sizeof m, "downloading %s: %s  %s/s", label.c_str(), human((double)have).c_str(), human(rate).c_str());
            log_progress(m, expect ? (double)have / expect : -1.0);
            if (final) log_progress_end();
            last = t;
        };
        HttpGetResult r = http_get(url, o, [&](const char* p, size_t n) {
            if (!f) return true;   // error page: ignore the body
            if (fwrite(p, 1, n, f) != n) { werr = "cannot write " + part; return false; }
            have += n;
            progress(false);
            return true;
        }, [&](const HttpGetResult& h) {
            const bool resumed = o.range_from > 0 && h.status == 206;
            if (h.status == 200 || resumed) {   // a 200 to a range request means the server restarted from 0
                f = fopen(part.c_str(), resumed ? "ab" : "wb");
                if (!f) werr = "cannot write " + part;
                have = start = resumed ? (size_t)o.range_from : 0;
            }
        });
        progress(true);
        if (f && fclose(f) != 0) werr = "cannot write " + part;
        if (!werr.empty()) { err = werr; return false; }   // local disk problem: retrying will not help
        if (r.status == 401 || r.status == 403 || r.status == 404) { err = "HTTP " + std::to_string(r.status) + " for " + url; return false; }
        if (r.status == 416 && expect && file_size(part) == expect) { ok = true; break; }   // nothing left to fetch
        if (!r.error.empty()) err = r.error;
        else if (r.status != 200 && r.status != 206) err = "HTTP " + std::to_string(r.status) + " for " + url;
        else if (expect && file_size(part) != expect) err = "incomplete download (" + human((double)file_size(part)) + " of " + human((double)expect) + ")";
        else ok = true;
    }
    if (!ok) return false;
    if (rename(part.c_str(), dst.c_str()) != 0) { err = "cannot write " + dst; return false; }
    return true;
}

bool hf_download(const std::string& repo, const std::vector<HfFile>& files, const std::string& dir, std::string& err) {
    mkdirs(dir);
    size_t total = 0;
    for (auto& f : files) total += f.size;
    for (size_t i = 0; i < files.size(); i++) {
        const HfFile& f = files[i];
        std::string dst = dir + "/" + f.path;
        if (is_file(dst) && (f.size == 0 || file_size(dst) == f.size)) continue;
        logi("pulling %s (%zu/%zu, %s)", f.path.c_str(), i + 1, files.size(), human((double)f.size).c_str());
        std::string e;
        if (!hf_fetch("https://huggingface.co/" + repo + "/resolve/main/" + f.path, dst, f.size, e)) {
            err = "download of " + f.path + " failed: " + e;
            return false;
        }
    }
    logi("downloaded %s", human((double)total).c_str());
    return true;
}

// lock per model name so concurrent pulls (CLI + server) do not race
struct PullLock {
    int fd = -1;
    explicit PullLock(const std::string& name) {
        mkdirs(models_dir());
        fd = open((models_dir() + "/." + slug(name) + ".lock").c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) return;
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            logi("waiting for another pull of %s...", name.c_str());
            flock(fd, LOCK_EX);
        }
    }
    ~PullLock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
};

int do_pull(const std::string& ref_str, bool keep_download, const std::string& device) {
    ModelRef ref = parse_ref(ref_str);
    if (!ref.known) {
        loge("unknown model '%s': use a name from `tachyon catalog`, a Hugging Face repo (org/name or hf.co/org/name), "
             "or `tachyon import <file.tach|hf_dir> <name>`", ref_str.c_str());
        return 1;
    }
    PullLock lock(ref.name);
    LocalModel lm;
    if (find_local(ref.name, lm)) { logi("%s is up to date (%s)", ref.name.c_str(), lm.path.c_str()); return 0; }
    std::string hf_dir = hf_cache_dir(ref.repo), canonical = ref.repo, dl_dir;
    if (!hf_dir.empty()) logi("using Hugging Face cache %s", hf_dir.c_str());
    else {
        std::vector<HfFile> files;
        std::string err;
        logi("pulling manifest for %s", ref.repo.c_str());
        if (!hf_list(ref.repo, files, canonical, err)) { loge("%s", err.c_str()); return 1; }
        hf_dir = hf_cache_dir(canonical);
        if (hf_dir.empty()) {
            size_t total = 0;
            for (auto& f : files) total += f.size;
            dl_dir = home_dir() + "/downloads/" + slug(canonical);
            struct statvfs vs;
            mkdirs(dl_dir);
            if (statvfs(dl_dir.c_str(), &vs) == 0) {
                double free_b = (double)vs.f_bavail * vs.f_frsize;
                double need = total * (ref.quant == "q8" ? 1.6 : 1.4) + (1ull << 30);   // checkpoint + temporary Q8 + output
                if (free_b < need)
                    logw("%s free in %s, about %s needed (set TACHYON_HOME to a bigger disk)", human(free_b).c_str(),
                         home_dir().c_str(), human(need).c_str());
            }
            if (!hf_download(canonical, files, dl_dir, err)) { loge("%s (rerun to resume)", err.c_str()); return 1; }
            hf_dir = dl_dir;
        }
    }
    mkdirs(models_dir());
    const std::string out = models_dir() + "/" + slug(ref.name) + ".tach";
    logi("converting %s -> %s (%s%s)", canonical.c_str(), out.c_str(), ref.quant.c_str(),
         ref.quant == "q8" ? "" : " + importance matrix");
    const double t0 = now_sec();
    int rc;
    if (ref.quant == "q8") {
        std::vector<std::string> a = {hf_dir, out, "-q", "q8"};
        std::vector<char*> av;
        for (auto& x : a) av.push_back((char*)x.c_str());
        rc = convert_main((int)av.size(), av.data());
    } else {
        rc = convert_calibrated(hf_dir, out, {"-q", ref.quant}, kCalibText, device, 40);
        remove((out + ".imatrix").c_str());
    }
    if (rc != 0 || !is_file(out)) { loge("conversion failed"); return 1; }
    write_manifest(ref.name, out, canonical, ref.quant);
    if (!dl_dir.empty() && !keep_download) remove_tree(dl_dir);
    logi("%s (%s) ready in %.0fs", ref.name.c_str(), human((double)file_size(out)).c_str(), now_sec() - t0);
    return 0;
}

// ------------------------------------------------------------------------------------------------ manager server

struct Endpoint { std::string host = "127.0.0.1"; int port = 28657; };
Endpoint endpoint() {
    Endpoint e;
    const char* h = getenv("TACHYON_HOST");
    if (h && *h) {
        std::string s = h;
        if (s.rfind("http://", 0) == 0) s = s.substr(7);
        while (!s.empty() && s.back() == '/') s.pop_back();
        size_t c = s.rfind(':');
        if (c != std::string::npos) { e.host = s.substr(0, c); e.port = atoi(s.c_str() + c + 1); }
        else e.host = s;
        if (e.host.empty()) e.host = "127.0.0.1";
    }
    return e;
}

int free_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    bind(fd, (sockaddr*)&a, sizeof a);
    socklen_t l = sizeof a;
    getsockname(fd, (sockaddr*)&a, &l);
    int p = ntohs(a.sin_port);
    close(fd);
    return p;
}

std::string tail_file(const std::string& p, size_t n = 1500) {
    if (!is_file(p)) return "";
    std::string s = read_file(p);
    return s.size() > n ? s.substr(s.size() - n) : s;
}

struct ManagerOptions {
    Endpoint ep;
    double keep_alive = 300;          // seconds an idle model stays loaded (<0: forever)
    int max_loaded = 1;               // models loaded at once (one GPU: 1)
    std::string default_model;        // used for requests naming an unknown model (e.g. Claude Code's haiku calls)
    std::string api_key;
    std::vector<std::string> runner_args;
    bool auto_pull = true;
};

struct Runner {
    std::string name, path, log;
    pid_t pid = -1;
    int port = 0;
    int inflight = 0;
    bool ready = false;
    bool unload_requested = false;
    double last_used = 0, started = 0;
    double keep_alive = 300;
};

class Manager {
public:
    explicit Manager(const ManagerOptions& o) : o_(o) {}

    void reaper() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::lock_guard<std::mutex> lk(mu_);
            for (auto it = runners_.begin(); it != runners_.end();) {
                auto& r = it->second;
                int st;
                // acquire() owns waitpid while a runner is loading.
                if (r->ready && r->pid > 0 && waitpid(r->pid, &st, WNOHANG) == r->pid) {
                    char how[64];
                    if (WIFSIGNALED(st)) snprintf(how, sizeof how, "killed by signal %d (%s)", WTERMSIG(st), strsignal(WTERMSIG(st)));
                    else snprintf(how, sizeof how, "exit code %d", WEXITSTATUS(st));
                    logw("model %s exited unexpectedly (pid %d, %s); log: %s", r->name.c_str(), r->pid, how, r->log.c_str());
                    r->pid = -1;
                    it = runners_.erase(it);
                    cv_.notify_all();
                    continue;
                }
                if (r->ready && r->inflight == 0 && r->keep_alive >= 0 && now_sec() - r->last_used >= r->keep_alive) {
                    logi("unloading %s (idle %.0fs)", r->name.c_str(), now_sec() - r->last_used);
                    kill_runner(*r);
                    it = runners_.erase(it);
                    cv_.notify_all();
                    continue;
                }
                ++it;
            }
        }
    }

    void handle(int fd, const std::string& peer) {
        Conn c(fd);
        c.peer = peer;
        try {
            while (true) {
                HttpRequest r;
                int st = 0;
                if (!c.read_request(r, &st)) {
                    if (st) {
                        logw("rejected request from %s: %d %s", peer.c_str(), st, Conn::status_text(st));
                        c.send_response(st, "application/json", err_json(Conn::status_text(st), false).dump(), false);
                    }
                    return;
                }
                LogScope scope(log_next_request_id());
                c.status = 0; c.bytes_out = 0; c.note.clear();
                const double t0 = now_sec();
                bool ok = route(c, r);
                http_log_access(c, r, now_sec() - t0);
                if (!ok || !r.keep_alive) return;
            }
        } catch (const std::exception& ex) {
            loge("unhandled exception while serving %s: %s", peer.c_str(), ex.what());
        } catch (...) {
            loge("unhandled exception while serving %s", peer.c_str());
        }
    }

    void stop_all() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& kv : runners_) kill_runner(*kv.second);
        runners_.clear();
    }

private:
    static Json err_json(const std::string& msg, bool anthropic, const std::string& type = "invalid_request_error") {
        Json in = Json::object();
        Json e = Json::object();
        if (anthropic) { in.set("type", type); in.set("message", msg); e.set("type", "error"); e.set("error", in); }
        else { in.set("message", msg); in.set("type", type); e.set("error", in); }
        return e;
    }
    bool reply(Conn& c, const HttpRequest& r, int code, const Json& body) { return c.send_response(code, "application/json", body.dump(), r.keep_alive); }
    bool error(Conn& c, const HttpRequest& r, int code, const std::string& msg) {
        c.note = msg.size() > 300 ? msg.substr(0, 300) + "..." : msg;
        if (r.path.rfind("/api/", 0) == 0) { Json j = Json::object(); j.set("error", msg); return reply(c, r, code, j); }
        const bool anth = r.path.find("/messages") != std::string::npos;
        std::string type = code == 404 ? (anth ? "not_found_error" : "model_not_found") : code >= 500 ? "api_error" : code == 401 ? "authentication_error" : "invalid_request_error";
        return reply(c, r, code, err_json(msg, anth, type));
    }

    void kill_runner(Runner& r) {
        if (r.pid <= 0) return;
        kill(r.pid, SIGTERM);
        for (int i = 0; i < 100; i++) {   // up to 5 s, then SIGKILL
            int st;
            if (waitpid(r.pid, &st, WNOHANG) == r.pid) { r.pid = -1; return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        kill(r.pid, SIGKILL);
        int st;
        waitpid(r.pid, &st, 0);
        r.pid = -1;
    }

    // PR_SET_PDEATHSIG fires when the *thread* that forked the child exits, so every runner is forked from this
    // one long-lived thread (connection threads come and go)
    pid_t spawn_runner(const LocalModel& m, int port, const std::string& log) {
        static std::mutex smu;
        static std::condition_variable scv;
        static std::deque<std::packaged_task<pid_t()>> jobs;
        static std::once_flag once;
        std::call_once(once, [] {
            std::thread([] {
                while (true) {
                    std::packaged_task<pid_t()> job;
                    {
                        std::unique_lock<std::mutex> lk(smu);
                        scv.wait(lk, [] { return !jobs.empty(); });
                        job = std::move(jobs.front());
                        jobs.pop_front();
                    }
                    job();
                }
            }).detach();
        });
        std::packaged_task<pid_t()> job([this, m, port, log] { return spawn_runner_now(m, port, log); });
        auto fut = job.get_future();
        {
            std::lock_guard<std::mutex> lk(smu);
            jobs.push_back(std::move(job));
        }
        scv.notify_one();
        return fut.get();
    }

    pid_t spawn_runner_now(const LocalModel& m, int port, const std::string& log) {
        std::vector<std::string> a = {self_exe(), "serve", "-m", m.path, "--host", "127.0.0.1", "--port", std::to_string(port),
                                      "--alias", m.name};
        if (g_log_level != 1) a.push_back(g_log_level >= 2 ? "-v" : "-q");   // before the user's args, which win
        for (auto& x : m.args.arr) if (x.is_str()) a.push_back(x.str);
        for (auto& x : o_.runner_args) a.push_back(x);
        pid_t pid = fork();
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGTERM);   // runners die with the manager
            int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
            int nul = open("/dev/null", O_RDONLY);
            if (nul >= 0) { dup2(nul, 0); close(nul); }
            std::vector<char*> av;
            for (auto& x : a) av.push_back((char*)x.c_str());
            av.push_back(nullptr);
            execv(av[0], av.data());
            _exit(127);
        }
        return pid;
    }

    // resolve the request's model name to a local model; pulls catalog models on first use
    bool resolve(const std::string& name, LocalModel& m, std::string& err) {
        if (!name.empty() && find_local(name, m)) return true;
        if (!name.empty() && o_.auto_pull && parse_ref(name).known) {
            logi("pulling %s", name.c_str());
            std::string log = home_dir() + "/logs/pull-" + slug(parse_ref(name).name) + ".log";
            mkdirs(home_dir() + "/logs");
            int rc = run_prog({self_exe(), "pull", name}, log);
            if (rc == 0 && find_local(name, m)) return true;
            err = "pulling " + name + " failed: " + tail_file(log, 400);
            return false;
        }
        // unknown name (e.g. a client's hard-coded model id): default model, else the loaded one, else the only one
        if (!o_.default_model.empty() && o_.default_model != name) return resolve(o_.default_model, m, err);
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (runners_.size() == 1) {
                auto& r = runners_.begin()->second;
                m = LocalModel();
                m.name = r->name;
                m.path = r->path;
                return true;
            }
        }
        auto all = list_local();
        if (all.size() == 1) { m = all[0]; return true; }
        std::string names;
        for (auto& x : all) names += (names.empty() ? "" : ", ") + x.name;
        err = "model '" + name + "' not found" + (names.empty() ? std::string(" (no local models: tachyon pull <model>)") : ". Local models: " + names);
        return false;
    }

    std::shared_ptr<Runner> acquire(const LocalModel& m, std::string& err) {
        std::unique_lock<std::mutex> lk(mu_);
        while (true) {
            auto it = runners_.find(m.name);
            if (it != runners_.end()) {
                auto r = it->second;
                if (r->ready) { r->inflight++; r->last_used = now_sec(); return r; }
                cv_.wait_for(lk, std::chrono::milliseconds(200));   // another request is starting it
                continue;
            }
            if ((int)runners_.size() >= o_.max_loaded) {
                // unload the least recently used idle model; wait if every loaded model is busy
                std::shared_ptr<Runner> lru;
                for (auto& kv : runners_)
                    if (kv.second->ready && kv.second->inflight == 0 && (!lru || kv.second->last_used < lru->last_used)) lru = kv.second;
                if (!lru) { cv_.wait_for(lk, std::chrono::milliseconds(200)); continue; }
                logi("unloading %s to make room for %s", lru->name.c_str(), m.name.c_str());
                kill_runner(*lru);
                runners_.erase(lru->name);
                continue;
            }
            auto r = std::make_shared<Runner>();
            r->name = m.name;
            r->path = m.path;
            r->port = free_port();
            mkdirs(home_dir() + "/logs");
            r->log = home_dir() + "/logs/" + slug(m.name) + ".log";
            r->started = now_sec();
            r->keep_alive = o_.keep_alive;
            r->pid = spawn_runner(m, r->port, r->log);
            runners_[m.name] = r;
            logi("loading %s (pid %d, port %d, log %s)", m.name.c_str(), r->pid, r->port, r->log.c_str());
            lk.unlock();
            bool ok = false;
            std::string body;
            while (true) {
                if (r->pid <= 0) break;
                int st;
                if (waitpid(r->pid, &st, WNOHANG) == r->pid) { r->pid = -1; break; }
                body.clear();
                if (http_call("127.0.0.1", r->port, "GET", "/health", "", [&](const std::string& d) { body += d; return true; }, nullptr, 2) == 200) { ok = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            lk.lock();
            if (!ok) {
                runners_.erase(m.name);
                cv_.notify_all();
                err = "model " + m.name + " failed to load:\n" + tail_file(r->log);
                loge("%s failed to load after %.1fs (see %s)", m.name.c_str(), now_sec() - r->started, r->log.c_str());
                return nullptr;
            }
            logi("%s ready in %.1fs", m.name.c_str(), now_sec() - r->started);
            r->ready = true;
            r->inflight++;
            r->last_used = now_sec();
            cv_.notify_all();
            return r;
        }
    }
    void release(const std::shared_ptr<Runner>& r, double keep_alive) {
        std::lock_guard<std::mutex> lk(mu_);
        r->inflight--;
        auto it = runners_.find(r->name);
        if (it == runners_.end() || it->second != r) return;
        r->last_used = now_sec();
        if (r->unload_requested) keep_alive = 0;
        r->keep_alive = keep_alive;
        if (r->inflight == 0 && keep_alive == 0) {
            kill_runner(*r);
            runners_.erase(r->name);
        }
        cv_.notify_all();
    }

    Json model_entry(const LocalModel& m) {
        Json j = Json::object();
        j.set("id", m.name);
        j.set("object", "model");
        j.set("type", "model");
        j.set("created", (int64_t)file_mtime(m.path));
        j.set("created_at", m.created);
        j.set("owned_by", "tachyon");
        j.set("display_name", m.name);
        return j;
    }
    static time_t file_mtime(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 ? st.st_mtime : 0; }

    bool authorized(const HttpRequest& r) const {
        if (o_.api_key.empty()) return true;
        std::string a = r.header("Authorization");
        if (a.size() > 7 && strncasecmp(a.c_str(), "Bearer ", 7) == 0 && a.substr(7) == o_.api_key) return true;
        return r.header("x-api-key") == o_.api_key;
    }

    bool route(Conn& c, HttpRequest& r) {
        logd("%s %s", r.method.c_str(), r.path.c_str());
        if (r.method == "OPTIONS") return c.send_response(204, "text/plain", "", r.keep_alive);
        if (r.method == "GET" && r.path == "/") return c.send_response(200, "text/plain", "Tachyon is running\n", r.keep_alive);
        if (r.method == "GET" && (r.path == "/health" || r.path == "/api/version")) {
            Json j = Json::object();
            j.set("status", "ok");
            j.set("manager", true);
            j.set("version", TACH_VERSION);
            return reply(c, r, 200, j);
        }
        if (!authorized(r)) return error(c, r, 401, "invalid API key");
        if (r.method == "GET" && (r.path == "/v1/models" || r.path == "/models")) {
            Json arr = Json::array();
            for (auto& m : list_local()) arr.push(model_entry(m));
            Json l = Json::object();
            l.set("object", "list");
            l.set("data", arr);
            l.set("has_more", false);
            if (arr.size()) { l.set("first_id", arr.arr.front()["id"]); l.set("last_id", arr.arr.back()["id"]); }
            return reply(c, r, 200, l);
        }
        if (r.method == "GET" && r.path.rfind("/v1/models/", 0) == 0) {
            LocalModel m;
            if (!find_local(r.path.substr(11), m)) return error(c, r, 404, "model not found");
            return reply(c, r, 200, model_entry(m));
        }
        if (r.method == "GET" && r.path == "/api/tags") {
            Json arr = Json::array();
            for (auto& m : list_local()) {
                Json j = Json::object();
                j.set("name", m.name);
                j.set("model", m.name);
                j.set("size", (int64_t)m.size);
                j.set("modified_at", m.created);
                j.set("digest", "");
                ModelFile mf; mf.open(m.path);
                Json d = ollama_details(mf);
                d.set("source", m.source);
                j.set("details", d);
                arr.push(j);
            }
            Json o = Json::object();
            o.set("models", arr);
            return reply(c, r, 200, o);
        }
        if (r.method == "GET" && r.path == "/api/ps") {
            Json arr = Json::array();
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& kv : runners_) {
                auto& x = *kv.second;
                Json j = Json::object();
                j.set("name", x.name);
                j.set("model", x.name);
                j.set("size", (int64_t)file_size(x.path));
                j.set("digest", "");
                const double remaining = x.keep_alive - (now_sec() - x.last_used);
                j.set("expires_at", x.keep_alive < 0 || x.inflight || remaining > 6e9 ? "9999-12-31T23:59:59Z" : iso_now(time(nullptr) + (time_t)std::max(0.0, remaining)));
                ModelFile mf; mf.open(x.path); j.set("details", ollama_details(mf));
                j.set("pid", (int)x.pid);
                j.set("port", x.port);
                j.set("status", x.ready ? (x.inflight ? "busy" : "idle") : "loading");
                j.set("active_requests", x.inflight);
                j.set("idle_seconds", x.inflight ? 0.0 : now_sec() - x.last_used);
                j.set("unload_in_seconds", x.keep_alive < 0 || x.inflight ? -1.0 : std::max(0.0, x.keep_alive - (now_sec() - x.last_used)));
                arr.push(j);
            }
            Json o = Json::object();
            o.set("models", arr);
            return reply(c, r, 200, o);
        }
        Json body;
        if (r.method == "POST" || r.method == "DELETE") {
            std::string err;
            body = Json::parse(r.body.empty() ? "{}" : r.body, &err);
            if (!err.empty() || !body.is_obj()) return error(c, r, 400, "invalid JSON body: " + err);
        }
        const std::string name = body.get_str("model", body.get_str("name", ""));
        const bool ollama = r.path == "/api/chat" || r.path == "/api/generate";
        double keep_alive = o_.keep_alive;
        if (ollama) {
            if (name.empty()) return error(c, r, 400, "model is required");
            if (body.has("keep_alive")) {
                const Json& k = body["keep_alive"];
                if (k.is_num()) keep_alive = k.num;
                else if (k.is_str()) {
                    char* end = nullptr;
                    keep_alive = strtod(k.str.c_str(), &end);
                    const std::string unit = end ? end : "";
                    double scale = unit == "ns" ? 1e-9 : unit == "us" ? 1e-6 : unit == "ms" ? 1e-3 : unit == "s" ? 1 : unit == "m" ? 60 : unit == "h" ? 3600 : unit.empty() && keep_alive == 0 ? 1 : 0;
                    if (end == k.str.c_str() || !scale) return error(c, r, 400, "invalid keep_alive duration");
                    keep_alive *= scale;
                } else return error(c, r, 400, "keep_alive must be seconds or a duration string");
                if (!std::isfinite(keep_alive)) return error(c, r, 400, "invalid keep_alive duration");
            }
        }
        if (r.method == "POST" && r.path == "/api/show") {
            LocalModel m;
            if (!find_local(name, m)) return error(c, r, 404, "model not found");
            ModelFile mf; mf.open(m.path);
            return reply(c, r, 200, ollama_show(mf));
        }
        const bool unload = ollama && keep_alive == 0 && (r.path == "/api/generate" ? body.get_str("prompt", "").empty() : body["messages"].is_arr() && body["messages"].arr.empty());
        if (r.method == "POST" && (r.path == "/api/stop" || unload)) {
            std::lock_guard<std::mutex> lk(mu_);
            std::string key = name;
            LocalModel m;
            if (find_local(name, m)) key = m.name;
            auto it = runners_.find(key);
            if (it == runners_.end() && !unload) return error(c, r, 404, "model " + name + " is not loaded");
            if (it != runners_.end()) {
                if (unload && it->second->inflight) { it->second->keep_alive = 0; it->second->unload_requested = true; }
                else { kill_runner(*it->second); runners_.erase(it); }
            }
            cv_.notify_all();
            if (unload) {
                Json j = Json::object(); j.set("model", key); j.set("done", true); j.set("done_reason", "unload"); j.set("created_at", iso_now());
                if (r.path == "/api/generate") j.set("response", "");
                else { Json m = Json::object(); m.set("role", "assistant"); m.set("content", ""); j.set("message", m); }
                return reply(c, r, 200, j);
            }
            return reply(c, r, 200, Json::object());
        }
        if (r.path == "/api/pull") {
            if (name.empty()) return error(c, r, 400, "model is required");
            const bool stream = body.get_bool("stream", true);
            if (!stream) {
                mkdirs(home_dir() + "/logs");
                std::string log = home_dir() + "/logs/pull-" + slug(parse_ref(name).name) + ".log";
                int rc = run_prog({self_exe(), "pull", name}, log);
                if (rc) return error(c, r, 500, tail_file(log, 600));
                Json j = Json::object(); j.set("status", "success"); return reply(c, r, 200, j);
            }
            if (!c.start_chunked("application/x-ndjson")) return false;
            mkdirs(home_dir() + "/logs");
            std::string log = home_dir() + "/logs/pull-" + slug(parse_ref(name).name) + ".log";
            Json st = Json::object();
            st.set("status", "pulling " + name);
            c.chunk(st.dump() + "\n");
            int rc = run_prog({self_exe(), "pull", name}, log);
            Json fin = Json::object();
            if (rc == 0) fin.set("status", "success");
            else fin.set("error", tail_file(log, 600));
            c.chunk(fin.dump() + "\n");
            return c.end_chunked();
        }
        if (r.path == "/api/delete") {
            LocalModel m;
            if (!find_local(name, m) || m.manifest.empty()) return error(c, r, 404, "model " + name + " not found");
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = runners_.find(m.name);
                if (it != runners_.end()) { kill_runner(*it->second); runners_.erase(it); }
            }
            remove_model(m);
            return reply(c, r, 200, Json::object());
        }
        if (r.method != "POST") return error(c, r, 404, "not found: " + r.method + " " + r.path);
        static const std::set<std::string> inference_paths = {
            "/api/chat", "/api/generate", "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions",
            "/v1/messages", "/messages", "/v1/messages/count_tokens", "/messages/count_tokens", "/tokenize", "/detokenize"};
        if (!inference_paths.count(r.path)) return error(c, r, 404, "not found: " + r.method + " " + r.path);
        // inference endpoints: route by model
        LocalModel m;
        std::string err;
        if (ollama && !find_local(name, m) && !(o_.auto_pull && parse_ref(name).known)) return error(c, r, 404, "model '" + name + "' not found");
        if (!resolve(name, m, err)) return error(c, r, 404, err);
        if (ollama) { body.set("model", m.name); r.body = body.dump(); }
        auto run = acquire(m, err);
        if (!run) return error(c, r, 500, err);
        bool keep = http_relay(r, "127.0.0.1", run->port, c, &err);
        release(run, keep_alive);
        if (!err.empty()) return error(c, r, 502, err);
        return keep;
    }

public:
    static void remove_model(const LocalModel& m) {
        // models imported by path (outside the store) keep their file
        if (m.path.rfind(models_dir() + "/", 0) == 0) remove(m.path.c_str());
        remove(m.manifest.c_str());
    }

private:
    ManagerOptions o_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<std::string, std::shared_ptr<Runner>> runners_;
};

int manager_main(const ManagerOptions& o) {
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(o.ep.port);
    if (inet_pton(AF_INET, o.ep.host == "localhost" ? "127.0.0.1" : o.ep.host.c_str(), &addr.sin_addr) != 1) die("bad host %s", o.ep.host.c_str());
    if (bind(fd, (sockaddr*)&addr, sizeof addr) < 0) die("cannot bind %s:%d (%s)", o.ep.host.c_str(), o.ep.port, strerror(errno));
    if (listen(fd, 128) < 0) die("listen failed");
    static Manager* mgr = new Manager(o);
    struct sigaction sa{};
    sa.sa_handler = [](int) { mgr->stop_all(); _exit(0); };
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    std::thread([] { mgr->reaper(); }).detach();
    auto local = list_local();
    logi("Tachyon %s model server on http://%s:%d  (%zu local models in %s)", TACH_VERSION, o.ep.host.c_str(), o.ep.port, local.size(), models_dir().c_str());
    logi("OpenAI API: /v1/chat/completions   Anthropic API: /v1/messages   models load on first request, unload after %s idle",
         o.keep_alive < 0 ? "never" : ago(o.keep_alive).c_str());
    while (true) {
        sockaddr_in cli{};
        socklen_t cl = sizeof cli;
        int c = accept(fd, (sockaddr*)&cli, &cl);
        if (c < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {   // transient: keep serving
                logw("accept failed (%s): too many open connections?", strerror(errno));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            loge("accept failed (%s): server stopping", strerror(errno));
            break;
        }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::string peer = http_peer_name(cli);
        std::thread([c, peer] { mgr->handle(c, peer); }).detach();
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ CLI client

bool manager_up(const Endpoint& ep, bool* is_manager = nullptr) {
    std::string body;
    int st = http_call(ep.host, ep.port, "GET", "/health", "", [&](const std::string& d) { body += d; return true; }, nullptr, 2);
    if (st != 200) return false;
    if (is_manager) *is_manager = Json::parse(body).get_bool("manager", false);
    return true;
}

// start `tachyon serve` (manager) in the background when nothing listens yet
bool ensure_manager(const Endpoint& ep) {
    bool mgr = false;
    if (manager_up(ep, &mgr)) {
        if (!mgr) { loge("%s:%d is a single-model `tachyon serve -m`; stop it or set TACHYON_HOST", ep.host.c_str(), ep.port); return false; }
        return true;
    }
    mkdirs(home_dir() + "/logs");
    std::string log = home_dir() + "/logs/server.log";
    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        if (fork() != 0) _exit(0);   // daemonize
        int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        int nul = open("/dev/null", O_RDONLY);
        if (nul >= 0) { dup2(nul, 0); close(nul); }
        std::string exe = self_exe();
        execl(exe.c_str(), exe.c_str(), "serve", (char*)nullptr);
        _exit(127);
    }
    int st;
    waitpid(pid, &st, 0);
    for (int i = 0; i < 100; i++) {
        if (manager_up(ep)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    loge("could not start the tachyon server (log: %s)", log.c_str());
    return false;
}

struct ChatOpts {
    std::string model, system;
    int think = -1;   // -1 model default
    bool show_thinking = true, verbose = false;
};

// stream one chat turn; prints as it goes. Returns false on error. Appends the assistant turn to history.
bool chat_turn(const Endpoint& ep, const ChatOpts& o, Json& history) {
    Json body = Json::object();
    body.set("model", o.model);
    body.set("messages", history);
    body.set("stream", true);
    Json so = Json::object();
    so.set("include_usage", true);
    body.set("stream_options", so);
    if (o.think >= 0) { Json kw = Json::object(); kw.set("enable_thinking", o.think == 1); body.set("chat_template_kwargs", kw); }
    std::string line_buf, content, reasoning, err_msg, raw;
    bool in_reasoning = false, got_any = false;
    Json usage, timings;
    const bool tty = isatty(1);
    std::atomic<bool> first{false};
    std::thread spinner;
    if (tty && isatty(2)) {
        spinner = std::thread([&] {
            const char* fr[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            for (int i = 0; !first.load(); i++) {
                fprintf(stderr, "\r%s ", fr[i % 10]);
                fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
        });
    }
    auto stop_spinner = [&]() {
        if (!first.exchange(true) && spinner.joinable()) { spinner.join(); fprintf(stderr, "\r  \r"); fflush(stderr); }
    };
    std::string err;
    int st = http_call(ep.host, ep.port, "POST", "/v1/chat/completions", body.dump(), [&](const std::string& d) {
        raw += d;
        line_buf += d;
        size_t p;
        while ((p = line_buf.find('\n')) != std::string::npos) {
            std::string line = line_buf.substr(0, p);
            line_buf.erase(0, p + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data: ", 0) != 0) continue;
            std::string payload = line.substr(6);
            if (payload == "[DONE]") return true;
            Json j = Json::parse(payload);
            if (j.has("error")) { err_msg = j["error"].get_str("message", "?"); return false; }
            if (j["usage"].is_obj()) usage = j["usage"];
            if (j.has("timings")) timings = j["timings"];
            if (!j["choices"].is_arr() || j["choices"].arr.empty()) continue;
            const Json& delta = j["choices"][0]["delta"];
            std::string r = delta.get_str("reasoning_content", "");
            std::string c = delta.get_str("content", "");
            if (r.empty() && c.empty()) continue;
            stop_spinner();
            got_any = true;
            reasoning += r;
            content += c;
            if (!r.empty() && o.show_thinking) {
                if (!in_reasoning) { printf(tty ? "\033[2mThinking...\n" : "Thinking...\n"); in_reasoning = true; }
                printf("%s", r.c_str());
            }
            if (!c.empty()) {
                if (in_reasoning) { printf(tty ? "\n...done thinking.\033[0m\n\n" : "\n...done thinking.\n\n"); in_reasoning = false; }
                printf("%s", c.c_str());
            }
            fflush(stdout);
        }
        return true;
    }, &err);
    stop_spinner();
    if (in_reasoning && tty) printf("\033[0m");
    printf("\n");
    fflush(stdout);
    if (st < 0) { loge("%s", err.c_str()); return false; }
    if (st != 200) {
        Json j = Json::parse(raw);
        loge("%s", j["error"].get_str("message", raw).c_str());
        return false;
    }
    if (!err_msg.empty()) { loge("%s", err_msg.c_str()); return false; }
    if (o.verbose && usage.is_obj())
        fprintf(stderr, "%s[%lld prompt tokens (%lld cached) | %lld generated | prompt %.0f tok/s | generation %.1f tok/s]%s\n", isatty(2) ? "\033[36m" : "",
                (long long)usage.get_int("prompt_tokens", 0), (long long)usage["prompt_tokens_details"].get_int("cached_tokens", 0),
                (long long)usage.get_int("completion_tokens", 0), timings.get_num("prompt_per_second", 0), timings.get_num("predicted_per_second", 0),
                isatty(2) ? "\033[0m" : "");
    Json a = Json::object();
    a.set("role", "assistant");
    a.set("content", content);
    if (!reasoning.empty()) a.set("reasoning_content", reasoning);
    history.push(a);
    return got_any || true;
}

int run_cli(int argc, char** argv) {
    ChatOpts o;
    std::string prompt;
    bool keep = false;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for %s", a.c_str()); return argv[++i]; };
        if (a == "--think") o.think = 1;
        else if (a == "--no-think" || a == "--think=false") o.think = 0;
        else if (a == "--hide-thinking") o.show_thinking = false;
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "-s" || a == "--system") o.system = next();
        else if (a == "--keep-download") keep = true;
        else if (o.model.empty()) o.model = a;
        else prompt += (prompt.empty() ? "" : " ") + a;
    }
    if (o.model.empty()) die("usage: tachyon run <model> [prompt]");
    LocalModel lm;
    if (!find_local(o.model, lm)) {
        if (do_pull(o.model, keep, "auto") != 0) return 1;
        if (!find_local(o.model, lm)) return 1;
    }
    o.model = lm.name;
    Endpoint ep = endpoint();
    if (!ensure_manager(ep)) return 1;
    auto base = [&]() {
        Json h = Json::array();
        if (!o.system.empty()) { Json s = Json::object(); s.set("role", "system"); s.set("content", o.system); h.push(s); }
        return h;
    };
    if (prompt.empty() && !isatty(0)) {   // piped input
        std::string s, line;
        while (std::getline(std::cin, line)) s += line + "\n";
        prompt = s;
    }
    if (!prompt.empty()) {
        Json h = base();
        Json u = Json::object();
        u.set("role", "user");
        u.set("content", prompt);
        h.push(u);
        return chat_turn(ep, o, h) ? 0 : 1;
    }
    Json history = base();
    fprintf(stderr, "%s>>> chatting with %s. /? for help, /bye to exit%s\n", isatty(2) ? "\033[2m" : "", o.model.c_str(), isatty(2) ? "\033[0m" : "");
    std::string line;
    while (true) {
        printf(">>> ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) { printf("\n"); break; }
        if (line.rfind("\"\"\"", 0) == 0) {   // multi-line input until a closing """
            std::string text = line.substr(3), l2;
            while (text.find("\"\"\"") == std::string::npos && std::getline(std::cin, l2)) text += "\n" + l2;
            size_t e = text.find("\"\"\"");
            line = e == std::string::npos ? text : text.substr(0, e);
        }
        if (line.empty()) continue;
        if (line == "/bye" || line == "/exit" || line == "/quit") break;
        if (line == "/?" || line == "/help") {
            fprintf(stderr, "  /clear            clear the conversation\n  /think on|off     toggle reasoning (thinking models)\n"
                            "  /system TEXT      set the system prompt (clears the conversation)\n  /verbose          toggle speed stats\n"
                            "  /bye              exit\n  \"\"\"               start / end multi-line input\n");
            continue;
        }
        if (line == "/clear") { history = base(); fprintf(stderr, "Cleared session context\n"); continue; }
        if (line == "/think on" || line == "/set think") { o.think = 1; continue; }
        if (line == "/think off" || line == "/set nothink") { o.think = 0; continue; }
        if (line == "/verbose" || line == "/set verbose") { o.verbose = !o.verbose; continue; }
        if (line.rfind("/system ", 0) == 0) { o.system = line.substr(8); history = base(); fprintf(stderr, "Set system message\n"); continue; }
        if (line[0] == '/') { fprintf(stderr, "Unknown command '%s'. Type /? for help\n", line.c_str()); continue; }
        Json u = Json::object();
        u.set("role", "user");
        u.set("content", line);
        history.push(u);
        if (!chat_turn(ep, o, history)) history.arr.pop_back();
        printf("\n");
    }
    return 0;
}

void print_table(const std::vector<std::vector<std::string>>& rows) {
    std::vector<size_t> w;
    for (auto& r : rows) for (size_t i = 0; i < r.size(); i++) { if (w.size() <= i) w.push_back(0); w[i] = std::max(w[i], r[i].size()); }
    for (auto& r : rows) {
        std::string s;
        for (size_t i = 0; i < r.size(); i++) s += r[i] + (i + 1 < r.size() ? std::string(w[i] - r[i].size() + 4, ' ') : "");
        printf("%s\n", s.c_str());
    }
}

}  // namespace

bool runner_is_command(const std::string& cmd) {
    static const char* cmds[] = {"pull", "list", "ls", "rm", "ps", "stop", "show", "import", "catalog", "cp"};
    for (auto c : cmds) if (cmd == c) return true;
    return false;
}

void runner_usage() {
    fprintf(stderr,
        "model runner (ollama-style):\n"
        "  tachyon run <model> [prompt]   chat with a model (pulls it and starts the background server when needed)\n"
        "  tachyon serve                  model server on $TACHYON_HOST (default 127.0.0.1:28657): any local model by name,\n"
        "                                 loaded on demand  [--keep-alive SECS] [--max-loaded N] [--default-model NAME]\n"
        "                                 [--runner-args \"--kv8 ...\"] [--api-key KEY] [--no-auto-pull]\n"
        "  tachyon pull <model>           download from Hugging Face + convert (Q4 + importance matrix)\n"
        "  tachyon list                   local models          tachyon catalog   models that can be pulled\n"
        "  tachyon ps                     loaded models         tachyon stop <model>   unload\n"
        "  tachyon show <model>           model details         tachyon rm <model>     delete\n"
        "  tachyon import <file.tach|hf_dir> <name>   register an existing .tach (no copy) or convert a local HF folder\n"
        "  models: catalog names (qwen3:8b, gemma4:e4b, qwen3.5:9b, ...; add -q8 for 8-bit), or any Hugging Face repo\n"
        "  (org/name) of a supported architecture. Store: $TACHYON_HOME (default ~/.tachyon); HF_TOKEN for gated repos.\n");
}

// `tachyon serve` without -m
int runner_serve(int argc, char** argv) {
    ManagerOptions o;
    o.ep = endpoint();
    if (getenv("TACHYON_KEEP_ALIVE")) o.keep_alive = atof(getenv("TACHYON_KEEP_ALIVE"));
    if (getenv("TACHYON_MAX_LOADED")) o.max_loaded = std::max(1, atoi(getenv("TACHYON_MAX_LOADED")));
    if (getenv("TACHYON_DEFAULT_MODEL")) o.default_model = getenv("TACHYON_DEFAULT_MODEL");
    if (getenv("TACHYON_API_KEY")) o.api_key = getenv("TACHYON_API_KEY");
    auto split = [&](const std::string& s) { std::string w; for (char c : s + " ") { if (c == ' ') { if (!w.empty()) o.runner_args.push_back(w); w.clear(); } else w += c; } };
    if (getenv("TACHYON_RUNNER_ARGS")) split(getenv("TACHYON_RUNNER_ARGS"));
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for %s", a.c_str()); return argv[++i]; };
        if (a == "--host") o.ep.host = next();
        else if (a == "--port") o.ep.port = atoi(next().c_str());
        else if (a == "--keep-alive") o.keep_alive = atof(next().c_str());
        else if (a == "--max-loaded") o.max_loaded = std::max(1, atoi(next().c_str()));
        else if (a == "--default-model") o.default_model = next();
        else if (a == "--api-key") o.api_key = next();
        else if (a == "--runner-args") split(next());
        else if (a == "--no-auto-pull") o.auto_pull = false;
        else if (log_arg(a, next)) {}
        else die("unknown argument %s for `tachyon serve` (model server). To serve one file: tachyon serve -m model.tach", a.c_str());
    }
    return manager_main(o);
}

int runner_run(int argc, char** argv) { return run_cli(argc, argv); }

int runner_main(const std::string& cmd, int argc, char** argv) {
    Endpoint ep = endpoint();
    if (cmd == "pull") {
        bool keep = false;
        std::string device = "auto";
        std::vector<std::string> names;
        for (int i = 0; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--keep-download") keep = true;
            else if (a == "--device" && i + 1 < argc) device = argv[++i];
            else names.push_back(a);
        }
        if (names.empty()) die("usage: tachyon pull <model>");
        for (auto& n : names) if (do_pull(n, keep, device) != 0) return 1;
        return 0;
    }
    if (cmd == "list" || cmd == "ls") {
        std::vector<std::vector<std::string>> rows = {{"NAME", "SIZE", "QUANT", "SOURCE", "MODIFIED"}};
        for (auto& m : list_local()) {
            struct stat st;
            stat(m.manifest.c_str(), &st);
            rows.push_back({m.name, human((double)m.size), m.quant, m.source, ago(difftime(time(nullptr), st.st_mtime)) + " ago"});
        }
        print_table(rows);
        return 0;
    }
    if (cmd == "catalog") {
        std::vector<std::vector<std::string>> rows = {{"NAME", "HUGGING FACE REPO", "QUANT", "", "DESCRIPTION"}};
        for (auto& c : kCatalog) {
            LocalModel lm;
            rows.push_back({c.name, c.repo, c.quant, find_local(c.name, lm) ? "pulled" : "", c.about});
        }
        print_table(rows);
        return 0;
    }
    if (cmd == "rm") {
        if (argc < 1) die("usage: tachyon rm <model>");
        int rc = 0;
        for (int i = 0; i < argc; i++) {
            LocalModel m;
            if (!find_local(argv[i], m) || m.manifest.empty()) { loge("model '%s' not found", argv[i]); rc = 1; continue; }
            if (manager_up(ep)) {   // unload first
                Json b = Json::object(); b.set("model", m.name);
                http_call(ep.host, ep.port, "POST", "/api/stop", b.dump(), nullptr, nullptr, 10);
            }
            Manager::remove_model(m);
            printf("deleted '%s'\n", m.name.c_str());
        }
        return rc;
    }
    if (cmd == "ps") {
        std::string body;
        if (http_call(ep.host, ep.port, "GET", "/api/ps", "", [&](const std::string& d) { body += d; return true; }, nullptr, 5) != 200) {
            loge("tachyon server is not running (start: tachyon serve)");
            return 1;
        }
        std::vector<std::vector<std::string>> rows = {{"NAME", "SIZE", "STATUS", "PID", "UNTIL"}};
        for (auto& m : Json::parse(body)["models"].arr) {
            double u = m.get_num("unload_in_seconds", -1);
            rows.push_back({m.get_str("name", ""), human(m.get_num("size", 0)), m.get_str("status", ""), std::to_string(m.get_int("pid", 0)),
                            u < 0 ? (m.get_str("status", "") == "busy" ? "busy" : "forever") : ago(u) + " from now"});
        }
        print_table(rows);
        return 0;
    }
    if (cmd == "stop") {
        if (argc < 1) die("usage: tachyon stop <model>");
        Json b = Json::object();
        b.set("model", argv[0]);
        std::string body;
        int st = http_call(ep.host, ep.port, "POST", "/api/stop", b.dump(), [&](const std::string& d) { body += d; return true; }, nullptr, 30);
        if (st != 200) { loge("%s", st < 0 ? "tachyon server is not running" : Json::parse(body)["error"].get_str("message", body).c_str()); return 1; }
        return 0;
    }
    if (cmd == "show") {
        if (argc < 1) die("usage: tachyon show <model>");
        LocalModel m;
        if (!find_local(argv[0], m)) die("model '%s' not found", argv[0]);
        ModelFile mf;
        mf.open(m.path);
        const ModelConfig& c = mf.config;
        const Json& t = mf.header()["tokenizer"];
        printf("  name           %s\n  file           %s (%s)\n", m.name.c_str(), m.path.c_str(), human((double)m.size).c_str());
        if (!m.source.empty()) printf("  source         https://huggingface.co/%s\n", m.source.c_str());
        printf("  architecture   %s\n  weights        %s\n  layers         %d (hidden %d)\n  context        %d\n  vocab          %d\n",
               c.arch.c_str(), qtype_name(c.wtype), c.n_layers, c.hidden, c.max_ctx, c.vocab);
        printf("  chat format    %s%s\n", t.get_str("chat_format", "chatml").c_str(), t.get_bool("thinking", false) ? " (thinking)" : "");
        return 0;
    }
    if (cmd == "import" || cmd == "cp") {
        if (argc < 2) die("usage: tachyon import <file.tach | hf_dir> <name> [-q q4|q8]");
        std::string src = argv[0], name = lower(argv[1]), quant = "q4";
        for (int i = 2; i + 1 < argc; i++) if (std::string(argv[i]) == "-q") quant = argv[i + 1];
        char abs[4096];
        if (!realpath(src.c_str(), abs)) die("%s not found", src.c_str());
        src = abs;
        if (is_file(src)) {
            ModelFile mf;
            mf.open(src);   // validates the file
            write_manifest(name, src, "", qtype_name(mf.config.wtype));
            printf("imported %s -> %s\n", src.c_str(), name.c_str());
            return 0;
        }
        if (!is_dir(src)) die("%s is neither a .tach file nor a directory", src.c_str());
        const std::string out = models_dir() + "/" + slug(name) + ".tach";
        mkdirs(models_dir());
        int rc;
        if (quant == "q8") {
            std::vector<std::string> a = {src, out, "-q", "q8"};
            std::vector<char*> av;
            for (auto& x : a) av.push_back((char*)x.c_str());
            rc = convert_main((int)av.size(), av.data());
        } else {
            rc = convert_calibrated(src, out, {"-q", quant}, kCalibText, "auto", 40);
            remove((out + ".imatrix").c_str());
        }
        if (rc != 0) return rc;
        write_manifest(name, out, src, quant);
        printf("imported %s -> %s\n", src.c_str(), name.c_str());
        return 0;
    }
    runner_usage();
    return 1;
}

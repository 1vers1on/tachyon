// Full-vocabulary logits for externally supplied token IDs; no sampling or prompt-template differences.
#include "backend.h"
#include <algorithm>
#include <sstream>
#include <climits>
#include <cerrno>
#include <memory>
int g_log_level = 0;
int main(int argc, char** argv) {
    if (argc < 7 || argc > 8) return 2;
    auto positive = [](const char* text) {
        char* end = nullptr;
        errno = 0;
        long value = strtol(text, &end, 10);
        return !errno && end != text && !*end && value > 0 && value <= INT_MAX ? int(value) : 0;
    };
    const int chunk = positive(argv[4]), ctx = positive(argv[5]);
    const std::string device = argv[6];
    if (!chunk || !ctx || (device != "cuda" && device != "cpu") ||
        (argc == 8 && std::string(argv[7]) != "q8")) {
        fprintf(stderr, "expected positive CHUNK/CTX, cuda|cpu, and optional q8\n");
        return 2;
    }
    // Replay the same sequence with several chunk sizes while keeping weights resident.
    std::vector<int> chunks{chunk};
    if (const char* list = getenv("PROBE_CHUNKS")) {
        chunks.clear();
        std::stringstream stream(list);
        std::string part;
        while (std::getline(stream, part, ',')) {
            int n = positive(part.c_str());
            if (!n) return 2;
            chunks.push_back(n);
        }
        if (chunks.empty()) return 2;
    }
    ModelFile mf; mf.open(argv[1]);
    std::string raw = read_file(argv[2]);
    if (raw.empty() || raw.size() % 4) return 2;
    std::vector<int> ids(raw.size() / 4);
    memcpy(ids.data(), raw.data(), raw.size());
    auto backend = std::unique_ptr<Backend>(device == "cuda" ? create_cuda_backend() : create_cpu_backend());
    BackendOptions opt;
    opt.ctx = ctx; opt.max_batch = std::max(*std::max_element(chunks.begin(), chunks.end()), 32); opt.n_slots = 1; opt.threads = 8;
    opt.use_graphs = false; opt.state_rollback = getenv("PROBE_ROLLBACK") != nullptr; opt.kv_int8 = argc == 8;
    backend->init(mf, opt);
    for (int chunk : chunks) {
        std::string output = argv[3];
        if (getenv("PROBE_CHUNKS")) output += ".chunk" + std::to_string(chunk);
        FILE* f = fopen(output.c_str(), "wb");
        if (!f) return 2;
        std::vector<float> logits(mf.config.vocab);
        for (size_t p = 0; p < ids.size(); p += chunk) {
            std::vector<BatchToken> batch;
            for (size_t i = p; i < std::min(p + chunk, ids.size()); i++) batch.push_back({ids[i], 0, int(i), true});
            backend->forward(batch);
            for (int i = 0; i < (int)batch.size(); i++) {
                backend->get_logits(i, logits.data());
                if (fwrite(logits.data(), 4, logits.size(), f) != logits.size()) { fclose(f); return 2; }
            }
        }
        if (fclose(f)) return 2;
    }
    return 0;
}

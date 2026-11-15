// Emit every logits row for a deterministic sequence, including across ring-buffer wraps.
#include "backend.h"
#include <memory>

int g_log_level = 0;

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) return 2;
    ModelFile mf; mf.open(argv[1]);
    const bool cuda = argc == 6 && std::string(argv[5]) == "cuda";
    const int chunk = atoi(argv[3]), n = atoi(argv[4]);
    if (chunk <= 0 || n <= 0 || n > 128) return 2;
    std::unique_ptr<Backend> b(cuda ? create_cuda_backend() : create_cpu_backend());
    BackendOptions opt;
    opt.ctx = 128; opt.max_batch = std::max(4, chunk); opt.threads = 2; opt.n_slots = 2;
    b->init(mf, opt);
    FILE* f = fopen(argv[2], "wb");
    if (!f) return 2;
    std::vector<float> logits(mf.config.vocab);
    for (int pos = 0; pos < n; pos += chunk) {
        std::vector<BatchToken> batch;
        for (int i = pos; i < n && i < pos + chunk; i++) batch.push_back({65 + i % 23, 0, i, true});
        b->forward(batch);
        for (size_t i = 0; i < batch.size(); i++) {
            b->get_logits(i, logits.data());
            if (fwrite(logits.data(), sizeof(float), logits.size(), f) != logits.size()) return 2;
        }
    }
    return fclose(f) ? 2 : 0;
}

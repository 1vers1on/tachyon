// Check compact candidates against full logits, including overflow and >16 logits rows.
#include "backend.h"
#include <memory>
#include <algorithm>
#include <cassert>
#include <chrono>

int g_log_level = 0;

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ModelFile mf; mf.open(argv[1]);
    std::unique_ptr<Backend> b(create_cuda_backend());
    if (!b) { fprintf(stderr, "%s\n", cuda_backend_error().c_str()); return 1; }
    BackendOptions opt; opt.ctx = 128; opt.max_batch = 64;
    b->init(mf, opt);
    for (int rows : {1, 4, 32, 64}) {
        std::vector<BatchToken> batch;
        for (int i = 0; i < rows; i++) batch.push_back({65, 0, i, true});
        b->forward(batch);
        std::vector<int> greedy(rows);
        b->argmax_all(greedy.data());
        std::vector<std::vector<std::pair<float, int>>> out;
        assert(!b->candidates(rows, 1e9f, 1, out));
        assert(b->candidates(rows, 0.f, 1024, out));
        std::vector<float> logits(b->vocab());
        for (int row = 0; row < rows; row++) {
            b->get_logits(row, logits.data());
            const auto winner = std::max_element(logits.begin(), logits.end());
            assert(greedy[row] == winner - logits.begin());
            const float mx = *winner;
            std::vector<std::pair<float, int>> expected;
            for (int t = 0; t < b->vocab(); t++) if (logits[t] >= mx) expected.push_back({logits[t], t});
            std::sort(out[row].begin(), out[row].end());
            assert(out[row] == expected);
        }
        const bool complete = b->candidates(rows, 1e9f, 1024, out);
        assert(complete == (b->vocab() <= 1024));
        if (complete) for (const auto& row : out) assert((int)row.size() == b->vocab());
        assert(!b->candidates(rows, 0.f, 0, out));
        assert(!b->candidates(rows + 1, 0.f, 1024, out));
    }
    puts("CUDA sampling candidates: exact, overflow and 64-row checks passed");
}

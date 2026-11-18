#include "cpu_moe.h"
#include <cassert>

int g_log_level = 0;
void cpu_selftest();

// Two diagonal Q8_0 experts exercise the same host path used when GPU memory fills.
static void test_packed_expert_offload() {
    constexpr int width = 64, experts = 2, tokens = 10;
    auto diagonal = [](std::vector<uint8_t>& weights, bool gate_up) {
        TensorInfo t;
        t.type = QType::Q8_0;
        t.cols = width;
        t.rows = experts * width * (gate_up ? 2 : 1);
        const size_t stride = qtype_data_bytes(t.type, 1, width);
        weights.resize(t.rows * stride);
        for (int r = 0; r < t.rows; r++) {
            const int feature = (gate_up ? r / 2 : r) % width;
            uint8_t* block = weights.data() + r * stride + (feature / 32) * 34;
            const uint16_t scale = f2h(feature < 32 ? 0.25f : 0.5f);
            memcpy(block, &scale, 2);
            const int expert = r / (t.rows / experts);
            block[2 + feature % 32] = gate_up ? (r % 2 ? -4 : 4) : 4 * (expert + 1);
        }
        t.data = weights.data();
        t.data_bytes = weights.size();
        return t;
    };
    CpuMoe moe(2);
    std::vector<uint8_t> gu, down;
    int layer = moe.add_layer(diagonal(gu, true), diagonal(down, false), experts, false, "test");
    uint16_t input[tokens * width];
    int ids[tokens];
    float routing[tokens], output[tokens * width];
    for (int m = 0; m < tokens; m++) {
        ids[m] = m % experts;
        routing[m] = 1.0f;
        for (int j = 0; j < width; j++) input[m * width + j] = f2h((m + j % 7 - 8) * 0.125f);
    }
    moe.forward(layer, tokens, 1, input, ids, routing, output);
    for (int i = 0; i < tokens * width; i++) {
        const float factor = i % width < 32 ? 1.0f : 2.0f;
        const float x = h2f(input[i]) * factor;
        const float expected = -x * x / (1.0f + std::exp(-x)) * factor * (1 + ids[i / width]);
        assert(std::abs(output[i] - expected) < 1e-5f);
    }
}

int main() {
    cpu_selftest();
    test_packed_expert_offload();
}

#include "quant.h"
#include "threadpool.h"
#include <algorithm>
#include <cassert>

int g_log_level = 0;

int main() {
    constexpr int rows = 37, cols = 128;
    std::vector<float> weights(rows * cols), importance(cols);
    uint32_t state = 0x12345678;
    for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) {
        state = state * 1664525u + 1013904223u;
        float v = ((int)(state >> 16) % 511 - 255) / 256.f;
        switch (r % 6) {
            case 0: v = (r / 6 - 2) * .25f; break;       // constant groups
            case 1: v = (k % 16) / 16.f - .5f; break;   // exactly representable Q4 grid
            case 2: v *= 0x1p-20f; break;              // fp16 subnormal parameters
            case 3: v = k % 2 ? -1.f : 1.f; break;
            case 4: v += 2.f; break;
            default: break;
        }
        weights[r * cols + k] = v;
    }
    for (int k = 0; k < cols; k++) importance[k] = k % 7 ? ((k * 13) % 31 + 1) / 8.f : 0.f;

    struct Case { QType type; bool weighted; uint64_t hash; };
    // Frozen outputs from the original scalar quantizer, identical with and without FMA/F16C.
    const Case cases[] = {
        {QType::Q4, false, 0xaa5f849649825e03ULL}, {QType::Q4, true, 0xc2450bc0d519ab2bULL},
        {QType::Q3, false, 0xf81805eef3dcee14ULL}, {QType::Q3, true, 0x0f788561080bab6eULL},
        {QType::Q8, false, 0xdd51af6b6be7bf27ULL}, {QType::F16, false, 0x18dab747c5ca9210ULL},
        {QType::F32, false, 0x6c053cb973097718ULL},
    };
    for (const auto& test : cases) {
        std::vector<uint8_t> expected, expected_scales;
        const float* imp = test.weighted ? importance.data() : nullptr;
        auto source = [&](int64_t r, float* b) { memcpy(b, weights.data() + r * cols, cols * sizeof(float)); };
        quantize_matrix(test.type, rows, cols, source, expected, expected_scales, 1, imp);
        assert(fnv1a64(expected_scales.data(), expected_scales.size(), fnv1a64(expected.data(), expected.size())) == test.hash);

        // Reuse both spinning and sleeping pools across full, partial, single-row and empty jobs.
        for (int spin_limit : {0, 200000}) {
            ThreadPool pool(4, spin_limit);
            std::vector<uint8_t> data, scales;
            for (int n : {rows, 1, 17, 0, rows}) {
                std::vector<std::atomic<int>> seen(n);
                auto counted_source = [&](int64_t r, float* b) {
                    assert(r >= 0 && r < n);
                    seen[r].fetch_add(1, std::memory_order_relaxed);
                    source(r, b);
                };
                data.assign(8, 0xff); scales.assign(8, 0xff);
                quantize_matrix(test.type, n, cols, counted_source, data, scales, pool, imp);
                for (const auto& count : seen) assert(count.load() == 1);
                assert(data.size() == qtype_data_bytes(test.type, n, cols));
                assert(scales.size() == qtype_scale_bytes(test.type, n, cols));
                assert(std::equal(data.begin(), data.end(), expected.begin()));
                assert(std::equal(scales.begin(), scales.end(), expected_scales.begin()));
            }
        }
        // The convenience overload also caps workers for short matrices.
        std::vector<uint8_t> data, scales;
        quantize_matrix(test.type, rows, cols, source, data, scales, 32, imp);
        assert(data == expected && scales == expected_scales);
    }

    // Verify the packed integer planes and zero-scale constant groups independently of hashes.
    for (QType type : {QType::Q3, QType::Q4}) {
        float input[Q4_GROUP], restored[Q4_GROUP];
        uint8_t data[Q4_GROUP / 2], scales[4];
        int qmax = type == QType::Q3 ? 7 : 15;
        for (int i = 0; i < Q4_GROUP; i++) input[i] = .125f * (i % (qmax + 1)) - .5f;
        quantize_row(type, input, Q4_GROUP, data, scales);
        dequantize_row(type, data, scales, Q4_GROUP, restored);
        assert(memcmp(input, restored, sizeof input) == 0);
        for (int i = 0; i < Q4_GROUP; i++) input[i] = -.5f;
        quantize_row(type, input, Q4_GROUP, data, scales, importance.data());
        dequantize_row(type, data, scales, Q4_GROUP, restored);
        assert(memcmp(input, restored, sizeof input) == 0);
    }
    puts("quantization: frozen outputs, packed values and reusable thread pools passed");
}

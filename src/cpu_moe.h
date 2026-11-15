// Tachyon - CPU-side mixture-of-experts FFN (experts kept in system RAM, used by the CUDA backend for models
// whose experts do not fit in VRAM). Inputs are already normalized on the GPU; routing is done there too.
#pragma once
#include "common.h"
#include "model_file.h"
#include <memory>

class CpuMoe {
public:
    explicit CpuMoe(int threads);
    ~CpuMoe();
    // Copies the expert matrices of one layer into anonymous memory (Q3 widened to Q4). Returns a layer handle.
    // wgu_e: [E*2F][H] (gate/up interleaved), wdown_e: [E*H][F].
    int add_layer(const TensorInfo& wgu_e, const TensorInfo& wdown_e, int n_experts, bool gelu, const std::string& name_prefix);
    // out[m][:] = sum_k w[m][k] * expert_{ids[m][k]}(xn[m][:])   (xn: normalized fp16 activations [M][H])
    void forward(int layer, int M, int K, const uint16_t* xn, const int* ids, const float* w, float* out);
    size_t bytes() const;
    // host copies of one layer's experts (for streaming them to the GPU): which 0 = wgu_e, 1 = wdown_e
    struct HostMat { QType t; int rows, cols; const uint8_t* d; const uint8_t* s; size_t db, sb; };
    HostMat mat(int layer, int which) const;
    // anonymous memory regions holding the expert weights (to page-lock them)
    std::vector<std::pair<void*, size_t>> regions() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

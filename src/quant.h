// Tachyon - weight quantization (converter side) and reference dequantization
#pragma once
#include "common.h"

// Quantize one row (cols elements) into dst_data / dst_scales of the given type.
// For Q4: dst_data gets cols/2 bytes, dst_scales cols/64 half2 (4 bytes each)
// For Q8: dst_data gets cols bytes,   dst_scales cols/32 halves
void quantize_row(QType t, const float* src, int64_t cols, uint8_t* dst_data, uint8_t* dst_scales, const float* imp = nullptr);
void dequantize_row(QType t, const uint8_t* data, const uint8_t* scales, int64_t cols, float* dst);

// Multithreaded quantization of a [rows, cols] matrix. src_row(r, buf) fills a row of floats.
#include <functional>
class ThreadPool;
// Reuse a pool when converting multiple tensors.
void quantize_matrix(QType t, int64_t rows, int64_t cols,
                     const std::function<void(int64_t, float*)>& src_row,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& scales, ThreadPool& pool, const float* imp = nullptr);
void quantize_matrix(QType t, int64_t rows, int64_t cols,
                     const std::function<void(int64_t, float*)>& src_row,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& scales, int nthreads, const float* imp = nullptr);

// ---- importance matrix (per-input-channel mean of x^2, collected on calibration text) ----
#include <map>
#include <mutex>
struct IMatrix {
    std::mutex mu;
    std::map<std::string, std::pair<std::vector<double>, int64_t>> acc;  // sum x^2, count
    void add(const std::string& name, const float* x, int M, int K);
    void save(const std::string& path);
    static std::map<std::string, std::vector<float>> load(const std::string& path);
};
extern IMatrix* g_imatrix;

// Lossless layout conversion: blocks of 128 weights, 28-byte PTQ1 -> 34-byte PQ2.
// src/dst must not overlap. Scales are copied bit for bit.
void repack_ptq1_to_pq2(const uint8_t* src, uint8_t* dst, size_t blocks);

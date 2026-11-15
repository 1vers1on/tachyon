// Tachyon - .tach model file format (single mmap-able file: JSON header + aligned blobs)
//
//   [0..8)   magic "TACHYON1"
//   [8..16)  u64 header_len
//   [16..16+header_len) JSON header
//   ... zero padding to DATA_ALIGN ...
//   data blobs (each aligned to DATA_ALIGN), offsets in header are relative to data start
#pragma once
#include "common.h"
#include "json.h"
#include <map>
#include <string>

constexpr size_t DATA_ALIGN = 4096;

struct TensorInfo {
    std::string name;
    QType type = QType::F32;
    int64_t rows = 0, cols = 0;   // 1-D tensors: rows=1
    uint64_t offset = 0;          // data offset (relative to data start)
    uint64_t data_bytes = 0;
    uint64_t scale_offset = 0;    // scales (Q4/Q8)
    uint64_t scale_bytes = 0;
    const uint8_t* data = nullptr;    // resolved pointers (mmap)
    const uint8_t* scales = nullptr;
};

struct BlobInfo {
    std::string name;
    uint64_t offset = 0, size = 0;
    const uint8_t* data = nullptr;
};

struct ModelConfig {
    std::string arch;          // qwen3 | qwen2 | llama
    std::string name;
    int n_layers = 0, hidden = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, ffn = 0, vocab = 0;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.f;
    bool qk_norm = false, attn_bias = false, tied_embed = false;
    // llama3-style rope scaling
    std::string rope_scaling;  // "" | "llama3" | "linear" | "yarn"
    float rope_factor = 1.f, rope_low_freq = 1.f, rope_high_freq = 4.f;
    int rope_orig_ctx = 8192;
    int max_ctx = 4096;
    QType wtype = QType::Q4;   // main weight type
    Json to_json() const;
    static ModelConfig from_json(const Json& j);
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
};

class ModelFile {
public:
    ~ModelFile();
    void open(const std::string& path);
    const TensorInfo* tensor(const std::string& name, bool required = true) const;
    const BlobInfo* blob(const std::string& name, bool required = true) const;
    const Json& header() const { return header_; }
    ModelConfig config;
    std::string path;
    size_t file_size = 0;
    std::map<std::string, TensorInfo> tensors;
    std::map<std::string, BlobInfo> blobs;
private:
    Json header_;
    void* map_ = nullptr;
};

// Streaming writer used by the converter.
class ModelWriter {
public:
    void begin(const std::string& path);
    // reserve a slot; returns offset (relative to data start)
    uint64_t write_blob(const void* data, size_t n);
    void add_tensor(const TensorInfo& t) { tensors_.push_back(t); }
    void add_blob(const BlobInfo& b) { blobs_.push_back(b); }
    void finish(Json header);  // header gets "tensors"/"blobs" added
private:
    std::string path_, tmp_path_;
    FILE* f_ = nullptr;
    uint64_t pos_ = 0;
    std::vector<TensorInfo> tensors_;
    std::vector<BlobInfo> blobs_;
};

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
#include <algorithm>
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

// Per-layer description. Old files without "layers" get uniform attention layers from the globals.
struct LayerCfg {
    std::string type = "attn";   // attn | conv (LFM2 short conv) | deltanet (Qwen3.5 gated delta net)
    // --- attention ---
    int n_heads = 0, n_kv_heads = 0, head_dim = 0;
    float rope_theta = 10000.f;
    int rope_half = 0;           // rotation pairs are (i, i + rope_half); dims >= 2*rope_half untouched
    int rope_rot = 0;            // number of frequencies actually rotating (proportional rope: rest are 0)
    int window = 0;              // sliding window (0 = full causal)
    int qk_norm = 0;             // 0 none, 1 per head, 2 over the full q / k vectors (OLMoE)
    bool v_norm = false;         // Gemma 4: unweighted RMSNorm on values
    bool attn_gate = false;      // Qwen3.5: q_proj also emits a per-head output gate (sigmoid)
    bool k_eq_v = false;         // Gemma 4 global layers: v = raw k projection
    float attn_scale = 0.f;      // softmax scale (0 = 1/sqrt(head_dim))
    int kv_src = -1;             // Gemma 4: reuse the KV cache of this (earlier) layer
    // --- feed forward ---
    int ffn = 0;                 // dense gated FFN width (0 = none)
    int n_experts = 0, top_k = 0, expert_ffn = 0;
    std::string router = "softmax";   // softmax | sigmoid
    bool norm_topk = false;
    float routed_scale = 1.f;
    bool expert_bias = false;
    bool sandwich = false;       // Gemma: post-attention and post-FFN norms
    float layer_scalar = 1.f;    // Gemma 4: output multiplier
    // --- short conv (LFM2) ---
    int conv_k = 0;
    // --- gated delta net (Qwen3.5) ---
    int dn_nk = 0, dn_nv = 0, dn_dk = 0, dn_dv = 0, dn_conv = 0;
    bool is_attn() const { return type == "attn"; }
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int qkv_rows() const { return q_dim() * (attn_gate ? 2 : 1) + (kv_src >= 0 ? 0 : (k_eq_v ? 1 : 2) * kv_dim()); }
    int dn_key_dim() const { return dn_nk * dn_dk; }
    int dn_val_dim() const { return dn_nv * dn_dv; }
    int dn_conv_dim() const { return 2 * dn_key_dim() + dn_val_dim(); }
    int dn_in_rows() const { return dn_conv_dim() + dn_val_dim() + 2 * dn_nv; }
    Json to_json() const;
    static LayerCfg from_json(const Json& j);
};

struct ModelConfig {
    std::string arch;          // qwen3 | qwen2 | llama | gemma3 | gemma4 | lfm2 | lfm2_moe | olmoe | qwen3_5
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
    // generalized model description
    std::string act = "silu";  // silu | gelu (tanh approximation)
    float embed_scale = 1.f;
    float final_softcap = 0.f;
    bool stateful = false;     // has recurrent state (conv / delta net): no KV rollback
    std::vector<LayerCfg> layers;
    int max_head_dim() const { int m = 0; for (auto& l : layers) if (l.is_attn()) m = std::max(m, l.head_dim); return m; }
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

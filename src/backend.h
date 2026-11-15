// Tachyon - backend interface (CPU and CUDA implement this)
#pragma once
#include "common.h"
#include "model_file.h"

struct BatchToken {
    int token;
    int slot;   // KV-cache slot (sequence)
    int pos;    // position within the sequence
    bool logits;  // produce logits for this token
};

struct BackendOptions {
    int n_slots = 1;     // number of concurrent sequences (KV cache slots)
    int ctx = 4096;      // context length per slot
    int threads = 0;     // CPU threads (0 = auto)
    int max_batch = 512; // max tokens per forward call (prefill chunk)
    bool kv_int8 = false;  // experimental 8-bit KV cache (CUDA)
    bool use_graphs = true;
    int device = 0;
    size_t reserve_bytes = 0;      // GPU memory to leave free (e.g. draft model weights)
    size_t extra_kv_per_tok = 0;   // additional KV bytes per token per slot to budget (draft model KV)
    int vocab_limit = 0;   // >0: only compute logits for token ids < vocab_limit (draft models)
};

class Backend {
public:
    virtual ~Backend() {}
    virtual const char* name() const = 0;
    virtual void init(const ModelFile& mf, const BackendOptions& opt) = 0;
    // Run the model on a batch of tokens. Rows of logits are produced (in order) for tokens with .logits set.
    virtual void forward(const std::vector<BatchToken>& batch) = 0;
    // Copy logits row i (index among tokens with .logits) to host buffer (vocab floats).
    virtual void get_logits(int i, float* dst) = 0;
    // Greedy argmax for all logits rows of the last forward. Optional penalty: tokens list with penalized logits.
    virtual void argmax_all(int* out) = 0;
    // Copy the first n positions of the KV cache from one slot to another (prefix sharing).
    virtual void copy_kv(int src_slot, int dst_slot, int n) = 0;
    virtual int vocab() const = 0;
    // Candidate logits (value, token) with logit >= rowmax - delta for logits rows [0, n). Returns false if
    // unsupported (caller falls back to get_logits). Rows whose candidate count overflows `cap` are truncated.
    virtual bool candidates(int n, float delta, int cap, std::vector<std::vector<std::pair<float, int>>>& out) { return false; }
    virtual void imat_flush() {}  // move device-side importance statistics into g_imatrix
    BackendOptions opt;
};

Backend* create_cpu_backend();
Backend* create_cuda_backend();

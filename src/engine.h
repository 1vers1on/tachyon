// Tachyon - inference engine: request scheduling, continuous batching, prefix caching, speculative decoding
#pragma once
#include "common.h"
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <functional>
#include <atomic>

struct GenRequest {
    std::vector<int> prompt;
    SamplingParams sp;
    int max_tokens = 512;
    std::vector<std::string> stop;
    bool ignore_eos = false;
    // Called on the engine thread with each new chunk of text. Return false to cancel.
    std::function<bool(const std::string& delta)> on_text;

    // ---- results (valid after done) ----
    std::string text;
    std::string finish_reason;  // stop | length | cancelled | error
    std::string error;
    int n_prompt = 0, n_cached = 0, n_gen = 0;
    int n_draft = 0, n_draft_accepted = 0;
    double t_submit = 0, t_first = 0, t_done = 0, t_prefill_done = 0;

    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return done; });
    }
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::atomic<bool> cancel{false};
};

struct EngineOptions {
    BackendOptions backend;
    std::string device = "auto";  // auto | cuda | cpu
    int spec_draft = -1;          // n-gram speculative tokens per step (-1 = auto)
    int prefill_chunk = 0;        // 0 = auto (512; 2048 when MoE experts are likely streamed from system RAM)
    std::string draft_model;      // optional small model for speculative decoding (same tokenizer)
    int draft_k = 4;              // initial draft length (adapts per slot)
    int draft_vocab = 0;          // >0: draft model scores only token ids below this (experimental)
};

class Engine {
public:
    ~Engine();
    void load(const std::string& model_path, const EngineOptions& o);
    void start();   // background scheduling thread
    void stop();
    void submit(std::shared_ptr<GenRequest> r);
    // run one request synchronously (no background thread needed)
    void run_sync(std::shared_ptr<GenRequest> r);

    Tokenizer tok;
    ModelFile mf;
    std::unique_ptr<Backend> be;
    std::unique_ptr<Backend> dbe;   // draft model backend (optional)
    ModelFile dmf;
    EngineOptions opt;
    std::string model_name;
    Json gen_defaults;

private:
    struct Slot {
        std::shared_ptr<GenRequest> req;
        std::vector<int> cache;   // tokens whose KV entries are valid in this slot
        std::vector<int> hist;    // full token history (prompt + generated) of the active request
        int n_prompt_done = 0;    // prompt tokens already in KV
        int last = -1;            // last sampled token (not yet in KV)
        bool generating = false;
        Sampler sampler;
        std::unique_ptr<StreamDecoder> dec;
        std::string pending_text;  // decoded but not yet emitted (stop-string holdback)
        int spec_k = 0;
        std::vector<int> dcache;  // tokens whose KV is valid in the draft model's slot
        int dk = 0;               // current draft-model depth
        int dsaved = -1;          // stateful draft model: dcache length at its last state backup
        int spec_miss = 0;
        double last_used = 0;
    };
    void loop();
    bool step();  // returns false if idle
    void admit();
    void finish(Slot& s, const std::string& reason);
    bool emit_token(Slot& s, int tok);  // returns false when the request is finished
    std::vector<int> draft_ngram(const Slot& s, int k, int min_n) const;
    void draft_with_model(const std::vector<std::pair<int, int>>& need, std::vector<std::vector<int>>& out);

    std::vector<Slot> slots_;
    std::deque<std::shared_ptr<GenRequest>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread th_;
    bool running_ = false;
    std::vector<float> logits_buf_;
};

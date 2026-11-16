// Tachyon - token sampling
#pragma once
#include "common.h"
#include "json.h"
#include <random>
#include <unordered_map>
#include <deque>

struct SamplingParams {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float typical_p = 1.0f;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int repeat_last_n = -1;  // -1: all history, 0: disabled, >0: rolling token window
    bool penalize_prompt = true;
    std::unordered_map<int, float> logit_bias;
    uint64_t seed = 0;  // 0 = random
    bool seed_set = false;  // an explicit seed, including zero
    bool greedy() const { return temperature <= 0.f || top_k == 1; }
    bool has_penalties() const { return !logit_bias.empty() || (repeat_last_n != 0 && (repetition_penalty != 1.f || presence_penalty != 0.f || frequency_penalty != 0.f)); }
};

// Shared defaults / request controls. update() validates only recognized keys unless strict is true.
// Ollama aliases (repeat_penalty, num_predict, draft_num_predict) are accepted too.
struct InferenceSettings {
    SamplingParams sp;
    int max_tokens = -1;
    std::vector<std::string> stop;
    bool ignore_eos = false, cache_prompt = true;
    int speculative_tokens = -1;  // -1 server default; 0 disables both n-gram and model drafting
    std::string update(const Json& values, bool strict = false);
    std::string validate_vocab(int vocab) const;
    Json to_json() const;
};

// Shared CLI flags and JSON configuration; flags/config files apply in command-line order.
bool inference_arg(const std::string& arg, const std::function<std::string()>& next, Json& overrides);

class Sampler {
public:
    void init(const SamplingParams& p);
    // apply penalties in-place to a full logits row
    void apply_penalties(float* logits, int n_vocab) const;
    // sample from a logits row (penalties must already be applied)
    int sample(const float* logits, int n_vocab);
    // speculative verification of a deterministic draft token: returns the sampled token; it equals `draft`
    // when the draft is accepted. (Rejection sampling for a point-mass proposal.)
    int sample_with_draft(const float* logits, int n_vocab, int draft);
    // same, from a pre-filtered candidate list (logit, token) produced on the GPU
    int sample_cands(std::vector<std::pair<float, int>>& c);
    int sample_cands_with_draft(std::vector<std::pair<float, int>>& c, int draft);
    void accept(int token);
    SamplingParams params;
private:
    void candidates(const float* logits, int n_vocab);
    void finalize_candidates(float mx);
    int draw();
    int draw_with_draft(int draft);
    std::mt19937_64 rng_;
    std::unordered_map<int, int> counts_;
    std::deque<int> recent_;
    std::vector<std::pair<float, int>> cand_;  // (prob, id)
};

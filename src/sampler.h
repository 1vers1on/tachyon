// Tachyon - token sampling
#pragma once
#include "common.h"
#include <random>
#include <unordered_map>

struct SamplingParams {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    uint64_t seed = 0;  // 0 = random
    bool greedy() const { return temperature <= 0.f || top_k == 1; }
    bool has_penalties() const { return repetition_penalty != 1.f || presence_penalty != 0.f || frequency_penalty != 0.f; }
};

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
    void accept(int token) { counts_[token]++; }
    SamplingParams params;
private:
    void candidates(const float* logits, int n_vocab);
    void finalize_candidates(float mx);
    int draw();
    int draw_with_draft(int draft);
    std::mt19937_64 rng_;
    std::unordered_map<int, int> counts_;
    std::vector<std::pair<float, int>> cand_;  // (prob, id)
};

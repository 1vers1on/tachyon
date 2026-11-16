#include "sampler.h"
#include <algorithm>

void Sampler::init(const SamplingParams& p) {
    params = p;
    uint64_t seed = p.seed_set || p.seed ? p.seed : std::random_device{}() ^ ((uint64_t)std::random_device{}() << 32);
    rng_.seed(seed);
    counts_.clear();
}

void Sampler::apply_penalties(float* l, int n) const {
    for (auto& kv : counts_) {
        int id = kv.first;
        if (id < 0 || id >= n) continue;
        float& v = l[id];
        if (params.repetition_penalty != 1.f) v = v > 0 ? v / params.repetition_penalty : v * params.repetition_penalty;
        v -= params.presence_penalty + params.frequency_penalty * kv.second;
    }
}

// Build the truncated, normalized candidate distribution (after temperature/top-k/top-p/min-p).
// Speed trick: tokens whose logit is more than ~20 temperatures below the max have relative probability
// < 2e-9 and are never considered (no full-vocabulary sort or softmax).
void Sampler::candidates(const float* l, int n) {
    const float T = std::max(params.temperature, 1e-4f);
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) mx = std::max(mx, l[i]);
    float thr = mx - 20.f * T;
    cand_.clear();
    for (int i = 0; i < n; i++) if (l[i] >= thr) cand_.push_back({l[i], i});
    finalize_candidates(mx);
}

void Sampler::finalize_candidates(float mx) {
    const float T = std::max(params.temperature, 1e-4f);
    int k = params.top_k > 0 ? std::min<int>(params.top_k, (int)cand_.size()) : (int)cand_.size();
    if (k < (int)cand_.size()) {
        std::nth_element(cand_.begin(), cand_.begin() + k, cand_.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        cand_.resize(k);
    }
    std::sort(cand_.begin(), cand_.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    double sum = 0;
    for (auto& c : cand_) { c.first = expf((c.first - mx) / T); sum += c.first; }
    for (auto& c : cand_) c.first = (float)(c.first / sum);
    // min-p
    if (params.min_p > 0) {
        float lim = cand_[0].first * params.min_p;
        size_t j = 1;
        while (j < cand_.size() && cand_[j].first >= lim) j++;
        cand_.resize(j);
    }
    // top-p
    if (params.top_p < 1.f) {
        double c = 0;
        size_t j = 0;
        for (; j < cand_.size(); j++) { c += cand_[j].first; if (c >= params.top_p) { j++; break; } }
        cand_.resize(std::max<size_t>(1, std::min(j, cand_.size())));
    }
    double s2 = 0;
    for (auto& c : cand_) s2 += c.first;
    for (auto& c : cand_) c.first = (float)(c.first / s2);
}

int Sampler::sample_cands(std::vector<std::pair<float, int>>& c) {
    float mx = -INFINITY;
    for (auto& x : c) mx = std::max(mx, x.first);
    cand_.swap(c);
    finalize_candidates(mx);
    return draw();
}
int Sampler::sample_cands_with_draft(std::vector<std::pair<float, int>>& c, int draft) {
    float mx = -INFINITY;
    for (auto& x : c) mx = std::max(mx, x.first);
    cand_.swap(c);
    finalize_candidates(mx);
    return draw_with_draft(draft);
}

int Sampler::sample(const float* l, int n) {
    if (params.greedy()) return (int)(std::max_element(l, l + n) - l);
    candidates(l, n);
    return draw();
}

int Sampler::draw() {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    float r = u(rng_), c = 0;
    for (auto& x : cand_) { c += x.first; if (r < c) return x.second; }
    return cand_.back().second;
}

int Sampler::sample_with_draft(const float* l, int n, int draft) {
    if (params.greedy()) return (int)(std::max_element(l, l + n) - l);
    candidates(l, n);
    return draw_with_draft(draft);
}

int Sampler::draw_with_draft(int draft) {
    float pd = 0;
    for (auto& x : cand_) if (x.second == draft) pd = x.first;
    std::uniform_real_distribution<float> u(0.f, 1.f);
    if (pd > 0 && u(rng_) < pd) return draft;
    // residual distribution: p with the draft removed
    float tot = 0;
    for (auto& x : cand_) if (x.second != draft) tot += x.first;
    if (tot <= 0) return draft;
    float r = u(rng_) * tot, c = 0;
    for (auto& x : cand_) {
        if (x.second == draft) continue;
        c += x.first;
        if (r < c) return x.second;
    }
    for (auto it = cand_.rbegin(); it != cand_.rend(); ++it) if (it->second != draft) return it->second;
    return draft;
}

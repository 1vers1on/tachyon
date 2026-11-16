#include "sampler.h"
#include <algorithm>
#include <charconv>

void Sampler::init(const SamplingParams& p) {
    params = p;
    uint64_t seed = p.seed_set || p.seed ? p.seed : std::random_device{}() ^ ((uint64_t)std::random_device{}() << 32);
    rng_.seed(seed);
    counts_.clear();
    recent_.clear();
}

void Sampler::accept(int token) {
    if (params.repeat_last_n == 0) return;
    counts_[token]++;
    if (params.repeat_last_n > 0) {
        recent_.push_back(token);
        if (recent_.size() > (size_t)params.repeat_last_n) {
            const int old = recent_.front(); recent_.pop_front();
            auto it = counts_.find(old);
            if (--it->second == 0) counts_.erase(it);
        }
    }
}

void Sampler::apply_penalties(float* l, int n) const {
    for (auto& kv : counts_) {
        int id = kv.first;
        if (id < 0 || id >= n) continue;
        float& v = l[id];
        if (params.repetition_penalty != 1.f) v = v > 0 ? v / params.repetition_penalty : v * params.repetition_penalty;
        v -= params.presence_penalty + params.frequency_penalty * kv.second;
    }
    for (auto [id, bias] : params.logit_bias) if (id >= 0 && id < n) l[id] += bias;
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
    // Locally typical sampling: retain the requested mass nearest the distribution's entropy.
    if (params.typical_p < 1.f) {
        double entropy = 0;
        for (auto& c : cand_) if (c.first > 0) entropy -= c.first * log(c.first);
        auto distance = [&](float p) { return p > 0 ? fabs(-log(p) - entropy) : INFINITY; };
        std::sort(cand_.begin(), cand_.end(), [&](auto& a, auto& b) {
            double da = distance(a.first), db = distance(b.first);
            return da != db ? da < db : a.second < b.second;
        });
        double mass = 0; size_t keep = 0;
        do { mass += cand_[keep++].first; } while (keep < cand_.size() && mass < params.typical_p);
        cand_.resize(keep);
        std::sort(cand_.begin(), cand_.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        double retained = 0;
        for (auto& c : cand_) retained += c.first;
        for (auto& c : cand_) c.first = (float)(c.first / retained);
    }
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

std::string InferenceSettings::update(const Json& values, bool strict) {
    if (!values.is_obj()) return "inference settings must be an object";
    auto s = *this;
    for (const auto& [input, value] : values.obj) {
        if (!strict && value.is_null()) continue;  // optional API parameters may be explicitly null
        std::string key = input;
        if (key == "repeat_penalty") key = "repetition_penalty";
        if (key == "num_predict" || key == "max_completion_tokens") key = "max_tokens";
        if (key == "draft_num_predict") key = "speculative_tokens";
        bool found = false;
        for (auto [name, target] : std::vector<std::pair<const char*, float*>>{
            {"temperature", &s.sp.temperature}, {"top_p", &s.sp.top_p}, {"min_p", &s.sp.min_p}, {"typical_p", &s.sp.typical_p},
            {"repetition_penalty", &s.sp.repetition_penalty}, {"presence_penalty", &s.sp.presence_penalty}, {"frequency_penalty", &s.sp.frequency_penalty}}) {
            if (key != name) continue;
            if (!value.is_num() || !std::isfinite(value.num) || !std::isfinite((float)value.num)) return "invalid numeric setting: " + input;
            *target = (float)value.num; found = true; break;
        }
        if (found) continue;
        if (key == "top_k" || key == "repeat_last_n" || key == "max_tokens" || key == "speculative_tokens") {
            if (!value.is_num() || !std::isfinite(value.num) || value.num < -2 || value.num > INT32_MAX || value.num != value.as_int()) return "invalid integer setting: " + input;
            int n = (int)value.num;
            if (key == "top_k") { if (n < 0) return "top_k must be >= 0"; s.sp.top_k = n; }
            if (key == "repeat_last_n") { if (n < -1) return "repeat_last_n must be >= -1"; s.sp.repeat_last_n = n; }
            if (key == "max_tokens") s.max_tokens = n;
            if (key == "speculative_tokens") { if (n < -1 || n > 8) return "speculative_tokens must be between -1 and 8"; s.speculative_tokens = n; }
        } else if (key == "seed") {
            if (!value.is_num() || !std::isfinite(value.num) || value.num < -1 || value.num > 9007199254740991.0 || value.num != value.as_int()) return "seed must be an integer from -1 to 2^53-1";
            s.sp.seed_set = value.num >= 0; s.sp.seed = s.sp.seed_set ? (uint64_t)value.num : 0;
        } else if (key == "penalize_prompt" || key == "ignore_eos" || key == "cache_prompt") {
            if (!value.is_bool()) return input + " must be a boolean";
            if (key == "penalize_prompt") s.sp.penalize_prompt = value.b;
            if (key == "ignore_eos") s.ignore_eos = value.b;
            if (key == "cache_prompt") s.cache_prompt = value.b;
        } else if (key == "stop") {
            s.stop.clear();
            if (value.is_str()) s.stop.push_back(value.str);
            else if (value.is_arr()) { for (const auto& v : value.arr) { if (!v.is_str()) return "stop must contain strings"; s.stop.push_back(v.str); } }
            else return "stop must be a string or an array of strings";
        } else if (key == "logit_bias") {
            if (!value.is_obj()) return "logit_bias must be an object keyed by token id";
            s.sp.logit_bias.clear();
            for (const auto& [token, bias] : value.obj) {
                int id = -1;
                auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), id);
                if (ec != std::errc() || end != token.data() + token.size() || id < 0 || !bias.is_num() || !std::isfinite(bias.num) || bias.num < -100 || bias.num > 100)
                    return "logit_bias requires nonnegative token ids and biases between -100 and 100";
                s.sp.logit_bias[id] = (float)bias.num;
            }
        } else if (strict) return "unknown inference setting: " + input;
    }
    if (s.sp.temperature < 0 || s.sp.top_p < 0 || s.sp.top_p > 1 || s.sp.min_p < 0 || s.sp.min_p > 1 ||
        s.sp.typical_p <= 0 || s.sp.typical_p > 1 || s.sp.repetition_penalty <= 0) return "invalid sampling range";
    *this = std::move(s);
    return "";
}

std::string InferenceSettings::validate_vocab(int vocab) const {
    for (auto [id, bias] : sp.logit_bias) if (id >= vocab) return "logit_bias token id exceeds model vocabulary";
    return "";
}

Json InferenceSettings::to_json() const {
    Json j = Json::object();
    j.set("temperature", sp.temperature); j.set("top_k", sp.top_k); j.set("top_p", sp.top_p);
    j.set("min_p", sp.min_p); j.set("typical_p", sp.typical_p); j.set("repetition_penalty", sp.repetition_penalty);
    j.set("presence_penalty", sp.presence_penalty); j.set("frequency_penalty", sp.frequency_penalty);
    j.set("repeat_last_n", sp.repeat_last_n); j.set("penalize_prompt", sp.penalize_prompt);
    j.set("seed", sp.seed_set ? Json(sp.seed) : Json(-1));
    Json biases = Json::object();
    for (auto [id, bias] : sp.logit_bias) biases.set(std::to_string(id), bias);
    j.set("logit_bias", biases); j.set("max_tokens", max_tokens);
    Json stops = Json::array(); for (auto& text : stop) stops.push(text);
    j.set("stop", stops); j.set("ignore_eos", ignore_eos); j.set("cache_prompt", cache_prompt);
    j.set("speculative_tokens", speculative_tokens);
    return j;
}

bool inference_arg(const std::string& arg, const std::function<std::string()>& next, Json& overrides) {
    if (arg == "--sampling-config") {
        const std::string path = next();
        std::string err;
        Json values = Json::parse(read_file(path), &err);
        InferenceSettings check;
        if (err.empty()) err = check.update(values, true);
        if (!err.empty()) die("invalid sampling config %s: %s", path.c_str(), err.c_str());
        // Normalize aliases so later flags replace the same key.
        for (const auto& [name, value] : values.obj) {
            const std::string key = name == "repeat_penalty" ? "repetition_penalty" : name == "num_predict" || name == "max_completion_tokens" ? "max_tokens" : name == "draft_num_predict" ? "speculative_tokens" : name;
            overrides.set(key, value);
        }
        return true;
    }
    static const std::vector<std::pair<std::string, std::string>> flags = {
        {"--temp", "temperature"}, {"--temperature", "temperature"}, {"--top-k", "top_k"}, {"--top-p", "top_p"},
        {"--min-p", "min_p"}, {"--typical-p", "typical_p"}, {"--repeat-penalty", "repetition_penalty"},
        {"--repeat-last-n", "repeat_last_n"}, {"--presence-penalty", "presence_penalty"}, {"--frequency-penalty", "frequency_penalty"},
        {"--seed", "seed"}, {"--max-tokens", "max_tokens"}, {"--logit-bias", "logit_bias"}, {"--speculative-tokens", "speculative_tokens"}};
    if (arg == "--stop") {
        Json stops = overrides.has("stop") ? overrides["stop"] : Json::array();
        if (stops.is_str()) { Json a = Json::array(); a.push(stops); stops = a; }
        stops.push(next()); overrides.set("stop", stops); return true;
    }
    for (auto [flag, key] : std::vector<std::pair<std::string, std::string>>{{"--ignore-eos", "ignore_eos"}, {"--no-prompt-cache", "cache_prompt"}, {"--no-penalize-prompt", "penalize_prompt"}})
        if (arg == flag) { overrides.set(key, arg == "--ignore-eos"); return true; }
    for (const auto& [flag, key] : flags) {
        if (arg != flag) continue;
        std::string err;
        Json value = Json::parse(next(), &err);
        if (!err.empty()) die("invalid value for %s: %s", arg.c_str(), err.c_str());
        Json one = Json::object(); one.set(key, value);
        InferenceSettings check;
        if (!(err = check.update(one, true)).empty()) die("%s: %s", arg.c_str(), err.c_str());
        overrides.set(key, value);
        return true;
    }
    return false;
}

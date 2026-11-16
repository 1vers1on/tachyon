#include "tokenizer.h"
#include "sampler.h"
#include <cassert>
#include <limits>

int g_log_level = 0;

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ModelFile mf; mf.open(argv[1]);
    Tokenizer tok; tok.load(mf);
    const std::string replacement = "\xef\xbf\xbd";
    for (const auto& [bytes, expected] : std::vector<std::pair<std::string, std::string>>{
        {"hello", "hello"}, {"\xe2\x82\xac", "\xe2\x82\xac"}, {"\xf0\x9f\x98\x80", "\xf0\x9f\x98\x80"},
        {"\xe2\x82", replacement}, {"\xff" "A\xc3(", replacement + "A" + replacement + "("},
        {"\xc0\xaf", replacement + replacement}, {"\xed\xa0\x80", replacement + replacement + replacement},
        {"\xf4\x90\x80\x80", replacement + replacement + replacement + replacement}}) {
        StreamDecoder dec(&tok);
        std::string out;
        for (unsigned char c : bytes) out += dec.push(c);
        out += dec.flush();
        assert(out == expected);
    }
    StreamDecoder dec(&tok);
    assert(dec.push(0xe2).empty());
    assert(dec.push(0x82).empty());
    assert(dec.push(0xac) == "\xe2\x82\xac");
    assert(dec.flush().empty());
    SamplingParams params; params.seed = 0; params.seed_set = true; params.top_k = 3; params.top_p = 1;
    Sampler a, b; a.init(params); b.init(params);
    for (int i = 0; i < 100; i++) {
        std::vector<std::pair<float, int>> x = {{0, 0}, {0, 1}, {0, 2}};
        std::vector<std::pair<float, int>> y = {{0, 2}, {0, 0}, {0, 1}};
        assert(a.sample_cands(x) == b.sample_cands(y));
    }
    for (double value : {1e100, -1e100, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Json j(value);
        assert(j.as_int(123) == 123);
        assert(!j.dump().empty());
    }
    params.repetition_penalty = 2; params.presence_penalty = 3; params.frequency_penalty = 1;
    params.repeat_last_n = 2; a.init(params);
    a.accept(1); a.accept(2); a.accept(1);  // token 1 occurs once within the last two tokens
    float logits[] = {10, -8, 10}; a.apply_penalties(logits, 3);
    assert(logits[0] == 10 && logits[1] == -20 && logits[2] == 1);
    a.accept(0);
    float expired[] = {10, -8, 10}; a.apply_penalties(expired, 3);
    assert(expired[0] == 1 && expired[1] == -20 && expired[2] == 10);
    params.repeat_last_n = 0; a.init(params); a.accept(1);
    float unpenalized[] = {10, -8, 10}; a.apply_penalties(unpenalized, 3);
    assert(!params.has_penalties() && unpenalized[1] == -8);
    params.logit_bias[1] = 30; params.temperature = 0; a.init(params); a.apply_penalties(unpenalized, 3);
    assert(params.has_penalties() && a.sample(unpenalized, 3) == 1);
    params = SamplingParams{}; params.seed_set = true; params.temperature = 1;
    params.typical_p = .1f; params.top_p = 1; params.top_k = 0;
    a.init(params);
    const float typical_logits[] = {logf(.5f), logf(.3f), logf(.2f)};
    for (int i = 0; i < 100; i++) assert(a.sample(typical_logits, 3) == 1);
    InferenceSettings settings;
    assert(settings.update(Json::parse(R"({"typical_p":0.8,"repeat_last_n":64,"seed":0,"cache_prompt":false,"draft_num_predict":0,"logit_bias":{"1":4},"num_predict":10})"), true).empty());
    InferenceSettings restored; assert(restored.update(settings.to_json(), true).empty());
    assert(restored.sp.typical_p == settings.sp.typical_p && restored.sp.seed_set && !restored.cache_prompt && restored.max_tokens == 10);
    assert(!settings.update(Json::parse(R"({"typical_p":0})")).empty());
    assert(!settings.update(Json::parse(R"({"repeat_last_n":-2})")).empty());
    assert(!settings.update(Json::parse(R"({"unknown_setting":1})"), true).empty());
    assert(!settings.validate_vocab(1).empty());
    puts("UTF-8 streaming, seeded sampling and JSON numeric checks passed");
}

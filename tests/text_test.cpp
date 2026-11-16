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
    puts("UTF-8 streaming, seeded sampling and JSON numeric checks passed");
}

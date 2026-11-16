#include "json.h"
#include <bit>
#include <cstdio>
#include <limits>
#include <random>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) { fprintf(stderr, "%s\n", message); ++failures; }
    };
    auto round_trip = [&](double number) {
        std::string error;
        auto parsed = Json::parse(Json(number).dump(), &error);
        check(error.empty() && parsed.is_num() && parsed.num == number,
              "JSON number lost precision during serialization");
    };
    for (double number : {0.0, -0.0, 0.1, 1.2345678901234567, 9007199254740991.0,
                          9007199254740992.0, 9007199254740994.0,
                          std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::denorm_min(),
                          std::numeric_limits<double>::max()}) round_trip(number);
    std::mt19937_64 rng(123);
    for (int i = 0; i < 1000; ++i) {
        double number = std::bit_cast<double>(rng());
        if (std::isfinite(number)) round_trip(number);
    }
    const std::string replacement = "\xef\xbf\xbd";
    const std::string emoji = "\xf0\x9f\x98\x80";
    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
        {R"("\ud83d\ude00")", emoji},
        {R"("\ud800")", replacement},
        {R"("\udc00")", replacement},
        {R"("\ud800\ud800")", replacement + replacement},
        {R"("\ud800\ud83d\ude00")", replacement + emoji},
        {R"("\ud800\u0041")", replacement + "A"},
        {R"("\ud800A")", replacement + "A"},
        {R"("\udc00\ud83d\ude00")", replacement + emoji}}) {
        std::string error;
        Json parsed = Json::parse(input, &error);
        check(error.empty() && parsed.is_str() && parsed.str == expected,
              "JSON surrogate recovery corrupted Unicode text");
    }
    for (const char* input : {R"("\ud800\uZZZZ")", R"("\ud800\u12")"}) {
        std::string error;
        Json::parse(input, &error);
        check(!error.empty(), "Malformed Unicode escape was accepted");
    }
    if (failures) { fprintf(stderr, "%d JSON checks failed\n", failures); return 1; }
    puts("JSON numeric round trips and Unicode surrogate recovery passed");
}

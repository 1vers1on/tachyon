#include "http.h"
#include "https.h"
#include <cstdio>
#include <cstdlib>

int g_log_level = 0;

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::string body, error;
    int status;
    if (std::string(argv[1]).starts_with("get")) {
        HttpGetOptions options;
        if (std::string(argv[1]) == "get-auth") options.auth = "Bearer test";
        auto result = http_get_string("http://127.0.0.1:" + std::string(argv[2]) + argv[3], options, body);
        error = result.error;
        status = result.status;
    } else {
        status = http_call("127.0.0.1", atoi(argv[2]), "GET", argv[3], "",
            [&](const std::string& bytes) { body += bytes; return true; }, &error, 2);
    }
    if (status < 0 || !error.empty()) { fprintf(stderr, "%s\n", error.c_str()); return 1; }
    printf("%s", body.c_str());
}

// Tachyon - HTTP(S) GET client for model downloads. TLS comes from the system OpenSSL (libssl), loaded at runtime,
// so neither curl nor OpenSSL headers are needed; certificates and host names are always verified.
#pragma once
#include <cstdint>
#include <functional>
#include <string>

struct HttpGetOptions {
    std::string auth;          // "Authorization" header value; sent only to the original host, never across redirects
    int64_t range_from = 0;    // > 0: request bytes from this offset (resume); the server may still answer 200
    double timeout_s = 60;     // connect / idle-read timeout
    int max_redirects = 10;
};

struct HttpGetResult {
    int status = -1;           // final HTTP status, -1 on transport errors
    int64_t content_length = -1;
    std::string error;         // transport error (or TLS / certificate failure)
};

// GET `url` (http:// or https://), following redirects. Body bytes of the final response go to on_data (return false
// to abort); on_head is called once with the final status before any body bytes.
HttpGetResult http_get(const std::string& url, const HttpGetOptions& opt,
                       const std::function<bool(const char*, size_t)>& on_data,
                       const std::function<void(const HttpGetResult&)>& on_head = nullptr);

// Convenience: whole body as a string.
HttpGetResult http_get_string(const std::string& url, const HttpGetOptions& opt, std::string& body);

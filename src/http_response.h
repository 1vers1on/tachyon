// Shared response framing for the CLI and download clients.
#pragma once
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <algorithm>
#include <cstdio>
#include <limits>

inline bool http_response_head(const std::string& head, int& status, bool& chunked, int64_t& length) {
    status = 0; chunked = false; length = -1;
    if (sscanf(head.c_str(), "HTTP/%*s %d", &status) != 1 || status < 100 || status > 599) return false;
    bool has_length = false, has_encoding = false;
    size_t p = head.find("\r\n");
    if (p == std::string::npos) return false;
    p += 2;
    while (p < head.size()) {
        size_t e = head.find("\r\n", p);
        if (e == std::string::npos) return false;
        if (e == p) break;
        size_t c = head.find(':', p);
        if (c == std::string::npos || c >= e || c == p) return false;
        std::string key = head.substr(p, c - p), value = head.substr(c + 1, e - c - 1);
        for (char& ch : key) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
        size_t a = value.find_first_not_of(" \t"), b = value.find_last_not_of(" \t");
        value = a == std::string::npos ? "" : value.substr(a, b - a + 1);
        if (key == "content-length") {
            if (has_length || value.empty()) return false;
            has_length = true;
            auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), length);
            if (ec != std::errc() || end != value.data() + value.size() || length < 0) return false;
        } else if (key == "transfer-encoding") {
            if (has_encoding) return false;
            has_encoding = true;
            for (char& ch : value) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
            if (value != "chunked") return false;
            chunked = true;
        }
        p = e + 2;
    }
    return !(has_length && has_encoding);
}

// fill() appends bytes to buf: positive on success, zero on EOF, negative on error.
// Returns an error string, including explicit callback cancellation.
template<class Fill, class Emit>
std::string http_response_body(std::string& buf, bool chunked, int64_t length, Fill fill, Emit emit) {
    auto consume = [&](uint64_t remaining) -> std::string {
        while (remaining) {
            if (buf.empty() && fill() <= 0) return "connection closed mid-response or read timed out";
            size_t n = (size_t)std::min<uint64_t>(remaining, buf.size());
            if (!emit(buf.data(), n)) return "aborted";
            buf.erase(0, n);
            remaining -= n;
        }
        return "";
    };
    if (!chunked) {
        if (length >= 0) return consume((uint64_t)length);
        for (;;) {
            if (!buf.empty() && !emit(buf.data(), buf.size())) return "aborted";
            buf.clear();
            int n = fill();
            if (n == 0) return "";
            if (n < 0) return "response read failed or timed out";
        }
    }
    for (;;) {
        size_t e;
        while ((e = buf.find("\r\n")) == std::string::npos) {
            if (buf.size() > 8192) return "chunk header too large";
            if (fill() <= 0) return "connection closed mid-chunk header";
        }
        if (e > 8192) return "chunk header too large";
        std::string_view size(buf.data(), e);
        size = size.substr(0, size.find(';'));
        uint64_t n = 0;
        auto [end, ec] = std::from_chars(size.data(), size.data() + size.size(), n, 16);
        if (size.empty() || ec != std::errc() || end != size.data() + size.size()) return "invalid chunk size";
        buf.erase(0, e + 2);
        if (n == 0) {
            size_t total = 0;
            for (;;) {
                while ((e = buf.find("\r\n")) == std::string::npos) {
                    if (total + buf.size() > (1 << 20)) return "response trailers too large";
                    if (fill() <= 0) return "connection closed mid-trailer";
                }
                total += e + 2;
                if (total > (1 << 20)) return "response trailers too large";
                buf.erase(0, e + 2);
                if (e == 0) return "";
            }
        }
        std::string error = consume(n);
        if (!error.empty()) return error;
        while (buf.size() < 2) if (fill() <= 0) return "connection closed mid-chunk";
        if (buf.compare(0, 2, "\r\n") != 0) return "invalid chunk terminator";
        buf.erase(0, 2);
    }
}

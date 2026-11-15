// Tachyon - common utilities
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <immintrin.h>

#define TACH_VERSION "0.1.0"

[[noreturn]] inline void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[tachyon] fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

extern int g_log_level;  // 0=quiet,1=info,2=debug
inline void logi(const char* fmt, ...) {
    if (g_log_level < 1) return;
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[tachyon] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
inline void logd(const char* fmt, ...) {
    if (g_log_level < 2) return;
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[tachyon:dbg] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

#define TACH_CHECK(c, ...) do { if (!(c)) die(__VA_ARGS__); } while (0)

inline double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- half / bfloat16 helpers (host) ----
inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }
inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

inline std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open %s", path.c_str());
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(n, '\0');
    if (n > 0 && fread(&s[0], 1, n, f) != (size_t)n) die("read failed %s", path.c_str());
    fclose(f);
    return s;
}
inline bool file_exists(const std::string& p) { FILE* f = fopen(p.c_str(), "rb"); if (f) { fclose(f); return true; } return false; }

inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// Quantization formats used by Tachyon weights
enum class QType : int {
    F32 = 0,
    F16 = 1,
    Q8 = 2,   // symmetric int8, group 32, fp16 scale.  w = s*q
    Q4 = 3,   // asymmetric uint4, group 64, half2(scale,min). w = s*q + m
    Q3 = 4,   // asymmetric 3-bit, group 64, half2(scale,min); 12-byte blocks of 32 (2-bit plane + 1-bit plane)
};
inline const char* qtype_name(QType t) {
    switch (t) { case QType::F32: return "f32"; case QType::F16: return "f16"; case QType::Q8: return "q8"; case QType::Q4: return "q4"; case QType::Q3: return "q3"; }
    return "?";
}
inline QType qtype_from_name(const std::string& s) {
    if (s == "f32") return QType::F32;
    if (s == "f16") return QType::F16;
    if (s == "q8") return QType::Q8;
    if (s == "q4") return QType::Q4;
    if (s == "q3") return QType::Q3;
    die("unknown qtype %s", s.c_str());
}
constexpr int Q4_GROUP = 64;
constexpr int Q8_GROUP = 32;
constexpr int ACT_BLOCK = 32;  // activation int8 quant block

// Byte sizes for a [rows, cols] matrix in a given qtype.
// Layout (row-major per matrix): data section then scale section.
//  Q4: data = rows*cols/2 bytes (per 32-block: byte j = w[j] | w[j+16]<<4), scales = rows*(cols/64) half2
//  Q8: data = rows*cols int8, scales = rows*(cols/32) half
//  Q3: data = rows*cols*3/8 bytes: per 32-block 3 words {L0, L1, H}: element e = 16w+4k+b has its low 2 bits
//      at bit (8b+2k) of Lw and its high bit at bit (8b+i) of H where e = 4i+b; scales as Q4
inline size_t qtype_data_bytes(QType t, size_t rows, size_t cols) {
    switch (t) {
        case QType::F32: return rows * cols * 4;
        case QType::F16: return rows * cols * 2;
        case QType::Q8: return rows * cols;
        case QType::Q4: return rows * cols / 2;
        case QType::Q3: return rows * cols * 3 / 8;
    }
    return 0;
}
inline size_t qtype_scale_bytes(QType t, size_t rows, size_t cols) {
    switch (t) {
        case QType::Q8: return rows * (cols / Q8_GROUP) * 2;
        case QType::Q4: case QType::Q3: return rows * (cols / Q4_GROUP) * 4;
        default: return 0;
    }
}
inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

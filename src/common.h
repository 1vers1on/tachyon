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
#include <atomic>
#include <functional>
#include <mutex>
#include <cerrno>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

#define TACH_VERSION "1.0.0"

#define TACH_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))

// ---- logging ----
// Four levels: error, warn, info, debug. g_log_level picks the verbosity:
//   -1 errors only, 0 quiet (errors + warnings), 1 info (default), 2 debug.
// Every line is formatted once and written with a single write(2), so lines from different threads (and, with a shared
// O_APPEND log file, different processes) never interleave. Output goes to stderr (colored on a terminal) and, with
// --log-file / TACHYON_LOG_FILE, also to a file. --log-format json (TACHYON_LOG_FORMAT=json) writes one JSON object
// per line. Threads that serve a request set g_log_req (see LogScope) and every line they log carries "req=N".
extern int g_log_level;
enum LogLevel { LOG_ERROR = 0, LOG_WARN = 1, LOG_INFO = 2, LOG_DEBUG = 3 };

struct LogState {
    std::mutex mu;
    int file_fd = -1;
    bool json = false;
    int tty = -1;   // stderr is a terminal (resolved on first use)
    bool color = false;
    // in-place progress line (see log_progress)
    bool progress_on = false;      // a progress line is drawn on the terminal right now
    std::string progress_text;     // ... this one, without a newline
    int progress_step = -1;        // last 10% step that was logged for the current task
    double progress_t = 0;         // when the last line was logged, for tasks of unknown size
};
inline LogState& log_state() { static LogState s; return s; }
inline thread_local uint64_t g_log_req = 0;   // id of the request this thread is working on (0 = none)
inline uint64_t log_next_request_id() { static std::atomic<uint64_t> n{0}; return ++n; }

// tags the log lines of the current thread with a request id for the lifetime of the object
struct LogScope {
    uint64_t prev;
    explicit LogScope(uint64_t id) : prev(g_log_req) { g_log_req = id; }
    ~LogScope() { g_log_req = prev; }
    LogScope(const LogScope&) = delete;
    LogScope& operator=(const LogScope&) = delete;
};

inline bool log_enabled(LogLevel l) {
    switch (l) {
        case LOG_ERROR: return true;
        case LOG_WARN: return g_log_level >= 0;
        case LOG_INFO: return g_log_level >= 1;
        default: return g_log_level >= 2;
    }
}

inline void log_write_fd(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::write(fd, s.data() + off, s.size() - off);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return;   // logging must never take the program down
        off += (size_t)n;
    }
}

inline std::string log_json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

// Formats one message as a log line (with its newline): `line` is for stderr, `fline` for the log file. Needs st.mu held.
inline void log_format(LogState& st, LogLevel lvl, const std::string& msg_in, std::string& line, std::string& fline) {
    static const char* const kTag[] = {"ERROR", "WARN ", "INFO ", "DEBUG"};
    static const char* const kName[] = {"error", "warn", "info", "debug"};
    static const char* const kColor[] = {"\033[1;31m", "\033[33m", "\033[32m", "\033[2m"};
    std::string msg = msg_in;
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm t;
    int ms = (int)(ts.tv_nsec / 1000000);
    if (st.tty < 0) { st.tty = isatty(2) ? 1 : 0; st.color = st.tty && !getenv("NO_COLOR"); }
    char tbuf[48], req[32] = "";
    if (g_log_req) snprintf(req, sizeof req, "req=%llu", (unsigned long long)g_log_req);
    if (st.json) {
        gmtime_r(&ts.tv_sec, &t);
        snprintf(tbuf, sizeof tbuf, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, ms);
        line = std::string("{\"ts\":\"") + tbuf + "\",\"level\":\"" + kName[lvl] + "\",\"pid\":" + std::to_string((int)getpid());
        if (g_log_req) line += ",\"req\":" + std::to_string((unsigned long long)g_log_req);
        line += ",\"msg\":\"" + log_json_escape(msg) + "\"}\n";
        fline = line;
    } else {
        localtime_r(&ts.tv_sec, &t);
        char full[48];
        snprintf(full, sizeof full, "%04d-%02d-%02d %02d:%02d:%02d.%03d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, ms);
        snprintf(tbuf, sizeof tbuf, "%02d:%02d:%02d.%03d", t.tm_hour, t.tm_min, t.tm_sec, ms);
        std::string tail = std::string(req[0] ? req : "") + (req[0] ? " " : "") + msg + "\n";
        // a terminal gets the short time of day; files and pipes get the full date and the pid
        if (st.color) line = std::string("\033[2m") + tbuf + "\033[0m " + kColor[lvl] + kTag[lvl] + "\033[0m " + tail;
        else line = std::string(st.tty ? tbuf : full) + " " + kTag[lvl] + " " + (st.tty ? "" : "[" + std::to_string((int)getpid()) + "] ") + tail;
        fline = std::string(full) + " " + kTag[lvl] + " [" + std::to_string((int)getpid()) + "] " + tail;
    }
}

// Needs st.mu held. While a progress line is on screen it is lifted off for the message and redrawn below it.
inline void log_emit_locked(LogState& st, LogLevel lvl, const std::string& msg) {
    std::string line, fline;
    log_format(st, lvl, msg, line, fline);
    if (st.progress_on) line = "\r\033[K" + line + st.progress_text;
    log_write_fd(2, line);
    if (st.file_fd >= 0) log_write_fd(st.file_fd, fline);
}

// Writes one finished message (no trailing newline needed) at the given level.
inline void log_emit(LogLevel lvl, const std::string& msg) {
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    log_emit_locked(st, lvl, msg);
}

inline std::string log_vformat(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    char small[512];
    int n = vsnprintf(small, sizeof small, fmt, ap);
    std::string out;
    if (n < 0) out = fmt;
    else if ((size_t)n < sizeof small) out.assign(small, (size_t)n);
    else { out.resize((size_t)n); vsnprintf(&out[0], (size_t)n + 1, fmt, ap2); }
    va_end(ap2);
    return out;
}

#define TACH_DEFINE_LOG(fn, level)                                  \
    TACH_PRINTF(1, 2) inline void fn(const char* fmt, ...) {        \
        if (!log_enabled(level)) return;                            \
        va_list ap; va_start(ap, fmt);                              \
        std::string m = log_vformat(fmt, ap);                       \
        va_end(ap);                                                 \
        log_emit(level, m);                                         \
    }
TACH_DEFINE_LOG(loge, LOG_ERROR)
TACH_DEFINE_LOG(logw, LOG_WARN)
TACH_DEFINE_LOG(logi, LOG_INFO)
TACH_DEFINE_LOG(logd, LOG_DEBUG)
#undef TACH_DEFINE_LOG

TACH_PRINTF(1, 2) [[noreturn]] inline void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::string m = log_vformat(fmt, ap);
    va_end(ap);
    log_emit(LOG_ERROR, "fatal: " + m);
    exit(1);
}

// -1 = unknown level name
inline int log_parse_level(const std::string& s) {
    if (s == "error") return -1;
    if (s == "warn" || s == "warning" || s == "quiet") return 0;
    if (s == "info") return 1;
    if (s == "debug" || s == "verbose") return 2;
    return -2;
}
inline void log_set_file(const std::string& path) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) die("cannot open log file %s (%s)", path.c_str(), strerror(errno));
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    if (st.file_fd >= 0) close(st.file_fd);
    st.file_fd = fd;
}
inline void log_set_format(const std::string& f) {
    if (f != "text" && f != "json") die("invalid log format '%s' (use text or json)", f.c_str());
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    st.json = f == "json";
}
// Environment defaults; command line flags are applied afterwards and win.
inline void log_init_from_env() {
    if (const char* v = getenv("TACHYON_LOG_LEVEL"); v && *v) {
        int l = log_parse_level(v);
        if (l == -2) loge("ignoring TACHYON_LOG_LEVEL=%s (use error, warn, info or debug)", v);
        else g_log_level = l;
    }
    if (const char* v = getenv("TACHYON_LOG_FORMAT"); v && *v) log_set_format(v);
    if (const char* v = getenv("TACHYON_LOG_FILE"); v && *v) log_set_file(v);
}
// Handles -v, -q, --log-level, --log-file and --log-format. Returns true when `a` was one of them.
inline bool log_arg(const std::string& a, const std::function<std::string()>& next) {
    if (a == "-v") { g_log_level = 2; return true; }
    if (a == "-q") { g_log_level = 0; return true; }
    if (a == "--log-level") {
        std::string v = next();
        int l = log_parse_level(v);
        if (l == -2) die("invalid --log-level '%s' (use error, warn, info or debug)", v.c_str());
        g_log_level = l;
        return true;
    }
    if (a == "--log-file") { log_set_file(next()); return true; }
    if (a == "--log-format") { log_set_format(next()); return true; }
    return false;
}

#define TACH_CHECK(c, ...) do { if (!(c)) die(__VA_ARGS__); } while (0)

inline double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Progress of a long task (a download, a conversion): `msg` says what and how far, `frac` is the fraction done (0..1, or
// negative when the size is unknown). On a terminal it is one line redrawn in place, in the usual log line format, that
// stays at the bottom while other lines are logged above it. Anywhere else (stderr redirected, JSON format) it is logged as
// ordinary info lines, one per 10% (one per 5 seconds when the size is unknown), so logs do not fill up with \r updates.
// The log file gets those same sparse lines. Call log_progress_end() when the task is over. Shown at info level and above.
inline void log_progress(const std::string& msg, double frac) {
    if (!log_enabled(LOG_INFO)) return;
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    const double now = now_sec();
    const int step = frac < 0 ? -1 : std::min(10, (int)(frac * 10));
    const bool due = frac < 0 ? (st.progress_t == 0 || now - st.progress_t >= 5) : step > st.progress_step;
    std::string line, fline;
    log_format(st, LOG_INFO, msg, line, fline);   // also resolves st.tty
    if (st.tty == 1 && !st.json) {
        line.pop_back();   // no newline: the line is redrawn in place
        st.progress_text = line;
        log_write_fd(2, "\r\033[K" + line);
        st.progress_on = true;
        if (due && st.file_fd >= 0) log_write_fd(st.file_fd, fline);
    } else if (due) {
        log_emit_locked(st, LOG_INFO, msg);
    }
    if (due) { if (frac >= 0) st.progress_step = step; st.progress_t = now; }
}
inline void log_progress_end() {
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    if (st.progress_on) log_write_fd(2, "\r\033[K");
    st.progress_on = false;
    st.progress_text.clear();
    st.progress_step = -1;
    st.progress_t = 0;
}

// ---- half / bfloat16 helpers (host) ----
#ifdef __F16C__
inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }
#else
// Software fallback for CPUs / builds without F16C. Matches the instructions: round to nearest even, signalling NaNs quieted.
inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff, u;
    if (e == 0x1f) u = sign | 0x7f800000 | (m ? 0x400000 | (m << 13) : 0);   // inf / NaN
    else if (e) u = sign | ((e + 112) << 23) | (m << 13);                       // normal
    else if (!m) u = sign;                                                       // zero
    else {                                                                       // subnormal: normalize
        int s = 0;
        while (!(m & 0x400)) { m <<= 1; s++; }
        u = sign | ((uint32_t)(113 - s) << 23) | ((m & 0x3ff) << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}
inline uint16_t f2h(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000, a = u & 0x7fffffff;
    if (a >= 0x7f800000) return sign | 0x7c00 | (a > 0x7f800000 ? 0x200 | ((a >> 13) & 0x3ff) : 0);  // inf / NaN
    if (a >= 0x477ff000) return sign | 0x7c00;                                  // >= 65520 rounds to inf
    if (a <= 0x33000000) return sign;                                           // <= 2^-25 rounds to zero
    if (a < 0x38800000) {                                                       // half subnormal
        const uint32_t m = (a & 0x7fffff) | 0x800000, shift = 126 - (a >> 23);
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return sign | r;
    }
    uint32_t h = (a - (112u << 23)) >> 13;                                      // normal
    const uint32_t rem = a & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return sign | h;
}
#endif
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
    PQ2_0 = 5, PTQ1_0 = 6, Q4_0 = 7, Q8_0 = 8, Q4_K = 9, Q5_K = 10, Q6_K = 11, BF16 = 12,
    Q3 = 4,   // asymmetric 3-bit, group 64, half2(scale,min); 12-byte blocks of 32 (2-bit plane + 1-bit plane)
};
inline const char* qtype_name(QType t) {
    switch (t) { case QType::F32: return "f32"; case QType::F16: return "f16"; case QType::Q8: return "q8"; case QType::Q4: return "q4"; case QType::Q3: return "q3"; default: break; }
    switch (t) { case QType::PQ2_0: return "pq2_0"; case QType::PTQ1_0: return "ptq1_0";
    case QType::Q4_0: return "q4_0"; case QType::Q8_0: return "q8_0"; case QType::Q4_K: return "q4_k";
    case QType::Q5_K: return "q5_k"; case QType::Q6_K: return "q6_k"; case QType::BF16: return "bf16"; default: return "?"; }
}
inline QType qtype_from_name(const std::string& s) {
    if (s == "f32") return QType::F32;
    if (s == "f16") return QType::F16;
    if (s == "q8") return QType::Q8;
    if (s == "q4") return QType::Q4;
    if (s == "q3") return QType::Q3;
    for (int i = 5; i <= 12; i++) if (s == qtype_name((QType)i)) return (QType)i;
    die("unknown qtype %s", s.c_str());
}
constexpr int Q4_GROUP = 64;
constexpr int Q8_GROUP = 32;
constexpr int ACT_BLOCK = 32;  // activation int8 quant block

#include "packed.h"
inline bool qtype_packed(QType t) { return (int)t >= 5 && (int)t <= 11; }

// Byte sizes for a [rows, cols] matrix in a given qtype.
// Layout (row-major per matrix): data section then scale section.
//  Q4: data = rows*cols/2 bytes (per 32-block: byte j = w[j] | w[j+16]<<4), scales = rows*(cols/64) half2
//  Q8: data = rows*cols int8, scales = rows*(cols/32) half
//  Q3: data = rows*cols*3/8 bytes: per 32-block 3 words {L0, L1, H}: element e = 16w+4k+b has its low 2 bits
//      at bit (8b+2k) of Lw and its high bit at bit (8b+i) of H where e = 4i+b; scales as Q4
inline size_t qtype_data_bytes(QType t, size_t rows, size_t cols) {
    if (qtype_packed(t)) return rows * (cols / packed_group((int)t)) * packed_bytes((int)t);
    switch (t) {
        case QType::F32: return rows * cols * 4;
        case QType::BF16: case QType::F16: return rows * cols * 2;
        case QType::Q8: return rows * cols;
        case QType::Q4: return rows * cols / 2;
        case QType::Q3: return rows * cols * 3 / 8;
        default: return 0;
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

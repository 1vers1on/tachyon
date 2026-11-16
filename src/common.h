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

#define TACH_VERSION "0.1.0"

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

// Writes one finished message (no trailing newline needed) at the given level.
inline void log_emit(LogLevel lvl, const std::string& msg_in) {
    static const char* const kTag[] = {"ERROR", "WARN ", "INFO ", "DEBUG"};
    static const char* const kName[] = {"error", "warn", "info", "debug"};
    static const char* const kColor[] = {"\033[1;31m", "\033[33m", "\033[32m", "\033[2m"};
    std::string msg = msg_in;
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm t;
    int ms = (int)(ts.tv_nsec / 1000000);
    LogState& st = log_state();
    std::lock_guard<std::mutex> lk(st.mu);
    if (st.tty < 0) { st.tty = isatty(2) ? 1 : 0; st.color = st.tty && !getenv("NO_COLOR"); }
    char tbuf[48], req[32] = "";
    if (g_log_req) snprintf(req, sizeof req, "req=%llu", (unsigned long long)g_log_req);
    std::string line, fline;
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
    log_write_fd(2, line);
    if (st.file_fd >= 0) log_write_fd(st.file_fd, fline);
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

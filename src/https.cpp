// Tachyon - HTTP(S) GET client with TLS from a runtime-loaded libssl
#include "https.h"
#include "http.h"
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <memory>
#include <mutex>

namespace {

// ---------------------------------------------------------------------------------------------- OpenSSL (dlopen)
// The few entry points needed, declared with opaque pointers so no OpenSSL headers are required. All exist in
// OpenSSL 1.1.x and 3.x with these signatures.
struct Ssl {
    int (*OPENSSL_init_ssl)(uint64_t, const void*) = nullptr;
    const void* (*TLS_client_method)() = nullptr;
    void* (*SSL_CTX_new)(const void*) = nullptr;
    void (*SSL_CTX_free)(void*) = nullptr;
    long (*SSL_CTX_ctrl)(void*, int, long, void*) = nullptr;
    int (*SSL_CTX_set_default_verify_paths)(void*) = nullptr;
    int (*SSL_CTX_load_verify_locations)(void*, const char*, const char*) = nullptr;
    void (*SSL_CTX_set_verify)(void*, int, void*) = nullptr;
    void* (*SSL_new)(void*) = nullptr;
    void (*SSL_free)(void*) = nullptr;
    int (*SSL_set_fd)(void*, int) = nullptr;
    long (*SSL_ctrl)(void*, int, long, void*) = nullptr;
    int (*SSL_set1_host)(void*, const char*) = nullptr;
    int (*SSL_connect)(void*) = nullptr;
    int (*SSL_read)(void*, void*, int) = nullptr;
    int (*SSL_write)(void*, const void*, int) = nullptr;
    int (*SSL_shutdown)(void*) = nullptr;
    int (*SSL_get_error)(const void*, int) = nullptr;
    long (*SSL_get_verify_result)(const void*) = nullptr;
    unsigned long (*ERR_get_error)() = nullptr;
    void (*ERR_error_string_n)(unsigned long, char*, size_t) = nullptr;
    const char* (*X509_verify_cert_error_string)(long) = nullptr;
    void* ctx = nullptr;
    std::string error;

    std::string last_error() {
        std::string s;
        while (unsigned long e = ERR_get_error()) {
            char b[256];
            ERR_error_string_n(e, b, sizeof b);
            s += (s.empty() ? "" : "; ") + std::string(b);
        }
        return s;
    }
};

constexpr int SSL_VERIFY_PEER = 1;
constexpr int SSL_CTRL_SET_TLSEXT_HOSTNAME = 55;
constexpr int SSL_CTRL_SET_MIN_PROTO_VERSION = 123;
constexpr int TLS1_2_VERSION = 0x0303;
constexpr int SSL_ERROR_ZERO_RETURN = 6;

Ssl g_ssl;

// loads libssl once; nullptr (with g_ssl.error set) when unavailable
Ssl* ssl_lib() {
    Ssl& s = g_ssl;
    static std::once_flag once;
    std::call_once(once, [] {
        void* h = nullptr;
        for (const char* n : {"libssl.so.3", "libssl.so.1.1", "libssl.so"})
            if ((h = dlopen(n, RTLD_NOW | RTLD_LOCAL))) break;
        if (!h) { s.error = "HTTPS needs the system OpenSSL library (libssl.so.3 or libssl.so.1.1), which was not found"; return; }
        bool ok = true;
#define L(f) s.f = (decltype(s.f))dlsym(h, #f); ok &= s.f != nullptr;
        L(OPENSSL_init_ssl) L(TLS_client_method) L(SSL_CTX_new) L(SSL_CTX_free) L(SSL_CTX_ctrl)
        L(SSL_CTX_set_default_verify_paths) L(SSL_CTX_load_verify_locations) L(SSL_CTX_set_verify)
        L(SSL_new) L(SSL_free) L(SSL_set_fd) L(SSL_ctrl) L(SSL_set1_host) L(SSL_connect) L(SSL_read) L(SSL_write)
        L(SSL_shutdown) L(SSL_get_error) L(SSL_get_verify_result)
        L(ERR_get_error) L(ERR_error_string_n) L(X509_verify_cert_error_string)   // libcrypto, via libssl's dependencies
#undef L
        if (!ok) { s.error = "the system OpenSSL library is too old (need 1.1.0 or newer)"; return; }
        signal(SIGPIPE, SIG_IGN);   // a peer that closes mid-write must fail SSL_write, not kill the process
        s.OPENSSL_init_ssl(0, nullptr);
        s.ctx = s.SSL_CTX_new(s.TLS_client_method());
        if (!s.ctx) { s.error = "cannot create TLS context: " + s.last_error(); return; }
        s.SSL_CTX_ctrl(s.ctx, SSL_CTRL_SET_MIN_PROTO_VERSION, TLS1_2_VERSION, nullptr);
        s.SSL_CTX_set_verify(s.ctx, SSL_VERIFY_PEER, nullptr);
        // OpenSSL's own defaults (honours SSL_CERT_FILE / SSL_CERT_DIR), plus the usual distribution CA bundles in
        // case this libssl was built with a different OPENSSLDIR
        s.SSL_CTX_set_default_verify_paths(s.ctx);
        for (const char* f : {"/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                              "/etc/ssl/ca-bundle.pem", "/etc/ssl/cert.pem"}) {
            struct stat st;
            if (stat(f, &st) == 0) { s.SSL_CTX_load_verify_locations(s.ctx, f, nullptr); break; }
        }
        s.last_error();   // clear queued errors from bundles that failed to load
    });
    return s.ctx ? &s : nullptr;
}

// ---------------------------------------------------------------------------------------------- connection
struct Url {
    bool tls = true;
    std::string host, path;
    int port = 443;
};

bool parse_url(const std::string& u, Url& r) {
    size_t p;
    if (u.rfind("https://", 0) == 0) { r.tls = true; r.port = 443; p = 8; }
    else if (u.rfind("http://", 0) == 0) { r.tls = false; r.port = 80; p = 7; }
    else return false;
    size_t slash = u.find('/', p);
    std::string hp = u.substr(p, slash == std::string::npos ? std::string::npos : slash - p);
    r.path = slash == std::string::npos ? "/" : u.substr(slash);
    if (size_t at = hp.rfind('@'); at != std::string::npos) hp = hp.substr(at + 1);   // drop userinfo
    if (size_t c = hp.rfind(':'); c != std::string::npos && hp.find(']') == std::string::npos) {
        r.port = atoi(hp.c_str() + c + 1);
        hp = hp.substr(0, c);
    }
    r.host = hp;
    return !r.host.empty() && r.port > 0;
}

struct Stream {
    int fd = -1;
    void* ssl = nullptr;
    Ssl* lib = nullptr;
    std::string buf;   // bytes read but not consumed
    ~Stream() {
        if (ssl) { lib->SSL_shutdown(ssl); lib->SSL_free(ssl); }
        if (fd >= 0) close(fd);
    }
    bool open(const Url& u, double timeout, std::string& err) {
        fd = http_connect(u.host, u.port, timeout);
        if (fd < 0) { err = "cannot connect to " + u.host + ":" + std::to_string(u.port); return false; }
        timeval tv{(time_t)timeout, 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (!u.tls) return true;
        lib = ssl_lib();
        if (!lib) { err = g_ssl.error; return false; }
        ssl = lib->SSL_new(lib->ctx);
        if (!ssl) { err = "TLS: " + lib->last_error(); return false; }
        lib->SSL_set_fd(ssl, fd);
        lib->SSL_ctrl(ssl, SSL_CTRL_SET_TLSEXT_HOSTNAME, 0, (void*)u.host.c_str());   // SNI
        lib->SSL_set1_host(ssl, u.host.c_str());                                      // certificate must match the host
        if (lib->SSL_connect(ssl) != 1) {
            long v = lib->SSL_get_verify_result(ssl);
            err = "TLS handshake with " + u.host + " failed: " +
                  (v != 0 ? std::string("certificate verification: ") + lib->X509_verify_cert_error_string(v) : lib->last_error());
            return false;
        }
        return true;
    }
    bool write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            int n = ssl ? lib->SSL_write(ssl, s.data() + off, (int)std::min<size_t>(s.size() - off, 1 << 20))
                        : (int)send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
            if (n <= 0) { if (!ssl && n < 0 && errno == EINTR) continue; return false; }
            off += n;
        }
        return true;
    }
    // append more bytes to buf; 0 on clean EOF, -1 on error/timeout
    int fill() {
        char b[1 << 16];
        for (;;) {
            int n;
            if (ssl) {
                n = lib->SSL_read(ssl, b, sizeof b);
                if (n <= 0) return lib->SSL_get_error(ssl, n) == SSL_ERROR_ZERO_RETURN ? 0 : -1;
            } else {
                n = (int)recv(fd, b, sizeof b, 0);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) return n == 0 ? 0 : -1;
            }
            buf.append(b, n);
            return n;
        }
    }
};

std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }

std::string header_value(const std::string& head, const std::string& name) {   // name lower-case, without ':'
    std::string lh = lower(head);
    size_t p = 0;
    while ((p = lh.find("\r\n" + name + ":", p)) != std::string::npos) {
        size_t v = p + 3 + name.size(), e = lh.find("\r\n", v);
        std::string val = head.substr(v, e - v);
        size_t a = val.find_first_not_of(" \t"), b = val.find_last_not_of(" \t");
        return a == std::string::npos ? "" : val.substr(a, b - a + 1);
    }
    return "";
}

std::string resolve_location(const Url& base, const std::string& loc) {
    if (loc.rfind("http://", 0) == 0 || loc.rfind("https://", 0) == 0) return loc;
    std::string origin = std::string(base.tls ? "https://" : "http://") + base.host +
                         ((base.tls && base.port == 443) || (!base.tls && base.port == 80) ? "" : ":" + std::to_string(base.port));
    if (loc.rfind("//", 0) == 0) return std::string(base.tls ? "https:" : "http:") + loc;
    if (!loc.empty() && loc[0] == '/') return origin + loc;
    std::string dir = base.path.substr(0, base.path.find('?'));
    dir = dir.substr(0, dir.rfind('/') + 1);
    return origin + dir + loc;
}

}  // namespace

HttpGetResult http_get(const std::string& url0, const HttpGetOptions& opt,
                       const std::function<bool(const char*, size_t)>& on_data,
                       const std::function<void(const HttpGetResult&)>& on_head) {
    HttpGetResult res;
    std::string url = url0;
    Url first;
    if (!parse_url(url, first)) { res.error = "unsupported URL " + url; return res; }
    for (int hop = 0; hop <= opt.max_redirects; hop++) {
        Url u;
        if (!parse_url(url, u)) { res.error = "unsupported redirect URL " + url; return res; }
        if (first.tls && !u.tls) { res.error = "refusing redirect from HTTPS to plain HTTP (" + u.host + ")"; return res; }
        Stream s;
        if (!s.open(u, opt.timeout_s, res.error)) return res;
        std::string req = "GET " + u.path + " HTTP/1.1\r\nHost: " + u.host + "\r\nUser-Agent: tachyon/0.1\r\n"
                          "Accept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n";
        if (!opt.auth.empty() && u.host == first.host && u.tls == first.tls) req += "Authorization: " + opt.auth + "\r\n";
        if (opt.range_from > 0) req += "Range: bytes=" + std::to_string(opt.range_from) + "-\r\n";
        req += "\r\n";
        if (!s.write_all(req)) { res.error = "send to " + u.host + " failed"; return res; }
        size_t he;
        while ((he = s.buf.find("\r\n\r\n")) == std::string::npos) {
            if (s.buf.size() > (1 << 20)) { res.error = "response header too large"; return res; }
            if (s.fill() <= 0) { res.error = "connection to " + u.host + " closed before the response header"; return res; }
        }
        std::string head = s.buf.substr(0, he + 2);
        s.buf.erase(0, he + 4);
        res.status = 0;
        sscanf(head.c_str(), "HTTP/%*s %d", &res.status);
        if (res.status >= 300 && res.status < 400 && res.status != 304) {
            std::string loc = header_value(head, "location");
            if (loc.empty()) { res.error = "redirect without Location"; return res; }
            url = resolve_location(u, loc);
            continue;
        }
        bool chunked = lower(header_value(head, "transfer-encoding")).find("chunked") != std::string::npos;
        std::string cl = header_value(head, "content-length");
        res.content_length = !chunked && !cl.empty() ? strtoll(cl.c_str(), nullptr, 10) : -1;
        if (on_head) on_head(res);
        auto emit = [&](const char* p, size_t n) { return n == 0 || on_data(p, n); };
        if (chunked) {
            for (;;) {
                size_t le;
                while ((le = s.buf.find("\r\n")) == std::string::npos)
                    if (s.fill() <= 0) { res.error = "connection closed mid-response"; return res; }
                size_t n = strtoull(s.buf.c_str(), nullptr, 16);
                s.buf.erase(0, le + 2);
                if (n == 0) break;
                while (s.buf.size() < n + 2)
                    if (s.fill() <= 0) { res.error = "connection closed mid-response"; return res; }
                if (!emit(s.buf.data(), n)) { res.error = "aborted"; return res; }
                s.buf.erase(0, n + 2);
            }
        } else {
            int64_t left = res.content_length;
            for (;;) {
                size_t n = left >= 0 ? (size_t)std::min<int64_t>(left, (int64_t)s.buf.size()) : s.buf.size();
                if (!emit(s.buf.data(), n)) { res.error = "aborted"; return res; }
                s.buf.erase(0, n);
                if (left >= 0 && (left -= (int64_t)n) == 0) break;
                int r = s.fill();
                if (r == 0 && left < 0) break;   // body delimited by connection close
                if (r <= 0) { res.error = r == 0 ? "connection closed mid-response" : "read from " + u.host + " failed or timed out"; return res; }
            }
        }
        return res;
    }
    res.error = "too many redirects";
    return res;
}

HttpGetResult http_get_string(const std::string& url, const HttpGetOptions& opt, std::string& body) {
    body.clear();
    return http_get(url, opt, [&](const char* p, size_t n) { body.append(p, n); return true; });
}

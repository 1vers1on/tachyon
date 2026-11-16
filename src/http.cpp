#include "http.h"
#include "http_response.h"
#include "common.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <poll.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <strings.h>
#include <charconv>

std::string HttpRequest::header(const std::string& k) const {
    for (auto& h : headers) if (strcasecmp(h.first.c_str(), k.c_str()) == 0) return h.second;
    return "";
}

Conn::~Conn() { if (fd_ >= 0) ::close(fd_); }

const char* Conn::status_text(int c) {
    switch (c) {
        case 100: return "Continue"; case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request";
        case 401: return "Unauthorized"; case 404: return "Not Found"; case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large"; case 429: return "Too Many Requests"; case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error"; case 502: return "Bad Gateway"; case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
    }
    return "OK";
}

bool Conn::fill() {
    char tmp[65536];
    while (true) {
        ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf_.append(tmp, n);
        return true;
    }
}

bool Conn::read_request(HttpRequest& r, int* status) {
    r = HttpRequest{};
    if (status) *status = 0;
    auto bad = [&](int code = 400) { if (status) *status = code; return false; };
    auto number = [](const std::string& text, size_t& value, int base) {
        if (text.empty()) return false;
        auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
        return ec == std::errc() && end == text.data() + text.size();
    };
    size_t hdr_end;
    while ((hdr_end = buf_.find("\r\n\r\n")) == std::string::npos) {
        if (buf_.size() > (1 << 20)) { if (status) *status = 431; return false; }
        if (!fill()) return false;
    }
    if (hdr_end > (1 << 20)) return bad(431);
    std::string head = buf_.substr(0, hdr_end);
    buf_.erase(0, hdr_end + 4);
    size_t le = head.find("\r\n");
    std::string line = head.substr(0, le);
    size_t a = line.find(' '), b = line.rfind(' ');
    if (a == std::string::npos || b == a) { if (status) *status = 400; return false; }
    r.method = line.substr(0, a);
    r.path = line.substr(a + 1, b - a - 1);
    std::string ver = line.substr(b + 1);
    if ((ver != "HTTP/1.1" && ver != "HTTP/1.0") || r.path.empty() || r.path[0] != '/' || r.path.find(' ') != std::string::npos) return bad();
    size_t q = r.path.find('?');
    if (q != std::string::npos) { r.query = r.path.substr(q + 1); r.path = r.path.substr(0, q); }
    while (r.path.size() > 1 && r.path.back() == '/') r.path.pop_back();
    size_t p = le == std::string::npos ? head.size() : le + 2;
    while (p < head.size()) {
        size_t e = head.find("\r\n", p);
        if (e == std::string::npos) e = head.size();
        std::string h = head.substr(p, e - p);
        size_t c = h.find(':');
        if (c == std::string::npos || c == 0 || h.substr(0, c).find_first_of(" \t") != std::string::npos) return bad();
        {
            std::string v = h.substr(c + 1);
            while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(0, 1);
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
            r.headers.push_back({h.substr(0, c), v});
        }
        p = e + 2;
    }
    std::string conn = r.header("Connection");
    r.keep_alive = ver == "HTTP/1.1" ? strcasecmp(conn.c_str(), "close") != 0 : strcasecmp(conn.c_str(), "keep-alive") == 0;
    const size_t kMax = 256u << 20;
    int ncl = 0, nte = 0;
    for (const auto& h : r.headers) {
        ncl += strcasecmp(h.first.c_str(), "Content-Length") == 0;
        nte += strcasecmp(h.first.c_str(), "Transfer-Encoding") == 0;
    }
    if (ncl > 1 || nte > 1 || (ncl && nte)) return bad();
    const bool chunked = nte != 0;
    if (chunked && strcasecmp(r.header("Transfer-Encoding").c_str(), "chunked") != 0) return bad();
    size_t cl = 0;
    if (ncl && !number(r.header("Content-Length"), cl, 10)) return bad();
    if (cl > kMax) return bad(413);
    // clients (curl) wait for this before sending large bodies
    if (strcasecmp(r.header("Expect").c_str(), "100-continue") == 0 && !write_all("HTTP/1.1 100 Continue\r\n\r\n")) return false;
    if (chunked) {
        r.body.clear();
        while (true) {
            size_t e;
            while ((e = buf_.find("\r\n")) == std::string::npos) {
                if (buf_.size() > 8192) return bad();
                if (!fill()) return false;
            }
            if (e > 8192) return bad();
            std::string size = buf_.substr(0, e);
            size_t ext = size.find(';');
            if (ext != std::string::npos) size.resize(ext);
            size_t n = 0;
            if (!number(size, n, 16)) return bad();
            buf_.erase(0, e + 2);
            if (n == 0) {   // trailers until the empty line
                size_t trailers = 0;
                while (true) {
                    while ((e = buf_.find("\r\n")) == std::string::npos) {
                        if (trailers + buf_.size() > (1 << 20)) return bad(431);
                        if (!fill()) return false;
                    }
                    trailers += e + 2;
                    if (trailers > (1 << 20)) return bad(431);
                    bool empty = e == 0;
                    buf_.erase(0, e + 2);
                    if (empty) break;
                }
                break;
            }
            if (n > kMax - r.body.size()) return bad(413);
            while (buf_.size() < n + 2) if (!fill()) return false;
            if (buf_.compare(n, 2, "\r\n") != 0) return bad();
            r.body.append(buf_, 0, n);
            buf_.erase(0, n + 2);
        }
        return true;
    }
    while (buf_.size() < cl) if (!fill()) return false;
    r.body = buf_.substr(0, cl);
    buf_.erase(0, cl);
    return true;
}

bool Conn::write_all(const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd_, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { logd("write to %s failed after %zu of %zu bytes (%s)", peer.c_str(), off, s.size(), n < 0 ? strerror(errno) : "closed"); return false; }
        off += n;
        bytes_out += n;
    }
    return true;
}

static std::string cors() {
    return "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";
}

bool Conn::send_response(int code, const std::string& ctype, const std::string& body, bool keep_alive, const std::string& extra) {
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n";
    h += "Content-Type: " + ctype + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += cors() + extra;
    h += keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
    status = code;
    return write_all(h + body);
}

bool Conn::start_chunked(const std::string& ctype) {
    status = 200;
    return write_all("HTTP/1.1 200 OK\r\nContent-Type: " + ctype + "\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
                     "Transfer-Encoding: chunked\r\n" + cors() + "Connection: keep-alive\r\n\r\n");
}

bool Conn::chunk(const std::string& data) {
    if (data.empty()) return true;
    char hx[32];
    snprintf(hx, sizeof hx, "%zx\r\n", data.size());
    return write_all(std::string(hx) + data + "\r\n");
}

bool Conn::end_chunked() { return write_all("0\r\n\r\n"); }

bool Conn::peer_closed() {
    char c;
    ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return true;
    if (n < 0) return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
    return false;
}

// ------------------------------------------------------------------------------------------------ client

int http_connect(const std::string& host, int port, double timeout_s) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string h = host == "localhost" || host.empty() || host == "0.0.0.0" ? "127.0.0.1" : host;
    if (getaddrinfo(h.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return -1;
    int fd = -1;
    const double deadline = now_sec() + timeout_s;
    for (addrinfo* p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype | SOCK_CLOEXEC, p->ai_protocol);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        if (timeout_s > 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, p->ai_addr, p->ai_addrlen);
        if (rc < 0 && timeout_s > 0 && errno == EINPROGRESS) {
            for (;;) {
                double left = deadline - now_sec();
                if (left <= 0) break;
                pollfd pending{fd, POLLOUT, 0};
                int ready = poll(&pending, 1, (int)std::clamp(left * 1000, 1.0, (double)INT32_MAX));
                if (ready < 0 && errno == EINTR) continue;
                int error = 0;
                socklen_t n = sizeof error;
                if (ready > 0 && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &n) == 0 && !error) rc = 0;
                break;
            }
        }
        if (timeout_s > 0) fcntl(fd, F_SETFL, flags);
        if (rc == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (timeout_s > 0) {
        timeval tv{(time_t)timeout_s, (suseconds_t)((timeout_s - (time_t)timeout_s) * 1e6)};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    }
    return fd;
}

namespace {
struct Reader {
    int fd;
    std::string buf;
    Conn* watch = nullptr;   // relay: give up when this client disconnects while upstream is still working
    bool failed = false;
    bool fill() {
        char tmp[65536];
        while (watch) {
            pollfd p[2] = {{fd, POLLIN, 0}, {watch->fd(), POLLIN, 0}};
            int n = poll(p, 2, 1000);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) return false;
            if (p[0].revents) break;
            if (p[1].revents && watch->peer_closed()) return false;
            if (p[1].revents) usleep(50000);   // pipelined bytes from the client: not ours to read now
        }
        while (true) {
            ssize_t n = recv(fd, tmp, sizeof tmp, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { failed = n < 0; return false; }
            buf.append(tmp, n);
            return true;
        }
    }
    bool head(std::string& h) {
        size_t he;
        while ((he = buf.find("\r\n\r\n")) == std::string::npos) {
            if (buf.size() > (1 << 20) || !fill()) return false;
        }
        if (he > (1 << 20)) return false;
        h = buf.substr(0, he + 4);
        buf.erase(0, he + 4);
        return true;
    }
};
std::string lower(std::string s) { for (auto& c : s) c = tolower((unsigned char)c); return s; }
long header_num(const std::string& lhead, const char* name) {
    size_t p = lhead.find(name);
    return p == std::string::npos ? -1 : atol(lhead.c_str() + p + strlen(name));
}
bool send_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += n;
    }
    return true;
}
}  // namespace

int http_call(const std::string& host, int port, const std::string& method, const std::string& path, const std::string& body,
              const std::function<bool(const std::string&)>& on_data, std::string* err, double timeout_s,
              const std::string& extra_headers) {
    if (err) err->clear();
    int fd = http_connect(host, port, timeout_s);
    if (fd < 0) { if (err) *err = "cannot connect to " + host + ":" + std::to_string(port); return -1; }
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n" + extra_headers;
    if (!body.empty() || method == "POST") req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    if (!send_all(fd, req)) { close(fd); if (err) *err = "send failed"; return -1; }
    Reader rd{fd, {}};
    std::string head;
    int status = 0;
    bool chunked;
    int64_t clen;
    do {
        if (!rd.head(head) || !http_response_head(head, status, chunked, clen)) {
            close(fd); if (err) *err = "invalid or incomplete HTTP response header"; return -1;
        }
    } while (status < 200 && status != 101);
    std::string error;
    if (method != "HEAD" && status != 204 && status != 304) {
        error = http_response_body(rd.buf, chunked, clen,
            [&]() { return rd.fill() ? 1 : rd.failed ? -1 : 0; },
            [&](const char* p, size_t n) { return !on_data || on_data(std::string(p, n)); });
    }
    close(fd);
    if (!error.empty()) { if (err) *err = error; return -1; }
    return status;
}

bool http_relay(const HttpRequest& r, const std::string& host, int port, Conn& client, std::string* err) {
    int fd = http_connect(host, port);
    if (fd < 0) { if (err) *err = "cannot connect to model runner"; return true; }
    std::string req = r.method + " " + r.path + (r.query.empty() ? "" : "?" + r.query) + " HTTP/1.1\r\n";
    for (auto& h : r.headers) {
        std::string k = lower(h.first);
        if (k == "connection" || k == "content-length" || k == "transfer-encoding" || k == "expect" || k == "keep-alive") continue;
        req += h.first + ": " + h.second + "\r\n";
    }
    req += "Connection: close\r\nContent-Length: " + std::to_string(r.body.size()) + "\r\n\r\n" + r.body;
    if (!send_all(fd, req)) { close(fd); if (err) *err = "send to model runner failed"; return true; }
    Reader rd{fd, {}, &client};
    std::string head;
    if (!rd.head(head)) {
        close(fd);
        if (client.peer_closed()) return false;
        if (err) *err = "model runner closed the connection";
        return true;
    }
    if (err) err->clear();
    sscanf(head.c_str(), "HTTP/%*s %d", &client.status);
    // keep the client connection open: only the upstream hop is "Connection: close"
    std::string lh = lower(head);
    size_t cp = lh.find("\r\nconnection:");
    if (cp != std::string::npos) {
        size_t e = head.find("\r\n", cp + 2);
        head.replace(cp, e - cp, std::string("\r\nConnection: ") + (r.keep_alive ? "keep-alive" : "close"));
    }
    bool ok = client.write_all(head);
    bool chunked = lh.find("transfer-encoding: chunked") != std::string::npos;
    long clen = header_num(lh, "content-length:");
    if (chunked) {
        // forward raw chunks until the terminating zero-length chunk
        while (ok) {
            size_t le;
            while ((le = rd.buf.find("\r\n")) == std::string::npos) if (!rd.fill()) { close(fd); return false; }
            long n = strtol(rd.buf.substr(0, le).c_str(), nullptr, 16);
            size_t need = le + 2 + n + 2;
            while (rd.buf.size() < need) if (!rd.fill()) { close(fd); return false; }
            ok = client.write_all(rd.buf.substr(0, need));
            rd.buf.erase(0, need);
            if (n == 0) break;
        }
    } else if (clen >= 0) {
        long got = 0;
        while (ok && got < clen) {
            if (rd.buf.empty() && !rd.fill()) { close(fd); return false; }
            size_t n = std::min<size_t>(rd.buf.size(), clen - got);
            ok = client.write_all(rd.buf.substr(0, n));
            rd.buf.erase(0, n);
            got += n;
        }
    } else {
        while (ok) { if (!rd.buf.empty()) ok = client.write_all(rd.buf); rd.buf.clear(); if (!rd.fill()) break; }
        close(fd);
        return false;
    }
    close(fd);
    return ok && r.keep_alive;
}

std::string http_peer_name(const sockaddr_in& a) {
    char ip[INET_ADDRSTRLEN] = "?";
    inet_ntop(AF_INET, &a.sin_addr, ip, sizeof ip);
    return std::string(ip) + ":" + std::to_string(ntohs(a.sin_port));
}

void http_log_access(const Conn& c, const HttpRequest& r, double seconds) {
    const bool quiet = r.method == "OPTIONS" || r.path == "/" || r.path == "/health" || r.path == "/v1/health" || r.path == "/api/version";
    LogLevel lvl = c.status >= 500 ? LOG_ERROR : c.status >= 400 ? LOG_WARN : quiet ? LOG_DEBUG : LOG_INFO;
    if (!log_enabled(lvl)) return;
    char kb[32];
    snprintf(kb, sizeof kb, c.bytes_out >= 10240 ? "%.0f KB" : "%.1f KB", c.bytes_out / 1024.0);
    std::string m = r.method + " " + r.path + " -> ";
    m += c.status ? std::to_string(c.status) : "no response";
    m += " in " + std::to_string((int)(seconds * 1000)) + " ms, " + kb + " to " + c.peer;
    if (!c.note.empty()) m += ": " + c.note;
    log_emit(lvl, m);
}

// Tachyon - minimal HTTP/1.1 (server connections, a blocking client, response relaying for the model manager)
#pragma once
#include <string>
#include <vector>
#include <functional>
#include <utility>

struct HttpRequest {
    std::string method, path, query, body;
    std::vector<std::pair<std::string, std::string>> headers;
    bool keep_alive = true;
    std::string header(const std::string& k) const;
};

class Conn {
public:
    explicit Conn(int fd) : fd_(fd) {}
    ~Conn();
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;
    // false on EOF / malformed request (connection should be closed); sets *status for an error reply
    bool read_request(HttpRequest& r, int* status = nullptr);
    bool write_all(const std::string& s);
    bool send_response(int code, const std::string& ctype, const std::string& body, bool keep_alive,
                       const std::string& extra_headers = "");
    bool start_chunked(const std::string& ctype);
    bool chunk(const std::string& data);
    bool end_chunked();
    // true when the peer has closed its side (or the socket failed); never blocks
    bool peer_closed();
    int fd() const { return fd_; }
    static const char* status_text(int c);
private:
    bool fill();
    int fd_;
    std::string buf_;
};

// Blocking client. Returns the HTTP status (or -1 with *err). on_data receives decoded body bytes as they arrive;
// returning false aborts.
int http_call(const std::string& host, int port, const std::string& method, const std::string& path, const std::string& body,
              const std::function<bool(const std::string&)>& on_data, std::string* err, double timeout_s = 0,
              const std::string& extra_headers = "");

int http_connect(const std::string& host, int port, double timeout_s = 0);

// Model manager proxy: send `r` to an upstream server and stream its response back to `client` byte for byte
// (head with the upstream framing, then the body). Returns false if the client connection must be closed.
bool http_relay(const HttpRequest& r, const std::string& host, int port, Conn& client, std::string* err);

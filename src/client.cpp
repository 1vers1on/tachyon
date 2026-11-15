// tachyon-client: CLI test client for any OpenAI-compatible server (streaming chat REPL, one-shot, load test)
#include "json.h"
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <functional>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Url { std::string host = "127.0.0.1", port = "8080", base; };
static Url parse_url(std::string u) {
    Url r;
    if (u.rfind("http://", 0) == 0) u = u.substr(7);
    size_t slash = u.find('/');
    std::string hp = slash == std::string::npos ? u : u.substr(0, slash);
    r.base = slash == std::string::npos ? "" : u.substr(slash);
    while (!r.base.empty() && r.base.back() == '/') r.base.pop_back();
    size_t c = hp.find(':');
    r.host = c == std::string::npos ? hp : hp.substr(0, c);
    if (c != std::string::npos) r.port = hp.substr(c + 1);
    else r.port = "80";
    return r;
}

// Minimal HTTP/1.1 client. on_data receives decoded body bytes as they arrive (chunked or length-delimited).
static int http_request(const Url& url, const std::string& method, const std::string& path, const std::string& body,
                        const std::function<bool(const std::string&)>& on_data, std::string* err) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &res) != 0) { if (err) *err = "cannot resolve " + url.host; return -1; }
    int fd = -1;
    for (addrinfo* p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { if (err) *err = "cannot connect to " + url.host + ":" + url.port; return -1; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::string req = method + " " + url.base + path + " HTTP/1.1\r\nHost: " + url.host + "\r\nConnection: close\r\n";
    const char* key = getenv("OPENAI_API_KEY");
    if (key && *key) req += std::string("Authorization: Bearer ") + key + "\r\n";
    if (!body.empty()) req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    if (send(fd, req.data(), req.size(), MSG_NOSIGNAL) != (ssize_t)req.size()) { close(fd); if (err) *err = "send failed"; return -1; }
    std::string buf;
    char tmp[16384];
    auto fill = [&]() { ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) return false; buf.append(tmp, n); return true; };
    size_t he;
    while ((he = buf.find("\r\n\r\n")) == std::string::npos) if (!fill()) { close(fd); if (err) *err = "connection closed"; return -1; }
    std::string head = buf.substr(0, he);
    buf.erase(0, he + 4);
    int status = 0;
    sscanf(head.c_str(), "HTTP/%*s %d", &status);
    std::string lower = head;
    for (auto& ch : lower) ch = tolower(ch);
    bool chunked = lower.find("transfer-encoding: chunked") != std::string::npos;
    long clen = -1;
    size_t cp = lower.find("content-length:");
    if (cp != std::string::npos) clen = atol(lower.c_str() + cp + 15);
    bool go = true;
    if (chunked) {
        while (go) {
            size_t le;
            while ((le = buf.find("\r\n")) == std::string::npos) if (!fill()) { go = false; break; }
            if (!go) break;
            long n = strtol(buf.substr(0, le).c_str(), nullptr, 16);
            buf.erase(0, le + 2);
            if (n == 0) break;
            while ((long)buf.size() < n + 2) if (!fill()) { go = false; break; }
            if (!go) break;
            if (!on_data(buf.substr(0, n))) go = false;
            buf.erase(0, n + 2);
        }
    } else {
        long got = 0;
        if (!buf.empty()) { got += buf.size(); go = on_data(buf); buf.clear(); }
        while (go && (clen < 0 || got < clen) && fill()) { got += buf.size(); go = on_data(buf); buf.clear(); }
    }
    close(fd);
    return status;
}

struct Opts {
    Url url;
    std::string model, system, prompt;
    int max_tokens = 1024;
    double temp = -1, top_p = -1;
    bool think = true, show_reasoning = true, raw = false;
    int bench_n = 0, bench_conc = 4;
};

struct StreamResult { std::string content, reasoning; Json usage, timings; int chunks = 0; double t_first = 0; std::string finish; };

// Streams a chat completion; prints deltas when print=true.
static bool chat_stream(const Opts& o, const Json& messages, StreamResult& res, bool print) {
    Json body = Json::object();
    if (!o.model.empty()) body.set("model", o.model);
    body.set("messages", messages);
    body.set("stream", true);
    Json so = Json::object(); so.set("include_usage", true);
    body.set("stream_options", so);
    body.set("max_tokens", o.max_tokens);
    if (o.temp >= 0) body.set("temperature", o.temp);
    if (o.top_p >= 0) body.set("top_p", o.top_p);
    if (!o.think) { Json kw = Json::object(); kw.set("enable_thinking", false); body.set("chat_template_kwargs", kw); }
    std::string line_buf;
    bool in_reasoning = false;
    double t0 = now();
    std::string err;
    int st = http_request(o.url, "POST", "/v1/chat/completions", body.dump(), [&](const std::string& d) {
        line_buf += d;
        size_t p;
        while ((p = line_buf.find('\n')) != std::string::npos) {
            std::string line = line_buf.substr(0, p);
            line_buf.erase(0, p + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data: ", 0) != 0) continue;
            std::string payload = line.substr(6);
            if (payload == "[DONE]") return true;
            Json j = Json::parse(payload);
            if (j.has("error")) { fprintf(stderr, "\nserver error: %s\n", j["error"].get_str("message", "?").c_str()); return false; }
            if (j.has("usage") && j["usage"].is_obj()) res.usage = j["usage"];
            if (j.has("timings")) res.timings = j["timings"];
            if (!j["choices"].is_arr() || j["choices"].arr.empty()) continue;
            const Json& ch = j["choices"][0];
            if (ch["finish_reason"].is_str()) res.finish = ch["finish_reason"].str;
            const Json& delta = ch["delta"];
            std::string r = delta.get_str("reasoning_content", delta.get_str("reasoning", ""));
            std::string c = delta.get_str("content", "");
            if ((!r.empty() || !c.empty()) && res.t_first == 0) res.t_first = now() - t0;
            if (!r.empty()) res.chunks++;
            if (!c.empty()) res.chunks++;
            res.reasoning += r;
            res.content += c;
            if (print) {
                if (!r.empty() && o.show_reasoning) {
                    if (!in_reasoning) { printf("\033[2m[thinking] "); in_reasoning = true; }
                    printf("%s", r.c_str());
                }
                if (!c.empty()) {
                    if (in_reasoning) { printf("\033[0m\n"); in_reasoning = false; }
                    printf("%s", c.c_str());
                }
                fflush(stdout);
            }
        }
        return true;
    }, &err);
    if (print && in_reasoning) printf("\033[0m");
    if (print) fflush(stdout);
    if (st < 0) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }
    if (st != 200) { fprintf(stderr, "HTTP %d\n", st); return false; }
    return true;
}

static void print_stats(const StreamResult& r) {
    const Json& t = r.timings;
    const Json& u = r.usage;
    fprintf(stderr, "\033[36m[%lld prompt tok (%lld cached) | %lld gen tok | ttft %.0f ms | prompt %.0f tok/s | gen %.1f tok/s",
            (long long)u.get_int("prompt_tokens", 0), (long long)u["prompt_tokens_details"].get_int("cached_tokens", 0),
            (long long)u.get_int("completion_tokens", 0), t.get_num("ttft_ms", r.t_first * 1000), t.get_num("prompt_per_second", 0),
            t.get_num("predicted_per_second", 0));
    if (t.get_int("draft_tokens", 0) > 0)
        fprintf(stderr, " | spec %lld/%lld", (long long)t.get_int("draft_accepted", 0), (long long)t.get_int("draft_tokens", 0));
    fprintf(stderr, "]\033[0m\n");
}

static void usage() {
    fprintf(stderr,
        "tachyon-client - CLI test client for OpenAI-compatible servers\n\n"
        "usage: tachyon-client [options]                 interactive chat (streaming)\n"
        "       tachyon-client -p \"question\"             one-shot\n"
        "       tachyon-client --bench N [-c 4]           load test: N requests, c concurrent\n\n"
        "options:\n"
        "  --url URL          server (default http://127.0.0.1:8080)\n"
        "  -m, --model NAME   model name to send\n"
        "  -s, --system TEXT  system prompt\n"
        "  -n, --max-tokens N (default 1024)\n"
        "  --temp T, --top-p P\n"
        "  --no-think         disable reasoning for hybrid thinking models (Qwen3)\n"
        "  --hide-thinking    do not print reasoning\n\n"
        "chat commands: /reset  /think on|off  /system TEXT  /stats  /quit\n");
}

int main(int argc, char** argv) {
    Opts o;
    o.url = parse_url("http://127.0.0.1:8080");
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "--url") o.url = parse_url(next());
        else if (a == "-m" || a == "--model") o.model = next();
        else if (a == "-s" || a == "--system") o.system = next();
        else if (a == "-p" || a == "--prompt") o.prompt = next();
        else if (a == "-n" || a == "--max-tokens") o.max_tokens = atoi(next().c_str());
        else if (a == "--temp") o.temp = atof(next().c_str());
        else if (a == "--top-p") o.top_p = atof(next().c_str());
        else if (a == "--no-think") o.think = false;
        else if (a == "--hide-thinking") o.show_reasoning = false;
        else if (a == "--bench") o.bench_n = atoi(next().c_str());
        else if (a == "-c" || a == "--concurrency") o.bench_conc = atoi(next().c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    // show the server's model
    {
        std::string body, err;
        int st = http_request(o.url, "GET", "/v1/models", "", [&](const std::string& d) { body += d; return true; }, &err);
        if (st != 200) { fprintf(stderr, "cannot reach server at %s:%s (%s)\n", o.url.host.c_str(), o.url.port.c_str(), err.c_str()); return 1; }
        Json j = Json::parse(body);
        if (o.model.empty() && j["data"].is_arr() && !j["data"].arr.empty()) o.model = j["data"][0].get_str("id", "");
    }
    auto base_msgs = [&]() {
        Json m = Json::array();
        if (!o.system.empty()) { Json s = Json::object(); s.set("role", "system"); s.set("content", o.system); m.push(s); }
        return m;
    };
    if (o.bench_n > 0) {
        const char* prompts[] = {
            "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
            "Explain how a CPU cache works, with examples.",
            "List 20 creative names for a coffee shop and explain each briefly.",
            "Write a Python function that parses a CSV file and computes column averages. Explain the code.",
            "What are the main causes of the French Revolution? Answer in detail.",
            "Describe the water cycle to a ten year old.",
        };
        std::atomic<int> next{0}, done_n{0};
        std::atomic<long> tokens{0};
        std::mutex mu;
        std::vector<double> ttfts, rates;
        double t0 = now();
        std::vector<std::thread> th;
        for (int c = 0; c < o.bench_conc; c++)
            th.emplace_back([&] {
                while (true) {
                    int k = next.fetch_add(1);
                    if (k >= o.bench_n) break;
                    Json msgs = base_msgs();
                    Json u = Json::object(); u.set("role", "user"); u.set("content", prompts[k % 6]);
                    msgs.push(u);
                    StreamResult r;
                    if (!chat_stream(o, msgs, r, false)) continue;
                    tokens += r.usage.get_int("completion_tokens", 0);
                    done_n++;
                    std::lock_guard<std::mutex> lk(mu);
                    ttfts.push_back(r.t_first);
                    rates.push_back(r.timings.get_num("predicted_per_second", 0));
                    fprintf(stderr, "  request %d: %lld tokens, %.1f tok/s, ttft %.0f ms\n", k, (long long)r.usage.get_int("completion_tokens", 0),
                            rates.back(), r.t_first * 1000);
                }
            });
        for (auto& t : th) t.join();
        double el = now() - t0;
        double avg_ttft = 0, avg_rate = 0;
        for (double x : ttfts) avg_ttft += x;
        for (double x : rates) avg_rate += x;
        int n = std::max<int>(1, ttfts.size());
        printf("bench: %d requests, concurrency %d, %ld tokens in %.2fs => %.1f tok/s aggregate | per-request %.1f tok/s | avg ttft %.0f ms\n",
               done_n.load(), o.bench_conc, tokens.load(), el, tokens / el, avg_rate / n, avg_ttft / n * 1000);
        return 0;
    }
    if (!o.prompt.empty()) {
        Json msgs = base_msgs();
        Json u = Json::object(); u.set("role", "user"); u.set("content", o.prompt);
        msgs.push(u);
        StreamResult r;
        bool ok = chat_stream(o, msgs, r, true);
        printf("\n");
        fflush(stdout);
        if (ok) print_stats(r);
        return ok ? 0 : 1;
    }
    // interactive REPL
    fprintf(stderr, "tachyon-client: connected to %s:%s, model %s. Type /help for commands.\n", o.url.host.c_str(), o.url.port.c_str(), o.model.c_str());
    Json history = base_msgs();
    bool stats = true;
    std::string line;
    while (true) {
        printf("\033[1;32m> \033[0m");
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/help") { usage(); continue; }
        if (line == "/reset") { history = base_msgs(); fprintf(stderr, "(history cleared)\n"); continue; }
        if (line == "/stats") { stats = !stats; continue; }
        if (line == "/think on") { o.think = true; continue; }
        if (line == "/think off") { o.think = false; continue; }
        if (line.rfind("/system ", 0) == 0) { o.system = line.substr(8); history = base_msgs(); fprintf(stderr, "(system prompt set, history cleared)\n"); continue; }
        Json u = Json::object(); u.set("role", "user"); u.set("content", line);
        history.push(u);
        StreamResult r;
        bool ok = chat_stream(o, history, r, true);
        printf("\n");
        fflush(stdout);
        if (!ok) { history.arr.pop_back(); continue; }
        if (stats) print_stats(r);
        Json a = Json::object(); a.set("role", "assistant"); a.set("content", r.content);
        history.push(a);
    }
    return 0;
}

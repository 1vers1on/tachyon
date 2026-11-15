// Tachyon - OpenAI-compatible HTTP server (no dependencies: raw sockets, HTTP/1.1 keep-alive, chunked SSE)
//
// Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions, POST /tokenize
#include "server.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <random>
#include <deque>
#include <atomic>
#include <algorithm>

namespace {

struct HttpRequest {
    std::string method, path, body;
    std::vector<std::pair<std::string, std::string>> headers;
    bool keep_alive = true;
    std::string header(const std::string& k) const {
        for (auto& h : headers) if (strcasecmp(h.first.c_str(), k.c_str()) == 0) return h.second;
        return "";
    }
};

class Conn {
public:
    explicit Conn(int fd) : fd_(fd) {}
    ~Conn() { if (fd_ >= 0) ::close(fd_); }
    bool read_request(HttpRequest& r) {
        size_t hdr_end;
        while ((hdr_end = buf_.find("\r\n\r\n")) == std::string::npos) {
            if (buf_.size() > (1 << 20) || !fill()) return false;
        }
        std::string head = buf_.substr(0, hdr_end);
        buf_.erase(0, hdr_end + 4);
        size_t le = head.find("\r\n");
        std::string line = head.substr(0, le);
        size_t a = line.find(' '), b = line.rfind(' ');
        if (a == std::string::npos || b == a) return false;
        r.method = line.substr(0, a);
        r.path = line.substr(a + 1, b - a - 1);
        std::string ver = line.substr(b + 1);
        size_t q = r.path.find('?');
        if (q != std::string::npos) r.path = r.path.substr(0, q);
        size_t p = le == std::string::npos ? head.size() : le + 2;
        while (p < head.size()) {
            size_t e = head.find("\r\n", p);
            if (e == std::string::npos) e = head.size();
            std::string h = head.substr(p, e - p);
            size_t c = h.find(':');
            if (c != std::string::npos) {
                std::string v = h.substr(c + 1);
                while (!v.empty() && v[0] == ' ') v.erase(0, 1);
                r.headers.push_back({h.substr(0, c), v});
            }
            p = e + 2;
        }
        std::string conn = r.header("Connection");
        r.keep_alive = ver == "HTTP/1.1" ? strcasecmp(conn.c_str(), "close") != 0 : strcasecmp(conn.c_str(), "keep-alive") == 0;
        size_t cl = atoll(r.header("Content-Length").c_str());
        if (cl > (256u << 20)) return false;
        while (buf_.size() < cl) if (!fill()) return false;
        r.body = buf_.substr(0, cl);
        buf_.erase(0, cl);
        return true;
    }
    bool write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::send(fd_, s.data() + off, s.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return false;
            off += n;
        }
        return true;
    }
    bool send_response(int code, const std::string& ctype, const std::string& body, bool keep_alive) {
        std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n";
        h += "Content-Type: " + ctype + "\r\n";
        h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        h += cors();
        h += keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
        return write_all(h + body);
    }
    bool start_chunked(const std::string& ctype) {
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: " + ctype + "\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n" + cors() +
                        "Connection: keep-alive\r\n\r\n";
        return write_all(h);
    }
    bool chunk(const std::string& data) {
        if (data.empty()) return true;
        char hx[32];
        snprintf(hx, sizeof hx, "%zx\r\n", data.size());
        return write_all(std::string(hx) + data + "\r\n");
    }
    bool end_chunked() { return write_all("0\r\n\r\n"); }

private:
    static std::string cors() {
        return "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    }
    static const char* status_text(int c) {
        switch (c) { case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request"; case 404: return "Not Found";
                     case 500: return "Internal Server Error"; case 503: return "Service Unavailable"; }
        return "OK";
    }
    bool fill() {
        char tmp[65536];
        ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf_.append(tmp, n);
        return true;
    }
    int fd_;
    std::string buf_;
};

std::string rand_id(const char* prefix) {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    char b[64];
    snprintf(b, sizeof b, "%s%016llx", prefix, (unsigned long long)rng());
    return b;
}

// Extract Qwen/Hermes-style <tool_call>{json}</tool_call> blocks from generated text.
// Returns the remaining text; fills `calls` with OpenAI tool_call objects.
std::string extract_tool_calls(const std::string& text, Json& calls) {
    calls = Json::array();
    std::string rest;
    size_t p = 0;
    while (true) {
        size_t a = text.find("<tool_call>", p);
        if (a == std::string::npos) { rest += text.substr(p); break; }
        rest += text.substr(p, a - p);
        size_t b = text.find("</tool_call>", a);
        std::string body = text.substr(a + 11, b == std::string::npos ? std::string::npos : b - a - 11);
        std::string err;
        Json j = Json::parse(body.substr(body.find_first_not_of(" \n\r\t") == std::string::npos ? 0 : body.find_first_not_of(" \n\r\t")), &err);
        if (!err.empty()) {
            // Qwen3.5 XML style: <function=NAME><parameter=K>\nV\n</parameter>...</function>
            size_t f = body.find("<function=");
            if (f != std::string::npos) {
                size_t fe = body.find('>', f);
                Json x = Json::object();
                x.set("name", body.substr(f + 10, fe - f - 10));
                Json args = Json::object();
                size_t q = fe;
                while ((q = body.find("<parameter=", q)) != std::string::npos) {
                    size_t qe = body.find('>', q);
                    std::string key = body.substr(q + 11, qe - q - 11);
                    size_t ve = body.find("</parameter>", qe);
                    std::string val = body.substr(qe + 1, ve == std::string::npos ? std::string::npos : ve - qe - 1);
                    if (!val.empty() && val[0] == '\n') val.erase(0, 1);
                    if (!val.empty() && val.back() == '\n') val.pop_back();
                    std::string e2;
                    Json pv = Json::parse(val, &e2);
                    args.set(key, e2.empty() && !pv.is_str() ? pv : Json(val));
                    if (ve == std::string::npos) break;
                    q = ve;
                }
                x.set("arguments", args);
                j = x;
                err.clear();
            }
        }
        if (err.empty() && j.is_obj() && j.has("name")) {
            Json c = Json::object();
            static std::atomic<uint64_t> ctr{0};
            char id[48];
            snprintf(id, sizeof id, "call_%llx%04llx", (unsigned long long)time(nullptr), (unsigned long long)(ctr++ & 0xffff));
            c.set("id", std::string(id));
            c.set("type", "function");
            Json f = Json::object();
            f.set("name", j["name"].as_str());
            const Json& args = j["arguments"];
            f.set("arguments", args.is_str() ? args.str : args.dump());
            c.set("function", f);
            calls.push(c);
        } else {
            rest += text.substr(a, b == std::string::npos ? std::string::npos : b + 12 - a);
        }
        if (b == std::string::npos) break;
        p = b + 12;
    }
    size_t s0 = rest.find_first_not_of(" \n"), s1 = rest.find_last_not_of(" \n");
    return s0 == std::string::npos ? "" : rest.substr(s0, s1 - s0 + 1);
}

Json error_json(const std::string& msg, const std::string& type = "invalid_request_error") {
    Json e = Json::object();
    Json in = Json::object();
    in.set("message", msg);
    in.set("type", type);
    e.set("error", in);
    return e;
}

// Splits "<think>...</think>" reasoning from the visible answer, robust to tags split across deltas.
class ThinkSplitter {
public:
    void init(bool active, bool starts_inside, const std::string& open = "<think>", const std::string& close = "</think>") {
        active_ = active; mode_ = active ? (starts_inside ? 1 : 0) : 2; open_ = open; close_ = close;
    }
    // returns (reasoning_delta, content_delta)
    std::pair<std::string, std::string> push(const std::string& d, bool final = false) {
        std::pair<std::string, std::string> out;
        buf_ += d;
        while (true) {
            if (mode_ == 2) { out.second += buf_; buf_.clear(); break; }
            if (mode_ == 0) {  // undecided: does output start with <think>?
                size_t i = 0;
                while (i < buf_.size() && (buf_[i] == '\n' || buf_[i] == ' ')) i++;
                std::string rest = buf_.substr(i);
                const std::string& tag = open_;
                if (rest.compare(0, tag.size(), tag) == 0) { buf_ = rest.substr(tag.size()); mode_ = 1; continue; }
                if (rest.size() < tag.size() && tag.compare(0, rest.size(), rest) == 0 && !final) break;
                mode_ = 2;
                continue;
            }
            // mode 1: inside reasoning
            const std::string& tag = close_;
            size_t p = buf_.find(tag);
            if (p != std::string::npos) {
                out.first += buf_.substr(0, p);
                buf_ = buf_.substr(p + tag.size());
                size_t a = 0;
                while (a < buf_.size() && buf_[a] == '\n') a++;
                buf_ = buf_.substr(a);
                mode_ = 3;
                continue;
            }
            size_t keep = 0;
            for (size_t k = std::min(tag.size() - 1, buf_.size()); k > 0; k--)
                if (buf_.compare(buf_.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
            if (final) keep = 0;
            out.first += buf_.substr(0, buf_.size() - keep);
            buf_ = buf_.substr(buf_.size() - keep);
            break;
        }
        if (mode_ == 3) {  // strip leading newlines right after </think>
            if (!seen_content_) {
                size_t a = 0;
                while (a < buf_.size() && buf_[a] == '\n') a++;
                buf_ = buf_.substr(a);
                if (!buf_.empty()) seen_content_ = true;
            }
            out.second += buf_;
            buf_.clear();
        }
        if (final && !buf_.empty()) { (mode_ == 1 ? out.first : out.second) += buf_; buf_.clear(); }
        return out;
    }
private:
    bool active_ = false, seen_content_ = false;
    int mode_ = 2;  // 0 undecided, 1 reasoning, 2 plain, 3 after reasoning
    std::string buf_, open_ = "<think>", close_ = "</think>";
};

class Server {
public:
    explicit Server(Engine& e) : e_(e) {}

    void handle(int fd) {
        Conn c(fd);
        while (true) {
            HttpRequest r;
            if (!c.read_request(r)) return;
            bool ok = route(c, r);
            if (!ok || !r.keep_alive) return;
        }
    }

private:
    bool route(Conn& c, const HttpRequest& r) {
        logd("%s %s", r.method.c_str(), r.path.c_str());
        if (r.method == "OPTIONS") return c.send_response(204, "text/plain", "", r.keep_alive);
        if (r.method == "GET" && (r.path == "/health" || r.path == "/v1/health")) {
            Json j = Json::object(); j.set("status", "ok");
            return c.send_response(200, "application/json", j.dump(), r.keep_alive);
        }
        if (r.method == "GET" && (r.path == "/v1/models" || r.path == "/models")) {
            Json m = Json::object();
            m.set("id", e_.model_name);
            m.set("object", "model");
            m.set("created", (int64_t)time(nullptr));
            m.set("owned_by", "tachyon");
            m.set("max_model_len", e_.opt.backend.ctx);
            Json l = Json::object();
            l.set("object", "list");
            Json arr = Json::array(); arr.push(m);
            l.set("data", arr);
            return c.send_response(200, "application/json", l.dump(), r.keep_alive);
        }
        if (r.method == "GET" && r.path == "/") {
            std::string s = "Tachyon " TACH_VERSION " serving " + e_.model_name + " on " + e_.be->name() + "\n";
            return c.send_response(200, "text/plain", s, r.keep_alive);
        }
        if (r.method == "POST" && (r.path == "/v1/chat/completions" || r.path == "/chat/completions")) return completions(c, r, true);
        if (r.method == "POST" && (r.path == "/v1/completions" || r.path == "/completions")) return completions(c, r, false);
        if (r.method == "POST" && r.path == "/tokenize") {
            std::string err;
            Json j = Json::parse(r.body, &err);
            auto ids = e_.tok.encode(j.get_str("content", j.get_str("prompt", "")));
            Json o = Json::object(); Json a = Json::array();
            for (int id : ids) a.push(id);
            o.set("tokens", a);
            return c.send_response(200, "application/json", o.dump(), r.keep_alive);
        }
        return c.send_response(404, "application/json", error_json("not found: " + r.path).dump(), r.keep_alive);
    }

    static std::string content_text(const Json& c) {
        if (c.is_str()) return c.str;
        std::string s;
        if (c.is_arr())
            for (auto& part : c.arr) {
                if (part.is_str()) s += part.str;
                else if (part.get_str("type", "") == "text") s += part.get_str("text", "");
            }
        return s;
    }

    bool completions(Conn& c, const HttpRequest& hr, bool chat) {
        std::string err;
        Json j = Json::parse(hr.body, &err);
        if (!err.empty() || !j.is_obj()) return c.send_response(400, "application/json", error_json("invalid JSON body: " + err).dump(), hr.keep_alive);
        auto req = std::make_shared<GenRequest>();
        // sampling defaults: model generation_config, then request overrides
        const Json& gd = e_.gen_defaults;
        SamplingParams sp;
        sp.temperature = (float)gd.get_num("temperature", 0.7);
        sp.top_k = (int)gd.get_int("top_k", 40);
        sp.top_p = (float)gd.get_num("top_p", 0.95);
        sp.min_p = (float)gd.get_num("min_p", 0.0);
        sp.repetition_penalty = (float)gd.get_num("repetition_penalty", 1.0);
        sp.temperature = (float)j.get_num("temperature", sp.temperature);
        sp.top_p = (float)j.get_num("top_p", sp.top_p);
        sp.top_k = (int)j.get_int("top_k", sp.top_k);
        sp.min_p = (float)j.get_num("min_p", sp.min_p);
        sp.presence_penalty = (float)j.get_num("presence_penalty", 0);
        sp.frequency_penalty = (float)j.get_num("frequency_penalty", 0);
        sp.repetition_penalty = (float)j.get_num("repetition_penalty", sp.repetition_penalty);
        sp.seed = (uint64_t)j.get_int("seed", 0);
        req->sp = sp;
        int max_tokens = (int)j.get_int("max_completion_tokens", j.get_int("max_tokens", -1));
        if (max_tokens <= 0) max_tokens = e_.opt.backend.ctx;
        req->max_tokens = max_tokens;
        const Json& stop = j["stop"];
        if (stop.is_str()) req->stop.push_back(stop.str);
        else for (auto& s : stop.arr) if (s.is_str()) req->stop.push_back(s.str);
        req->ignore_eos = j.get_bool("ignore_eos", false);
        const bool stream = j.get_bool("stream", false);
        const bool include_usage = j["stream_options"].get_bool("include_usage", false);

        bool thinking = false;
        std::vector<std::string> tools;
        if (chat && j["tools"].is_arr() && j.get_str("tool_choice", "auto") != "none")
            for (auto& t : j["tools"].arr) tools.push_back(t.dump(-2));
        const bool use_tools = !tools.empty();
        if (chat) {
            std::vector<ChatMessage> msgs;
            for (auto& m : j["messages"].arr) {
                ChatMessage cm;
                cm.role = m.get_str("role", "user");
                if (cm.role == "developer") cm.role = "system";
                cm.content = content_text(m["content"]);
                cm.reasoning = m.get_str("reasoning_content", m.get_str("reasoning", ""));
                if (cm.role == "assistant" && m["tool_calls"].is_arr()) {
                    bool firstc = true;
                    for (auto& tc : m["tool_calls"].arr) {
                        const Json& f = tc.has("function") ? tc["function"] : tc;
                        if (!firstc || !cm.content.empty()) cm.content += "\n";
                        firstc = false;
                        const Json& args = f["arguments"];
                        if (e_.tok.xml_tools()) {
                            std::string e2;
                            Json av = args.is_str() ? Json::parse(args.str, &e2) : args;
                            cm.content += "<tool_call>\n<function=" + f.get_str("name", "") + ">\n";
                            for (auto& kv : av.obj)
                                cm.content += "<parameter=" + kv.first + ">\n" + (kv.second.is_str() ? kv.second.str : kv.second.dump(-2)) + "\n</parameter>\n";
                            cm.content += "</function>\n</tool_call>";
                        } else {
                            cm.content += "<tool_call>\n{\"name\": \"" + f.get_str("name", "") + "\", \"arguments\": " +
                                          (args.is_str() ? args.str : args.dump(-2)) + "}\n</tool_call>";
                        }
                    }
                }
                msgs.push_back(cm);
            }
            if (msgs.empty()) return c.send_response(400, "application/json", error_json("messages must be a non-empty array").dump(), hr.keep_alive);
            bool enable_thinking = e_.tok.think_default();
            const Json& kw = j["chat_template_kwargs"];
            if (kw.has("enable_thinking")) enable_thinking = kw.get_bool("enable_thinking", true);
            if (j.has("enable_thinking")) enable_thinking = j.get_bool("enable_thinking", true);
            if (j.get_str("reasoning_effort", "") == "none") enable_thinking = false;
            std::string prompt = e_.tok.render_chat(msgs, true, enable_thinking, tools);
            req->prompt = e_.tok.encode(prompt);
            thinking = e_.tok.thinking_model() && enable_thinking;
        } else {
            const Json& p = j["prompt"];
            if (p.is_str()) req->prompt = e_.tok.encode(p.str);
            else if (p.is_arr() && !p.arr.empty() && p[0].is_num()) for (auto& t : p.arr) req->prompt.push_back((int)t.as_int());
            else if (p.is_arr() && !p.arr.empty() && p[0].is_str()) req->prompt = e_.tok.encode(p[0].str);
            else return c.send_response(400, "application/json", error_json("prompt is required").dump(), hr.keep_alive);
        }
        const std::string id = rand_id(chat ? "chatcmpl-" : "cmpl-");
        const int64_t created = time(nullptr);
        const std::string object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

        // deltas from the engine thread -> this connection thread
        struct Q { std::mutex mu; std::condition_variable cv; std::deque<std::string> items; bool done = false; };
        auto q = std::make_shared<Q>();
        if (stream) {
            req->on_text = [q](const std::string& d) {
                std::lock_guard<std::mutex> lk(q->mu);
                q->items.push_back(d);
                q->cv.notify_one();
                return true;
            };
        }
        e_.submit(req);

        ThinkSplitter split;
        split.init(chat && thinking, chat && e_.tok.think_opens_in_prompt(thinking), e_.tok.think_open_tag(), e_.tok.think_close_tag());
        std::string tool_buf;
        bool in_tool = false;

        auto make_chunk = [&](const std::string& reasoning, const std::string& content, const char* finish, bool role) {
            Json o = Json::object();
            o.set("id", id);
            o.set("object", object);
            o.set("created", created);
            o.set("model", e_.model_name);
            Json ch = Json::object();
            ch.set("index", 0);
            if (chat) {
                Json d = Json::object();
                if (role) d.set("role", "assistant");
                if (!reasoning.empty()) d.set("reasoning_content", reasoning);
                if (!content.empty() || (!role && reasoning.empty() && !finish)) d.set("content", content);
                ch.set("delta", d);
            } else {
                ch.set("text", content);
            }
            ch.set("logprobs", Json());
            ch.set("finish_reason", finish ? Json(finish) : Json());
            Json arr = Json::array(); arr.push(ch);
            o.set("choices", arr);
            return "data: " + o.dump() + "\n\n";
        };
        auto usage_json = [&]() {
            Json u = Json::object();
            u.set("prompt_tokens", req->n_prompt);
            u.set("completion_tokens", req->n_gen);
            u.set("total_tokens", req->n_prompt + req->n_gen);
            Json det = Json::object();
            det.set("cached_tokens", req->n_cached);
            u.set("prompt_tokens_details", det);
            return u;
        };
        auto timings_json = [&]() {
            Json t = Json::object();
            double pp = req->t_prefill_done - req->t_submit, gen = req->t_done - req->t_first;
            t.set("prompt_ms", pp * 1000);
            t.set("prompt_per_second", (req->n_prompt - req->n_cached) / std::max(pp, 1e-9));
            t.set("predicted_ms", gen * 1000);
            t.set("predicted_per_second", req->n_gen > 1 ? (req->n_gen - 1) / std::max(gen, 1e-9) : 0.0);
            t.set("ttft_ms", (req->t_first - req->t_submit) * 1000);
            t.set("draft_tokens", req->n_draft);
            t.set("draft_accepted", req->n_draft_accepted);
            return t;
        };

        if (stream) {
            if (!c.start_chunked("text/event-stream")) { req->cancel = true; return false; }
            bool alive = true;
            if (chat) alive = c.chunk(make_chunk("", "", nullptr, true));
            while (true) {
                std::deque<std::string> items;
                bool fin;
                {
                    std::unique_lock<std::mutex> lk(q->mu);
                    q->cv.wait_for(lk, std::chrono::milliseconds(20), [&] { return !q->items.empty(); });
                    items.swap(q->items);
                    std::lock_guard<std::mutex> lk2(req->mu);
                    fin = req->done;
                }
                std::string out;
                auto emit = [&](std::pair<std::string, std::string> pr) {
                    if (use_tools) {
                        // content is streamed until a tool call starts; tool calls are sent whole at the end
                        tool_buf += pr.second;
                        pr.second.clear();
                        if (!in_tool) {
                            size_t a = tool_buf.find("<tool_call>");
                            size_t safe = a != std::string::npos ? a : tool_buf.size();
                            if (a == std::string::npos)
                                for (size_t k = std::min<size_t>(10, tool_buf.size()); k > 0; k--)
                                    if (tool_buf.compare(tool_buf.size() - k, k, "<tool_call>", 0, k) == 0) { safe = tool_buf.size() - k; break; }
                            pr.second = tool_buf.substr(0, safe);
                            tool_buf.erase(0, safe);
                            if (a != std::string::npos) in_tool = true;
                        }
                    }
                    if (!pr.first.empty() || !pr.second.empty()) out += make_chunk(pr.first, pr.second, nullptr, false);
                };
                for (auto& d : items) emit(split.push(d));
                if (fin) {
                    // drain anything that arrived after the swap
                    std::lock_guard<std::mutex> lk(q->mu);
                    for (auto& d : q->items) emit(split.push(d));
                    q->items.clear();
                    emit(split.push("", true));
                }
                if (alive && !out.empty() && !c.chunk(out)) { alive = false; req->cancel = true; }
                if (fin) break;
            }
            if (!alive) { log_request(*req); return false; }
            if (req->finish_reason == "error") {
                c.chunk("data: " + error_json(req->error).dump() + "\n\n");
            } else {
                std::string fr = req->finish_reason == "cancelled" ? "stop" : req->finish_reason;
                if (use_tools && !tool_buf.empty()) {
                    Json calls;
                    std::string rest = extract_tool_calls(tool_buf, calls);
                    if (calls.size()) {
                        for (size_t k = 0; k < calls.arr.size(); k++) calls.arr[k].set("index", (int)k);
                        Json o = Json::object();
                        o.set("id", id); o.set("object", object); o.set("created", created); o.set("model", e_.model_name);
                        Json ch = Json::object(); ch.set("index", 0);
                        Json d = Json::object(); d.set("tool_calls", calls);
                        if (!rest.empty()) d.set("content", rest);
                        ch.set("delta", d); ch.set("finish_reason", Json());
                        Json arr = Json::array(); arr.push(ch); o.set("choices", arr);
                        c.chunk("data: " + o.dump() + "\n\n");
                        if (fr == "stop") fr = "tool_calls";
                    } else c.chunk(make_chunk("", tool_buf, nullptr, false));
                }
                std::string last = make_chunk("", "", fr.c_str(), false);
                c.chunk(last);
                if (include_usage) {
                    Json o = Json::object();
                    o.set("id", id); o.set("object", object); o.set("created", created); o.set("model", e_.model_name);
                    o.set("choices", Json::array());
                    o.set("usage", usage_json());
                    o.set("timings", timings_json());
                    c.chunk("data: " + o.dump() + "\n\n");
                }
            }
            c.chunk("data: [DONE]\n\n");
            log_request(*req);
            return c.end_chunked();
        }
        req->wait();
        log_request(*req);
        if (req->finish_reason == "error") return c.send_response(400, "application/json", error_json(req->error).dump(), hr.keep_alive);
        Json o = Json::object();
        o.set("id", id);
        o.set("object", object);
        o.set("created", created);
        o.set("model", e_.model_name);
        Json ch = Json::object();
        ch.set("index", 0);
        auto pr = split.push(req->text, true);
        std::string finish = req->finish_reason == "cancelled" ? "stop" : req->finish_reason;
        if (chat) {
            Json msg = Json::object();
            msg.set("role", "assistant");
            Json calls;
            std::string content = use_tools ? extract_tool_calls(pr.second, calls) : pr.second;
            msg.set("content", content.empty() && use_tools && calls.size() ? Json() : Json(content));
            if (!pr.first.empty()) msg.set("reasoning_content", pr.first);
            if (use_tools && calls.size()) {
                msg.set("tool_calls", calls);
                if (finish == "stop") finish = "tool_calls";
            }
            ch.set("message", msg);
        } else {
            ch.set("text", pr.first + pr.second);
        }
        ch.set("logprobs", Json());
        ch.set("finish_reason", finish);
        Json arr = Json::array(); arr.push(ch);
        o.set("choices", arr);
        o.set("usage", usage_json());
        o.set("timings", timings_json());
        return c.send_response(200, "application/json", o.dump(), hr.keep_alive);
    }

    void log_request(const GenRequest& r) {
        double pp = r.t_prefill_done - r.t_submit, gen = r.t_done - r.t_first;
        logi("request: prompt %d (%d cached, %.0f tok/s), gen %d tok @ %.1f tok/s, spec %d/%d, %s", r.n_prompt, r.n_cached,
             (r.n_prompt - r.n_cached) / std::max(pp, 1e-9), r.n_gen, r.n_gen > 1 ? (r.n_gen - 1) / std::max(gen, 1e-9) : 0.0,
             r.n_draft_accepted, r.n_draft, r.finish_reason.c_str());
    }

    Engine& e_;
};

}  // namespace

int server_main(Engine& e, const std::string& host, int port) {
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket failed");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host == "localhost" ? "127.0.0.1" : host.c_str(), &addr.sin_addr) != 1) die("bad host %s", host.c_str());
    if (bind(fd, (sockaddr*)&addr, sizeof addr) < 0) die("cannot bind %s:%d (%s)", host.c_str(), port, strerror(errno));
    if (listen(fd, 128) < 0) die("listen failed");
    logi("listening on http://%s:%d  (OpenAI API at /v1)", host.c_str(), port);
    static Server srv(e);
    while (true) {
        sockaddr_in cli{};
        socklen_t cl = sizeof cli;
        int c = accept(fd, (sockaddr*)&cli, &cl);
        if (c < 0) { if (errno == EINTR) continue; break; }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([c] { srv.handle(c); }).detach();
    }
    return 0;
}

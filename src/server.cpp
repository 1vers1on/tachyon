// Tachyon - HTTP server: OpenAI-compatible and Anthropic-compatible APIs (no dependencies: raw sockets,
// HTTP/1.1 keep-alive, chunked SSE).
//
// OpenAI:    POST /v1/chat/completions, POST /v1/completions, GET /v1/models[/{id}]
// Anthropic: POST /v1/messages, POST /v1/messages/count_tokens
// Other:     GET /health, POST /tokenize, POST /detokenize
#include "server.h"
#include "http.h"
#include "chat.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <random>
#include <deque>
#include <map>
#include <atomic>
#include <algorithm>

namespace {

std::string rand_id(const char* prefix, int n = 24) {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mu;
    static const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::lock_guard<std::mutex> lk(mu);
    std::string s = prefix;
    for (int i = 0; i < n; i++) s += al[rng() % 62];
    return s;
}

std::string iso_time(time_t t) {
    char b[64];
    struct tm tmv;
    gmtime_r(&t, &tmv);
    strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return b;
}

Json oai_error(const std::string& msg, const std::string& type = "invalid_request_error") {
    Json in = Json::object();
    in.set("message", msg);
    in.set("type", type);
    in.set("code", Json());
    Json e = Json::object();
    e.set("error", in);
    return e;
}

Json anth_error(const std::string& msg, const std::string& type = "invalid_request_error") {
    Json in = Json::object();
    in.set("type", type);
    in.set("message", msg);
    Json e = Json::object();
    e.set("type", "error");
    e.set("error", in);
    return e;
}

// text of an OpenAI / Anthropic content value: string, or array of parts (text parts concatenated)
std::string content_text(const Json& c, const char* sep = "") {
    if (c.is_str()) return c.str;
    std::string s;
    if (c.is_arr())
        for (auto& part : c.arr) {
            std::string t;
            if (part.is_str()) t = part.str;
            else {
                const std::string ty = part.get_str("type", "");
                if (ty == "text" || ty == "input_text" || ty == "output_text") t = part.get_str("text", "");
                else if (ty == "image" || ty == "image_url" || ty == "input_image") t = "[image omitted: this model cannot see images]";
                else if (ty == "document" && part["source"].get_str("type", "") == "text") t = part["source"].get_str("data", "");
                else if (ty == "tool_result") t = content_text(part["content"], sep);
            }
            if (t.empty()) continue;
            if (!s.empty()) s += sep;
            s += t;
        }
    return s;
}

// protocol-independent chat request
struct ChatSpec {
    std::vector<ChatMessage> msgs;
    std::vector<Json> tools;          // OpenAI shape {"type": "function", "function": {...}}
    std::string tool_choice = "auto"; // auto | none | required | named
    std::string tool_name;            // tool_choice == named
    int thinking = -1;                // -1 model default, 0 off, 1 on
    std::string format_instruction;   // JSON output requested (response_format / output_config.format)
    SamplingParams sp;
    int max_tokens = -1;
    std::vector<std::string> stop;
    bool ignore_eos = false;
};

struct Prepared {
    std::shared_ptr<GenRequest> req;
    bool reasoning = false, starts_inside = false, tools = false;
    std::string prefill;   // forced tool call: generated text starts with this (already in the prompt)
};

// receiver of the processed output stream
struct Events {
    virtual ~Events() {}
    virtual void reasoning(const std::string&) {}
    virtual void text(const std::string&) {}
    virtual void tool_call(ToolCall&) {}
};

std::string json_instruction(const Json& schema) {
    std::string s = "Respond only with a single valid JSON value, without code fences or any other text.";
    if (schema.is_obj() && schema.size()) s += " It must conform to this JSON schema:\n" + schema.dump(-2);
    return s;
}

class Server {
public:
    Server(Engine& e, const ServerOptions& o) : e_(e), o_(o) { alias_ = o.alias.empty() ? e.model_name : o.alias; }

    void handle(int fd) {
        Conn c(fd);
        while (true) {
            HttpRequest r;
            int st = 0;
            if (!c.read_request(r, &st)) {
                if (st) c.send_response(st, "application/json", oai_error(Conn::status_text(st)).dump(), false);
                return;
            }
            bool ok = route(c, r);
            if (!ok || !r.keep_alive) return;
        }
    }

private:
    static bool is_anthropic(const HttpRequest& r) { return r.path.find("/messages") != std::string::npos; }

    bool send_error(Conn& c, const HttpRequest& r, int code, const std::string& msg, const std::string& type = "invalid_request_error") {
        std::string atype = code == 401 ? "authentication_error" : code == 404 ? "not_found_error" : code >= 500 ? "api_error" : type;
        return c.send_response(code, "application/json", is_anthropic(r) ? anth_error(msg, atype).dump() : oai_error(msg, type).dump(), r.keep_alive);
    }

    bool authorized(const HttpRequest& r) const {
        if (o_.api_key.empty()) return true;
        std::string a = r.header("Authorization");
        if (a.size() > 7 && strncasecmp(a.c_str(), "Bearer ", 7) == 0 && a.substr(7) == o_.api_key) return true;
        return r.header("x-api-key") == o_.api_key;
    }

    Json model_json() const {
        Json m = Json::object();
        const time_t created = time(nullptr);
        m.set("id", alias_);
        m.set("object", "model");
        m.set("type", "model");
        m.set("created", (int64_t)created);
        m.set("created_at", iso_time(created));
        m.set("owned_by", "tachyon");
        m.set("display_name", alias_);
        m.set("max_model_len", e_.opt.backend.ctx);
        m.set("context_length", e_.opt.backend.ctx);
        m.set("max_input_tokens", e_.opt.backend.ctx);
        return m;
    }

    bool route(Conn& c, HttpRequest& r) {
        logd("%s %s", r.method.c_str(), r.path.c_str());
        if (r.method == "OPTIONS") return c.send_response(204, "text/plain", "", r.keep_alive);
        if (r.method == "GET" && (r.path == "/health" || r.path == "/v1/health")) {
            auto st = e_.stats();
            Json j = Json::object();
            j.set("status", "ok");
            j.set("model", alias_);
            j.set("slots", st.slots);
            j.set("active", st.active);
            j.set("queued", st.queued);
            j.set("ctx", e_.opt.backend.ctx);
            return c.send_response(200, "application/json", j.dump(), r.keep_alive);
        }
        if (r.method == "GET" && r.path == "/") {
            std::string s = "Tachyon " TACH_VERSION " serving " + alias_ + " on " + e_.be->name() + "\n";
            return c.send_response(200, "text/plain", s, r.keep_alive);
        }
        if (!authorized(r)) return send_error(c, r, 401, "invalid API key");
        if (r.method == "GET" && (r.path == "/v1/models" || r.path == "/models")) {
            Json l = Json::object();
            l.set("object", "list");
            Json arr = Json::array();
            arr.push(model_json());
            l.set("data", arr);
            l.set("has_more", false);
            l.set("first_id", alias_);
            l.set("last_id", alias_);
            return c.send_response(200, "application/json", l.dump(), r.keep_alive);
        }
        if (r.method == "GET" && r.path.rfind("/v1/models/", 0) == 0) return c.send_response(200, "application/json", model_json().dump(), r.keep_alive);
        if (r.method == "POST" && (r.path == "/v1/chat/completions" || r.path == "/chat/completions")) return openai_chat(c, r);
        if (r.method == "POST" && (r.path == "/v1/completions" || r.path == "/completions")) return openai_completions(c, r);
        if (r.method == "POST" && (r.path == "/v1/messages" || r.path == "/messages")) return anthropic_messages(c, r, false);
        if (r.method == "POST" && (r.path == "/v1/messages/count_tokens" || r.path == "/messages/count_tokens")) return anthropic_messages(c, r, true);
        if (r.method == "POST" && r.path == "/tokenize") {
            Json j = Json::parse(r.body);
            auto ids = e_.tok.encode(j.get_str("content", j.get_str("prompt", "")));
            Json o = Json::object(), a = Json::array();
            for (int id : ids) a.push(id);
            o.set("tokens", a);
            return c.send_response(200, "application/json", o.dump(), r.keep_alive);
        }
        if (r.method == "POST" && r.path == "/detokenize") {
            Json j = Json::parse(r.body);
            std::vector<int> ids;
            for (auto& t : j["tokens"].arr) ids.push_back((int)t.as_int());
            Json o = Json::object();
            o.set("content", e_.tok.decode(ids));
            return c.send_response(200, "application/json", o.dump(), r.keep_alive);
        }
        return send_error(c, r, 404, "not found: " + r.method + " " + r.path, "not_found_error");
    }

    // sampling defaults: model generation_config
    SamplingParams default_sampling() const {
        const Json& gd = e_.gen_defaults;
        SamplingParams sp;
        sp.temperature = (float)gd.get_num("temperature", 0.7);
        sp.top_k = (int)gd.get_int("top_k", 40);
        sp.top_p = (float)gd.get_num("top_p", 0.95);
        sp.min_p = (float)gd.get_num("min_p", 0.0);
        sp.repetition_penalty = (float)gd.get_num("repetition_penalty", 1.0);
        return sp;
    }

    // render + tokenize; returns an error message on failure
    std::string prepare(const ChatSpec& spec, Prepared& p, bool count_only = false) {
        auto req = std::make_shared<GenRequest>();
        std::vector<ChatMessage> msgs = spec.msgs;
        if (msgs.empty()) return "messages must be a non-empty array";
        if (!spec.format_instruction.empty()) {
            if (msgs[0].role == "system") msgs[0].content += "\n\n" + spec.format_instruction;
            else { ChatMessage s; s.role = "system"; s.content = spec.format_instruction; msgs.insert(msgs.begin(), s); }
        }
        p.tools = !spec.tools.empty() && spec.tool_choice != "none";
        const bool forced = p.tools && (spec.tool_choice == "required" || spec.tool_choice == "named");
        if (spec.tool_choice == "named") {
            bool found = false;
            for (auto& t : spec.tools) found |= (t.has("function") ? t["function"] : t).get_str("name", "") == spec.tool_name;
            if (!found) return "tool_choice names an unknown tool: " + spec.tool_name;
        }
        bool think = spec.thinking < 0 ? e_.tok.think_default() : spec.thinking > 0;
        if (forced) think = false;   // the forced call is prefilled right after the generation prompt
        const std::vector<Json> no_tools;
        const std::vector<Json>& tools = p.tools ? spec.tools : no_tools;
        std::string prompt = e_.tok.render_chat(msgs, true, think, tools);
        if (forced) {
            p.prefill = e_.tok.tool_call_prefix(spec.tool_choice == "named" ? spec.tool_name : "");
            prompt += p.prefill;
        }
        req->prompt = e_.tok.encode(prompt);
        const int ctx = e_.opt.backend.ctx;
        if (!count_only && (int)req->prompt.size() + 1 > ctx)
            return "prompt is too long: " + std::to_string(req->prompt.size()) + " tokens > " + std::to_string(ctx - 1) + " maximum";
        if (!count_only && e_.mf.config.stateful) {
            // conversation prefix (all messages, no generation prompt): the engine checkpoints the recurrent state there
            auto pre = e_.tok.encode(e_.tok.render_chat(msgs, false, think, tools));
            if (pre.size() < req->prompt.size() && std::equal(pre.begin(), pre.end(), req->prompt.begin())) req->checkpoint = (int)pre.size();
        }
        p.reasoning = e_.tok.thinking_model() && think;
        if (p.reasoning) {   // generation prompt already opened the reasoning block?
            auto rstrip = [](std::string s) { while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; };
            const std::string tag = rstrip(e_.tok.think_open_tag()), pr = rstrip(prompt);
            p.starts_inside = pr.size() >= tag.size() && pr.compare(pr.size() - tag.size(), tag.size(), tag) == 0;
        }
        req->sp = spec.sp;
        req->max_tokens = spec.max_tokens > 0 ? spec.max_tokens : ctx;
        req->stop = spec.stop;
        req->ignore_eos = spec.ignore_eos;
        p.req = req;
        return "";
    }

    // Runs a prepared request, feeding processed output into `ev`. Streaming: flush() writes what the events
    // buffered (returns -1 on a write error, else bytes written), ping() keeps idle connections alive.
    // Returns false when the client went away (the request is cancelled).
    bool pump(Conn& c, Prepared& p, Events& ev, bool stream, const std::function<long()>& flush, const std::function<bool()>& ping) {
        auto req = p.req;
        struct Q { std::mutex mu; std::condition_variable cv; std::deque<std::string> items; };
        auto q = std::make_shared<Q>();
        req->on_text = [q](const std::string& d) {
            std::lock_guard<std::mutex> lk(q->mu);
            q->items.push_back(d);
            q->cv.notify_one();
            return true;
        };
        e_.submit(req);
        ThinkSplitter split;
        split.init(p.reasoning, p.starts_inside, e_.tok.think_open_tag(), e_.tok.think_close_tag());
        ToolCallParser tp;
        if (p.tools) tp.init(e_.tok.tool_format());
        std::vector<ToolCall> calls;
        auto process = [&](const std::string& d, bool fin) {
            auto pr = split.push(d, fin);
            if (!pr.first.empty()) ev.reasoning(pr.first);
            std::string content = pr.second;
            if (p.tools) { calls.clear(); content = tp.push(pr.second, fin, calls); }
            if (!content.empty()) ev.text(content);
            for (auto& tc : calls) ev.tool_call(tc);
        };
        if (!p.prefill.empty()) process(p.prefill, false);
        double last_io = now_sec(), last_poll = last_io;
        while (true) {
            std::deque<std::string> items;
            {
                std::unique_lock<std::mutex> lk(q->mu);
                q->cv.wait_for(lk, std::chrono::milliseconds(25), [&] { return !q->items.empty(); });
                items.swap(q->items);
            }
            bool fin;
            {
                std::lock_guard<std::mutex> lk(req->mu);
                fin = req->done;
            }
            if (fin) {   // text is always queued before `done` is set
                std::lock_guard<std::mutex> lk(q->mu);
                for (auto& d : q->items) items.push_back(d);
                q->items.clear();
            }
            for (auto& d : items) process(d, false);
            if (fin) process("", true);
            const double t = now_sec();
            if (stream) {
                long n = flush();
                if (n < 0) { req->cancel = true; return false; }
                if (n > 0) last_io = t;
            }
            if (fin) return true;
            if (t - last_poll > 0.2) {
                last_poll = t;
                if (c.peer_closed()) { req->cancel = true; return false; }
            }
            if (stream && t - last_io > 10) {
                if (!ping()) { req->cancel = true; return false; }
                last_io = t;
            }
        }
    }

    Json usage_oai(const GenRequest& r) const {
        Json u = Json::object();
        u.set("prompt_tokens", r.n_prompt);
        u.set("completion_tokens", r.n_gen);
        u.set("total_tokens", r.n_prompt + r.n_gen);
        Json det = Json::object();
        det.set("cached_tokens", r.n_cached);
        u.set("prompt_tokens_details", det);
        return u;
    }
    Json timings_json(const GenRequest& r) const {
        Json t = Json::object();
        double pp = r.t_prefill_done - r.t_submit, gen = r.t_done - r.t_first;
        t.set("prompt_ms", pp * 1000);
        t.set("prompt_per_second", (r.n_prompt - r.n_cached) / std::max(pp, 1e-9));
        t.set("predicted_ms", gen * 1000);
        t.set("predicted_per_second", r.n_gen > 1 ? (r.n_gen - 1) / std::max(gen, 1e-9) : 0.0);
        t.set("ttft_ms", (r.t_first - r.t_submit) * 1000);
        t.set("draft_tokens", r.n_draft);
        t.set("draft_accepted", r.n_draft_accepted);
        return t;
    }

    // ------------------------------------------------------------------------------------------ OpenAI

    void oai_sampling(const Json& j, ChatSpec& s) const {
        s.sp = default_sampling();
        s.sp.temperature = (float)j.get_num("temperature", s.sp.temperature);
        s.sp.top_p = (float)j.get_num("top_p", s.sp.top_p);
        s.sp.top_k = (int)j.get_int("top_k", s.sp.top_k);
        s.sp.min_p = (float)j.get_num("min_p", s.sp.min_p);
        s.sp.presence_penalty = (float)j.get_num("presence_penalty", 0);
        s.sp.frequency_penalty = (float)j.get_num("frequency_penalty", 0);
        s.sp.repetition_penalty = (float)j.get_num("repetition_penalty", s.sp.repetition_penalty);
        s.sp.seed = (uint64_t)j.get_int("seed", 0);
        s.max_tokens = (int)j.get_int("max_completion_tokens", j.get_int("max_tokens", -1));
        const Json& stop = j["stop"];
        if (stop.is_str()) s.stop.push_back(stop.str);
        else for (auto& x : stop.arr) if (x.is_str()) s.stop.push_back(x.str);
        s.ignore_eos = j.get_bool("ignore_eos", false);
    }

    std::string oai_spec(const Json& j, ChatSpec& s) const {
        oai_sampling(j, s);
        if (!j["messages"].is_arr()) return "messages must be an array";
        for (auto& m : j["messages"].arr) {
            ChatMessage cm;
            cm.role = m.get_str("role", "user");
            if (cm.role == "developer") cm.role = "system";
            if (cm.role == "function") cm.role = "tool";
            cm.content = content_text(m["content"]);
            cm.reasoning = m.get_str("reasoning_content", m.get_str("reasoning", ""));
            if (m["tool_calls"].is_arr())
                for (auto& tc : m["tool_calls"].arr) {
                    const Json& f = tc.has("function") ? tc["function"] : tc;
                    ToolCall c;
                    c.id = tc.get_str("id", "");
                    c.name = f.get_str("name", "");
                    const Json& a = f["arguments"];
                    c.arguments = a.is_str() ? a.str : a.is_null() ? "{}" : a.dump();
                    cm.tool_calls.push_back(c);
                }
            cm.tool_call_id = m.get_str("tool_call_id", "");
            cm.name = m.get_str("name", "");
            if (cm.role == "tool" && cm.name.empty())   // resolve the function name from the call id
                for (auto it = s.msgs.rbegin(); it != s.msgs.rend() && cm.name.empty(); ++it)
                    for (auto& tc : it->tool_calls) if (tc.id == cm.tool_call_id) cm.name = tc.name;
            s.msgs.push_back(cm);
        }
        if (j["tools"].is_arr())
            for (auto& t : j["tools"].arr) if (t.get_str("type", "function") == "function") s.tools.push_back(t);
        if (j["functions"].is_arr())   // legacy function calling
            for (auto& f : j["functions"].arr) { Json t = Json::object(); t.set("type", "function"); t.set("function", f); s.tools.push_back(t); }
        const Json& tc = j.has("tool_choice") ? j["tool_choice"] : j["function_call"];
        if (tc.is_str()) s.tool_choice = tc.str == "any" ? "required" : tc.str;
        else if (tc.is_obj()) {
            s.tool_name = tc.has("function") ? tc["function"].get_str("name", "") : tc.get_str("name", "");
            s.tool_choice = s.tool_name.empty() ? "auto" : "named";
        }
        if (s.tool_choice != "auto" && s.tool_choice != "none" && s.tool_choice != "required" && s.tool_choice != "named") s.tool_choice = "auto";
        const Json& kw = j["chat_template_kwargs"];
        if (kw.has("enable_thinking")) s.thinking = kw.get_bool("enable_thinking", true);
        if (j.has("enable_thinking")) s.thinking = j.get_bool("enable_thinking", true);
        if (j["think"].is_bool()) s.thinking = j["think"].b;
        const std::string effort = j.get_str("reasoning_effort", "");
        if (effort == "none") s.thinking = 0;
        else if (!effort.empty()) s.thinking = 1;
        const Json& rf = j["response_format"];
        const std::string rft = rf.get_str("type", "");
        if (rft == "json_object") s.format_instruction = json_instruction(Json());
        else if (rft == "json_schema") s.format_instruction = json_instruction(rf["json_schema"]["schema"]);
        return "";
    }

    std::string req_model(const Json& j) const { std::string m = j.get_str("model", ""); return m.empty() ? alias_ : m; }

    bool openai_chat(Conn& c, const HttpRequest& hr) {
        std::string err;
        Json j = Json::parse(hr.body, &err);
        if (!err.empty() || !j.is_obj()) return send_error(c, hr, 400, "invalid JSON body: " + err);
        ChatSpec spec;
        if (!(err = oai_spec(j, spec)).empty()) return send_error(c, hr, 400, err);
        Prepared p;
        if (!(err = prepare(spec, p)).empty()) return send_error(c, hr, 400, err);
        const bool stream = j.get_bool("stream", false);
        const bool include_usage = j["stream_options"].get_bool("include_usage", false);
        const std::string id = rand_id("chatcmpl-");
        const int64_t created = time(nullptr);
        const std::string model = req_model(j);

        struct Sink : Events {
            bool stream;
            Json head;
            std::string out, content, reasoning_text;
            Json calls = Json::array();
            std::string chunk(const Json& delta, const Json& finish = Json()) {
                Json o = head;
                Json ch = Json::object();
                ch.set("index", 0);
                ch.set("delta", delta);
                ch.set("logprobs", Json());
                ch.set("finish_reason", finish);
                Json arr = Json::array();
                arr.push(ch);
                o.set("choices", arr);
                return "data: " + o.dump() + "\n\n";
            }
            void reasoning(const std::string& r) override {
                if (!stream) { reasoning_text += r; return; }
                Json d = Json::object(); d.set("reasoning_content", r); out += chunk(d);
            }
            void text(const std::string& t) override {
                if (!stream) { content += t; return; }
                Json d = Json::object(); d.set("content", t); out += chunk(d);
            }
            void tool_call(ToolCall& tc) override {
                tc.id = rand_id("call_");
                Json call = Json::object();
                if (stream) call.set("index", (int)calls.size());
                call.set("id", tc.id);
                call.set("type", "function");
                Json f = Json::object();
                f.set("name", tc.name);
                f.set("arguments", tc.arguments);
                call.set("function", f);
                calls.push(call);
                if (stream) { Json d = Json::object(); Json a = Json::array(); a.push(call); d.set("tool_calls", a); out += chunk(d); }
            }
        } sink;
        sink.stream = stream;
        sink.head = Json::object();
        sink.head.set("id", id);
        sink.head.set("object", stream ? "chat.completion.chunk" : "chat.completion");
        sink.head.set("created", created);
        sink.head.set("model", model);
        sink.head.set("system_fingerprint", "tachyon-" TACH_VERSION);

        auto finish_reason = [&]() -> std::string {
            const std::string& f = p.req->finish_reason;
            if (f == "length") return "length";
            return sink.calls.size() ? "tool_calls" : "stop";
        };
        if (stream) {
            if (!c.start_chunked("text/event-stream")) { return false; }
            Json d = Json::object(); d.set("role", "assistant"); d.set("content", "");
            if (!c.chunk(sink.chunk(d))) return false;
            bool alive = pump(c, p, sink, true,
                [&]() -> long { std::string s; s.swap(sink.out); if (s.empty()) return 0; return c.chunk(s) ? (long)s.size() : -1; },
                [&]() { return c.chunk(": keep-alive\n\n"); });
            log_request(*p.req, "chat");
            if (!alive) return false;
            if (p.req->finish_reason == "error") {
                c.chunk("data: " + oai_error(p.req->error, "server_error").dump() + "\n\n");
            } else {
                std::string last = sink.chunk(Json::object(), Json(finish_reason()));
                if (include_usage) {
                    Json o = sink.head;
                    o.set("choices", Json::array());
                    o.set("usage", usage_oai(*p.req));
                    o.set("timings", timings_json(*p.req));
                    last += "data: " + o.dump() + "\n\n";
                }
                c.chunk(last);
            }
            c.chunk("data: [DONE]\n\n");
            return c.end_chunked();
        }
        bool alive = pump(c, p, sink, false, nullptr, nullptr);
        log_request(*p.req, "chat");
        if (!alive) return false;
        if (p.req->finish_reason == "error") return send_error(c, hr, 500, p.req->error, "server_error");
        Json o = sink.head;
        Json msg = Json::object();
        msg.set("role", "assistant");
        msg.set("content", sink.content.empty() && sink.calls.size() ? Json() : Json(sink.content));
        if (!sink.reasoning_text.empty()) msg.set("reasoning_content", sink.reasoning_text);
        if (sink.calls.size()) msg.set("tool_calls", sink.calls);
        Json ch = Json::object();
        ch.set("index", 0);
        ch.set("message", msg);
        ch.set("logprobs", Json());
        ch.set("finish_reason", finish_reason());
        Json arr = Json::array();
        arr.push(ch);
        o.set("choices", arr);
        o.set("usage", usage_oai(*p.req));
        o.set("timings", timings_json(*p.req));
        return c.send_response(200, "application/json", o.dump(), hr.keep_alive);
    }

    bool openai_completions(Conn& c, const HttpRequest& hr) {
        std::string err;
        Json j = Json::parse(hr.body, &err);
        if (!err.empty() || !j.is_obj()) return send_error(c, hr, 400, "invalid JSON body: " + err);
        ChatSpec spec;
        oai_sampling(j, spec);
        Prepared p;
        p.req = std::make_shared<GenRequest>();
        const Json& pj = j["prompt"];
        if (pj.is_str()) p.req->prompt = e_.tok.encode(pj.str);
        else if (pj.is_arr() && !pj.arr.empty() && pj[0].is_num()) for (auto& t : pj.arr) p.req->prompt.push_back((int)t.as_int());
        else if (pj.is_arr() && !pj.arr.empty() && pj[0].is_str()) p.req->prompt = e_.tok.encode(pj[0].str);
        else return send_error(c, hr, 400, "prompt is required");
        const int ctx = e_.opt.backend.ctx;
        if ((int)p.req->prompt.size() + 1 > ctx)
            return send_error(c, hr, 400, "prompt is too long: " + std::to_string(p.req->prompt.size()) + " tokens > " + std::to_string(ctx - 1) + " maximum");
        p.req->sp = spec.sp;
        p.req->max_tokens = spec.max_tokens > 0 ? spec.max_tokens : ctx;
        p.req->stop = spec.stop;
        p.req->ignore_eos = spec.ignore_eos;
        const bool stream = j.get_bool("stream", false);
        const bool include_usage = j["stream_options"].get_bool("include_usage", false);
        const bool echo = j.get_bool("echo", false);
        struct Sink : Events {
            bool stream;
            Json head;
            std::string out, all;
            std::string chunk(const std::string& t, const Json& finish = Json()) {
                Json o = head;
                Json ch = Json::object();
                ch.set("index", 0);
                ch.set("text", t);
                ch.set("logprobs", Json());
                ch.set("finish_reason", finish);
                Json arr = Json::array();
                arr.push(ch);
                o.set("choices", arr);
                return "data: " + o.dump() + "\n\n";
            }
            void text(const std::string& t) override { if (stream) out += chunk(t); else all += t; }
        } sink;
        sink.stream = stream;
        sink.head = Json::object();
        sink.head.set("id", rand_id("cmpl-"));
        sink.head.set("object", "text_completion");
        sink.head.set("created", (int64_t)time(nullptr));
        sink.head.set("model", req_model(j));
        if (echo && pj.is_str()) sink.text(pj.str);
        auto fr = [&]() { return p.req->finish_reason == "length" ? "length" : "stop"; };
        if (stream) {
            if (!c.start_chunked("text/event-stream")) return false;
            bool alive = pump(c, p, sink, true,
                [&]() -> long { std::string s; s.swap(sink.out); if (s.empty()) return 0; return c.chunk(s) ? (long)s.size() : -1; },
                [&]() { return c.chunk(": keep-alive\n\n"); });
            log_request(*p.req, "completion");
            if (!alive) return false;
            if (p.req->finish_reason == "error") c.chunk("data: " + oai_error(p.req->error, "server_error").dump() + "\n\n");
            else {
                std::string last = sink.chunk("", Json(fr()));
                if (include_usage) {
                    Json o = sink.head;
                    o.set("choices", Json::array());
                    o.set("usage", usage_oai(*p.req));
                    last += "data: " + o.dump() + "\n\n";
                }
                c.chunk(last);
            }
            c.chunk("data: [DONE]\n\n");
            return c.end_chunked();
        }
        bool alive = pump(c, p, sink, false, nullptr, nullptr);
        log_request(*p.req, "completion");
        if (!alive) return false;
        if (p.req->finish_reason == "error") return send_error(c, hr, 500, p.req->error, "server_error");
        Json o = sink.head;
        Json ch = Json::object();
        ch.set("index", 0);
        ch.set("text", sink.all);
        ch.set("logprobs", Json());
        ch.set("finish_reason", fr());
        Json arr = Json::array();
        arr.push(ch);
        o.set("choices", arr);
        o.set("usage", usage_oai(*p.req));
        o.set("timings", timings_json(*p.req));
        return c.send_response(200, "application/json", o.dump(), hr.keep_alive);
    }

    // ------------------------------------------------------------------------------------------ Anthropic

    std::string anth_spec(const Json& j, ChatSpec& s) const {
        s.sp = default_sampling();
        s.sp.temperature = (float)j.get_num("temperature", s.sp.temperature);
        s.sp.top_p = (float)j.get_num("top_p", s.sp.top_p);
        s.sp.top_k = (int)j.get_int("top_k", s.sp.top_k);
        s.max_tokens = (int)j.get_int("max_tokens", -1);
        for (auto& x : j["stop_sequences"].arr) if (x.is_str() && !x.str.empty()) s.stop.push_back(x.str);
        const std::string sys = content_text(j["system"], "\n");
        if (!sys.empty()) { ChatMessage m; m.role = "system"; m.content = sys; s.msgs.push_back(m); }
        if (!j["messages"].is_arr()) return "messages: field required";
        std::map<std::string, std::string> id2name;
        for (auto& m : j["messages"].arr) {
            const std::string role = m.get_str("role", "user");
            const Json& content = m["content"];
            if (role != "user" && role != "assistant" && role != "system") return "messages: unexpected role \"" + role + "\"";
            if (content.is_str() || role == "system") {
                ChatMessage cm;
                cm.role = role;
                cm.content = content_text(content, "\n");
                s.msgs.push_back(cm);
                continue;
            }
            if (role == "assistant") {
                ChatMessage a;
                a.role = "assistant";
                for (auto& b : content.arr) {
                    const std::string ty = b.get_str("type", "");
                    if (ty == "text") a.content += b.get_str("text", "");
                    else if (ty == "thinking") a.reasoning += b.get_str("thinking", "");
                    else if (ty == "tool_use" || ty == "server_tool_use") {
                        ToolCall tc;
                        tc.id = b.get_str("id", "");
                        tc.name = b.get_str("name", "");
                        tc.arguments = b["input"].is_obj() ? b["input"].dump() : "{}";
                        id2name[tc.id] = tc.name;
                        a.tool_calls.push_back(tc);
                    }
                }
                s.msgs.push_back(a);
                continue;
            }
            // user: tool results first (as tool messages), then the remaining content as one user turn
            std::string text;
            for (auto& b : content.arr) {
                const std::string ty = b.get_str("type", "");
                if (ty == "tool_result") {
                    ChatMessage t;
                    t.role = "tool";
                    t.tool_call_id = b.get_str("tool_use_id", "");
                    t.name = id2name.count(t.tool_call_id) ? id2name[t.tool_call_id] : "";
                    t.content = content_text(b["content"], "\n");
                    if (b.get_bool("is_error", false)) t.content = "Error: " + t.content;
                    s.msgs.push_back(t);
                } else {
                    Json one = Json::array();
                    one.push(b);
                    std::string x = content_text(one);
                    if (x.empty()) continue;
                    if (!text.empty()) text += "\n";
                    text += x;
                }
            }
            if (!text.empty()) { ChatMessage u; u.role = "user"; u.content = text; s.msgs.push_back(u); }
        }
        for (auto& t : j["tools"].arr) {
            if (!t.has("input_schema")) continue;   // server tools (web search, ...) cannot run here
            Json f = Json::object();
            f.set("name", t.get_str("name", ""));
            f.set("description", t.get_str("description", ""));
            f.set("parameters", t["input_schema"]);
            Json w = Json::object();
            w.set("type", "function");
            w.set("function", f);
            s.tools.push_back(w);
        }
        const Json& tc = j["tool_choice"];
        const std::string tct = tc.get_str("type", "auto");
        if (tct == "any") s.tool_choice = "required";
        else if (tct == "tool") { s.tool_choice = "named"; s.tool_name = tc.get_str("name", ""); }
        else if (tct == "none") s.tool_choice = "none";
        const std::string tht = j["thinking"].get_str("type", "");
        if (tht == "enabled" || tht == "adaptive") s.thinking = 1;
        else if (tht == "disabled") s.thinking = 0;
        const Json& fmt = j["output_config"]["format"];
        if (fmt.get_str("type", "") == "json_schema") s.format_instruction = json_instruction(fmt["schema"]);
        return "";
    }

    bool anthropic_messages(Conn& c, const HttpRequest& hr, bool count_only) {
        std::string err;
        Json j = Json::parse(hr.body, &err);
        if (!err.empty() || !j.is_obj()) return send_error(c, hr, 400, "invalid JSON body: " + err);
        ChatSpec spec;
        if (!(err = anth_spec(j, spec)).empty()) return send_error(c, hr, 400, err);
        Prepared p;
        if (count_only) {
            // count_tokens never fails on length (clients use it to decide when to compact)
            err = prepare(spec, p, true);
            if (!err.empty()) return send_error(c, hr, 400, err);
            Json o = Json::object();
            o.set("input_tokens", (int64_t)p.req->prompt.size());
            return c.send_response(200, "application/json", o.dump(), hr.keep_alive);
        }
        if (!(err = prepare(spec, p)).empty()) return send_error(c, hr, 400, err);
        const bool stream = j.get_bool("stream", false);
        const std::string id = rand_id("msg_");
        const std::string model = req_model(j);

        struct Sink : Events {
            bool stream;
            std::string out;
            Json content = Json::array();   // non-streaming result blocks
            int idx = -1;
            std::string cur;                 // type of the open block
            int n_calls = 0;
            void event(const char* name, const Json& data) { out += std::string("event: ") + name + "\ndata: " + data.dump() + "\n\n"; }
            void close() {
                if (cur.empty()) return;
                if (stream) {
                    if (cur == "thinking") {
                        Json d = Json::object(); d.set("type", "signature_delta"); d.set("signature", "tachyon");
                        Json e = Json::object(); e.set("type", "content_block_delta"); e.set("index", idx); e.set("delta", d);
                        event("content_block_delta", e);
                    }
                    Json e = Json::object(); e.set("type", "content_block_stop"); e.set("index", idx);
                    event("content_block_stop", e);
                }
                cur.clear();
            }
            void open(const std::string& type) {
                if (cur == type) return;
                close();
                idx++;
                cur = type;
                Json b = Json::object();
                b.set("type", type);
                if (type == "thinking") { b.set("thinking", ""); b.set("signature", ""); }
                else b.set("text", "");
                if (stream) {
                    Json e = Json::object(); e.set("type", "content_block_start"); e.set("index", idx); e.set("content_block", b);
                    event("content_block_start", e);
                } else {
                    if (type == "thinking") b.set("signature", "tachyon");
                    content.push(b);
                }
            }
            void delta(const char* dtype, const char* field, const std::string& v) {
                if (stream) {
                    Json d = Json::object(); d.set("type", dtype); d.set(field, v);
                    Json e = Json::object(); e.set("type", "content_block_delta"); e.set("index", idx); e.set("delta", d);
                    event("content_block_delta", e);
                } else {
                    Json& b = content.arr.back();
                    b.set(field, b.get_str(field, "") + v);
                }
            }
            void reasoning(const std::string& r) override { open("thinking"); delta("thinking_delta", "thinking", r); }
            void text(const std::string& t) override { open("text"); delta("text_delta", "text", t); }
            void tool_call(ToolCall& tc) override {
                close();
                idx++;
                n_calls++;
                tc.id = rand_id("toolu_");
                std::string e2;
                Json input = Json::parse(tc.arguments, &e2);
                if (!e2.empty() || !input.is_obj()) input = Json::object();
                Json b = Json::object();
                b.set("type", "tool_use");
                b.set("id", tc.id);
                b.set("name", tc.name);
                if (!stream) { b.set("input", input); content.push(b); return; }
                b.set("input", Json::object());
                Json e = Json::object(); e.set("type", "content_block_start"); e.set("index", idx); e.set("content_block", b);
                event("content_block_start", e);
                Json d = Json::object(); d.set("type", "input_json_delta"); d.set("partial_json", input.dump());
                Json e3 = Json::object(); e3.set("type", "content_block_delta"); e3.set("index", idx); e3.set("delta", d);
                event("content_block_delta", e3);
                Json e4 = Json::object(); e4.set("type", "content_block_stop"); e4.set("index", idx);
                event("content_block_stop", e4);
            }
        } sink;
        sink.stream = stream;

        auto usage = [&](bool final) {
            const GenRequest& r = *p.req;
            Json u = Json::object();
            u.set("input_tokens", final ? r.n_prompt - r.n_cached : (int)r.prompt.size());
            u.set("cache_creation_input_tokens", 0);
            u.set("cache_read_input_tokens", final ? r.n_cached : 0);
            u.set("output_tokens", final ? r.n_gen : 0);
            return u;
        };
        auto stop_reason = [&]() -> std::pair<std::string, Json> {
            const GenRequest& r = *p.req;
            if (r.finish_reason == "length") return {"max_tokens", Json()};
            if (sink.n_calls) return {"tool_use", Json()};
            if (!r.stop_hit.empty()) return {"stop_sequence", Json(r.stop_hit)};
            return {"end_turn", Json()};
        };
        auto message = [&]() {
            Json m = Json::object();
            m.set("id", id);
            m.set("type", "message");
            m.set("role", "assistant");
            m.set("model", model);
            m.set("content", Json::array());
            m.set("stop_reason", Json());
            m.set("stop_sequence", Json());
            m.set("usage", usage(false));
            return m;
        };
        if (stream) {
            if (!c.start_chunked("text/event-stream")) return false;
            Json ms = Json::object();
            ms.set("type", "message_start");
            ms.set("message", message());
            sink.event("message_start", ms);
            Json ping = Json::object();
            ping.set("type", "ping");
            sink.event("ping", ping);
            if (!c.chunk(sink.out)) return false;
            sink.out.clear();
            bool alive = pump(c, p, sink, true,
                [&]() -> long { std::string s; s.swap(sink.out); if (s.empty()) return 0; return c.chunk(s) ? (long)s.size() : -1; },
                [&]() { return c.chunk("event: ping\ndata: {\"type\": \"ping\"}\n\n"); });
            log_request(*p.req, "messages");
            if (!alive) return false;
            if (p.req->finish_reason == "error") {
                sink.event("error", anth_error(p.req->error, "api_error"));
            } else {
                sink.close();
                auto sr = stop_reason();
                Json d = Json::object();
                d.set("stop_reason", sr.first);
                d.set("stop_sequence", sr.second);
                Json md = Json::object();
                md.set("type", "message_delta");
                md.set("delta", d);
                md.set("usage", usage(true));
                sink.event("message_delta", md);
                Json stop = Json::object();
                stop.set("type", "message_stop");
                sink.event("message_stop", stop);
            }
            c.chunk(sink.out);
            return c.end_chunked();
        }
        bool alive = pump(c, p, sink, false, nullptr, nullptr);
        log_request(*p.req, "messages");
        if (!alive) return false;
        if (p.req->finish_reason == "error") return send_error(c, hr, 500, p.req->error, "api_error");
        Json m = message();
        sink.close();
        m.set("content", sink.content);
        auto sr = stop_reason();
        m.set("stop_reason", sr.first);
        m.set("stop_sequence", sr.second);
        m.set("usage", usage(true));
        return c.send_response(200, "application/json", m.dump(), hr.keep_alive);
    }

    void log_request(const GenRequest& r, const char* kind) {
        if (r.t_done == 0) { logi("%s: client disconnected, request cancelled (prompt %d tokens)", kind, (int)r.prompt.size()); return; }
        double pp = r.t_prefill_done - r.t_submit, gen = r.t_done - r.t_first;
        logi("%s: prompt %d (%d cached, %.0f tok/s), gen %d tok @ %.1f tok/s, spec %d/%d, %s", kind, r.n_prompt, r.n_cached,
             (r.n_prompt - r.n_cached) / std::max(pp, 1e-9), r.n_gen, r.n_gen > 1 ? (r.n_gen - 1) / std::max(gen, 1e-9) : 0.0,
             r.n_draft_accepted, r.n_draft, r.finish_reason.empty() ? "cancelled" : r.finish_reason.c_str());
    }

    Engine& e_;
    ServerOptions o_;
    std::string alias_;
};

}  // namespace

int server_main(Engine& e, const ServerOptions& o) {
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket failed");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(o.port);
    if (inet_pton(AF_INET, o.host == "localhost" ? "127.0.0.1" : o.host.c_str(), &addr.sin_addr) != 1) die("bad host %s", o.host.c_str());
    if (bind(fd, (sockaddr*)&addr, sizeof addr) < 0) die("cannot bind %s:%d (%s)", o.host.c_str(), o.port, strerror(errno));
    if (listen(fd, 128) < 0) die("listen failed");
    logi("listening on http://%s:%d  (OpenAI API at /v1/chat/completions, Anthropic API at /v1/messages)", o.host.c_str(), o.port);
    static Server* srv = new Server(e, o);
    while (true) {
        sockaddr_in cli{};
        socklen_t cl = sizeof cli;
        int c = accept(fd, (sockaddr*)&cli, &cl);
        if (c < 0) { if (errno == EINTR) continue; break; }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([c] { srv->handle(c); }).detach();
    }
    return 0;
}

#include "chat.h"
#include <algorithm>

// ------------------------------------------------------------------------------------------------ ThinkSplitter

void ThinkSplitter::init(bool active, bool starts_inside, const std::string& open, const std::string& close) {
    mode_ = active ? (starts_inside ? 1 : 0) : 2;
    open_ = open; close_ = close;
    buf_.clear();
    r_nl_.clear();
    seen_content_ = r_started_ = false;
}

std::pair<std::string, std::string> ThinkSplitter::push(const std::string& d, bool final) {
    std::pair<std::string, std::string> out;
    buf_ += d;
    while (true) {
        if (mode_ == 3) break;  // after the reasoning block: handled below
        if (mode_ == 2) { out.second += buf_; buf_.clear(); break; }
        if (mode_ == 0) {  // undecided: does the output start with the open tag?
            size_t i = 0;
            while (i < buf_.size() && (buf_[i] == '\n' || buf_[i] == ' ')) i++;
            std::string rest = buf_.substr(i);
            if (rest.compare(0, open_.size(), open_) == 0) { buf_ = rest.substr(open_.size()); mode_ = 1; continue; }
            if (rest.size() < open_.size() && open_.compare(0, rest.size(), rest) == 0 && !final) break;
            mode_ = 2;
            continue;
        }
        // mode 1: inside reasoning
        size_t p = buf_.find(close_);
        if (p != std::string::npos) {
            out.first += buf_.substr(0, p);
            buf_ = buf_.substr(p + close_.size());
            mode_ = 3;
            continue;
        }
        size_t keep = 0;
        for (size_t k = std::min(close_.size() - 1, buf_.size()); k > 0; k--)
            if (buf_.compare(buf_.size() - k, k, close_, 0, k) == 0) { keep = k; break; }
        if (final) keep = 0;
        out.first += buf_.substr(0, buf_.size() - keep);
        buf_ = buf_.substr(buf_.size() - keep);
        break;
    }
    if (mode_ == 3) {  // strip leading newlines right after the close tag
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
    // reasoning text: drop the newlines that follow the open tag / precede the close tag (Qwen writes "<think>\n")
    std::string r;
    for (char c : out.first) {
        if (c == '\n') { if (r_started_) r_nl_ += c; continue; }
        r += r_nl_;
        r_nl_.clear();
        r += c;
        r_started_ = true;
    }
    if (mode_ != 1) r_nl_.clear();
    out.first = r;
    return out;
}

// ------------------------------------------------------------------------------------------------ tool calls

namespace {

std::string trim(const std::string& x) {
    size_t a = x.find_first_not_of(" \t\n\r"), b = x.find_last_not_of(" \t\n\r");
    return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
}

void add_call(std::vector<ToolCall>& calls, const std::string& name, const Json& args) {
    ToolCall c;
    c.name = name;
    c.arguments = args.is_str() ? args.str : (args.is_null() ? std::string("{}") : args.dump());
    calls.push_back(c);
}

// Hermes JSON ({"name":..,"arguments":..}, or a list of them) or Qwen3.5 XML (<function=..><parameter=..>)
bool parse_hermes(const std::string& body0, std::vector<ToolCall>& calls) {
    std::string body = trim(body0);
    std::string err;
    Json j = Json::parse(body, &err);
    if (err.empty()) {
        std::vector<const Json*> items;
        if (j.is_arr()) for (auto& x : j.arr) items.push_back(&x); else items.push_back(&j);
        size_t n0 = calls.size();
        for (auto* x : items) {
            if (!x->is_obj() || !(*x)["name"].is_str()) { calls.resize(n0); return false; }
            const Json& a = x->has("arguments") ? (*x)["arguments"] : (*x)["parameters"];
            add_call(calls, (*x)["name"].str, a.is_null() ? Json::object() : a);
        }
        return !items.empty();
    }
    size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    size_t fe = body.find('>', f);
    if (fe == std::string::npos) return false;
    Json args = Json::object();
    size_t q = fe;
    while ((q = body.find("<parameter=", q)) != std::string::npos) {
        size_t qe = body.find('>', q);
        if (qe == std::string::npos) break;
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
    add_call(calls, trim(body.substr(f + 10, fe - f - 10)), args);
    return true;
}

// Python literals (LFM2 calls): 'str' "str" numbers True False None [..] (..) {..}
struct PyParser {
    const char* p;
    const char* e;
    void ws() { while (p < e && isspace((unsigned char)*p)) p++; }
    bool lit(const char* s) {
        size_t n = strlen(s);
        if ((size_t)(e - p) >= n && memcmp(p, s, n) == 0) { p += n; return true; }
        return false;
    }
    bool ident(std::string& out) {
        ws();
        const char* s = p;
        while (p < e && (isalnum((unsigned char)*p) || *p == '_' || *p == '.' || *p == '-' || (unsigned char)*p >= 0x80)) p++;
        out.assign(s, p);
        return !out.empty();
    }
    bool str(std::string& out) {
        char q = *p++;
        out.clear();
        while (p < e && *p != q) {
            if (*p == '\\' && p + 1 < e) {
                p++;
                switch (*p) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'u': {
                        if (e - p < 5) return false;
                        unsigned cp = strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16);
                        p += 4;
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                        else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                        break;
                    }
                    default: out += *p;
                }
                p++;
            } else out += *p++;
        }
        if (p >= e) return false;
        p++;
        return true;
    }
    bool value(Json& v, int depth = 0) {
        ws();
        if (p >= e || depth > 64) return false;
        char c = *p;
        if (c == '\'' || c == '"') { std::string s; if (!str(s)) return false; v = Json(s); return true; }
        if (c == '[' || c == '(') {
            char close = c == '[' ? ']' : ')';
            p++;
            v = Json::array();
            ws();
            if (p < e && *p == close) { p++; return true; }
            while (true) {
                Json x;
                if (!value(x, depth + 1)) return false;
                v.push(x);
                ws();
                if (p < e && *p == ',') { p++; ws(); if (p < e && *p == close) { p++; return true; } continue; }
                if (p < e && *p == close) { p++; return true; }
                return false;
            }
        }
        if (c == '{') {
            p++;
            v = Json::object();
            ws();
            if (p < e && *p == '}') { p++; return true; }
            while (true) {
                Json k;
                if (!value(k, depth + 1)) return false;
                ws();
                if (p >= e || *p != ':') return false;
                p++;
                Json x;
                if (!value(x, depth + 1)) return false;
                v.set(k.is_str() ? k.str : k.dump(), x);
                ws();
                if (p < e && *p == ',') { p++; ws(); if (p < e && *p == '}') { p++; return true; } continue; }
                if (p < e && *p == '}') { p++; return true; }
                return false;
            }
        }
        if (lit("True") || lit("true")) { v = Json(true); return true; }
        if (lit("False") || lit("false")) { v = Json(false); return true; }
        if (lit("None") || lit("null")) { v = Json(); return true; }
        char* end = nullptr;
        std::string rest(p, std::min<size_t>(e - p, 64));
        double d = strtod(rest.c_str(), &end);
        if (end == rest.c_str()) return false;
        p += end - rest.c_str();
        v = Json(d);
        return true;
    }
};

bool parse_lfm2(const std::string& body0, std::vector<ToolCall>& calls) {
    std::string body = trim(body0);
    PyParser ps{body.data(), body.data() + body.size()};
    ps.ws();
    bool bracket = ps.p < ps.e && *ps.p == '[';
    if (bracket) ps.p++;
    size_t n0 = calls.size();
    while (true) {
        ps.ws();
        if (ps.p >= ps.e || (bracket && *ps.p == ']')) break;
        std::string name;
        if (!ps.ident(name)) { calls.resize(n0); return false; }
        ps.ws();
        if (ps.p >= ps.e || *ps.p != '(') { calls.resize(n0); return false; }
        ps.p++;
        Json args = Json::object();
        int pos = 0;
        while (true) {
            ps.ws();
            if (ps.p < ps.e && *ps.p == ')') { ps.p++; break; }
            const char* save = ps.p;
            std::string key;
            bool kw = ps.ident(key);
            ps.ws();
            if (kw && ps.p < ps.e && *ps.p == '=') ps.p++;
            else { ps.p = save; key = "arg" + std::to_string(pos); }
            Json v;
            if (!ps.value(v)) { calls.resize(n0); return false; }
            args.set(key, v);
            pos++;
            ps.ws();
            if (ps.p < ps.e && *ps.p == ',') { ps.p++; continue; }
            if (ps.p < ps.e && *ps.p == ')') { ps.p++; break; }
            calls.resize(n0);
            return false;
        }
        add_call(calls, name, args);
        ps.ws();
        if (ps.p < ps.e && *ps.p == ',') ps.p++;
    }
    return calls.size() > n0;
}

// Gemma 4: call:NAME{key:<|"|>text<|"|>,n:1,obj:{k:v},list:[..]}
struct G4Parser {
    const char* p;
    const char* e;
    static constexpr const char* Q = "<|\"|>";
    void ws() { while (p < e && isspace((unsigned char)*p)) p++; }
    bool at(const char* s) const { size_t n = strlen(s); return (size_t)(e - p) >= n && memcmp(p, s, n) == 0; }
    bool key(std::string& k) {
        ws();
        if (at(Q)) {
            p += 5;
            const char* s = p;
            while (p < e && !at(Q)) p++;
            if (p >= e) return false;
            k.assign(s, p);
            p += 5;
            return true;
        }
        const char* s = p;
        while (p < e && *p != ':' && *p != '}' && *p != ',') p++;
        k = trim(std::string(s, p));
        return !k.empty();
    }
    bool value(Json& v, int depth = 0) {
        ws();
        if (p >= e || depth > 64) return false;
        if (at(Q)) {
            p += 5;
            const char* s = p;
            while (p < e && !at(Q)) p++;
            if (p >= e) return false;
            v = Json(std::string(s, p));
            p += 5;
            return true;
        }
        if (*p == '{') {
            p++;
            v = Json::object();
            ws();
            if (p < e && *p == '}') { p++; return true; }
            while (true) {
                std::string k;
                if (!key(k)) return false;
                ws();
                if (p >= e || *p != ':') return false;
                p++;
                Json x;
                if (!value(x, depth + 1)) return false;
                v.set(k, x);
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == '}') { p++; return true; }
                return false;
            }
        }
        if (*p == '[') {
            p++;
            v = Json::array();
            ws();
            if (p < e && *p == ']') { p++; return true; }
            while (true) {
                Json x;
                if (!value(x, depth + 1)) return false;
                v.push(x);
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == ']') { p++; return true; }
                return false;
            }
        }
        if (at("true")) { p += 4; v = Json(true); return true; }
        if (at("false")) { p += 5; v = Json(false); return true; }
        if (at("null")) { p += 4; v = Json(); return true; }
        std::string rest(p, std::min<size_t>(e - p, 64));
        char* end = nullptr;
        double d = strtod(rest.c_str(), &end);
        if (end != rest.c_str()) { p += end - rest.c_str(); v = Json(d); return true; }
        // unquoted text up to the next delimiter
        const char* s = p;
        while (p < e && *p != ',' && *p != '}' && *p != ']') p++;
        v = Json(trim(std::string(s, p)));
        return true;
    }
};

bool parse_gemma4(const std::string& body0, std::vector<ToolCall>& calls) {
    std::string body = trim(body0);
    if (body.compare(0, 5, "call:") == 0) body = body.substr(5);
    size_t b = body.find('{');
    std::string name = trim(body.substr(0, b));
    if (name.empty()) return false;
    Json args = Json::object();
    if (b != std::string::npos) {
        G4Parser ps{body.data() + b, body.data() + body.size()};
        if (!ps.value(args) || !args.is_obj()) return false;
    }
    add_call(calls, name, args);
    return true;
}

bool parse_mistral(const std::string& body0, std::vector<ToolCall>& calls) {
    std::string body = trim(body0);
    size_t a = body.find_first_of("[{");
    if (a == std::string::npos) return false;
    // the first balanced JSON value; models sometimes keep talking after the call list
    int depth = 0;
    bool in_str = false, esc = false;
    size_t z = std::string::npos;
    for (size_t i = a; i < body.size() && z == std::string::npos; i++) {
        char ch = body[i];
        if (in_str) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') in_str = false; continue; }
        if (ch == '"') in_str = true;
        else if (ch == '[' || ch == '{') depth++;
        else if ((ch == ']' || ch == '}') && --depth == 0) z = i;
    }
    if (z == std::string::npos) return false;
    std::string err;
    Json j = Json::parse(body.substr(a, z - a + 1), &err);
    if (!err.empty()) return false;
    return parse_hermes(j.dump(), calls);
}

}  // namespace

bool parse_tool_calls(ToolFormat f, const std::string& body, std::vector<ToolCall>& calls) {
    switch (f) {
        case ToolFormat::Lfm2: return parse_lfm2(body, calls);
        case ToolFormat::Gemma4: return parse_gemma4(body, calls);
        case ToolFormat::Mistral: return parse_mistral(body, calls);
        default: return parse_hermes(body, calls);
    }
}

void ToolCallParser::init(ToolFormat f) {
    fmt_ = f;
    switch (f) {
        case ToolFormat::Lfm2: start_ = "<|tool_call_start|>"; end_ = "<|tool_call_end|>"; break;
        case ToolFormat::Gemma4: start_ = "<|tool_call>"; end_ = "<tool_call|>"; break;
        case ToolFormat::Mistral: start_ = "[TOOL_CALLS]"; end_.clear(); break;
        default: start_ = "<tool_call>"; end_ = "</tool_call>"; break;
    }
    buf_.clear(); ws_.clear();
    in_call_ = any_text_ = false;
    n_calls_ = 0;
}

bool ToolCallParser::parse_block(const std::string& body, std::vector<ToolCall>& calls) const {
    return parse_tool_calls(fmt_, body, calls);
}

std::string ToolCallParser::push(const std::string& d, bool final, std::vector<ToolCall>& calls) {
    buf_ += d;
    std::string text;   // plain text found in this push (before whitespace handling)
    std::string out;
    auto emit_text = [&](const std::string& t) {
        // whitespace is held back until non-space text follows, so text/call separators never reach the client
        for (char c : t) {
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { ws_ += c; continue; }
            if (any_text_) out += ws_;
            ws_.clear();
            out += c;
            any_text_ = true;
        }
    };
    while (true) {
        if (!in_call_) {
            size_t a = buf_.find(start_);
            if (a != std::string::npos) {
                emit_text(buf_.substr(0, a));
                buf_.erase(0, a + start_.size());
                ws_.clear();
                in_call_ = true;
                continue;
            }
            size_t hold = 0;
            if (!final)
                for (size_t k = std::min(start_.size() - 1, buf_.size()); k > 0; k--)
                    if (buf_.compare(buf_.size() - k, k, start_, 0, k) == 0) { hold = k; break; }
            emit_text(buf_.substr(0, buf_.size() - hold));
            buf_.erase(0, buf_.size() - hold);
            break;
        }
        size_t b = end_.empty() ? std::string::npos : buf_.find(end_);
        if (b == std::string::npos && !final) break;
        std::string body = b == std::string::npos ? buf_ : buf_.substr(0, b);
        buf_ = b == std::string::npos ? std::string() : buf_.substr(b + end_.size());
        in_call_ = false;
        size_t n0 = calls.size();
        if (parse_block(body, calls)) n_calls_ += (int)(calls.size() - n0);
        else emit_text(start_ + body + (b == std::string::npos ? std::string() : end_));   // not a call: show it as text
        if (b == std::string::npos) break;
    }
    if (final) ws_.clear();
    return out;
}

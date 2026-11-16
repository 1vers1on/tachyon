#include "json.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace {
struct Parser {
    const char* p;
    const char* end;
    std::string err;

    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
    bool fail(const char* m) { if (err.empty()) err = m; return false; }

    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    }
    bool hex4(uint32_t& v) {
        if (end - p < 4) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return fail("bad hex");
        }
        return true;
    }
    bool str(std::string& o) {
        if (p >= end || *p != '"') return fail("expected string");
        p++;
        while (p < end) {
            const char* s = p;
            while (p < end && *p != '"' && *p != '\\') p++;
            o.append(s, p - s);
            if (p >= end) break;
            if (*p == '"') { p++; return true; }
            p++;  // backslash
            if (p >= end) break;
            char c = *p++;
            switch (c) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            p += 2;
                            uint32_t lo;
                            if (!hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else { put_utf8(o, 0xFFFD); cp = lo; }
                        } else cp = 0xFFFD;
                    } else if (cp >= 0xDC00 && cp < 0xE000) cp = 0xFFFD;
                    put_utf8(o, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }
    bool value(Json& v, int depth) {
        if (depth > 512) return fail("too deep");
        ws();
        if (p >= end) return fail("unexpected end");
        char c = *p;
        if (c == '{') {
            p++;
            v.type = Json::Object;
            ws();
            if (p < end && *p == '}') { p++; return true; }
            while (true) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected :");
                p++;
                v.obj.emplace_back(std::move(k), Json());
                if (!value(v.obj.back().second, depth + 1)) return false;
                ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == '}') { p++; return true; }
                return fail("expected , or }");
            }
        } else if (c == '[') {
            p++;
            v.type = Json::Array;
            ws();
            if (p < end && *p == ']') { p++; return true; }
            while (true) {
                v.arr.emplace_back();
                if (!value(v.arr.back(), depth + 1)) return false;
                ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == ']') { p++; return true; }
                return fail("expected , or ]");
            }
        } else if (c == '"') {
            v.type = Json::String;
            return str(v.str);
        } else if (c == 't') {
            if (end - p >= 4 && !memcmp(p, "true", 4)) { p += 4; v.type = Json::Bool; v.b = true; return true; }
            return fail("bad literal");
        } else if (c == 'f') {
            if (end - p >= 5 && !memcmp(p, "false", 5)) { p += 5; v.type = Json::Bool; v.b = false; return true; }
            return fail("bad literal");
        } else if (c == 'n') {
            if (end - p >= 4 && !memcmp(p, "null", 4)) { p += 4; v.type = Json::Null; return true; }
            return fail("bad literal");
        } else if (c == 'N') {  // NaN (python json)
            if (end - p >= 3 && !memcmp(p, "NaN", 3)) { p += 3; v.type = Json::Number; v.num = NAN; return true; }
            return fail("bad literal");
        } else if (c == 'I') {
            if (end - p >= 8 && !memcmp(p, "Infinity", 8)) { p += 8; v.type = Json::Number; v.num = INFINITY; return true; }
            return fail("bad literal");
        } else {
            char* e = nullptr;
            std::string tmp;
            const char* s = p;
            while (p < end && (isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) p++;
            if (p == s) {
                if (end - p >= 9 && !memcmp(p, "-Infinity", 9)) { p += 9; v.type = Json::Number; v.num = -INFINITY; return true; }
                return fail("unexpected character");
            }
            tmp.assign(s, p - s);
            v.type = Json::Number;
            v.num = strtod(tmp.c_str(), &e);
            if (!e || *e) return fail("bad number");
            return true;
        }
    }
};

void dump_num(std::string& o, double d) {
    if (std::isnan(d) || std::isinf(d)) { o += "null"; return; }
    char buf[64];
    if (fabs(d) < 9.007199254740992e15 && d == (double)(int64_t)d) snprintf(buf, sizeof buf, "%lld", (long long)d);
    else snprintf(buf, sizeof buf, "%.9g", d);
    o += buf;
}

// indent == -2: compact single line with Python json.dumps separators (", ", ": ")
void dump_rec(const Json& j, std::string& o, int indent, int level) {
    auto nl = [&](int lv) {
        if (indent < 0) return;
        o += '\n';
        o.append((size_t)(indent * lv), ' ');
    };
    switch (j.type) {
        case Json::Null: o += "null"; break;
        case Json::Bool: o += j.b ? "true" : "false"; break;
        case Json::Number: dump_num(o, j.num); break;
        case Json::String: json_escape_to(o, j.str); break;
        case Json::Array:
            o += '[';
            for (size_t i = 0; i < j.arr.size(); i++) {
                if (i) o += indent == -2 ? ", " : ",";
                nl(level + 1);
                dump_rec(j.arr[i], o, indent, level + 1);
            }
            if (!j.arr.empty()) nl(level);
            o += ']';
            break;
        case Json::Object:
            o += '{';
            for (size_t i = 0; i < j.obj.size(); i++) {
                if (i) o += indent == -2 ? ", " : ",";
                nl(level + 1);
                json_escape_to(o, j.obj[i].first);
                o += indent >= 0 || indent == -2 ? ": " : ":";
                dump_rec(j.obj[i].second, o, indent, level + 1);
            }
            if (!j.obj.empty()) nl(level);
            o += '}';
            break;
    }
}
}  // namespace

void json_escape_to(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    o += '"';
}

Json Json::parse(const std::string& s, std::string* err) {
    Parser ps{s.data(), s.data() + s.size(), {}};
    Json v;
    bool ok = ps.value(v, 0);
    if (ok) { ps.ws(); if (ps.p != ps.end) { ok = false; ps.err = "trailing characters"; } }
    if (!ok) {
        if (err) *err = ps.err + " at offset " + std::to_string(ps.p - s.data());
        return Json();
    }
    if (err) err->clear();
    return v;
}

std::string Json::dump(int indent) const {
    std::string o;
    dump_rec(*this, o, indent, 0);
    return o;
}

// Tachyon - minimal JSON DOM, parser and serializer
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <cstdint>
#include <cmath>
#include <limits>

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    Json() {}
    Json(std::nullptr_t) {}
    Json(bool v) : type(Bool), b(v) {}
    Json(int v) : type(Number), num(v) {}
    Json(int64_t v) : type(Number), num((double)v) {}
    Json(uint64_t v) : type(Number), num((double)v) {}
    Json(size_t v, int) : type(Number), num((double)v) {}
    Json(double v) : type(Number), num(v) {}
    Json(const char* s) : type(String), str(s) {}
    Json(const std::string& s) : type(String), str(s) {}
    Json(std::string&& s) : type(String), str(std::move(s)) {}

    static Json array() { Json j; j.type = Array; return j; }
    static Json object() { Json j; j.type = Object; return j; }

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_num() const { return type == Number; }
    bool is_str() const { return type == String; }
    bool is_arr() const { return type == Array; }
    bool is_obj() const { return type == Object; }

    const Json* find(const std::string& k) const {
        if (type != Object) return nullptr;
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool has(const std::string& k) const { return find(k) != nullptr; }
    const Json& operator[](const std::string& k) const {
        static Json null_json;
        const Json* p = find(k);
        return p ? *p : null_json;
    }
    Json& set(const std::string& k, Json v) {
        if (type == Null) type = Object;
        for (auto& kv : obj) if (kv.first == k) { kv.second = std::move(v); return kv.second; }
        obj.emplace_back(k, std::move(v));
        return obj.back().second;
    }
    Json& push(Json v) { if (type == Null) type = Array; arr.push_back(std::move(v)); return arr.back(); }
    const Json& operator[](size_t i) const { return arr[i]; }
    size_t size() const { return type == Array ? arr.size() : type == Object ? obj.size() : 0; }

    double as_num(double def = 0) const { return type == Number ? num : (type == Bool ? (b ? 1 : 0) : def); }
    int64_t as_int(int64_t def = 0) const {
        return type == Number && std::isfinite(num) && num >= (double)std::numeric_limits<int64_t>::min() &&
               num < -(double)std::numeric_limits<int64_t>::min() ? (int64_t)num : def;
    }
    bool as_bool(bool def = false) const { return type == Bool ? b : (type == Number ? num != 0 : def); }
    std::string as_str(const std::string& def = "") const { return type == String ? str : def; }

    // get with default
    double get_num(const std::string& k, double def) const { auto p = find(k); return p && p->is_num() ? p->num : def; }
    int64_t get_int(const std::string& k, int64_t def) const { auto p = find(k); return p ? p->as_int(def) : def; }
    bool get_bool(const std::string& k, bool def) const { auto p = find(k); return p && (p->is_bool() || p->is_num()) ? p->as_bool() : def; }
    std::string get_str(const std::string& k, const std::string& def) const { auto p = find(k); return p && p->is_str() ? p->str : def; }

    static Json parse(const std::string& s, std::string* err = nullptr);
    std::string dump(int indent = -1) const;
};

void json_escape_to(std::string& out, const std::string& s);

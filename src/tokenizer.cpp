#include "tokenizer.h"
#include "unicode_tables.h"
#include <algorithm>
#include <ctime>

static bool in_ranges(const uint32_t (*r)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

// fast ASCII paths
static uint8_t g_ascii_class[128];  // bit0 letter, bit1 number, bit2 space
static struct AsciiInit {
    AsciiInit() {
        for (int c = 0; c < 128; c++) {
            uint8_t v = 0;
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) v |= 1;
            if (c >= '0' && c <= '9') v |= 2;
            if (c == ' ' || (c >= 9 && c <= 13)) v |= 4;
            g_ascii_class[c] = v;
        }
    }
} g_ascii_init;

bool unicode_is_letter(uint32_t cp) {
    if (cp < 128) return g_ascii_class[cp] & 1;
    return in_ranges(k_unicode_letter_ranges, k_unicode_letter_count, cp);
}
bool unicode_is_number(uint32_t cp) {
    if (cp < 128) return g_ascii_class[cp] & 2;
    return in_ranges(k_unicode_number_ranges, k_unicode_number_count, cp);
}
bool unicode_is_mark(uint32_t cp) {
    if (cp < 0x300) return false;
    return in_ranges(k_unicode_mark_ranges, k_unicode_mark_count, cp);
}
bool unicode_is_space(uint32_t cp) {
    if (cp < 128) return g_ascii_class[cp] & 4;
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<int>& offs) {
    cps.clear(); offs.clear();
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + len > s.size()) len = 1;
        uint32_t cp;
        if (len == 1) cp = c;
        else if (len == 2) cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
        else if (len == 3) cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        else cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        cps.push_back(cp);
        offs.push_back((int)i);
        i += len;
    }
    offs.push_back((int)s.size());
}

size_t utf8_valid_prefix(const std::string& s) {
    // length of the longest prefix that does not end in an incomplete UTF-8 sequence
    size_t n = s.size();
    if (n == 0) return 0;
    // look back up to 3 bytes for a lead byte
    for (size_t k = 1; k <= 4 && k <= n; k++) {
        unsigned char c = s[n - k];
        if ((c & 0xC0) == 0x80) continue;  // continuation
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if ((size_t)len > k) return n - k;  // incomplete
        return n;
    }
    return n;
}

void Tokenizer::load(const ModelFile& mf) {
    const Json& tj = mf.header()["tokenizer"];
    int nv = (int)tj.get_int("n_vocab", 0);
    const BlobInfo* vd = mf.blob("tok.vocab_data");
    const BlobInfo* vo = mf.blob("tok.vocab_off");
    const BlobInfo* ty = mf.blob("tok.type");
    const BlobInfo* mg = mf.blob("tok.merges");
    const uint32_t* offs = (const uint32_t*)vo->data;
    vocab_.resize(nv);
    ttype_.assign(ty->data, ty->data + nv);
    spm_ = tj.get_str("type", "") == "bpe_spm";
    for (int i = 0; i < nv; i++) {
        vocab_[i].assign((const char*)vd->data + offs[i], offs[i + 1] - offs[i]);
        if (ttype_[i] == 0) tok2id_.emplace(vocab_[i], i);
    }
    if (spm_) {
        // decoded form: U+2581 -> ' ', <0xNN> -> byte NN
        dec_.resize(nv);
        for (int i = 0; i < nv; i++) {
            const std::string& v = vocab_[i];
            if (v.size() == 6 && v.compare(0, 3, "<0x") == 0 && v[5] == '>' && ttype_[i] == 0) {
                dec_[i] = std::string(1, (char)strtol(v.substr(3, 2).c_str(), nullptr, 16));
                continue;
            }
            std::string d;
            for (size_t k = 0; k < v.size();) {
                if (k + 2 < v.size() && (unsigned char)v[k] == 0xE2 && (unsigned char)v[k + 1] == 0x96 && (unsigned char)v[k + 2] == 0x81) { d += ' '; k += 3; }
                else d += v[k++];
            }
            dec_[i] = d;
        }
    }
    for (int b = 0; b < 256; b++) {
        if (spm_) {
            char nm[8]; snprintf(nm, sizeof nm, "<0x%02X>", b);
            auto it = tok2id_.find(nm);
            byte_tok_[b] = it == tok2id_.end() ? -1 : it->second;
        } else {
            auto it = tok2id_.find(std::string(1, (char)b));
            byte_tok_[b] = it == tok2id_.end() ? -1 : it->second;
        }
    }
    const int32_t* m = (const int32_t*)mg->data;
    int64_t nm = mg->size / 12;
    merges_.reserve(nm * 2);
    for (int64_t i = 0; i < nm; i++) {
        uint64_t key = ((uint64_t)(uint32_t)m[3 * i] << 32) | (uint32_t)m[3 * i + 1];
        merges_.emplace(key, std::make_pair((int)i, (int)m[3 * i + 2]));
    }
    for (auto& s : tj["specials"].arr) specials_.push_back({s.get_str("content", ""), (int)s.get_int("id", -1), s.get_bool("special", true)});
    std::stable_sort(specials_.begin(), specials_.end(), [](const Special& a, const Special& b) { return a.s.size() > b.s.size(); });
    for (size_t i = 0; i < specials_.size(); i++) if (!specials_[i].s.empty()) by_first_[(unsigned char)specials_[i].s[0]].push_back((int)i);
    pretok_ = tj.get_str("pretokenizer", "qwen2");
    for (auto& e : tj["eos"].arr) eos_.push_back((int)e.as_int());
    bos_ = (int)tj.get_int("bos", -1);
    add_bos_ = tj.get_bool("add_bos", false);
    thinking_ = tj.get_bool("thinking", false);
    chat_format_ = tj.get_str("chat_format", "chatml");
    chat_prefix_ = tj.get_str("chat_prefix", "");
    think_tag_open_ = tj.get_str("think_tag_open", "<think>");
    think_tag_close_ = tj.get_str("think_tag_close", "</think>");
    think_system_token_ = tj.get_str("think_system_token", "");
    tool_style_ = tj.get_str("tool_style", "");
    if (tool_style_.empty())  // files converted before the flag existed: derive from the stored template
        tool_style_ = tj.get_str("chat_template", "").find("<function=example_function_name>") != std::string::npos ? "xml" : "json";
    think_default_ = tj.get_bool("think_default", true);
    think_open_ = tj.get_bool("think_open", false);
    think_disable_block_ = tj.get_bool("think_disable_block", true);
    // chat formats added later: detect from the stored template (files converted before keep working)
    if (chat_format_ == "raw" || chat_format_ == "tulu") {
        const std::string ct = tj.get_str("chat_template", "");
        if (ct.find("<|start_of_role|>") != std::string::npos) { chat_format_ = "granite"; thinking_ = false; }
        else if (ct.find("<|end|>") != std::string::npos && ct.find("<|assistant|>") != std::string::npos) chat_format_ = "phi";
        else if (ct.find("｜User｜") != std::string::npos) {
            chat_format_ = "deepseek"; thinking_ = true; think_open_ = true; think_default_ = true;
        }
        else if (ct.find("[INST]") != std::string::npos) chat_format_ = "mistral";
    }
    {
        const std::string ct = tj.get_str("chat_template", "");
        if (chat_format_ == "chatml")
            tool_format_ = tool_style_ == "xml" ? ToolFormat::QwenXml
                         : ct.find("<|tool_call_start|>") != std::string::npos ? ToolFormat::Lfm2 : ToolFormat::Hermes;
        else if (chat_format_ == "gemma4") tool_format_ = ToolFormat::Gemma4;
        else if (chat_format_ == "mistral" && token_id("[TOOL_CALLS]") >= 0) tool_format_ = ToolFormat::Mistral;
        else tool_format_ = ToolFormat::Generic;
    }
    // Templates that write the BOS text themselves (Gemma 4, Llama 3) were converted with add_bos = false and the
    // BOS as a chat prefix, which left raw prompts (completions, perplexity) without it. Treat the model as
    // BOS-prefixed everywhere instead: encode() adds it, the chat renderer no longer writes it.
    // Only for genuine start tokens: OLMoE's "bos" is an unrelated added token that HF never prepends to raw text.
    auto is_start_tok = [](const std::string& t) {
        return t.find("bos") != std::string::npos || t.find("start") != std::string::npos || t.find("begin") != std::string::npos;
    };
    if (!add_bos_ && bos_ >= 0 && !chat_prefix_.empty() && is_start_tok(chat_prefix_))
        for (auto& sp : specials_)
            if (sp.id == bos_ && sp.s == chat_prefix_) { add_bos_ = true; chat_prefix_.clear(); break; }
}

bool Tokenizer::is_eos(int id) const {
    for (int e : eos_) if (e == id) return true;
    return false;
}

int Tokenizer::token_id(const std::string& s) const {
    for (auto& sp : specials_) if (sp.s == s) return sp.id;
    auto it = tok2id_.find(s);
    return it == tok2id_.end() ? -1 : it->second;
}

// Hand-written equivalent of the HF pre-tokenizer regexes. Produces [start,end) codepoint spans.
void Tokenizer::pretokenize(const uint32_t* c, int n, std::vector<std::pair<int, int>>& out) const {
    const bool gpt2 = pretok_ == "gpt2";
    const bool q35 = pretok_ == "qwen35";
    const int max_digits = pretok_ == "llama3" ? 3 : (gpt2 ? 1 << 30 : 1);
    auto L = [&](int i) { return i < n && (unicode_is_letter(c[i]) || (q35 && unicode_is_mark(c[i]))); };
    auto N = [&](int i) { return i < n && unicode_is_number(c[i]); };
    auto S = [&](int i) { return i < n && unicode_is_space(c[i]); };
    auto lower = [](uint32_t x) { return x >= 'A' && x <= 'Z' ? x + 32 : x; };
    int i = 0;
    while (i < n) {
        int len = 0;
        // contractions
        if (c[i] == '\'' && i + 1 < n) {
            uint32_t a = gpt2 ? c[i + 1] : lower(c[i + 1]);
            uint32_t b = i + 2 < n ? (gpt2 ? c[i + 2] : lower(c[i + 2])) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) len = 3;
        }
        if (!len) {
            if (gpt2) {
                int j = i + (c[i] == ' ' ? 1 : 0);
                if (L(j)) { while (L(j)) j++; len = j - i; }
                else if (N(j)) { while (N(j)) j++; len = j - i; }
                else if (j < n && !S(j)) { while (j < n && !S(j) && !L(j) && !N(j)) j++; len = j - i; }
            } else {
                // [^\r\n\p{L}\p{N}]?\p{L}+
                if (L(i)) { int j = i; while (L(j)) j++; len = j - i; }
                else if (c[i] != '\r' && c[i] != '\n' && !N(i) && L(i + 1)) { int j = i + 1; while (L(j)) j++; len = j - i; }
                // \p{N}{1,max}
                else if (N(i)) { int j = i; while (N(j) && j - i < max_digits) j++; len = j - i; }
                else {
                    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                    int j = i + (c[i] == ' ' ? 1 : 0);
                    int k = j;
                    while (k < n && !S(k) && !L(k) && !N(k)) k++;
                    if (k > j) {
                        while (k < n && (c[k] == '\r' || c[k] == '\n')) k++;
                        len = k - i;
                    }
                }
            }
        }
        if (!len && S(i)) {
            int j = i;
            while (S(j)) j++;
            if (!gpt2) {
                // \s*[\r\n]+  -> ends after the last newline in the whitespace run
                int last_nl = -1;
                for (int k = i; k < j; k++) if (c[k] == '\r' || c[k] == '\n') last_nl = k;
                if (last_nl >= 0) len = last_nl + 1 - i;
            }
            if (!len) {
                // \s+(?!\S) | \s+
                if (j == n) len = j - i;
                else if (j - i >= 2) len = j - i - 1;
                else len = j - i;
            }
        }
        if (!len) len = 1;
        out.emplace_back(i, i + len);
        i += len;
    }
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    {
        std::lock_guard<std::mutex> lk(cache_mu_);
        auto it = cache_.find(piece);
        if (it != cache_.end()) { out.insert(out.end(), it->second.begin(), it->second.end()); return; }
    }
    std::vector<int> sym;
    auto whole = tok2id_.find(piece);
    if (whole != tok2id_.end()) sym.push_back(whole->second);
    else {
        sym.reserve(piece.size());
        for (unsigned char b : piece) sym.push_back(byte_tok_[b]);
        bpe_merge(sym);
    }
    out.insert(out.end(), sym.begin(), sym.end());
    std::lock_guard<std::mutex> lk(cache_mu_);
    if (cache_.size() > 200000) cache_.clear();
    cache_.emplace(piece, std::move(sym));
}

// Lowest-rank-first BPE with a heap over adjacent pairs and a linked list of symbols: O(n log n).
void Tokenizer::bpe_merge(std::vector<int>& sym) const {
    const int n = (int)sym.size();
    if (n < 2) return;
    std::vector<int> prv(n), nxt(n);
    for (int i = 0; i < n; i++) { prv[i] = i - 1; nxt[i] = i + 1 < n ? i + 1 : -1; }
    struct Item { int rank, left, a, b; };
    auto cmp = [](const Item& x, const Item& y) { return x.rank != y.rank ? x.rank > y.rank : x.left > y.left; };
    std::vector<Item> heap;
    auto push = [&](int left) {
        if (left < 0 || nxt[left] < 0) return;
        int a = sym[left], b = sym[nxt[left]];
        if (a < 0 || b < 0) return;
        auto it = merges_.find(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
        if (it == merges_.end()) return;
        heap.push_back({it->second.first, left, a, b});
        std::push_heap(heap.begin(), heap.end(), cmp);
    };
    for (int i = 0; i + 1 < n; i++) push(i);
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        Item it = heap.back();
        heap.pop_back();
        int l = it.left, r = nxt[l];
        if (sym[l] != it.a || r < 0 || sym[r] != it.b) continue;  // stale entry
        auto m = merges_.find(((uint64_t)(uint32_t)it.a << 32) | (uint32_t)it.b);
        sym[l] = m->second.second;
        sym[r] = -2;  // dead
        nxt[l] = nxt[r];
        if (nxt[r] >= 0) prv[nxt[r]] = l;
        push(prv[l]);
        push(l);
    }
    int w = 0;
    for (int i = 0; i >= 0 && i < n; i = nxt[i]) sym[w++] = sym[i];
    sym.resize(w);
}

// SentencePiece-style BPE (Gemma): spaces -> U+2581, the whole text is one word, characters missing from
// the vocabulary fall back to <0xNN> byte tokens.
void Tokenizer::encode_spm(const std::string& text, std::vector<int>& out) const {
    std::vector<int> sym;
    size_t i = 0;
    auto add_bytes = [&](const char* p, size_t n) { for (size_t k = 0; k < n; k++) sym.push_back(byte_tok_[(unsigned char)p[k]]); };
    std::string ch;
    while (i < text.size()) {
        unsigned char c = text[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + len > text.size()) len = 1;
        if (c == ' ') ch = "\xE2\x96\x81"; else ch.assign(text, i, len);
        auto it = tok2id_.find(ch);
        if (it != tok2id_.end()) sym.push_back(it->second);
        else add_bytes(ch.data(), ch.size());
        i += len;
    }
    bpe_merge(sym);
    out.insert(out.end(), sym.begin(), sym.end());
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    if (spm_) { encode_spm(text, out); return; }
    std::vector<uint32_t> cps;
    std::vector<int> offs;
    utf8_decode(text, cps, offs);
    std::vector<std::pair<int, int>> spans;
    pretokenize(cps.data(), (int)cps.size(), spans);
    for (auto& sp : spans) bpe(text.substr(offs[sp.first], offs[sp.second] - offs[sp.first]), out);
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    // Added tokens are split out before BPE, longest first (like HF). Without parse_special only the
    // non-special ones match (e.g. Gemma's "\n" and whitespace-run tokens), never control tokens.
    std::vector<int> out;
    if (add_bos_ && bos_ >= 0) out.push_back(bos_);
    if (specials_.empty()) { encode_plain(text, out); return out; }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (int si : by_first_[(unsigned char)text[i]]) {
            const Special& sp = specials_[si];
            if ((parse_special || !sp.special) && text.compare(i, sp.s.size(), sp.s) == 0) {
                encode_plain(text.substr(start, i - start), out);
                out.push_back(sp.id);
                i += sp.s.size();
                start = i;
                matched = true;
                break;
            }
        }
        if (!matched) i++;
    }
    encode_plain(text.substr(start), out);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) if (id >= 0 && id < (int)vocab_.size()) s += token_bytes(id);
    return s;
}

// Replace malformed byte sequences, retaining only a potentially valid incomplete suffix.
// Byte-fallback tokens can otherwise put invalid UTF-8 into JSON and break API clients.
static std::string stream_utf8(std::string& pending, bool final) {
    std::string out;
    size_t pos = 0;
    while (pos < pending.size()) {
        const unsigned char c = pending[pos];
        if (c < 0x80) { out += pending[pos++]; continue; }
        const int len = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (!len) { out += "\xef\xbf\xbd"; pos++; continue; }
        int j = 1;
        for (; j < len && pos + j < pending.size(); j++) {
            const unsigned char b = pending[pos + j];
            if (b < 0x80 || b > 0xbf || (j == 1 && ((c == 0xe0 && b < 0xa0) || (c == 0xed && b > 0x9f) ||
                (c == 0xf0 && b < 0x90) || (c == 0xf4 && b > 0x8f)))) break;
        }
        if (j == len) { out.append(pending, pos, len); pos += len; }
        else if (!final && pos + j == pending.size()) break;
        else { out += "\xef\xbf\xbd"; pos += j; }
    }
    pending.erase(0, pos);
    return out;
}

std::string StreamDecoder::push(int id) {
    if (id >= 0 && id < tok_->n_vocab()) pending_ += tok_->token_bytes(id);
    return stream_utf8(pending_, false);
}
std::string StreamDecoder::flush() {
    return stream_utf8(pending_, true);
}

static std::string strip_nl(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && s[a] == '\n') a++;
    while (b > a && s[b - 1] == '\n') b--;
    return s.substr(a, b - a);
}


// ---------------------------------------------------------------------------------------------------------------
// Chat templates. Each format mirrors its HF Jinja template (hard-coded; no Jinja engine), including tool calling.

static std::string trim_ws(const std::string& x) {
    size_t a = x.find_first_not_of(" \t\n\r"), b = x.find_last_not_of(" \t\n\r");
    return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
}
static const Json& tool_fn(const Json& t) { return t.has("function") ? t["function"] : t; }
static Json call_args(const ToolCall& c) {
    std::string err;
    Json a = Json::parse(c.arguments.empty() ? "{}" : c.arguments, &err);
    return err.empty() && a.is_obj() ? a : Json::object();
}
// Python str() of a JSON scalar (what Jinja's `| string` prints)
static std::string py_str(const Json& v) {
    if (v.is_str()) return v.str;
    if (v.is_bool()) return v.b ? "True" : "False";
    if (v.is_null()) return "None";
    return v.dump();
}
static std::vector<std::pair<std::string, const Json*>> dictsort(const Json& o) {
    std::vector<std::pair<std::string, const Json*>> v;
    for (auto& kv : o.obj) v.push_back({kv.first, &kv.second});
    std::stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return strcasecmp(a.first.c_str(), b.first.c_str()) < 0; });
    return v;
}

static const char* kHermesTools =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
static const char* kHermesToolsEnd =
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

std::string Tokenizer::tool_call_prefix(const std::string& name) const {
    switch (tool_format_) {
        case ToolFormat::QwenXml: return "<tool_call>\n<function=" + (name.empty() ? std::string() : name + ">\n");
        case ToolFormat::Lfm2: return "<|tool_call_start|>[" + (name.empty() ? std::string() : name + "(");
        case ToolFormat::Gemma4: return "<|tool_call>call:" + (name.empty() ? std::string() : name + "{");
        case ToolFormat::Mistral: return "[TOOL_CALLS] [" + (name.empty() ? std::string() : "{\"name\": \"" + name + "\", \"arguments\": ");
        default: return "<tool_call>\n" + (name.empty() ? std::string() : "{\"name\": \"" + name + "\", \"arguments\": ");
    }
}

// Models without a native tool format: describe the tools in the system prompt (Hermes style, which most
// instruction-tuned models follow), write earlier calls as <tool_call> blocks and results as user turns.
std::string Tokenizer::render_generic_tools(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                                            const std::vector<Json>& tools) const {
    std::vector<ChatMessage> out;
    std::string tblock;
    if (!tools.empty()) {
        tblock = kHermesTools;
        for (auto& t : tools) tblock += "\n" + t.dump(-2);
        tblock += kHermesToolsEnd;
    }
    size_t i0 = 0;
    if (!tblock.empty()) {
        ChatMessage sys;
        sys.role = "system";
        if (!msgs.empty() && msgs[0].role == "system") { sys.content = msgs[0].content + "\n\n"; i0 = 1; }
        sys.content += tblock;
        out.push_back(sys);
    }
    for (size_t i = i0; i < msgs.size(); i++) {
        const ChatMessage& m = msgs[i];
        if (m.role == "tool") {
            std::string r = "<tool_response>\n" + m.content + "\n</tool_response>";
            if (i > i0 && msgs[i - 1].role == "tool" && !out.empty()) out.back().content += "\n" + r;
            else { ChatMessage u; u.role = "user"; u.content = r; out.push_back(u); }
            continue;
        }
        ChatMessage c = m;
        c.tool_calls.clear();
        for (auto& tc : m.tool_calls) {
            if (!c.content.empty()) c.content += "\n";
            c.content += "<tool_call>\n{\"name\": \"" + tc.name + "\", \"arguments\": " + call_args(tc).dump(-2) + "}\n</tool_call>";
        }
        out.push_back(c);
    }
    return render_chat(out, add_gen, enable_thinking, {});
}

// --- Gemma 4: declarations and arguments in its own notation, strings delimited by <|"|> ---
static const char* kQ = "<|\"|>";
static std::string g4_arg(const Json& v, bool escape_keys = true) {
    if (v.is_null()) return "null";
    if (v.is_str()) return kQ + v.str + kQ;
    if (v.is_bool()) return v.b ? "true" : "false";
    if (v.is_obj()) {
        std::string o = "{";
        bool first = true;
        for (auto& kv : dictsort(v)) {
            if (!first) o += ",";
            first = false;
            o += escape_keys ? kQ + kv.first + kQ : kv.first;
            o += ":" + g4_arg(*kv.second, escape_keys);
        }
        return o + "}";
    }
    if (v.is_arr()) {
        std::string o = "[";
        for (size_t i = 0; i < v.arr.size(); i++) o += (i ? "," : "") + g4_arg(v.arr[i], escape_keys);
        return o + "]";
    }
    return v.dump();
}
static std::string upper(std::string s) { for (auto& c : s) c = toupper((unsigned char)c); return s; }
static std::string g4_type(const Json& t) {
    if (t.is_str()) return upper(t.str);
    if (t.is_arr()) for (auto& x : t.arr) if (x.is_str() && x.str != "null") return upper(x.str);
    return "";
}
static std::string g4_required(const Json& req) {
    std::string o = "required:[";
    for (size_t i = 0; i < req.arr.size(); i++) o += (i ? "," : "") + std::string(kQ) + req.arr[i].as_str() + kQ;
    return o + "]";
}
static std::string g4_params(const Json& props, bool filter_keys = false) {
    static const char* std_keys[] = {"description", "type", "properties", "required", "nullable"};
    std::string o;
    bool first = true;
    for (auto& kv : dictsort(props)) {
        if (filter_keys && std::find_if(std::begin(std_keys), std::end(std_keys), [&](const char* k) { return kv.first == k; }) != std::end(std_keys)) continue;
        const Json& v = kv.second->is_obj() ? *kv.second : Json::object();
        if (!first) o += ",";
        first = false;
        o += kv.first + ":{";
        bool comma = false;
        auto sep = [&]() { if (comma) o += ","; comma = true; };
        if (!v.get_str("description", "").empty()) { o += "description:" + std::string(kQ) + v.get_str("description", "") + kQ; comma = true; }
        const std::string T = g4_type(v["type"]);
        if (T == "STRING") {
            if (v["enum"].size()) { sep(); o += "enum:" + g4_arg(v["enum"]); }
        } else if (T == "ARRAY") {
            const Json& items = v["items"];
            if (items.is_obj() && items.size()) {
                sep();
                o += "items:{";
                bool f2 = true;
                for (auto& ik : dictsort(items)) {
                    if (ik.second->is_null()) continue;
                    if (!f2) o += ",";
                    f2 = false;
                    if (ik.first == "properties") o += "properties:{" + (ik.second->is_obj() ? g4_params(*ik.second) : std::string()) + "}";
                    else if (ik.first == "required") o += g4_required(*ik.second);
                    else if (ik.first == "type") o += "type:" + g4_arg(Json(g4_type(*ik.second)));
                    else o += ik.first + ":" + g4_arg(*ik.second);
                }
                o += "}";
            }
        }
        if (v.get_bool("nullable", false)) { sep(); o += "nullable:true"; }
        if (T == "OBJECT") {
            if (v["properties"].is_obj()) { sep(); o += "properties:{" + g4_params(v["properties"]) + "}"; }
            else { sep(); o += "properties:{" + g4_params(v, true) + "}"; }
            if (v["required"].size()) { sep(); o += g4_required(v["required"]); }
        }
        sep();
        o += "type:" + std::string(kQ) + T + kQ + "}";
    }
    return o;
}
static std::string g4_declaration(const Json& tool) {
    const Json& f = tool_fn(tool);
    std::string o = "declaration:" + f.get_str("name", "") + "{description:" + kQ + f.get_str("description", "") + kQ;
    const Json& p = f["parameters"];
    if (p.is_obj() && p.size()) {
        o += ",parameters:{";
        if (p["properties"].is_obj() && p["properties"].size()) o += "properties:{" + g4_params(p["properties"]) + "},";
        if (p["required"].size()) o += g4_required(p["required"]) + ",";
        o += "type:" + std::string(kQ) + (p.has("type") ? g4_type(p["type"]) : std::string("OBJECT")) + kQ + "}";
    }
    return o + "}";
}

std::string Tokenizer::render_gemma4(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                                     const std::vector<Json>& tools) const {
    std::string o = chat_prefix_;
    const bool think = enable_thinking && !think_system_token_.empty();
    const bool has_sys = !msgs.empty() && (msgs[0].role == "system" || msgs[0].role == "developer");
    std::string prev_type;
    size_t i0 = 0;
    if (think || !tools.empty() || has_sys) {
        o += "<|turn>system\n";
        if (think) { o += think_system_token_ + "\n"; prev_type = "think"; }
        if (has_sys) { o += trim_ws(msgs[0].content); i0 = 1; }
        for (auto& t : tools) o += "<|tool>" + g4_declaration(t) + "<tool|>";
        if (!tools.empty()) prev_type = "tool";
        o += "<turn|>\n";
    }
    int last_user = -1;
    for (size_t i = i0; i < msgs.size(); i++) if (msgs[i].role == "user") last_user = (int)i;
    std::string prev_non_tool;
    auto strip_thinking = [&](const std::string& t) {   // drop <|channel>...<channel|> blocks
        std::string r;
        size_t p = 0;
        while (true) {
            size_t a = t.find("<|channel>", p);
            if (a == std::string::npos) { r += t.substr(p); break; }
            r += t.substr(p, a - p);
            size_t b = t.find("<channel|>", a);
            if (b == std::string::npos) break;
            p = b + 10;
        }
        return trim_ws(r);
    };
    for (size_t i = i0; i < msgs.size(); i++) {
        const ChatMessage& m = msgs[i];
        if (m.role == "tool") continue;
        prev_type.clear();
        const std::string role = m.role == "assistant" ? "model" : m.role;
        if (!(role == "model" && prev_non_tool == "assistant")) o += "<|turn>" + role + "\n";
        if (!m.reasoning.empty() && (int)i > last_user) o += "<|channel>thought\n" + m.reasoning + "\n<channel|>";
        for (auto& tc : m.tool_calls) {
            o += "<|tool_call>call:" + tc.name + "{";
            bool first = true;
            for (auto& kv : dictsort(call_args(tc))) {
                if (!first) o += ",";
                first = false;
                o += kv.first + ":" + g4_arg(*kv.second, false);
            }
            o += "}<tool_call|>";
            prev_type = "tool_call";
        }
        bool tr_out = false;
        if (!m.tool_calls.empty())
            for (size_t k = i + 1; k < msgs.size() && msgs[k].role == "tool"; k++) {
                std::string name = msgs[k].name.empty() ? "unknown" : msgs[k].name;
                for (auto& tc : m.tool_calls) if (!tc.id.empty() && tc.id == msgs[k].tool_call_id) name = tc.name;
                o += "<|tool_response>response:" + name + "{value:" + g4_arg(Json(msgs[k].content), false) + "}<tool_response|>";
                tr_out = true;
                prev_type = "tool_response";
            }
        const std::string content = role == "model" ? strip_thinking(m.content) : trim_ws(m.content);
        o += content;
        std::string next_role;
        for (size_t j = i + 1; j < msgs.size(); j++) if (msgs[j].role != "tool") { next_role = msgs[j].role; break; }
        const bool continues = role == "model" && next_role == "assistant" && (m.tool_calls.empty() || tr_out);
        if (prev_type == "tool_call" && !tr_out) o += "<|tool_response>";
        else if (continues) {}
        else if (!(tr_out && content.empty() && next_role.empty())) o += "<turn|>\n";
        prev_non_tool = m.role;
    }
    if (add_gen) {
        if (prev_type != "tool_response" && prev_type != "tool_call") o += "<|turn>model\n";
        else if (prev_type == "tool_response" && think) o += think_tag_open_;
    }
    return o;
}

std::string Tokenizer::render_chat(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                                   const std::vector<Json>& tools) const {
    if (tool_format_ == ToolFormat::Generic) {
        bool tooly = !tools.empty();
        for (auto& m : msgs) tooly |= m.role == "tool" || !m.tool_calls.empty();
        if (tooly) return render_generic_tools(msgs, add_gen, enable_thinking, tools);
    }
    if (chat_format_ == "gemma4") return render_gemma4(msgs, add_gen, enable_thinking, tools);
    std::string o = chat_prefix_;
    if (chat_format_ == "llama3") {
        if (!add_bos_) o += "<|begin_of_text|>";
        for (auto& m : msgs) o += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + trim_ws(m.content) + "<|eot_id|>";
        if (add_gen) o += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return o;
    }
    if (chat_format_ == "gemma") {
        // Gemma 3: the system prompt is prepended to the first user turn; assistant role is "model"
        std::string sys;
        size_t i0 = 0;
        if (!msgs.empty() && msgs[0].role == "system") { sys = msgs[0].content + "\n\n"; i0 = 1; }
        for (size_t i = i0; i < msgs.size(); i++) {
            const auto& m = msgs[i];
            std::string role = m.role == "assistant" ? "model" : m.role;
            std::string content = m.content;
            if (m.role == "assistant") {  // drop reasoning from history
                size_t p = content.find(think_tag_close_);
                if (p != std::string::npos) content = content.substr(p + think_tag_close_.size());
            }
            o += "<start_of_turn>" + role + "\n" + (i == i0 ? sys : std::string()) + trim_ws(content) + "<end_of_turn>\n";
        }
        if (add_gen) o += "<start_of_turn>model\n";
        return o;
    }
    auto strip_reasoning = [&](const std::string& c) {
        size_t p = c.find("</think>");
        return p == std::string::npos ? c : trim_ws(c.substr(p + 8));
    };
    if (chat_format_ == "granite") {
        size_t i0 = 0;
        if (!msgs.empty() && msgs[0].role == "system") { o += "<|start_of_role|>system<|end_of_role|>" + msgs[0].content + "<|end_of_text|>\n"; i0 = 1; }
        else {   // the template's default system prompt
            char date[64]; time_t t = time(nullptr); struct tm tmv; localtime_r(&t, &tmv);
            strftime(date, sizeof date, "%B %d, %Y", &tmv);
            o += std::string("<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: ") + date +
                 ".\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n";
        }
        for (size_t i = i0; i < msgs.size(); i++) o += "<|start_of_role|>" + msgs[i].role + "<|end_of_role|>" + msgs[i].content + "<|end_of_text|>\n";
        if (add_gen) o += "<|start_of_role|>assistant<|end_of_role|>";
        return o;
    }
    if (chat_format_ == "phi") {
        for (auto& m : msgs) o += "<|" + m.role + "|>" + m.content + "<|end|>";
        if (add_gen) o += "<|assistant|>";
        return o;
    }
    if (chat_format_ == "deepseek") {
        // DeepSeek-R1 distills: system text first, reasoning dropped from history, generation opens <think>
        const std::string U = "<｜User｜>", A = "<｜Assistant｜>";
        const std::string eos = eos_.empty() ? std::string() : token_bytes(eos_[0]);
        for (auto& m : msgs) if (m.role == "system") o += m.content;
        for (auto& m : msgs) {
            if (m.role == "user") o += U + m.content;
            else if (m.role == "assistant") o += A + strip_reasoning(m.content) + eos;
        }
        if (add_gen) o += A + "<think>\n";
        return o;
    }
    if (chat_format_ == "mistral") {
        // Mistral v0.3: [INST] user[/INST] answer</s>; the system prompt is prepended to the last user turn;
        // tools are listed right before the last user turn, calls / results use the v3 tool tokens
        std::string sys;
        for (auto& m : msgs) if (m.role == "system") sys = m.content;
        int last_user = -1;
        for (int i = (int)msgs.size() - 1; i >= 0; i--) if (msgs[i].role == "user") { last_user = i; break; }
        const std::string eos = eos_.empty() ? std::string() : token_bytes(eos_[0]);
        auto id9 = [](const std::string& id) {   // Mistral tool call ids are 9 alphanumerics
            static const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            bool ok = id.size() == 9;
            for (char c : id) ok &= isalnum((unsigned char)c) != 0;
            if (ok) return id;
            uint64_t h = fnv1a64(id.data(), id.size());
            std::string r;
            for (int k = 0; k < 9; k++) { r += al[h % 62]; h /= 62; }
            return r;
        };
        for (int i = 0; i < (int)msgs.size(); i++) {
            const auto& m = msgs[i];
            if (m.role == "user") {
                if (!tools.empty() && i == last_user) {
                    o += "[AVAILABLE_TOOLS] [";
                    for (size_t k = 0; k < tools.size(); k++) {
                        Json f = Json::object();
                        for (auto& kv : tool_fn(tools[k]).obj) if (kv.first != "return") f.set(kv.first, kv.second);
                        Json w = Json::object(); w.set("type", "function"); w.set("function", f);
                        o += w.dump(-2) + (k + 1 < tools.size() ? ", " : "]");
                    }
                    o += "[/AVAILABLE_TOOLS]";
                }
                o += "[INST] " + (i == last_user && !sys.empty() ? sys + "\n\n" : std::string()) + m.content + "[/INST]";
            } else if (m.role == "assistant" && !m.tool_calls.empty()) {
                o += "[TOOL_CALLS] [";
                for (size_t k = 0; k < m.tool_calls.size(); k++) {
                    Json f = Json::object();
                    f.set("name", m.tool_calls[k].name);
                    f.set("arguments", call_args(m.tool_calls[k]));
                    std::string d = f.dump(-2);
                    d.pop_back();
                    o += d + ", \"id\": \"" + id9(m.tool_calls[k].id) + "\"}" + (k + 1 < m.tool_calls.size() ? ", " : "]" + eos);
                }
            } else if (m.role == "assistant") {
                o += " " + trim_ws(m.content) + eos;
            } else if (m.role == "tool") {
                o += "[TOOL_RESULTS] {\"content\": " + m.content + ", \"call_id\": \"" + id9(m.tool_call_id) + "\"}[/TOOL_RESULTS]";
            }
        }
        return o;
    }
    if (chat_format_ == "tulu") {
        const std::string eos = eos_.empty() ? std::string() : token_bytes(eos_[0]);
        for (auto& m : msgs) o += "<|" + m.role + "|>\n" + m.content + (m.role == "assistant" ? eos + "\n" : "\n");
        if (add_gen) o += "<|assistant|>\n";
        return o;
    }
    if (chat_format_ == "raw") {
        for (auto& m : msgs) o += m.content + "\n";
        return o;
    }
    if (tool_format_ == ToolFormat::Lfm2) {
        // LFM2 / LFM2.5: tools as a JSON list in the system prompt, Python-style calls, reasoning kept for the last turn
        std::string sys;
        size_t i0 = 0;
        if (!msgs.empty() && msgs[0].role == "system") { sys = msgs[0].content; i0 = 1; }
        if (!tools.empty()) {
            sys += (sys.empty() ? "" : "\n") + std::string("List of tools: [");
            for (size_t k = 0; k < tools.size(); k++) sys += (k ? ", " : "") + tools[k].dump(-2);
            sys += "]";
        }
        if (!sys.empty()) o += "<|im_start|>system\n" + sys + "<|im_end|>\n";
        int last_asst = -1;
        for (size_t i = i0; i < msgs.size(); i++) if (msgs[i].role == "assistant") last_asst = (int)i;
        for (size_t i = i0; i < msgs.size(); i++) {
            const auto& m = msgs[i];
            o += "<|im_start|>" + m.role + "\n";
            if (m.role == "assistant") {
                const bool keep = (int)i == last_asst;
                if (!m.reasoning.empty() && keep) o += "<think>" + m.reasoning + "</think>";
                std::string content = m.content;
                size_t p = content.rfind("</think>");
                if (!keep && p != std::string::npos) content = trim_ws(content.substr(p + 8));
                o += content;
                if (!m.tool_calls.empty()) {
                    o += "<|tool_call_start|>[";
                    for (size_t k = 0; k < m.tool_calls.size(); k++) {
                        o += (k ? ", " : "") + m.tool_calls[k].name + "(";
                        bool first = true;
                        for (auto& kv : call_args(m.tool_calls[k]).obj) {
                            o += (first ? "" : ", ") + kv.first + "=";
                            first = false;
                            const Json& v = kv.second;
                            if (v.is_str()) {
                                std::string q = "'";
                                for (char c : v.str) {
                                    if (c == '\\') q += "\\\\"; else if (c == '\'') q += "\\'"; else if (c == '\n') q += "\\n";
                                    else if (c == '\r') q += "\\r"; else q += c;
                                }
                                o += q + "'";
                            } else if (v.is_obj() || v.is_arr()) o += v.dump(-2);
                            else o += py_str(v);
                        }
                        o += ")";
                    }
                    o += "]<|tool_call_end|>";
                }
                o += "<|im_end|>\n";
            } else {
                o += m.content + "<|im_end|>\n";
            }
        }
        if (add_gen) o += "<|im_start|>assistant\n";
        return o;
    }
    // ChatML (Qwen family). Mirrors the Qwen3 / Qwen3.5 templates: reasoning of earlier turns is dropped.
    const bool xml = tool_format_ == ToolFormat::QwenXml;
    int last_user = -1;
    for (int i = (int)msgs.size() - 1; i >= 0; i--) if (msgs[i].role == "user") { last_user = i; break; }
    size_t first = 0;
    if (!tools.empty() && xml) {
        // Qwen3.5 style: XML function calls, system prompt after the tool instructions
        o += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto& t : tools) o += "\n" + t.dump(-2);
        o += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
             "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
             "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
             "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within "
             "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function "
             "call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
             "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!msgs.empty() && msgs[0].role == "system") { o += "\n\n" + trim_ws(msgs[0].content); first = 1; }
        o += "<|im_end|>\n";
    } else if (!tools.empty()) {
        o += "<|im_start|>system\n";
        if (!msgs.empty() && msgs[0].role == "system") { o += msgs[0].content + "\n\n"; first = 1; }
        o += kHermesTools;
        for (auto& t : tools) o += "\n" + t.dump(-2);
        o += std::string(kHermesToolsEnd) + "<|im_end|>\n";
    }
    for (size_t i = first; i < msgs.size(); i++) {
        const auto& m = msgs[i];
        if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") o += "<|im_start|>user";
            o += "\n<tool_response>\n" + (xml ? trim_ws(m.content) : m.content) + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") o += "<|im_end|>\n";
            continue;
        }
        std::string content = xml ? trim_ws(m.content) : m.content;
        if (m.role == "assistant") {
            std::string reasoning = m.reasoning;
            size_t p = content.find("</think>");
            if (p != std::string::npos) {
                std::string before = content.substr(0, p);
                size_t q = before.rfind("<think>");
                if (reasoning.empty()) reasoning = strip_nl(q == std::string::npos ? before : before.substr(q + 7));
                content = content.substr(p + 8);
                size_t a = 0; while (a < content.size() && content[a] == '\n') a++;
                content = content.substr(a);
            }
            if (thinking_ && (int)i > last_user) {
                if (xml) content = "<think>\n" + trim_ws(reasoning) + "\n</think>\n\n" + content;
                else if (!reasoning.empty() || i + 1 == msgs.size()) content = "<think>\n" + strip_nl(reasoning) + "\n</think>\n\n" + content;
            }
            for (size_t k = 0; k < m.tool_calls.size(); k++) {
                const ToolCall& tc = m.tool_calls[k];
                if (xml) {
                    content += k > 0 ? "\n" : (trim_ws(m.content).empty() ? "" : "\n\n");
                    content += "<tool_call>\n<function=" + tc.name + ">\n";
                    for (auto& kv : call_args(tc).obj)
                        content += "<parameter=" + kv.first + ">\n" + (kv.second.is_obj() || kv.second.is_arr() ? kv.second.dump(-2) : py_str(kv.second)) + "\n</parameter>\n";
                    content += "</function>\n</tool_call>";
                } else {
                    if (k > 0 || !m.content.empty()) content += "\n";
                    std::string args = trim_ws(tc.arguments);
                    content += "<tool_call>\n{\"name\": \"" + tc.name + "\", \"arguments\": " + (args.empty() ? "{}" : args) + "}\n</tool_call>";
                }
            }
        }
        o += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
    }
    if (add_gen) {
        o += "<|im_start|>assistant\n";
        if (thinking_) {
            if (!enable_thinking && think_disable_block_) o += "<think>\n\n</think>\n\n";
            else if (enable_thinking && think_open_) o += "<think>\n";
        }
    }
    return o;
}

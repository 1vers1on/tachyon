#include "tokenizer.h"
#include "unicode_tables.h"
#include <algorithm>

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
    for (int i = 0; i < nv; i++) {
        vocab_[i].assign((const char*)vd->data + offs[i], offs[i + 1] - offs[i]);
        if (ttype_[i] == 0) tok2id_.emplace(vocab_[i], i);
    }
    for (int b = 0; b < 256; b++) {
        auto it = tok2id_.find(std::string(1, (char)b));
        byte_tok_[b] = it == tok2id_.end() ? -1 : it->second;
    }
    const int32_t* m = (const int32_t*)mg->data;
    int64_t nm = mg->size / 12;
    merges_.reserve(nm * 2);
    for (int64_t i = 0; i < nm; i++) {
        uint64_t key = ((uint64_t)(uint32_t)m[3 * i] << 32) | (uint32_t)m[3 * i + 1];
        merges_.emplace(key, std::make_pair((int)i, (int)m[3 * i + 2]));
    }
    for (auto& s : tj["specials"].arr) specials_.push_back({s.get_str("content", ""), (int)s.get_int("id", -1)});
    std::sort(specials_.begin(), specials_.end(), [](const Special& a, const Special& b) { return a.s.size() > b.s.size(); });
    pretok_ = tj.get_str("pretokenizer", "qwen2");
    for (auto& e : tj["eos"].arr) eos_.push_back((int)e.as_int());
    bos_ = (int)tj.get_int("bos", -1);
    add_bos_ = tj.get_bool("add_bos", false);
    thinking_ = tj.get_bool("thinking", false);
    chat_format_ = tj.get_str("chat_format", "chatml");
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
    const int max_digits = pretok_ == "llama3" ? 3 : (gpt2 ? 1 << 30 : 1);
    auto L = [&](int i) { return i < n && unicode_is_letter(c[i]); };
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
        while (sym.size() > 1) {
            int best = -1, best_rank = INT32_MAX, best_id = -1;
            for (size_t k = 0; k + 1 < sym.size(); k++) {
                uint64_t key = ((uint64_t)(uint32_t)sym[k] << 32) | (uint32_t)sym[k + 1];
                auto it = merges_.find(key);
                if (it != merges_.end() && it->second.first < best_rank) {
                    best_rank = it->second.first; best = (int)k; best_id = it->second.second;
                }
            }
            if (best < 0) break;
            sym[best] = best_id;
            sym.erase(sym.begin() + best + 1);
        }
    }
    out.insert(out.end(), sym.begin(), sym.end());
    std::lock_guard<std::mutex> lk(cache_mu_);
    if (cache_.size() > 200000) cache_.clear();
    cache_.emplace(piece, std::move(sym));
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    std::vector<uint32_t> cps;
    std::vector<int> offs;
    utf8_decode(text, cps, offs);
    std::vector<std::pair<int, int>> spans;
    pretokenize(cps.data(), (int)cps.size(), spans);
    for (auto& sp : spans) bpe(text.substr(offs[sp.first], offs[sp.second] - offs[sp.first]), out);
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    if (add_bos_ && bos_ >= 0) out.push_back(bos_);
    if (!parse_special || specials_.empty()) { encode_plain(text, out); return out; }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<' || text[i] == '[' || (unsigned char)text[i] >= 0x80 || text[i] == '|') {
            for (auto& sp : specials_) {
                if (!sp.s.empty() && text.compare(i, sp.s.size(), sp.s) == 0) {
                    encode_plain(text.substr(start, i - start), out);
                    out.push_back(sp.id);
                    i += sp.s.size();
                    start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) i++;
    }
    encode_plain(text.substr(start), out);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) if (id >= 0 && id < (int)vocab_.size()) s += vocab_[id];
    return s;
}

std::string StreamDecoder::push(int id) {
    if (id >= 0 && id < tok_->n_vocab()) pending_ += tok_->token_bytes(id);
    size_t k = utf8_valid_prefix(pending_);
    std::string out = pending_.substr(0, k);
    pending_.erase(0, k);
    return out;
}
std::string StreamDecoder::flush() {
    std::string o;
    o.swap(pending_);
    return o;
}

static std::string strip_nl(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && s[a] == '\n') a++;
    while (b > a && s[b - 1] == '\n') b--;
    return s.substr(a, b - a);
}

std::string Tokenizer::render_chat(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                                   const std::vector<std::string>& tools) const {
    std::string o;
    if (chat_format_ == "llama3") {
        o += "<|begin_of_text|>";
        for (auto& m : msgs) o += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
        if (add_gen) o += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return o;
    }
    if (chat_format_ == "raw") {
        for (auto& m : msgs) o += m.content + "\n";
        return o;
    }
    // ChatML (Qwen family). Mirrors the Qwen3 template: reasoning of earlier turns is dropped.
    int last_user = -1;
    for (int i = (int)msgs.size() - 1; i >= 0; i--) if (msgs[i].role == "user") { last_user = i; break; }
    size_t first = 0;
    if (!tools.empty()) {
        o += "<|im_start|>system\n";
        if (!msgs.empty() && msgs[0].role == "system") { o += msgs[0].content + "\n\n"; first = 1; }
        o += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
             "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (auto& t : tools) o += "\n" + t;
        o += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
             "</tool_call><|im_end|>\n";
    }
    for (size_t i = first; i < msgs.size(); i++) {
        const auto& m = msgs[i];
        if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") o += "<|im_start|>user";
            o += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") o += "<|im_end|>\n";
            continue;
        }
        std::string content = m.content;
        if (m.role == "assistant" && thinking_) {
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
            if ((int)i > last_user && !reasoning.empty())
                content = "<think>\n" + strip_nl(reasoning) + "\n</think>\n\n" + content;
        }
        o += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
    }
    if (add_gen) {
        o += "<|im_start|>assistant\n";
        if (thinking_ && !enable_thinking) o += "<think>\n\n</think>\n\n";
    }
    return o;
}

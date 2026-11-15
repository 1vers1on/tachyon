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
    for (auto& s : tj["specials"].arr) specials_.push_back({s.get_str("content", ""), (int)s.get_int("id", -1)});
    std::sort(specials_.begin(), specials_.end(), [](const Special& a, const Special& b) { return a.s.size() > b.s.size(); });
    for (auto& sp : specials_) if (!sp.s.empty()) special_first_[(unsigned char)sp.s[0]] = true;
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
    think_default_ = tj.get_bool("think_default", true);
    think_open_ = tj.get_bool("think_open", false);
    think_disable_block_ = tj.get_bool("think_disable_block", true);
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
    std::vector<int> out;
    if (add_bos_ && bos_ >= 0) out.push_back(bos_);
    if (!parse_special || specials_.empty()) { encode_plain(text, out); return out; }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (special_first_[(unsigned char)text[i]]) {
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
    for (int id : ids) if (id >= 0 && id < (int)vocab_.size()) s += token_bytes(id);
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
    std::string o = chat_prefix_;
    if (chat_format_ == "llama3") {
        o += "<|begin_of_text|>";
        for (auto& m : msgs) o += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
        if (add_gen) o += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return o;
    }
    auto trim = [](const std::string& x) {
        size_t a = x.find_first_not_of(" \t\n\r"), b = x.find_last_not_of(" \t\n\r");
        return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
    };
    if (chat_format_ == "gemma" || chat_format_ == "gemma4") {
        // Gemma: the system prompt is prepended to the first user turn; assistant role is "model"
        const bool g4 = chat_format_ == "gemma4";
        const std::string so = g4 ? "<|turn>" : "<start_of_turn>", eo = g4 ? "<turn|>\n" : "<end_of_turn>\n";
        std::string sys;
        size_t i0 = 0;
        const bool has_sys = !msgs.empty() && msgs[0].role == "system";
        if (g4) {
            const bool think = enable_thinking && !think_system_token_.empty();
            if (think || has_sys) {
                o += so + "system\n";
                if (think) o += think_system_token_ + "\n";
                if (has_sys) o += trim(msgs[0].content);
                o += eo;
            }
            if (has_sys) i0 = 1;
        } else if (has_sys) {
            sys = msgs[0].content + "\n\n";
            i0 = 1;
        }
        for (size_t i = i0; i < msgs.size(); i++) {
            const auto& m = msgs[i];
            std::string role = m.role == "assistant" ? "model" : m.role;
            std::string content = m.content;
            if (m.role == "assistant") {  // drop reasoning from history
                size_t p = content.find(think_tag_close_);
                if (p != std::string::npos) content = content.substr(p + think_tag_close_.size());
            }
            o += so + role + "\n" + (i == i0 ? sys : std::string()) + trim(content) + eo;
        }
        if (add_gen) o += so + "model\n";
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
    // ChatML (Qwen / LFM2 family). Mirrors the Qwen3 template: reasoning of earlier turns is dropped.
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
        if (thinking_) {
            if (!enable_thinking && think_disable_block_) o += "<think>\n\n</think>\n\n";
            else if (enable_thinking && think_open_) o += "<think>\n";
        }
    }
    return o;
}

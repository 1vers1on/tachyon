// Tachyon - byte-level BPE tokenizer (Qwen2/Qwen3, Llama-3, GPT-2 style pre-tokenizers)
#pragma once
#include "common.h"
#include "json.h"
#include "model_file.h"
#include <unordered_map>
#include <mutex>

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning;  // optional reasoning_content
};

class Tokenizer {
public:
    void load(const ModelFile& mf);
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    std::string decode(const std::vector<int>& ids) const;
    const std::string& token_bytes(int id) const { return spm_ ? dec_[id] : vocab_[id]; }
    bool is_special(int id) const { return id >= 0 && id < (int)ttype_.size() && ttype_[id] == 1; }
    bool is_eos(int id) const;
    int n_vocab() const { return (int)vocab_.size(); }
    int bos() const { return bos_; }
    bool add_bos() const { return add_bos_; }
    int token_id(const std::string& s) const;  // exact token lookup (special or normal), -1 if none
    const std::vector<int>& eos_ids() const { return eos_; }

    // chat
    // tools: function definitions already serialized as JSON (one string per tool)
    std::string render_chat(const std::vector<ChatMessage>& msgs, bool add_generation_prompt, bool enable_thinking,
                            const std::vector<std::string>& tools = {}) const;
    bool thinking_model() const { return thinking_; }
    bool think_default() const { return think_default_; }
    // generation prompt already opens a <think> block (model output starts inside reasoning)
    bool think_opens_in_prompt(bool enable_thinking) const { return thinking_ && think_open_ && enable_thinking; }
    std::string chat_format() const { return chat_format_; }

private:
    void pretokenize(const uint32_t* cps, int n, std::vector<std::pair<int, int>>& out) const;
    void bpe(const std::string& piece, std::vector<int>& out) const;
    void encode_plain(const std::string& text, std::vector<int>& out) const;
    void bpe_merge(std::vector<int>& sym) const;   // heap-based BPE merge loop
    void encode_spm(const std::string& text, std::vector<int>& out) const;

    std::vector<std::string> vocab_;
    std::vector<uint8_t> ttype_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (a<<32|b) -> (rank, merged)
    int byte_tok_[256];
    struct Special { std::string s; int id; };
    std::vector<Special> specials_;  // sorted by length desc
    bool special_first_[256] = {};   // first bytes of added tokens (fast skip)
    std::string pretok_ = "qwen2";
    std::vector<int> eos_;
    int bos_ = -1;
    bool add_bos_ = false;
    bool thinking_ = false;
    std::string chat_format_ = "chatml";
    std::string chat_prefix_;
    bool think_default_ = true, think_open_ = false, think_disable_block_ = true;
    bool spm_ = false;                 // SentencePiece-style BPE (Gemma)
    std::vector<std::string> dec_;     // spm: decoded bytes per token
    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

// Incremental UTF-8 safe detokenizer for streaming
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer* t) : tok_(t) {}
    // returns newly completed text (may be empty if waiting for more bytes of a UTF-8 sequence)
    std::string push(int id);
    std::string flush();
private:
    const Tokenizer* tok_;
    std::string pending_;
};

bool unicode_is_letter(uint32_t cp);
bool unicode_is_number(uint32_t cp);
bool unicode_is_space(uint32_t cp);
bool unicode_is_mark(uint32_t cp);
size_t utf8_valid_prefix(const std::string& s);

// Tachyon - byte-level BPE tokenizer (Qwen2/Qwen3, Llama-3, GPT-2 style pre-tokenizers)
#pragma once
#include "common.h"
#include "json.h"
#include "model_file.h"
#include <unordered_map>
#include <mutex>

struct ToolCall {
    std::string id, name;
    std::string arguments;  // JSON object text
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning;  // optional reasoning_content
    std::vector<ToolCall> tool_calls;   // assistant turns
    std::string tool_call_id, name;     // role "tool": the call this result answers
};

// How a model writes tool calls (picked from the chat format / stored template)
enum class ToolFormat {
    Hermes,    // Qwen3 / Qwen2.5: <tool_call>{"name": .., "arguments": ..}</tool_call>
    QwenXml,   // Qwen3.5: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
    Lfm2,      // LFM2 / LFM2.5: <|tool_call_start|>[name(k='v', n=1)]<|tool_call_end|>
    Gemma4,    // <|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>
    Mistral,   // [TOOL_CALLS] [{"name": .., "arguments": ..}]
    Ifm,       // K2 Horizon: <ifm|tool_calls> with XML arguments
    Generic,   // models without a native format: Hermes-style instructions in the system prompt
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
    // tools: OpenAI-style definitions {"type": "function", "function": {name, description, parameters}}
    std::string render_chat(const std::vector<ChatMessage>& msgs, bool add_generation_prompt, bool enable_thinking,
                            const std::vector<Json>& tools = {}) const;
    ToolFormat tool_format() const { return tool_format_; }
    // text that opens a tool call in the model's format (forced tool_choice prefills the prompt with it);
    // with a name, also the start of that function's call
    std::string tool_call_prefix(const std::string& name = "") const;
    bool thinking_model() const { return thinking_; }
    bool think_default() const { return think_default_; }
    const std::string& think_open_tag() const { return think_tag_open_; }
    bool xml_tools() const { return tool_style_ == "xml"; }
    const std::string& think_close_tag() const { return think_tag_close_; }
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
    struct Special { std::string s; int id; bool special = true; };
    std::vector<Special> specials_;  // sorted by length desc
    std::vector<int> by_first_[256]; // indices into specials_ by first byte (length desc)
    std::string pretok_ = "qwen2";
    std::vector<int> eos_;
    int bos_ = -1;
    bool add_bos_ = false;
    bool thinking_ = false;
    std::string chat_format_ = "chatml";
    std::string chat_prefix_;
    std::string native_template_;
    std::string tool_style_ = "json";
    ToolFormat tool_format_ = ToolFormat::Hermes;
    std::string render_generic_tools(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                                     const std::vector<Json>& tools) const;
    std::string render_gemma4(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking,
                              const std::vector<Json>& tools) const;
    std::string think_tag_open_ = "<think>", think_tag_close_ = "</think>", think_system_token_;
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

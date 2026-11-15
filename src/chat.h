// Tachyon - protocol-independent chat output processing: reasoning split and tool-call parsing (streaming)
#pragma once
#include "tokenizer.h"

// Splits "<think>...</think>" reasoning from the visible answer, robust to tags split across deltas.
class ThinkSplitter {
public:
    void init(bool active, bool starts_inside, const std::string& open = "<think>", const std::string& close = "</think>");
    // returns (reasoning_delta, content_delta)
    std::pair<std::string, std::string> push(const std::string& d, bool final = false);
    bool in_reasoning() const { return mode_ == 1 || mode_ == 0; }
private:
    bool seen_content_ = false, r_started_ = false;
    std::string r_nl_;
    int mode_ = 2;  // 0 undecided, 1 reasoning, 2 plain, 3 after reasoning
    std::string buf_, open_ = "<think>", close_ = "</think>";
};

// Finds tool calls in the model's content stream (format of the model's chat template) and turns them into
// ToolCall objects. Text outside calls passes through; whitespace that only separates text from a call is dropped.
class ToolCallParser {
public:
    void init(ToolFormat f);
    // returns content text that is safe to show; completed calls are appended to `calls`
    std::string push(const std::string& d, bool final, std::vector<ToolCall>& calls);
    int n_calls() const { return n_calls_; }
private:
    bool parse_block(const std::string& body, std::vector<ToolCall>& calls) const;
    ToolFormat fmt_ = ToolFormat::Hermes;
    std::string start_, end_, buf_, ws_;
    bool in_call_ = false, any_text_ = false;
    int n_calls_ = 0;
};

// Parse one tool-call block body (between the format's start and end markers). Exposed for tests.
bool parse_tool_calls(ToolFormat f, const std::string& body, std::vector<ToolCall>& calls);

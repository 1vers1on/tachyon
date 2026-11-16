#pragma once
#include "model_file.h"

// Metadata shared by the model manager and the single-model server.
inline Json ollama_details(const ModelFile& mf) {
    Json d = Json::object(), families = Json::array();
    families.push(mf.config.arch);
    d.set("parent_model", "");
    d.set("format", "tach");
    d.set("family", mf.config.arch);
    d.set("families", families);
    int64_t parameters = 0;
    for (const auto& [name, t] : mf.tensors) parameters += t.rows * t.cols;
    char size[32];
    snprintf(size, sizeof size, parameters >= 1000000000 ? "%.1fB" : "%.1fM", parameters / (parameters >= 1000000000 ? 1e9 : 1e6));
    d.set("parameter_size", size);
    d.set("quantization_level", qtype_name(mf.config.wtype));
    return d;
}

inline Json ollama_show(const ModelFile& mf) {
    Json j = Json::object(), info = Json::object(), caps = Json::array();
    j.set("details", ollama_details(mf));
    j.set("template", mf.header()["tokenizer"].get_str("chat_template", ""));
    j.set("parameters", "");
    j.set("modelfile", "FROM " + mf.path + "\n");
    j.set("license", "");
    const auto& c = mf.config;
    info.set("general.architecture", c.arch);
    info.set(c.arch + ".context_length", c.max_ctx);
    info.set(c.arch + ".embedding_length", c.hidden);
    info.set(c.arch + ".block_count", c.n_layers);
    j.set("model_info", info);
    caps.push("completion");
    caps.push("tools");
    if (mf.header()["tokenizer"].get_bool("thinking", false)) caps.push("thinking");
    j.set("capabilities", caps);
    return j;
}

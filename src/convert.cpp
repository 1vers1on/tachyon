// Tachyon - HuggingFace (safetensors + tokenizer.json) -> .tach converter
#include "common.h"
#include "json.h"
#include "safetensors.h"
#include "model_file.h"
#include "quant.h"
#include <thread>
#include <unordered_map>
#include <algorithm>

// ---- GPT-2 byte-level unicode mapping ----
static std::vector<int> bytes_to_unicode_cp() {
    std::vector<int> cp(256, -1);
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    for (int b : bs) cp[b] = b;
    int n = 0;
    for (int b = 0; b < 256; b++) if (cp[b] < 0) cp[b] = 256 + n++;
    return cp;
}

static bool utf8_next(const std::string& s, size_t& i, uint32_t& cp) {
    unsigned char c = s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) { cp = c; i++; return false; }
    if (len == 1) cp = c;
    else if (len == 2) cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
    else if (len == 3) cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
    else cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    i += len;
    return true;
}

// decode a byte-level token string back to raw bytes
static bool bytelevel_decode(const std::string& tok, const std::unordered_map<uint32_t, int>& inv, std::string& out) {
    out.clear();
    size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp;
        utf8_next(tok, i, cp);
        auto it = inv.find(cp);
        if (it == inv.end()) return false;
        out += (char)it->second;
    }
    return true;
}

struct ConvertOptions {
    std::string in_dir, out_path;
    QType wtype = QType::Q4;
    QType embed_type = QType::Q4;   // token embedding / lm head
    int threads = 0;
    int max_ctx = 0;
    std::vector<std::pair<std::string, QType>> overrides;  // substring -> qtype
    std::string imatrix;
};

static std::string detect_pretok(const Json& tj) {
    std::string pat;
    std::function<void(const Json&)> walk = [&](const Json& j) {
        if (j.is_obj()) {
            if (j.get_str("type", "") == "Split" && j["pattern"].has("Regex")) pat = j["pattern"]["Regex"].str;
            for (auto& kv : j.obj) walk(kv.second);
        } else if (j.is_arr()) for (auto& x : j.arr) walk(x);
    };
    walk(tj["pre_tokenizer"]);
    if (pat.empty()) return "gpt2";
    if (pat.find("[\\p{L}\\p{M}]+") != std::string::npos) return "qwen35";
    if (pat.find("\\p{N}{1,3}") != std::string::npos) return "llama3";
    if (pat.find("\\p{N}") != std::string::npos && pat.find("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+") != std::string::npos) return "qwen2";
    return "gpt2";
}

static Json build_tokenizer(const std::string& dir, ModelWriter& w, int vocab_size, Json& gen_defaults) {
    Json tj = Json::parse(read_file(dir + "/tokenizer.json"));
    Json tc;
    if (file_exists(dir + "/tokenizer_config.json")) tc = Json::parse(read_file(dir + "/tokenizer_config.json"));
    std::string chat_template = tc.get_str("chat_template", "");
    if (chat_template.empty() && file_exists(dir + "/chat_template.jinja")) chat_template = read_file(dir + "/chat_template.jinja");
    const Json& model = tj["model"];
    if (model.get_str("type", "") != "BPE") die("only BPE tokenizers are supported (got %s)", model.get_str("type", "?").c_str());
    // SentencePiece-style BPE (Gemma): pieces use U+2581 for spaces and <0xNN> byte-fallback tokens; no byte-level mapping
    const bool spm = model.get_bool("byte_fallback", false);

    auto cpmap = bytes_to_unicode_cp();
    std::unordered_map<uint32_t, int> inv;
    for (int b = 0; b < 256; b++) inv[cpmap[b]] = b;

    int n_vocab = vocab_size;
    for (auto& kv : model["vocab"].obj) n_vocab = std::max(n_vocab, (int)kv.second.as_int() + 1);
    for (auto& a : tj["added_tokens"].arr) n_vocab = std::max(n_vocab, (int)a.get_int("id", 0) + 1);

    std::vector<std::string> tokens(n_vocab);
    std::vector<uint8_t> ttype(n_vocab, 0);  // 0=normal 1=special/added 2=unused
    for (int i = 0; i < n_vocab; i++) ttype[i] = 2;
    std::unordered_map<std::string, int> str2id;
    for (auto& kv : model["vocab"].obj) {
        int id = (int)kv.second.as_int();
        std::string raw;
        if (spm || !bytelevel_decode(kv.first, inv, raw)) raw = kv.first;
        tokens[id] = raw;
        ttype[id] = 0;
        str2id[kv.first] = id;
    }
    Json specials = Json::array();
    for (auto& a : tj["added_tokens"].arr) {
        int id = (int)a.get_int("id", 0);
        tokens[id] = a.get_str("content", "");
        ttype[id] = 1;
        Json s = Json::object();
        s.set("id", id);
        s.set("content", tokens[id]);
        s.set("special", a.get_bool("special", true));
        specials.push(s);
    }
    // merges -> (a, b, merged)
    std::vector<int32_t> merges;
    for (auto& m : model["merges"].arr) {
        std::string a, b;
        if (m.is_str()) {
            size_t sp = m.str.find(' ', 1);
            if (sp == std::string::npos) continue;
            a = m.str.substr(0, sp);
            b = m.str.substr(sp + 1);
        } else if (m.is_arr() && m.size() == 2) {
            a = m[0].str; b = m[1].str;
        } else continue;
        auto ia = str2id.find(a), ib = str2id.find(b), im = str2id.find(a + b);
        if (ia == str2id.end() || ib == str2id.end() || im == str2id.end()) continue;
        merges.push_back(ia->second);
        merges.push_back(ib->second);
        merges.push_back(im->second);
    }
    // blobs
    std::vector<uint32_t> offs(n_vocab + 1);
    std::string all;
    for (int i = 0; i < n_vocab; i++) { offs[i] = all.size(); all += tokens[i]; }
    offs[n_vocab] = all.size();
    auto add_blob = [&](const std::string& name, const void* d, size_t n) {
        BlobInfo b; b.name = name; b.size = n; b.offset = w.write_blob(d, n); w.add_blob(b);
    };
    add_blob("tok.vocab_data", all.data(), all.size());
    add_blob("tok.vocab_off", offs.data(), offs.size() * 4);
    add_blob("tok.type", ttype.data(), ttype.size());
    add_blob("tok.merges", merges.data(), merges.size() * 4);

    Json t = Json::object();
    t.set("type", spm ? "bpe_spm" : "bpe_bytelevel");
    t.set("pretokenizer", detect_pretok(tj));
    t.set("n_vocab", n_vocab);
    t.set("n_merges", (int64_t)(merges.size() / 3));
    t.set("specials", specials);
    auto tok_id = [&](const Json& v) -> int {
        std::string s = v.is_str() ? v.str : (v.is_obj() ? v.get_str("content", "") : "");
        if (s.empty()) return -1;
        for (auto& sp : specials.arr) if (sp.get_str("content", "") == s) return (int)sp.get_int("id", -1);
        auto it = str2id.find(s);
        return it == str2id.end() ? -1 : it->second;
    };
    int bos = tok_id(tc["bos_token"]);
    int eos = tok_id(tc["eos_token"]);
    t.set("bos", bos);
    t.set("add_bos", tc.get_bool("add_bos_token", false) && bos >= 0);
    Json eos_ids = Json::array();
    if (eos >= 0) eos_ids.push(eos);
    if (file_exists(dir + "/generation_config.json")) {
        Json g = Json::parse(read_file(dir + "/generation_config.json"));
        const Json& e = g["eos_token_id"];
        auto add = [&](int id) { for (auto& x : eos_ids.arr) if (x.as_int() == id) return; eos_ids.push(id); };
        if (e.is_num()) add((int)e.as_int());
        else for (auto& x : e.arr) add((int)x.as_int());
        for (auto k : {"temperature", "top_k", "top_p", "min_p", "repetition_penalty"})
            if (g.has(k)) gen_defaults.set(k, g[k]);
    }
    // common end-of-turn tokens
    for (auto s : {"<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>", "<|end|>"}) {
        int id = tok_id(Json(s));
        if (id < 0) continue;
        bool f = false;
        for (auto& x : eos_ids.arr) if (x.as_int() == id) f = true;
        if (!f) eos_ids.push(id);
    }
    t.set("eos", eos_ids);
    std::string fmt = "raw";
    if (chat_template.find("<|im_start|>") != std::string::npos) fmt = "chatml";
    else if (chat_template.find("<|start_header_id|>") != std::string::npos) fmt = "llama3";
    else if (chat_template.find("<start_of_turn>") != std::string::npos) fmt = "gemma";
    else if (chat_template.find("<|turn>") != std::string::npos) fmt = "gemma4";
    else if (chat_template.find("<|user|>") != std::string::npos) fmt = "tulu";
    t.set("chat_format", fmt);
    t.set("thinking", chat_template.find("<think>") != std::string::npos);
    // Qwen3: thinking on unless disabled (empty block injected when disabled)
    // Qwen3.5: thinking off unless enabled; enabling opens the block in the prompt
    const bool think_opt_in = chat_template.find("enable_thinking is defined and enable_thinking is true") != std::string::npos;
    const bool think_opt_out = chat_template.find("enable_thinking is defined and enable_thinking is false") != std::string::npos;
    t.set("think_default", !think_opt_in);
    t.set("think_open", think_opt_in);
    t.set("think_disable_block", think_opt_in || think_opt_out);
    // templates that start by emitting the BOS token text
    std::string tpl_trim = chat_template.substr(0, 40);
    if (tpl_trim.find("bos_token") != std::string::npos && !tc.get_bool("add_bos_token", false)) {
        const Json& bt = tc["bos_token"];
        std::string bs = bt.is_str() ? bt.str : bt.get_str("content", "");
        t.set("chat_prefix", bs);
    }
    t.set("chat_template", chat_template);
    return t;
}

static int nthreads_default() { int n = std::thread::hardware_concurrency(); return n > 0 ? n : 8; }

int convert_main(int argc, char** argv) {
    ConvertOptions o;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for %s", a.c_str()); return argv[++i]; };
        if (a == "-i" || a == "--input") o.in_dir = next();
        else if (a == "-o" || a == "--output") o.out_path = next();
        else if (a == "-q" || a == "--quant") { o.wtype = qtype_from_name(next()); o.embed_type = o.wtype == QType::Q3 ? QType::Q4 : o.wtype; }
        else if (a == "--embed-quant") o.embed_type = qtype_from_name(next());
        else if (a == "-t" || a == "--threads") o.threads = atoi(next().c_str());
        else if (a == "--max-ctx") o.max_ctx = atoi(next().c_str());
        else if (a == "--imatrix") o.imatrix = next();
        else if (a == "--tensor-quant") {  // e.g. wdown=q8,wqkv=q8
            std::string v = next();
            size_t st = 0;
            while (st < v.size()) {
                size_t e = v.find(',', st); if (e == std::string::npos) e = v.size();
                std::string item = v.substr(st, e - st);
                size_t eq = item.find('=');
                if (eq == std::string::npos) die("bad --tensor-quant item %s", item.c_str());
                o.overrides.push_back({item.substr(0, eq), qtype_from_name(item.substr(eq + 1))});
                st = e + 1;
            }
        }
        else if (o.in_dir.empty()) o.in_dir = a;
        else if (o.out_path.empty()) o.out_path = a;
        else die("unknown convert arg %s", a.c_str());
    }
    if (o.in_dir.empty() || o.out_path.empty())
        die("usage: tachyon convert <hf_model_dir> <out.tach> [-q q4|q8|f16] [--embed-quant q4|q8|f16]");
    if (o.threads <= 0) o.threads = nthreads_default();
    double t0 = now_sec();

    Json hc_root = Json::parse(read_file(o.in_dir + "/config.json"));
    const Json& hc = hc_root.has("text_config") ? hc_root["text_config"] : hc_root;  // multimodal wrappers: text part
    std::string mt = hc.get_str("model_type", hc_root.get_str("model_type", "llama"));
    ModelConfig c;
    if (mt == "qwen3") c.arch = "qwen3";
    else if (mt == "qwen2") c.arch = "qwen2";
    else if (mt == "llama" || mt == "mistral") c.arch = "llama";
    else if (mt == "gemma3_text" || mt == "gemma3") c.arch = "gemma3";
    else if (mt == "gemma4_text" || mt == "gemma4" || mt == "gemma4_unified_text" || mt == "gemma4_unified") c.arch = "gemma4";
    else if (mt == "lfm2") c.arch = "lfm2";
    else if (mt == "lfm2_moe") c.arch = "lfm2_moe";
    else if (mt == "olmoe") c.arch = "olmoe";
    else if (mt == "qwen3_5_text" || mt == "qwen3_5") c.arch = "qwen3_5";
    else die("unsupported model_type %s (supported: qwen3, qwen2, llama, mistral, gemma3, gemma4, lfm2, lfm2_moe, olmoe, qwen3_5)", mt.c_str());
    {
        std::string d = o.in_dir;
        while (!d.empty() && d.back() == '/') d.pop_back();
        // HF cache dir: .../models--Org--Name/snapshots/<hash>
        size_t p = d.find("models--");
        if (p != std::string::npos) {
            std::string s = d.substr(p + 8);
            s = s.substr(0, s.find('/'));
            size_t q = s.find("--");
            c.name = q == std::string::npos ? s : s.substr(q + 2);
        } else c.name = d.substr(d.rfind('/') + 1);
    }
    const std::string A = c.arch;
    const bool gemma = A == "gemma3" || A == "gemma4";
    c.n_layers = (int)hc.get_int("num_hidden_layers", 0);
    c.hidden = (int)hc.get_int("hidden_size", 0);
    c.n_heads = (int)hc.get_int("num_attention_heads", 0);
    c.n_kv_heads = (int)hc.get_int("num_key_value_heads", c.n_heads);
    c.head_dim = (int)hc.get_int("head_dim", c.hidden / std::max(1, c.n_heads));
    c.ffn = (int)hc.get_int("intermediate_size", 0);
    c.vocab = (int)hc.get_int("vocab_size", 0);
    c.rms_eps = (float)hc.get_num("rms_norm_eps", hc.get_num("norm_eps", 1e-6));
    // rope theta: legacy field or rope_parameters
    const Json& rp = hc["rope_parameters"];
    c.rope_theta = (float)hc.get_num("rope_theta", rp.get_num("rope_theta", 10000));
    c.tied_embed = hc.get_bool("tie_word_embeddings", hc_root.get_bool("tie_word_embeddings", gemma || A == "lfm2" || A == "lfm2_moe"));
    c.attn_bias = hc.get_bool("attention_bias", c.arch == "qwen2");
    c.qk_norm = c.arch == "qwen3";
    c.max_ctx = (int)hc.get_int("max_position_embeddings", 4096);
    if (o.max_ctx > 0) c.max_ctx = o.max_ctx;
    c.wtype = o.wtype;
    if (gemma) {
        c.act = "gelu";
        // HF multiplies by sqrt(hidden) rounded to the weight dtype (bf16)
        float sc = sqrtf((float)c.hidden);
        uint32_t u; memcpy(&u, &sc, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&sc, &u, 4);
        c.embed_scale = sc;
        c.final_softcap = (float)hc.get_num("final_logit_softcapping", 0);
    }
    if (A == "lfm2" || A == "lfm2_moe" || A == "qwen3_5") c.stateful = true;
    const Json& rs = hc["rope_scaling"];
    if (rs.is_obj()) {
        std::string rt = rs.get_str("rope_type", rs.get_str("type", ""));
        if (rt == "llama3") {
            c.rope_scaling = "llama3";
            c.rope_factor = (float)rs.get_num("factor", 8);
            c.rope_low_freq = (float)rs.get_num("low_freq_factor", 1);
            c.rope_high_freq = (float)rs.get_num("high_freq_factor", 4);
            c.rope_orig_ctx = (int)rs.get_int("original_max_position_embeddings", 8192);
        } else if (rt == "linear") {
            c.rope_scaling = "linear";
            c.rope_factor = (float)rs.get_num("factor", 1);
        } else if (!rt.empty() && rt != "default") {
            logi("warning: rope scaling type '%s' ignored", rt.c_str());
        }
    }
    // ---------------- per-layer configuration ----------------
    std::vector<std::string> ltypes;
    for (auto& t : hc["layer_types"].arr) ltypes.push_back(t.str);
    if (A == "gemma3" && ltypes.empty()) {
        int pat = (int)hc.get_int("sliding_window_pattern", hc.get_int("_sliding_window_pattern", 6));
        for (int i = 0; i < c.n_layers; i++) ltypes.push_back((i + 1) % pat ? "sliding_attention" : "full_attention");
    }
    int num_kv_shared = (int)hc.get_int("num_kv_shared_layers", 0);
    std::vector<int> kv_src(c.n_layers, -1);
    if (A == "gemma4" && num_kv_shared > 0) {
        int first = c.n_layers - num_kv_shared;
        for (int i = first; i < c.n_layers; i++)
            for (int j = first - 1; j >= 0; j--) if (ltypes[j] == ltypes[i]) { kv_src[i] = j; break; }
    }
    for (int i = 0; i < c.n_layers; i++) {
        LayerCfg L;
        std::string lt = i < (int)ltypes.size() ? ltypes[i] : "full_attention";
        L.n_heads = c.n_heads; L.n_kv_heads = c.n_kv_heads; L.head_dim = c.head_dim;
        L.rope_theta = c.rope_theta; L.rope_half = c.head_dim / 2; L.rope_rot = L.rope_half;
        L.ffn = c.ffn;
        if (A == "qwen3") L.qk_norm = 1;
        if (A == "olmoe") L.qk_norm = 2;
        if (A == "lfm2" || A == "lfm2_moe") {
            if (lt == "conv") { L.type = "conv"; L.conv_k = (int)hc.get_int("conv_L_cache", 3); }
            else L.qk_norm = 1;
        }
        if (A == "gemma3") {
            L.qk_norm = 1;
            L.sandwich = true;
            L.attn_scale = 1.f / sqrtf((float)hc.get_num("query_pre_attn_scalar", c.head_dim));
            if (lt == "sliding_attention") {
                L.window = (int)hc.get_int("sliding_window", 512);
                L.rope_theta = (float)hc.get_num("rope_local_base_freq", 10000);
            }
        }
        if (A == "gemma4") {
            L.qk_norm = 1;
            L.sandwich = true;
            L.v_norm = true;
            L.attn_scale = 1.f;
            const Json& rpl = rp[lt];
            L.rope_theta = (float)rpl.get_num("rope_theta", lt == "sliding_attention" ? 10000 : 1000000);
            if (lt == "sliding_attention") {
                L.window = (int)hc.get_int("sliding_window", 512);
            } else {
                L.head_dim = (int)hc.get_int("global_head_dim", c.head_dim);
                bool keqv = hc.get_bool("attention_k_eq_v", false);
                L.k_eq_v = keqv;
                if (keqv && hc["num_global_key_value_heads"].is_num()) L.n_kv_heads = (int)hc.get_int("num_global_key_value_heads", c.n_kv_heads);
                float prf = (float)rpl.get_num("partial_rotary_factor", 1.0);
                L.rope_half = L.head_dim / 2;                       // proportional: pairs span the whole head
                L.rope_rot = (int)(prf * L.head_dim / 2);           // only the first rope_rot frequencies rotate
            }
            L.kv_src = kv_src[i];
            if (L.kv_src >= 0 && hc.get_bool("use_double_wide_mlp", false)) L.ffn = 2 * c.ffn;
        }
        if (A == "qwen3_5") {
            if (lt == "linear_attention") {
                L.type = "deltanet";
                L.dn_nk = (int)hc.get_int("linear_num_key_heads", 16);
                L.dn_nv = (int)hc.get_int("linear_num_value_heads", 16);
                L.dn_dk = (int)hc.get_int("linear_key_head_dim", 128);
                L.dn_dv = (int)hc.get_int("linear_value_head_dim", 128);
                L.dn_conv = (int)hc.get_int("linear_conv_kernel_dim", 4);
            } else {
                L.qk_norm = 1;
                L.attn_gate = hc.get_bool("attn_output_gate", true);
                float prf = (float)rp.get_num("partial_rotary_factor", 1.0);
                int rot = (int)(c.head_dim * prf);
                L.rope_half = rot / 2;
                L.rope_rot = rot / 2;
            }
        }
        if (A == "olmoe") {
            L.ffn = 0;
            L.n_experts = (int)hc.get_int("num_experts", 64);
            L.top_k = (int)hc.get_int("num_experts_per_tok", 8);
            L.expert_ffn = c.ffn;
            L.router = "softmax";
            L.norm_topk = hc.get_bool("norm_topk_prob", false);
        }
        if (A == "lfm2_moe" && i >= (int)hc.get_int("num_dense_layers", 0)) {
            L.ffn = 0;
            L.n_experts = (int)hc.get_int("num_experts", 32);
            L.top_k = (int)hc.get_int("num_experts_per_tok", 4);
            L.expert_ffn = (int)hc.get_int("moe_intermediate_size", 1792);
            L.router = "sigmoid";
            L.norm_topk = hc.get_bool("norm_topk_prob", true);
            L.routed_scale = (float)hc.get_num("routed_scaling_factor", 1.0);
            L.expert_bias = hc.get_bool("use_expert_bias", true);
        }
        c.layers.push_back(L);
    }

    SafeTensors st;
    st.open_dir(o.in_dir);
    std::map<std::string, std::vector<float>> imat;
    if (!o.imatrix.empty()) { imat = IMatrix::load(o.imatrix); logi("using importance matrix %s (%zu entries)", o.imatrix.c_str(), imat.size()); }
    ModelWriter w;
    w.begin(o.out_path);
    Json gen_defaults = Json::object();
    Json tok = build_tokenizer(o.in_dir, w, c.vocab, gen_defaults);

    // checkpoint prefix: plain models use "model.", multimodal wrappers "model.language_model."
    std::string P = st.has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";
    auto need = [&](const std::string& n) -> const STTensor* {
        const STTensor* t = st.get(n);
        if (!t) die("missing tensor %s in checkpoint", n.c_str());
        return t;
    };
    auto first_of = [&](std::initializer_list<std::string> names) -> const STTensor* {
        for (auto& n : names) if (st.get(n)) return st.get(n);
        std::string all; for (auto& n : names) all += n + " ";
        die("missing tensor (tried: %s)", all.c_str());
    };
    size_t total_bytes = 0;
    // Write a matrix assembled from row sources
    auto write_matrix = [&](const std::string& name, QType qt, int64_t rows, int64_t cols,
                            const std::function<void(int64_t, float*)>& src_row) {
        if (qt != QType::F32) for (auto& ov : o.overrides) if (name.find(ov.first) != std::string::npos) qt = ov.second;
        if (qt == QType::Q3 && (name == "tok_embd" || name == "output")) qt = QType::Q4;  // no Q3 embedding kernels
        if ((qt == QType::Q4 || qt == QType::Q3) && cols % Q4_GROUP) qt = cols % Q8_GROUP ? QType::F16 : QType::Q8;
        std::vector<uint8_t> data, scales;
        const float* imp = nullptr;
        auto it = imat.find(name);
        std::vector<float> impv;
        if (it != imat.end() && (int64_t)it->second.size() == cols) {
            // regularize: importance = mean(x^2) + small fraction of its average
            double avg = 0;
            for (float v : it->second) avg += v;
            avg /= cols;
            impv.resize(cols);
            for (int64_t i = 0; i < cols; i++) impv[i] = it->second[i] + 0.01f * (float)avg;
            imp = impv.data();
        }
        quantize_matrix(qt, rows, cols, src_row, data, scales, o.threads, imp);
        TensorInfo ti;
        ti.name = name; ti.type = qt; ti.rows = rows; ti.cols = cols;
        ti.data_bytes = data.size();
        ti.offset = w.write_blob(data.data(), data.size());
        ti.scale_bytes = scales.size();
        ti.scale_offset = scales.empty() ? 0 : w.write_blob(scales.data(), scales.size());
        w.add_tensor(ti);
        total_bytes += data.size() + scales.size();
        logd("  %-24s %-4s [%lld x %lld]", name.c_str(), qtype_name(qt), (long long)rows, (long long)cols);
    };
    // 1-D float vectors; `add` is added to every element (Gemma/Qwen3.5 store norms as w-1)
    auto write_vec = [&](const std::string& name, const STTensor* t, float add = 0.f, float mul = 1.f) {
        int64_t n = t->numel();
        write_matrix(name, QType::F32, 1, n, [&, add, mul](int64_t, float* buf) {
            t->to_float(buf, 0, n);
            for (int64_t i = 0; i < n; i++) buf[i] = (buf[i] + add) * mul;
        });
    };
    auto write_const = [&](const std::string& name, int64_t n, float v) {
        write_matrix(name, QType::F32, 1, n, [&, v](int64_t, float* buf) { for (int64_t i = 0; i < n; i++) buf[i] = v; });
    };
    auto simple = [&](const STTensor* t) {
        int64_t cols = t->shape.back();
        return [t, cols](int64_t r, float* buf) { t->to_float(buf, r * cols, cols); };
    };
    const float norm_add = (A == "gemma3" || A == "qwen3_5") ? 1.f : 0.f;  // RMSNorm weight stored as (w - 1)

    logi("converting %s (%s) -> %s [%s]", c.name.c_str(), c.arch.c_str(), o.out_path.c_str(), qtype_name(o.wtype));
    const STTensor* emb = need(P + "embed_tokens.weight");
    int64_t V = emb->shape[0];
    c.vocab = (int)V;
    const int H = c.hidden;
    write_matrix("tok_embd", o.embed_type, V, H, simple(emb));
    write_vec("output_norm", first_of({P + "norm.weight", P + "embedding_norm.weight"}), norm_add);
    if (!c.tied_embed) {
        const STTensor* lm = st.get("lm_head.weight");
        if (lm) write_matrix("output", o.embed_type, lm->shape[0], H, simple(lm));
        else c.tied_embed = true;
    }
    for (int l = 0; l < c.n_layers; l++) {
        LayerCfg& L = c.layers[l];
        std::string p = P + "layers." + std::to_string(l) + ".";
        std::string o_ = "blk." + std::to_string(l) + ".";
        // ---- norms ----
        if (A == "lfm2" || A == "lfm2_moe") {
            write_vec(o_ + "attn_norm", need(p + "operator_norm.weight"));
            write_vec(o_ + "ffn_norm", need(p + "ffn_norm.weight"));
        } else if (gemma) {
            write_vec(o_ + "attn_norm", need(p + "input_layernorm.weight"), norm_add);
            write_vec(o_ + "post_attn_norm", need(p + "post_attention_layernorm.weight"), norm_add);
            write_vec(o_ + "ffn_norm", need(p + "pre_feedforward_layernorm.weight"), norm_add);
            write_vec(o_ + "post_ffn_norm", need(p + "post_feedforward_layernorm.weight"), norm_add);
            if (const STTensor* ls = st.get(p + "layer_scalar")) { float v; ls->to_float(&v, 0, 1); L.layer_scalar = v; }
        } else {
            write_vec(o_ + "attn_norm", need(p + "input_layernorm.weight"), norm_add);
            write_vec(o_ + "ffn_norm", need(p + "post_attention_layernorm.weight"), norm_add);
        }
        // ---- mixer ----
        if (L.type == "attn") {
            std::string ap = p + "self_attn.";
            const int qd = L.q_dim(), kvd = L.kv_dim(), hd = L.head_dim;
            const STTensor* wq = need(ap + "q_proj.weight");
            const STTensor* wk = L.kv_src >= 0 ? nullptr : need(ap + "k_proj.weight");
            const STTensor* wv = (L.kv_src >= 0 || L.k_eq_v) ? nullptr : need(ap + "v_proj.weight");
            const int qrows = qd * (L.attn_gate ? 2 : 1);
            // rows: [q (q and gate interleaved per head as in the checkpoint) ; k ; v]
            write_matrix(o_ + "wqkv", o.wtype, L.qkv_rows(), H, [&](int64_t r, float* buf) {
                if (r < qrows) wq->to_float(buf, r * H, H);
                else if (r < qrows + kvd) wk->to_float(buf, (r - qrows) * H, H);
                else wv->to_float(buf, (r - qrows - kvd) * H, H);
            });
            if (c.attn_bias) {
                const STTensor *bq = need(ap + "q_proj.bias"), *bk = need(ap + "k_proj.bias"), *bv = need(ap + "v_proj.bias");
                write_matrix(o_ + "bqkv", QType::F32, 1, qd + 2 * kvd, [&](int64_t, float* buf) {
                    bq->to_float(buf, 0, qd);
                    bk->to_float(buf + qd, 0, kvd);
                    bv->to_float(buf + qd + kvd, 0, kvd);
                });
            }
            if (L.qk_norm) {
                write_vec(o_ + "q_norm", first_of({ap + "q_norm.weight", ap + "q_layernorm.weight"}), norm_add);
                if (L.kv_src < 0) write_vec(o_ + "k_norm", first_of({ap + "k_norm.weight", ap + "k_layernorm.weight"}), norm_add);
            }
            (void)hd;
            write_matrix(o_ + "wo", o.wtype, H, qd, simple(first_of({ap + "o_proj.weight", ap + "out_proj.weight"})));
        } else if (L.type == "conv") {
            write_matrix(o_ + "conv_in", o.wtype, 3 * H, H, simple(need(p + "conv.in_proj.weight")));
            write_vec(o_ + "conv_w", need(p + "conv.conv.weight"));  // [H][1][K] -> [H][K]
            write_matrix(o_ + "conv_out", o.wtype, H, H, simple(need(p + "conv.out_proj.weight")));
        } else if (L.type == "deltanet") {
            std::string dp = p + "linear_attn.";
            const STTensor *wqkv = need(dp + "in_proj_qkv.weight"), *wz = need(dp + "in_proj_z.weight"),
                           *wb = need(dp + "in_proj_b.weight"), *wa = need(dp + "in_proj_a.weight");
            const int cd = L.dn_conv_dim(), vd = L.dn_val_dim(), nv = L.dn_nv;
            write_matrix(o_ + "dn_in", o.wtype, L.dn_in_rows(), H, [&](int64_t r, float* buf) {
                if (r < cd) wqkv->to_float(buf, r * H, H);
                else if (r < cd + vd) wz->to_float(buf, (r - cd) * H, H);
                else if (r < cd + vd + nv) wb->to_float(buf, (r - cd - vd) * H, H);
                else wa->to_float(buf, (r - cd - vd - nv) * H, H);
            });
            write_vec(o_ + "dn_conv", need(dp + "conv1d.weight"));
            // store A = -exp(A_log) directly
            const STTensor* alog = need(dp + "A_log");
            write_matrix(o_ + "dn_A", QType::F32, 1, nv, [&](int64_t, float* buf) {
                alog->to_float(buf, 0, nv);
                for (int i = 0; i < nv; i++) buf[i] = -expf(buf[i]);
            });
            write_vec(o_ + "dn_dt", need(dp + "dt_bias"));
            write_vec(o_ + "dn_norm", need(dp + "norm.weight"));
            write_matrix(o_ + "dn_out", o.wtype, H, vd, simple(need(dp + "out_proj.weight")));
        }
        // ---- feed forward ----
        if (L.n_experts) {
            std::string mp = (A == "lfm2_moe") ? p + "feed_forward." : p + "mlp.";
            const STTensor* router = first_of({mp + "gate.weight", mp + "router.weight"});
            write_matrix(o_ + "router", QType::F32, L.n_experts, H, simple(router));
            if (L.expert_bias) {
                const STTensor* eb = first_of({mp + "expert_bias", mp + "gate.expert_bias"});
                write_vec(o_ + "expert_bias", eb);
            }
            const int F = L.expert_ffn, E = L.n_experts;
            // experts: per-expert tensors (w1/w3/w2 or gate/up/down) or fused 3D tensors
            auto ex = [&](int e, const char* a, const char* b) -> const STTensor* {
                std::string base = mp + "experts." + std::to_string(e) + ".";
                if (st.get(base + a + ".weight")) return st.get(base + a + ".weight");
                return need(base + b + ".weight");
            };
            const STTensor* fused_gu = st.get(mp + "experts.gate_up_proj");
            const STTensor* fused_dn = st.get(mp + "experts.down_proj");
            write_matrix(o_ + "wgu_e", o.wtype, (int64_t)E * 2 * F, H, [&](int64_t r, float* buf) {
                int e = (int)(r / (2 * F)), rr = (int)(r % (2 * F));
                bool up = rr & 1; int row = rr >> 1;
                if (fused_gu) fused_gu->to_float(buf, ((int64_t)e * 2 * F + (up ? F + row : row)) * H, H);
                else ex(e, up ? "up_proj" : "gate_proj", up ? "w3" : "w1")->to_float(buf, (int64_t)row * H, H);
            });
            write_matrix(o_ + "wdown_e", o.wtype, (int64_t)E * H, F, [&](int64_t r, float* buf) {
                int e = (int)(r / H), row = (int)(r % H);
                if (fused_dn) fused_dn->to_float(buf, ((int64_t)e * H + row) * F, F);
                else ex(e, "down_proj", "w2")->to_float(buf, (int64_t)row * F, F);
            });
        } else if (L.ffn) {
            const STTensor *wg, *wu, *wd;
            if (A == "lfm2" || A == "lfm2_moe") {
                wg = need(p + "feed_forward.w1.weight"); wu = need(p + "feed_forward.w3.weight"); wd = need(p + "feed_forward.w2.weight");
            } else {
                wg = need(p + "mlp.gate_proj.weight"); wu = need(p + "mlp.up_proj.weight"); wd = need(p + "mlp.down_proj.weight");
            }
            L.ffn = (int)wg->shape[0];
            // interleaved: row 2i = gate_i, row 2i+1 = up_i
            write_matrix(o_ + "wgu", o.wtype, 2 * (int64_t)L.ffn, H, [&](int64_t r, float* buf) {
                (r & 1 ? wu : wg)->to_float(buf, (r >> 1) * H, H);
            });
            write_matrix(o_ + "wdown", o.wtype, H, L.ffn, simple(wd));
        }
        if (g_log_level >= 1) { fprintf(stderr, "\r[tachyon] layer %d/%d", l + 1, c.n_layers); fflush(stderr); }
    }
    if (g_log_level >= 1) fprintf(stderr, "\n");
    Json header = Json::object();
    header.set("format", "tachyon");
    header.set("version", 1);
    header.set("config", c.to_json());
    header.set("tokenizer", tok);
    header.set("generation", gen_defaults);
    w.finish(header);
    logi("done: %.2f GB weights in %.1fs", total_bytes / 1e9, now_sec() - t0);
    return 0;
}

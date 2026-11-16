#include "model_file.h"
#include "file_validation.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

Json LayerCfg::to_json() const {
    Json j = Json::object();
    j.set("type", type);
    if (type == "attn") {
        j.set("n_heads", n_heads); j.set("n_kv_heads", n_kv_heads); j.set("head_dim", head_dim);
        if(!rope_inv_freq.empty()){Json f=Json::array();for(float v:rope_inv_freq)f.push((double)v);j.set("rope_inv_freq",f);}
        j.set("rope_mscale",(double)rope_mscale);
        j.set("rope_theta", (double)rope_theta); j.set("rope_half", rope_half); j.set("rope_rot", rope_rot);
        if (rope_den) j.set("rope_den", rope_den);
        j.set("window", window); j.set("qk_norm", qk_norm); j.set("v_norm", v_norm); j.set("attn_gate", attn_gate);
        j.set("k_eq_v", k_eq_v); j.set("attn_scale", (double)attn_scale); j.set("kv_src", kv_src);
    } else if (type == "conv") {
        j.set("conv_k", conv_k);
    } else if (type == "deltanet") {
        j.set("dn_nk", dn_nk); j.set("dn_nv", dn_nv); j.set("dn_dk", dn_dk); j.set("dn_dv", dn_dv); j.set("dn_conv", dn_conv);
    }
    j.set("ffn", ffn);
    if (n_experts) {
        j.set("n_experts", n_experts); j.set("top_k", top_k); j.set("expert_ffn", expert_ffn); j.set("router", router);
        j.set("norm_topk", norm_topk); j.set("routed_scale", (double)routed_scale); j.set("expert_bias", expert_bias);
    }
    j.set("sandwich", sandwich);
    j.set("layer_scalar", (double)layer_scalar);
    return j;
}

LayerCfg LayerCfg::from_json(const Json& j) {
    LayerCfg l;
    l.type = j.get_str("type", "attn");
    l.n_heads = (int)j.get_int("n_heads", 0); l.n_kv_heads = (int)j.get_int("n_kv_heads", 0); l.head_dim = (int)j.get_int("head_dim", 0);
    l.rope_theta = (float)j.get_num("rope_theta", 10000); l.rope_half = (int)j.get_int("rope_half", l.head_dim / 2);
    l.rope_rot = (int)j.get_int("rope_rot", l.rope_half);
    for(auto& v:j["rope_inv_freq"].arr)l.rope_inv_freq.push_back(v.as_num());
    l.rope_mscale=j.get_num("rope_mscale",1.0);
    l.rope_den = (int)j.get_int("rope_den", 0);
    l.window = (int)j.get_int("window", 0); l.qk_norm = (int)j.get_int("qk_norm", 0); l.v_norm = j.get_bool("v_norm", false);
    l.attn_gate = j.get_bool("attn_gate", false); l.k_eq_v = j.get_bool("k_eq_v", false);
    l.attn_scale = (float)j.get_num("attn_scale", 0); l.kv_src = (int)j.get_int("kv_src", -1);
    l.conv_k = (int)j.get_int("conv_k", 0);
    l.dn_nk = (int)j.get_int("dn_nk", 0); l.dn_nv = (int)j.get_int("dn_nv", 0); l.dn_dk = (int)j.get_int("dn_dk", 0);
    l.dn_dv = (int)j.get_int("dn_dv", 0); l.dn_conv = (int)j.get_int("dn_conv", 0);
    l.ffn = (int)j.get_int("ffn", 0);
    l.n_experts = (int)j.get_int("n_experts", 0); l.top_k = (int)j.get_int("top_k", 0); l.expert_ffn = (int)j.get_int("expert_ffn", 0);
    l.router = j.get_str("router", "softmax"); l.norm_topk = j.get_bool("norm_topk", false);
    l.routed_scale = (float)j.get_num("routed_scale", 1); l.expert_bias = j.get_bool("expert_bias", false);
    l.sandwich = j.get_bool("sandwich", false);
    l.layer_scalar = (float)j.get_num("layer_scalar", 1);
    return l;
}

Json ModelConfig::to_json() const {
    Json j = Json::object();
    j.set("norm_groups",norm_groups);
    j.set("arch", arch);
    j.set("name", name);
    j.set("n_layers", n_layers);
    j.set("hidden", hidden);
    j.set("n_heads", n_heads);
    j.set("n_kv_heads", n_kv_heads);
    j.set("head_dim", head_dim);
    j.set("ffn", ffn);
    j.set("vocab", vocab);
    j.set("rms_eps", (double)rms_eps);
    j.set("rope_theta", (double)rope_theta);
    j.set("qk_norm", qk_norm);
    j.set("attn_bias", attn_bias);
    j.set("tied_embed", tied_embed);
    j.set("rope_scaling", rope_scaling);
    j.set("rope_factor", (double)rope_factor);
    j.set("rope_low_freq", (double)rope_low_freq);
    j.set("rope_high_freq", (double)rope_high_freq);
    j.set("rope_orig_ctx", rope_orig_ctx);
    if (!rope_short.empty()) {
        Json a = Json::array(), b = Json::array();
        for (float v : rope_short) a.push((double)v);
        for (float v : rope_long) b.push((double)v);
        j.set("rope_short", a); j.set("rope_long", b); j.set("rope_mscale", (double)rope_mscale);
    }
    j.set("max_ctx", max_ctx);
    j.set("wtype", qtype_name(wtype));
    j.set("act", act);
    j.set("embed_scale", (double)embed_scale);
    j.set("final_softcap", (double)final_softcap);
    j.set("stateful", stateful);
    j.set("ple_dim", ple_dim);
    Json ls = Json::array();
    for (auto& l : layers) ls.push(l.to_json());
    j.set("layers", ls);
    return j;
}

ModelConfig ModelConfig::from_json(const Json& j) {
    ModelConfig c;
    c.norm_groups=j.get_int("norm_groups",1);
    c.arch = j.get_str("arch", "llama");
    c.name = j.get_str("name", "model");
    c.n_layers = (int)j.get_int("n_layers", 0);
    c.hidden = (int)j.get_int("hidden", 0);
    c.n_heads = (int)j.get_int("n_heads", 0);
    c.n_kv_heads = (int)j.get_int("n_kv_heads", 0);
    c.head_dim = (int)j.get_int("head_dim", 0);
    c.ffn = (int)j.get_int("ffn", 0);
    c.vocab = (int)j.get_int("vocab", 0);
    c.rms_eps = (float)j.get_num("rms_eps", 1e-6);
    c.rope_theta = (float)j.get_num("rope_theta", 10000);
    c.qk_norm = j.get_bool("qk_norm", false);
    c.attn_bias = j.get_bool("attn_bias", false);
    c.tied_embed = j.get_bool("tied_embed", false);
    c.rope_scaling = j.get_str("rope_scaling", "");
    c.rope_factor = (float)j.get_num("rope_factor", 1);
    c.rope_low_freq = (float)j.get_num("rope_low_freq", 1);
    c.rope_high_freq = (float)j.get_num("rope_high_freq", 4);
    c.rope_orig_ctx = (int)j.get_int("rope_orig_ctx", 8192);
    for (auto& v : j["rope_short"].arr) c.rope_short.push_back((float)v.as_num());
    for (auto& v : j["rope_long"].arr) c.rope_long.push_back((float)v.as_num());
    c.rope_mscale = (float)j.get_num("rope_mscale", 1.0);
    c.max_ctx = (int)j.get_int("max_ctx", 4096);
    c.wtype = qtype_from_name(j.get_str("wtype", "q4"));
    c.act = j.get_str("act", "silu");
    c.embed_scale = (float)j.get_num("embed_scale", 1);
    c.final_softcap = (float)j.get_num("final_softcap", 0);
    c.stateful = j.get_bool("stateful", false);
    c.ple_dim = (int)j.get_int("ple_dim", 0);
    if(c.norm_groups<1||c.norm_groups>32||c.hidden%c.norm_groups)die("invalid grouped RMSNorm geometry");
    if (j["layers"].is_arr()) {
        for (auto& l : j["layers"].arr) c.layers.push_back(LayerCfg::from_json(l));
    } else {
        // legacy file: uniform Llama/Qwen3 attention layers
        for (int i = 0; i < c.n_layers; i++) {
            LayerCfg l;
            l.n_heads = c.n_heads; l.n_kv_heads = c.n_kv_heads; l.head_dim = c.head_dim;
            l.rope_theta = c.rope_theta; l.rope_half = c.head_dim / 2; l.rope_rot = l.rope_half;
            l.qk_norm = c.qk_norm ? 1 : 0;
            l.ffn = c.ffn;
            c.layers.push_back(l);
        }
    }
    return c;
}

ModelFile::~ModelFile() {
    if (map_) munmap(map_, file_size);
}

void ModelFile::open(const std::string& p) {
    if (map_) { munmap(map_, file_size); map_ = nullptr; }
    tensors.clear(); blobs.clear();
    path = p;
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open model %s", p.c_str());
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) die("%s is truncated or incomplete", p.c_str());
    file_size = st.st_size;
    map_ = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) die("mmap failed for %s", p.c_str());
    const uint8_t* base = (const uint8_t*)map_;
    std::string err;
    size_t data_start, data_end = file_size;
    if (file_size >= 24 && memcmp(base, "TACHYON2", 8) == 0) {
        // v2: [magic][u64 header_off][u64 header_len] ... data at DATA_ALIGN ... [header JSON]
        uint64_t hoff, hlen;
        memcpy(&hoff, base + 8, 8);
        memcpy(&hlen, base + 16, 8);
        if (hoff < DATA_ALIGN || hoff > file_size || hlen > file_size - hoff) die("%s is truncated or incomplete", p.c_str());
        header_ = Json::parse(std::string((const char*)base + hoff, hlen), &err);
        data_start = DATA_ALIGN;
        data_end = hoff;
    } else if (file_size >= 16 && memcmp(base, "TACHYON1", 8) == 0) {
        uint64_t hlen;
        memcpy(&hlen, base + 8, 8);
        if (hlen > file_size - 16) die("%s is truncated or incomplete", p.c_str());
        header_ = Json::parse(std::string((const char*)base + 16, hlen), &err);
        data_start = align_up(16 + hlen, DATA_ALIGN);
    } else die("%s is not a tachyon model file", p.c_str());
    if (!err.empty() || !header_.is_obj()) die("bad header: %s", err.c_str());
    if (data_start > data_end) die("%s is truncated or incomplete", p.c_str());
    const size_t payload = data_end - data_start;
    if (header_["config"].get_int("n_layers", 0) > (int64_t)header_["tensors"].size())
        die("invalid model layer count");
    config = ModelConfig::from_json(header_["config"]);
    for (auto& t : header_["tensors"].arr) {
        TensorInfo ti;
        ti.name = t.get_str("name", "");
        ti.hadamard_block = (int)t.get_int("hadamard_block", 0);
        ti.hadamard_inverse = t.get_bool("hadamard_inverse", false);
        ti.hadamard_signs = t.get_str("hadamard_signs", "");
        ti.type = qtype_from_name(t.get_str("type", "f32"));
        ti.rows = file_size_value(t["rows"], "tensor rows");
        ti.cols = file_size_value(t["cols"], "tensor cols");
        ti.offset = file_size_value(t["offset"], "tensor offset");
        ti.data_bytes = file_size_value(t["data_bytes"], "tensor data_bytes");
        ti.scale_offset = t.has("scale_offset") ? file_size_value(t["scale_offset"], "tensor scale_offset") : 0;
        ti.scale_bytes = t.has("scale_bytes") ? file_size_value(t["scale_bytes"], "tensor scale_bytes") : 0;
        if (ti.rows <= 0 || ti.cols <= 0 || ti.rows > INT32_MAX || ti.cols > INT32_MAX)
            die("invalid dimensions for tensor %s", ti.name.c_str());
        size_t n = file_product(ti.rows, ti.cols, "tensor");
        file_product(n, 4, "tensor");  // bound all weight format size calculations
        int group = ti.type == QType::Q8 ? Q8_GROUP : ti.type == QType::Q4 || ti.type == QType::Q3 ? Q4_GROUP : 1;
        if (qtype_packed(ti.type)) group = packed_group((int)ti.type);
        if (ti.hadamard_block < 0 || (ti.hadamard_block && ((ti.hadamard_block & (ti.hadamard_block - 1)) || ti.cols % ti.hadamard_block)))
            die("invalid Hadamard geometry for %s", ti.name.c_str());
        if (ti.cols % group || ti.data_bytes != qtype_data_bytes(ti.type, ti.rows, ti.cols) ||
            ti.scale_bytes != qtype_scale_bytes(ti.type, ti.rows, ti.cols))
            die("tensor %s byte size does not match shape/format", ti.name.c_str());
        if (!file_range(ti.offset, ti.data_bytes, payload) ||
            (ti.scale_bytes && !file_range(ti.scale_offset, ti.scale_bytes, payload)))
            die("tensor %s out of range", ti.name.c_str());
        ti.data = base + data_start + ti.offset;
        ti.scales = ti.scale_bytes ? base + data_start + ti.scale_offset : nullptr;
        if (!tensors.emplace(ti.name, ti).second) die("duplicate tensor %s", ti.name.c_str());
    }
    for (auto& b : header_["blobs"].arr) {
        BlobInfo bi;
        bi.name = b.get_str("name", "");
        bi.offset = file_size_value(b["offset"], "blob offset");
        bi.size = file_size_value(b["size"], "blob size");
        if (!file_range(bi.offset, bi.size, payload)) die("blob %s out of range", bi.name.c_str());
        bi.data = base + data_start + bi.offset;
        if (!blobs.emplace(bi.name, bi).second) die("duplicate blob %s", bi.name.c_str());
    }
    for (auto& [name, t] : tensors) if (!t.hadamard_signs.empty()) {
        const auto* b = blob(t.hadamard_signs);
        if (!t.hadamard_block || b->size != (size_t)t.cols * 4) die("invalid Hadamard sign blob");
        for (int i = 0; i < t.cols; i++) { float v; memcpy(&v, b->data + 4 * i, 4); if (v != 1.f && v != -1.f) die("invalid Hadamard sign value"); }
    }

}

void ModelFile::validate_inference() const {
    const auto& c = config;
    if (c.hidden <= 0 || c.vocab <= 0 || c.n_layers <= 0 || c.max_ctx <= 0 ||
        c.layers.size() != (size_t)c.n_layers || c.ffn < 0 || c.ple_dim < 0 ||
        !std::isfinite(c.rms_eps) || c.rms_eps <= 0)
        die("invalid model dimensions or layer count");
    auto shape = [&](const std::string& name, int64_t rows, int64_t cols, bool vec = false, bool required = true) {
        const auto* t = tensor(name, required);
        if (t && (t->rows != rows || t->cols != cols || (vec && t->type != QType::F32)))
            die("tensor %s shape/format does not match model config", name.c_str());
    };
    auto width = [&](int64_t n) {
        if (n <= 0 || n > INT32_MAX) die("invalid model projection dimensions");
        return n;
    };
    const int H = c.hidden;
    shape("tok_embd", c.vocab, H);
    if (!c.tied_embed) shape("output", c.vocab, H);
    shape("output_norm", 1, H, true);
    if (c.ple_dim) {
        int64_t n = width((int64_t)c.n_layers * c.ple_dim);
        shape("ple_embd", c.vocab, n); shape("ple_proj", n, H);
        shape("ple_proj_norm", 1, c.ple_dim, true);
    }
    for (int i = 0; i < c.n_layers; i++) {
        const auto& l = c.layers[i];
        std::string p = "blk." + std::to_string(i) + ".";
        if (l.ffn < 0 || l.n_experts < 0 || l.window < 0) die("invalid model layer dimensions");
        shape(p + "attn_norm", 1, H, true); shape(p + "ffn_norm", 1, H, true);
        shape(p + "post_attn_norm", 1, H, true, l.sandwich);
        shape(p + "post_ffn_norm", 1, H, true, l.sandwich);
        if (l.is_attn()) {
            if (l.n_heads <= 0 || l.n_kv_heads <= 0 || l.head_dim <= 0 || l.n_heads % l.n_kv_heads ||
                l.rope_half < 0 || l.rope_half > l.head_dim / 2 || l.rope_rot < 0 || l.rope_rot > l.rope_half ||
                l.qk_norm < 0 || l.qk_norm > 2 || !std::isfinite(l.rope_theta) || l.rope_theta <= 0 ||
                l.kv_src < -1 || l.kv_src >= i)
                die("invalid attention config for layer %d", i);
            int64_t qd = width((int64_t)l.n_heads * l.head_dim), kd = width((int64_t)l.n_kv_heads * l.head_dim);
            if (l.kv_src >= 0) {
                const auto& s = c.layers[l.kv_src];
                if (!s.is_attn() || s.n_kv_heads != l.n_kv_heads || s.head_dim != l.head_dim || s.window != l.window)
                    die("invalid shared KV config for layer %d", i);
            }
            shape(p + "wqkv", width(qd * (l.attn_gate ? 2 : 1) + (l.kv_src >= 0 ? 0 : kd * (l.k_eq_v ? 1 : 2))), H);
            shape(p + "wo", H, qd);
            shape(p + "bqkv", 1, qd + 2 * kd, true, c.attn_bias);
            shape(p + "q_norm", 1, l.qk_norm == 2 ? qd : l.head_dim, true, l.qk_norm != 0);
            shape(p + "k_norm", 1, l.qk_norm == 2 ? kd : l.head_dim, true, l.qk_norm != 0 && l.kv_src < 0);
        } else if (l.type == "conv") {
            if (l.conv_k < 2 || !c.stateful) die("invalid convolution config for layer %d", i);
            shape(p + "conv_in", width((int64_t)3 * H), H); shape(p + "conv_out", H, H);
            shape(p + "conv_w", 1, (int64_t)H * l.conv_k, true);
        } else if (l.type == "deltanet") {
            if (l.dn_nk <= 0 || l.dn_nv <= 0 || l.dn_dk <= 0 || l.dn_dv <= 0 || l.dn_conv < 2 ||
                l.dn_nv % l.dn_nk || !c.stateful) die("invalid delta net config for layer %d", i);
            int64_t kd = width((int64_t)l.dn_nk * l.dn_dk), vd = width((int64_t)l.dn_nv * l.dn_dv);
            int64_t cd = width(2 * kd + vd);
            shape(p + "dn_in", width(cd + vd + 2ll * l.dn_nv), H); shape(p + "dn_out", H, vd);
            shape(p + "dn_conv", 1, cd * l.dn_conv, true);
            shape(p + "dn_A", 1, l.dn_nv, true); shape(p + "dn_dt", 1, l.dn_nv, true);
            shape(p + "dn_norm", 1, l.dn_dv, true);
        } else die("unknown layer type %s", l.type.c_str());
        if (l.n_experts) {
            if (l.top_k <= 0 || l.top_k > l.n_experts || l.expert_ffn <= 0) die("invalid expert config for layer %d", i);
            shape(p + "router", l.n_experts, H, true);
            shape(p + "expert_bias", 1, l.n_experts, true, l.expert_bias);
            shape(p + "wgu_e", width((int64_t)l.n_experts * 2 * l.expert_ffn), H);
            shape(p + "wdown_e", width((int64_t)l.n_experts * H), l.expert_ffn);
        } else if (l.ffn) {
            shape(p + "wgu", width(2ll * l.ffn), H); shape(p + "wdown", H, l.ffn);
        }
        if (c.ple_dim) {
            shape(p + "ple_gate", c.ple_dim, H); shape(p + "ple_out", H, c.ple_dim);
            shape(p + "ple_post_norm", 1, H, true);
        }
    }
}

const TensorInfo* ModelFile::tensor(const std::string& name, bool required) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        if (required) die("missing tensor %s", name.c_str());
        return nullptr;
    }
    return &it->second;
}
const BlobInfo* ModelFile::blob(const std::string& name, bool required) const {
    auto it = blobs.find(name);
    if (it == blobs.end()) {
        if (required) die("missing blob %s", name.c_str());
        return nullptr;
    }
    return &it->second;
}

void ModelWriter::begin(const std::string& path) {
    path_ = path;
    tmp_path_ = path + ".partial";
    f_ = fopen(tmp_path_.c_str(), "wb");
    if (!f_) die("cannot write %s", tmp_path_.c_str());
    // placeholder preamble; header offset/len are patched in finish()
    std::vector<uint8_t> pre(DATA_ALIGN, 0);
    memcpy(pre.data(), "TACHYON2", 8);
    if (fwrite(pre.data(), 1, pre.size(), f_) != pre.size()) die("write failed for %s", tmp_path_.c_str());
    pos_ = 0;
}

uint64_t ModelWriter::write_blob(const void* data, size_t n) {
    static const uint8_t zeros[256] = {};
    uint64_t pad = align_up(pos_, 256) - pos_;
    if (pad && fwrite(zeros, 1, pad, f_) != pad) die("write failed (disk full?)");
    pos_ += pad;
    uint64_t off = pos_;
    if (n && fwrite(data, 1, n, f_) != n) die("write failed (disk full?)");
    pos_ += n;
    return off;
}

void ModelWriter::finish(Json header) {
    Json ts = Json::array();
    for (auto& t : tensors_) {
        Json j = Json::object();
        j.set("name", t.name);
        j.set("type", qtype_name(t.type));
        if (t.hadamard_block) { j.set("hadamard_block", t.hadamard_block); j.set("hadamard_inverse", t.hadamard_inverse); j.set("hadamard_signs", t.hadamard_signs); }
        j.set("rows", (int64_t)t.rows);
        j.set("cols", (int64_t)t.cols);
        j.set("offset", (uint64_t)t.offset);
        j.set("data_bytes", (uint64_t)t.data_bytes);
        j.set("scale_offset", (uint64_t)t.scale_offset);
        j.set("scale_bytes", (uint64_t)t.scale_bytes);
        ts.push(j);
    }
    Json bs = Json::array();
    for (auto& b : blobs_) {
        Json j = Json::object();
        j.set("name", b.name);
        j.set("offset", (uint64_t)b.offset);
        j.set("size", (uint64_t)b.size);
        bs.push(j);
    }
    header.set("tensors", ts);
    header.set("blobs", bs);
    std::string h = header.dump();
    uint64_t hoff = DATA_ALIGN + pos_, hlen = h.size();
    if (fwrite(h.data(), 1, h.size(), f_) != h.size()) die("write failed (disk full?)");
    if (fseek(f_, 8, SEEK_SET) != 0) die("seek failed");
    if (fwrite(&hoff, 8, 1, f_) != 1 || fwrite(&hlen, 8, 1, f_) != 1) die("write failed");
    if (fflush(f_) != 0 || ferror(f_)) die("write failed for %s", tmp_path_.c_str());
    if (fclose(f_) != 0) die("close failed for %s", tmp_path_.c_str());
    f_ = nullptr;
    if (rename(tmp_path_.c_str(), path_.c_str()) != 0) die("rename %s -> %s failed", tmp_path_.c_str(), path_.c_str());
}

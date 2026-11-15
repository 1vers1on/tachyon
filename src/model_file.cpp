#include "model_file.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

Json LayerCfg::to_json() const {
    Json j = Json::object();
    j.set("type", type);
    if (type == "attn") {
        j.set("n_heads", n_heads); j.set("n_kv_heads", n_kv_heads); j.set("head_dim", head_dim);
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
    path = p;
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open model %s", p.c_str());
    struct stat st;
    fstat(fd, &st);
    file_size = st.st_size;
    map_ = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) die("mmap failed for %s", p.c_str());
    const uint8_t* base = (const uint8_t*)map_;
    std::string err;
    size_t data_start;
    if (file_size >= 24 && memcmp(base, "TACHYON2", 8) == 0) {
        // v2: [magic][u64 header_off][u64 header_len] ... data at DATA_ALIGN ... [header JSON]
        uint64_t hoff, hlen;
        memcpy(&hoff, base + 8, 8);
        memcpy(&hlen, base + 16, 8);
        if (hoff + hlen > file_size || hoff == 0) die("%s is truncated or incomplete", p.c_str());
        header_ = Json::parse(std::string((const char*)base + hoff, hlen), &err);
        data_start = DATA_ALIGN;
    } else if (file_size >= 16 && memcmp(base, "TACHYON1", 8) == 0) {
        uint64_t hlen;
        memcpy(&hlen, base + 8, 8);
        header_ = Json::parse(std::string((const char*)base + 16, hlen), &err);
        data_start = align_up(16 + hlen, DATA_ALIGN);
    } else die("%s is not a tachyon model file", p.c_str());
    if (!err.empty()) die("bad header: %s", err.c_str());
    config = ModelConfig::from_json(header_["config"]);
    for (auto& t : header_["tensors"].arr) {
        TensorInfo ti;
        ti.name = t.get_str("name", "");
        ti.type = qtype_from_name(t.get_str("type", "f32"));
        ti.rows = t.get_int("rows", 1);
        ti.cols = t.get_int("cols", 1);
        ti.offset = t.get_int("offset", 0);
        ti.data_bytes = t.get_int("data_bytes", 0);
        ti.scale_offset = t.get_int("scale_offset", 0);
        ti.scale_bytes = t.get_int("scale_bytes", 0);
        ti.data = base + data_start + ti.offset;
        ti.scales = ti.scale_bytes ? base + data_start + ti.scale_offset : nullptr;
        if (data_start + ti.offset + ti.data_bytes > file_size) die("tensor %s out of range", ti.name.c_str());
        tensors[ti.name] = ti;
    }
    for (auto& b : header_["blobs"].arr) {
        BlobInfo bi;
        bi.name = b.get_str("name", "");
        bi.offset = b.get_int("offset", 0);
        bi.size = b.get_int("size", 0);
        bi.data = base + data_start + bi.offset;
        blobs[bi.name] = bi;
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

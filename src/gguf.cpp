#include "gguf.h"
#include "model_file.h"
#include "quant.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <set>
#include <numeric>

namespace {
struct Gguf {
    struct Tensor { std::string name; std::vector<uint64_t> shape; uint32_t type; uint64_t offset; QType qt; };
    int fd = -1; const uint8_t* data = nullptr; size_t size = 0, pos = 0, start = 0;
    Json meta = Json::object(); std::map<std::string, Tensor> tensors;
    ~Gguf() { if (data) munmap((void*)data, size); if (fd >= 0) close(fd); }
    void require(size_t n) { if (pos > size || n > size - pos) die("truncated GGUF at byte %zu", pos); }
    template<class T> T get() { require(sizeof(T)); T v; memcpy(&v, data + pos, sizeof v); pos += sizeof v; return v; }
    std::string string() { auto n = get<uint64_t>(); require(n); std::string s((const char*)data + pos, n); pos += n; return s; }
    Json value(int type, int depth = 0) {
        if (depth > 4) die("GGUF metadata nesting too deep");
        switch (type) {
            case 0: return Json((int)get<uint8_t>()); case 1: return Json((int)get<int8_t>());
            case 2: return Json((int)get<uint16_t>()); case 3: return Json((int)get<int16_t>());
            case 4: return Json((uint64_t)get<uint32_t>()); case 5: return Json((int64_t)get<int32_t>());
            case 6: return Json((double)get<float>()); case 7: return Json(get<uint8_t>() != 0);
            case 8: return Json(string()); case 10: return Json(get<uint64_t>()); case 11: return Json(get<int64_t>());
            case 12: return Json(get<double>());
            case 9: { auto sub = get<uint32_t>(); auto n = get<uint64_t>(); if (n > size - pos) die("invalid GGUF array length");
                Json a = Json::array(); a.arr.reserve(n); while (n--) a.push(value(sub, depth + 1)); return a; }
            default: die("unknown GGUF metadata type %d", type);
        }
    }
    static QType type(uint32_t t) {
        switch (t) { case 0: return QType::F32; case 1: return QType::F16; case 2: return QType::Q4_0; case 8: return QType::Q8_0;
        case 12: return QType::Q4_K; case 13: return QType::Q5_K; case 14: return QType::Q6_K;
        case 30: return QType::BF16; case 142: return QType::PQ2_0; case 143: return QType::PTQ1_0; default: die("unsupported GGUF tensor type %u (no implicit requantization)", t); }
    }
    void open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY); struct stat st;
        if (fd < 0 || fstat(fd, &st) || st.st_size < 24) die("cannot read GGUF %s", path.c_str());
        size = st.st_size; data = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; die("cannot mmap GGUF"); }
        if (get<uint32_t>() != 0x46554747 || get<uint32_t>() != 3) die("expected little-endian GGUF v3");
        auto nt = get<uint64_t>(), nk = get<uint64_t>();
        if (nt > 1000000 || nk > 1000000) die("invalid GGUF header counts");
        while (nk--) { auto key = string(); auto t = get<uint32_t>(); if (meta.has(key)) die("duplicate GGUF metadata %s", key.c_str()); meta.set(key, value(t)); }
        while (nt--) { Tensor t; t.name = string(); auto nd = get<uint32_t>(); if (nd < 1 || nd > 4) die("invalid GGUF tensor rank");
            while (nd--) { auto d = get<uint64_t>(); if (!d || d > INT32_MAX) die("invalid GGUF tensor dimension"); t.shape.push_back(d); }
            t.type = get<uint32_t>(); t.offset = get<uint64_t>();
            // Ignore optional vision/draft tensors only at the graph import stage, never reinterpret unknown types.
            if (!tensors.emplace(t.name, t).second) die("duplicate GGUF tensor"); }
        uint64_t align = meta.get_int("general.alignment", 32);
        if (!align || align > 65536 || (align & (align - 1))) die("invalid GGUF alignment");
        start = align_up(pos, align);
        if (start > size) die("truncated GGUF tensor directory");
        for (auto& [name,t] : tensors) if (t.offset > size - start) die("GGUF tensor offset out of range");
    }
    const Tensor& tensor(const std::string& n) const { auto it = tensors.find(n); if (it == tensors.end()) die("missing GGUF tensor %s", n.c_str()); return it->second; }
    const uint8_t* bytes(const Tensor& t) const {
        auto qt = type(t.type); size_t rows = 1; for (size_t i = 1; i < t.shape.size(); i++) {
            if (rows > SIZE_MAX / t.shape[i]) die("GGUF tensor size overflow"); rows *= t.shape[i]; }
        int group = qtype_packed(qt) ? packed_group((int)qt) : 1;
        if (t.shape[0] % group || rows > SIZE_MAX / t.shape[0] / 4) die("invalid GGUF block shape");
        size_t n = qtype_data_bytes(qt, rows, t.shape[0]);
        if (n > size - start - t.offset) die("truncated GGUF tensor %s", t.name.c_str());
        return data + start + t.offset;
    }
};

// GGUF byte-level BPE vocabulary uses GPT-2's reversible Unicode byte alphabet.
std::string unbyte(const std::string& s) {
    static const auto inverse = [] { std::map<int,int> m; int next=256; for(int b=0;b<256;b++) { bool visible=(b>=33&&b<=126)||(b>=161&&b<=172)||b>=174; m[visible?b:next++]=b; } return m; }();
    std::string out;
    for (size_t i = 0; i < s.size();) {
        unsigned c = (uint8_t)s[i++], cp = c; int n = c < 128 ? 0 : c < 224 ? 1 : c < 240 ? 2 : 3;
        if (i + n > s.size()) return s;
        if(n) cp &= (1u << (6 - n)) - 1;
        while (n--) cp = (cp << 6) | ((uint8_t)s[i++] & 63);
        auto it = inverse.find(cp); if (it == inverse.end()) return s; out += (char)it->second;
    }
    return out;
}
Json tokenizer(Gguf& g, ModelWriter& w) {
    const Json& words = g.meta["tokenizer.ggml.tokens"];
    if (!words.is_arr() || words.arr.empty() || words.size() > INT32_MAX) die("GGUF has no tokenizer vocabulary");
    bool spm = g.meta.get_str("tokenizer.ggml.model", "") == "gemma4" || g.meta.get_str("tokenizer.ggml.model", "") == "llama" || g.meta.get_str("tokenizer.ggml.pre", "").find("gemma") != std::string::npos;
    std::vector<uint32_t> offsets; std::vector<uint8_t> types; std::string all;
    std::unordered_map<std::string,int> ids; Json special = Json::array();
    for (int i = 0; i < (int)words.size(); i++) {
        auto s = words[i].str; ids[s] = i; offsets.push_back(all.size());
        int type = g.meta["tokenizer.ggml.token_type"][i].as_int();
        int ours = type == 3 || type == 4 ? 1 : type == 5 ? 2 : 0;
        all += spm || ours ? s : unbyte(s); types.push_back(ours);
        if (ours == 1) { Json j = Json::object(); j.set("content", s); j.set("id", i); j.set("special", type == 3); special.push(j); }
    }
    offsets.push_back(all.size()); std::vector<int32_t> merges;
    for (const auto& entry : g.meta["tokenizer.ggml.merges"].arr) {
        auto split = entry.str.find(' ', 1); if (split == std::string::npos) die("invalid GGUF BPE merge");
        auto a = entry.str.substr(0, split), b = entry.str.substr(split + 1);
        if (!ids.count(a) || !ids.count(b) || !ids.count(a + b)) continue;
        merges.insert(merges.end(), {ids[a], ids[b], ids[a+b]});
    }
    auto blob = [&](const std::string& n, const void* p, size_t size) { BlobInfo b; b.name=n; b.size=size; b.offset=w.write_blob(p,size); w.add_blob(b); };
    blob("tok.vocab_data", all.data(), all.size()); blob("tok.vocab_off", offsets.data(), offsets.size()*4);
    blob("tok.type", types.data(), types.size()); blob("tok.merges", merges.data(), merges.size()*4);
    Json t=Json::object(); t.set("type", spm?"bpe_spm":"bpe_bytelevel"); t.set("n_vocab", (int)words.size());
    t.set("pretokenizer", spm?"gemma":"qwen2"); t.set("specials", special);
    t.set("bos", g.meta.get_int("tokenizer.ggml.bos_token_id", -1)); t.set("add_bos", g.meta.get_bool("tokenizer.ggml.add_bos_token", spm));
    Json eos=Json::array(); for (auto key:{"tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id","tokenizer.ggml.eom_token_id"}) if(g.meta.has(key)) eos.push(g.meta[key]);
    for (const auto& name:{"<|im_end|>","<|endoftext|>","<turn|>"}) if (ids.count(name)) eos.push(ids[name]);
    t.set("eos",eos); auto ct=g.meta.get_str("tokenizer.chat_template", ""); t.set("chat_template", ct);
    t.set("template_engine","jinja");
    t.set("chat_format",spm?"gemma4":"chatml"); t.set("thinking",true); t.set("think_default",false); t.set("think_open",!spm);
    t.set("tool_style","xml");
    if(spm) {t.set("think_tag_open","<|channel>thought\n");t.set("think_tag_close","<channel|>");t.set("think_disable_block",false);}
    return t;
}
} // namespace

int convert_gguf(const std::string& input, const std::string& output) {
    Gguf g; g.open(input); auto& meta=g.meta;
    const auto arch=meta.get_str("general.architecture", "");
    if(arch!="qwen35" && arch!="gemma4") die("GGUF architecture %s is not supported",arch.c_str());
    const auto prefix=arch+".";
    auto integer=[&](const std::string& key,int fallback=0){return (int)meta.get_int(prefix+key,fallback);};
    auto number=[&](const std::string& key,float fallback){return (float)meta.get_num(prefix+key,fallback);};
    auto perlayer=[&](const std::string& key,int i,int fallback){const auto& v=meta[prefix+key];return v.is_arr()?(int)v[i].as_int():(int)meta.get_int(prefix+key,fallback);};
    ModelConfig c; c.arch=arch=="qwen35"?"qwen3_5":"gemma4"; c.name=meta.get_str("general.name","GGUF");
    c.hidden=integer("embedding_length");c.n_layers=integer("block_count");c.n_heads=integer("attention.head_count");
    c.n_kv_heads=perlayer("attention.head_count_kv",0,c.n_heads);c.head_dim=integer("attention.key_length",c.hidden/std::max(c.n_heads,1));
    c.ffn=integer("feed_forward_length");c.rms_eps=number("attention.layer_norm_rms_epsilon",1e-6f);
    c.max_ctx=integer("context_length",8192);c.rope_theta=number("rope.freq_base",10000.f);
    c.vocab=meta["tokenizer.ggml.tokens"].size();c.tied_embed=!g.tensors.count("output.weight");
    if(c.hidden<=0 || c.n_layers<=0 || c.n_layers>512 || c.n_heads<=0 || c.vocab<=0)die("invalid GGUF model dimensions");
    c.wtype=Gguf::type(g.tensor("token_embd.weight").type);
    c.stateful=arch=="qwen35";
    if(arch=="gemma4") {c.act="gelu";float sc=sqrtf((float)c.hidden);uint32_t bits;memcpy(&bits,&sc,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;memcpy(&c.embed_scale,&bits,4);c.final_softcap=number("final_logit_softcapping",0.f);c.ple_dim=integer("embedding_length_per_layer_input");if(c.ple_dim)die("GGUF per-layer input embeddings require an explicit mapping");}
    int hb=(int)meta.get_int("prism.hadamard.block_size",0); auto mode=meta.get_str("prism.hadamard.sign_mode","identity");
    std::set<std::string> folded,inverse,used;
    for(auto& n:meta["prism.hadamard.weight_names"].arr)folded.insert(n.str);
    for(auto& n:meta["prism.hadamard.inverse_weight_names"].arr)inverse.insert(n.str);
    if(hb && (meta.get_int("prism.hadamard.version",0)!=1 || hb>16384 || hb<1 || (hb&(hb-1)) ||
       meta.get_str("prism.hadamard.transform","")!="normalized-sylvester-walsh-hadamard" ||
       meta.get_str("prism.hadamard.axis","")!="input-last-dimension" || (mode!="explicit"&&mode!="identity")))die("unsupported Prism Hadamard metadata");
    if((!folded.empty()||!inverse.empty())&&!hb)die("missing Prism Hadamard block size");
    ModelWriter w;w.begin(output);auto tok=tokenizer(g,w);
    std::map<int,std::string> sign_names; size_t sign_offset=0;
    for(auto& width:meta["prism.hadamard.sign_widths"].arr) {
        int n=width.as_int();if(n<=0||n%hb||sign_offset+n>meta["prism.hadamard.sign_values"].size())die("invalid Prism sign dimensions");
        std::vector<float> values(n);for(int i=0;i<n;i++){values[i]=meta["prism.hadamard.sign_values"][sign_offset+i].as_num();if(values[i]!=1&&values[i]!=-1)die("invalid Prism sign");}
        sign_offset+=n;BlobInfo b;b.name="hadamard.signs."+std::to_string(n);b.size=n*4;b.offset=w.write_blob(values.data(),b.size);w.add_blob(b);sign_names[n]=b.name;
    }
    if(sign_offset!=meta["prism.hadamard.sign_values"].size())die("Prism sign data length mismatch");
    auto rotation=[&](TensorInfo& t,const std::vector<std::string>& names){
        int kind=-1;
        for(auto& n:names){used.insert(n);int k=inverse.count(n)?2:folded.count(n)?1:0;if(kind>=0&&k!=kind)die("cannot fuse tensors with different Hadamard transforms");kind=k;}
        if(kind>0){t.hadamard_block=hb;t.hadamard_inverse=kind==2;if(t.cols%hb)die("Hadamard width mismatch");if(mode=="explicit"){if(!sign_names.count(t.cols))die("missing Prism signs for width %lld",(long long)t.cols);t.hadamard_signs=sign_names[t.cols];}}
    };
    auto matrix=[&](const std::string& name,const std::vector<std::string>& sources,int rows,int cols,const std::function<std::pair<int,int>(int)>& select){
        std::vector<const Gguf::Tensor*> ts;for(auto& n:sources)ts.push_back(&g.tensor(n));
        QType qt=Gguf::type(ts[0]->type);size_t stride=qtype_data_bytes(qt,1,cols);
        for(auto* t:ts)if(Gguf::type(t->type)!=qt||t->shape[0]!=(uint64_t)cols)die("incompatible GGUF tensor fusion for %s",name.c_str());
        std::vector<const uint8_t*> bases;for(auto* t:ts)bases.push_back(g.bytes(*t));
        std::vector<uint8_t> bytes((size_t)rows*stride);
        for(int r=0;r<rows;r++){auto [source,row]=select(r);if(source<0){memset(bytes.data()+r*stride,0,stride);continue;}auto* t=ts.at(source);size_t count=1;for(size_t d=1;d<t->shape.size();d++)count*=t->shape[d];if(row<0||(size_t)row>=count)die("GGUF row selection out of range");memcpy(bytes.data()+r*stride,bases[source]+row*stride,stride);}
        TensorInfo t;t.name=name;t.type=qt;t.rows=rows;t.cols=cols;t.data_bytes=bytes.size();rotation(t,sources);t.offset=w.write_blob(bytes.data(),bytes.size());w.add_tensor(t);
    };
    auto plain=[&](const std::string& name,const std::string& source){auto& t=g.tensor(source);int rows=1;for(size_t i=1;i<t.shape.size();i++)rows*=t.shape[i];matrix(name,{source},rows,t.shape[0],[](int r){return std::pair{0,r};});};
    auto vec=[&](const std::string& name,const std::string& source,const std::function<int(int)>& select = {}){
        auto& s=g.tensor(source);size_t n=1;for(auto d:s.shape)n*=d;auto* bytes=g.bytes(s);QType qt=Gguf::type(s.type);
        std::vector<float> v(n);if(qt!=QType::F32&&qt!=QType::F16&&qt!=QType::BF16)die("expected floating-point GGUF vector %s",source.c_str());
        for(size_t i=0;i<n;i++){int j=select?select(i):(int)i;v[i]=qt==QType::F32?((const float*)bytes)[j]:qt==QType::BF16?bf2f(((const uint16_t*)bytes)[j]):h2f(((const uint16_t*)bytes)[j]);}
        TensorInfo t;t.name=name;t.type=QType::F32;t.rows=1;t.cols=n;t.data_bytes=n*4;t.offset=w.write_blob(v.data(),n*4);w.add_tensor(t);used.insert(source);
    };
    plain("tok_embd","token_embd.weight");vec("output_norm","output_norm.weight");if(!c.tied_embed)plain("output","output.weight");
    for(int i=0;i<c.n_layers;i++) {
        const auto p="blk."+std::to_string(i)+".";LayerCfg l;l.n_heads=c.n_heads;l.n_kv_heads=perlayer("attention.head_count_kv",i,c.n_kv_heads);l.head_dim=c.head_dim;
        l.rope_theta=c.rope_theta;l.rope_half=c.head_dim/2;l.rope_rot=l.rope_half;l.ffn=c.ffn;
        vec(p+"attn_norm",p+"attn_norm.weight");
        vec(p+"ffn_norm",p+(arch=="qwen35"?"post_attention_norm.weight":"ffn_norm.weight"));
        bool recurrent=arch=="qwen35" && ((i+1)%integer("full_attention_interval",4)!=0);
        if(recurrent) {
            l.type="deltanet";l.dn_nk=integer("ssm.group_count");l.dn_nv=integer("ssm.time_step_rank");
            l.dn_dk=l.dn_dv=integer("ssm.state_size");l.dn_conv=integer("ssm.conv_kernel");
            int nk=l.dn_nk,nv=l.dn_nv,dv=l.dn_dv,kd=l.dn_key_dim(),vd=l.dn_val_dim(),cd=l.dn_conv_dim();
            if(nk<1||nv<nk||nv%nk||dv<1)die("invalid GGUF delta-net dimensions");
            int rep=nv/nk;
            auto tile=[=](int row,int width){int h=row/width;return ((h%rep)*nk+h/rep)*width+row%width;};
            matrix(p+"dn_in",{p+"attn_qkv.weight",p+"attn_gate.weight"},l.dn_in_rows(),c.hidden,[=](int r){
                if(r<cd)return std::pair{0,r<2*kd?r:2*kd+tile(r-2*kd,dv)};
                if(r<cd+vd)return std::pair{1,tile(r-cd,dv)};
                return std::pair{-1,0};
            });
            // These two matrices stay F32: no ternary conversion of the recurrence controls.
            matrix(p+"dn_ab",{p+"ssm_beta.weight",p+"ssm_alpha.weight"},2*nv,c.hidden,[=](int r){return std::pair{r/nv,tile(r%nv,1)};});
            if(Gguf::type(g.tensor(p+"ssm_beta.weight").type)!=QType::F32 && Gguf::type(g.tensor(p+"ssm_beta.weight").type)!=QType::BF16)die("expected floating-point delta-net controls");
            vec(p+"dn_conv",p+"ssm_conv1d.weight",[=](int j){int channel=j/l.dn_conv,k=j%l.dn_conv;return (channel<2*kd?channel:2*kd+tile(channel-2*kd,dv))*l.dn_conv+k;});
            vec(p+"dn_A",p+"ssm_a",[=](int j){return tile(j,1);});
            vec(p+"dn_dt",p+"ssm_dt.bias",[=](int j){return tile(j,1);});
            vec(p+"dn_norm",p+"ssm_norm.weight");
            if(rep>1 && !meta.get_bool("prism.hadamard.gdn_v_grouped",false))die("GGUF tiled delta-net output columns need an explicit lossless mapping");
            plain(p+"dn_out",p+"ssm_out.weight");
        } else {
            l.qk_norm=1;
            if(arch=="qwen35") {l.attn_gate=true;l.rope_half=integer("rope.dimension_count",l.head_dim)/2;l.rope_rot=l.rope_half;l.rope_den=2*l.rope_half;}
            else {
                bool sliding=meta[prefix+"attention.sliding_window_pattern"][i].as_bool();
                l.sandwich=true;l.v_norm=true;l.attn_scale=1.f;
                l.window=sliding?integer("attention.sliding_window"):0;
                if(sliding){l.head_dim=integer("attention.key_length_swa",256);l.rope_theta=number("rope.freq_base_swa",10000.f);}
                l.rope_half=l.head_dim/2;l.rope_rot=l.rope_half;
                int shared=integer("attention.shared_kv_layers");
                if(i>=c.n_layers-shared)for(int j=c.n_layers-shared-1;j>=0;j--)if((c.layers[j].window>0)==sliding){l.kv_src=j;break;}
                l.k_eq_v=l.kv_src<0&&!g.tensors.count(p+"attn_v.weight");
                vec(p+"post_attn_norm",p+"post_attention_norm.weight");vec(p+"post_ffn_norm",p+"post_ffw_norm.weight");
                if(g.tensors.count(p+"layer_output_scale.weight")){auto& t=g.tensor(p+"layer_output_scale.weight");l.layer_scalar=*(const float*)g.bytes(t);}
                auto rf=p+"rope_freqs.weight";
                if(!sliding){if(!g.tensors.count(rf))rf="rope_freqs.weight";auto& t=g.tensor(rf);auto f=(const float*)g.bytes(t);l.rope_rot=0;for(int j=0;j<l.rope_half;j++){if(std::isfinite(f[j])&&f[j]!=0&&f[j]<1e20f)l.rope_rot++;else break;}used.insert(rf);}
            }
            std::vector<std::string> sources={p+"attn_q.weight"};if(l.kv_src<0){sources.push_back(p+"attn_k.weight");if(!l.k_eq_v)sources.push_back(p+"attn_v.weight");}
            int qr=l.q_dim()*(l.attn_gate?2:1),kv=l.kv_dim();
            matrix(p+"wqkv",sources,l.qkv_rows(),c.hidden,[=](int r){return r<qr?std::pair{0,r}:r<qr+kv?std::pair{1,r-qr}:std::pair{2,r-qr-kv};});
            plain(p+"wo",p+"attn_output.weight");vec(p+"q_norm",p+"attn_q_norm.weight");if(l.kv_src<0)vec(p+"k_norm",p+"attn_k_norm.weight");
        }
        int f=g.tensor(p+"ffn_gate.weight").shape[1];l.ffn=f;
        matrix(p+"wgu",{p+"ffn_gate.weight",p+"ffn_up.weight"},2*f,c.hidden,[](int r){return std::pair{r%2,r/2};});
        plain(p+"wdown",p+"ffn_down.weight");
        c.layers.push_back(l);
    }
    for(auto& n:folded)if(!used.count(n))die("unmapped Hadamard-folded tensor %s",n.c_str());
    for(auto& n:inverse)if(!used.count(n))die("unmapped inverse Hadamard tensor %s",n.c_str());
    Json defaults=Json::object();for(auto kv:{std::pair{"temp","temperature"},{"top_k","top_k"},{"top_p","top_p"},{"min_p","min_p"},{"penalty_repeat","repetition_penalty"}})if(meta.has(std::string("general.sampling.")+kv.first))defaults.set(kv.second,meta[std::string("general.sampling.")+kv.first]);
    Json header=Json::object();header.set("generation_defaults",defaults);header.set("format","tachyon");header.set("version",1);header.set("config",c.to_json());header.set("tokenizer",tok);
    header.set("source_format","gguf");header.set("source_metadata",meta);w.finish(header);return 0;
}

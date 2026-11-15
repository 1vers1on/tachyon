#include "safetensors.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <set>

SafeTensors::~SafeTensors() {
    for (auto& m : maps_) munmap(m.first, m.second);
}

void SafeTensors::open_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open %s", path.c_str());
    struct stat st;
    fstat(fd, &st);
    size_t sz = st.st_size;
    void* m = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) die("mmap failed %s", path.c_str());
    maps_.push_back({m, sz});
    const uint8_t* base = (const uint8_t*)m;
    uint64_t hlen;
    memcpy(&hlen, base, 8);
    std::string err;
    Json h = Json::parse(std::string((const char*)base + 8, hlen), &err);
    if (!err.empty()) die("bad safetensors header in %s: %s", path.c_str(), err.c_str());
    const uint8_t* data = base + 8 + hlen;
    for (auto& kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        STTensor t;
        t.dtype = kv.second.get_str("dtype", "");
        for (auto& s : kv.second["shape"].arr) t.shape.push_back(s.as_int());
        auto& offs = kv.second["data_offsets"];
        size_t a = offs[0].as_int(), b = offs[1].as_int();
        t.data = data + a;
        t.nbytes = b - a;
        tensors_[kv.first] = t;
    }
}

void SafeTensors::open_dir(const std::string& dir) {
    std::string idx;
    for (auto cand : {"/model.safetensors.index.json", "/model.safetensors-index.json"})
        if (file_exists(dir + cand)) { idx = dir + cand; break; }
    if (!idx.empty()) {
        Json j = Json::parse(read_file(idx));
        std::set<std::string> files;
        for (auto& kv : j["weight_map"].obj) files.insert(kv.second.str);
        for (auto& f : files) open_file(dir + "/" + f);
    } else if (file_exists(dir + "/model.safetensors")) {
        open_file(dir + "/model.safetensors");
    } else {
        die("no safetensors found in %s", dir.c_str());
    }
}

const STTensor* SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> r;
    for (auto& kv : tensors_) r.push_back(kv.first);
    return r;
}

void STTensor::to_float(float* out, int64_t start, int64_t count) const {
    if (dtype == "BF16") {
        const uint16_t* p = (const uint16_t*)data + start;
        for (int64_t i = 0; i < count; i++) out[i] = bf2f(p[i]);
    } else if (dtype == "F16") {
        const uint16_t* p = (const uint16_t*)data + start;
        for (int64_t i = 0; i < count; i++) out[i] = h2f(p[i]);
    } else if (dtype == "F32") {
        memcpy(out, (const float*)data + start, count * 4);
    } else {
        die("unsupported dtype %s", dtype.c_str());
    }
}

#include "safetensors.h"
#include "file_validation.h"
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
    if (fstat(fd, &st) != 0 || st.st_size < 8) die("truncated safetensors file %s", path.c_str());
    size_t sz = st.st_size;
    void* m = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) die("mmap failed %s", path.c_str());
    maps_.push_back({m, sz});
    const uint8_t* base = (const uint8_t*)m;
    uint64_t hlen;
    memcpy(&hlen, base, 8);
    if (hlen > sz - 8) die("truncated safetensors header in %s", path.c_str());
    std::string err;
    Json h = Json::parse(std::string((const char*)base + 8, hlen), &err);
    if (!err.empty() || !h.is_obj()) die("bad safetensors header in %s: %s", path.c_str(), err.c_str());
    const uint8_t* data = base + 8 + hlen;
    for (auto& kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        STTensor t;
        t.dtype = kv.second.get_str("dtype", "");
        const Json& shape = kv.second["shape"];
        if (!shape.is_arr()) die("missing shape for tensor %s", kv.first.c_str());
        size_t count = 1;
        for (auto& s : shape.arr) {
            size_t dim = file_size_value(s, "safetensors dimension");
            count = file_product(count, dim, "safetensors tensor");
            t.shape.push_back(dim);
        }
        auto& offs = kv.second["data_offsets"];
        if (!offs.is_arr() || offs.size() != 2) die("invalid offsets for tensor %s", kv.first.c_str());
        size_t a = file_size_value(offs[0], "safetensors offset"), b = file_size_value(offs[1], "safetensors offset");
        if (b < a || !file_range(a, b - a, sz - 8 - hlen)) die("tensor %s out of range", kv.first.c_str());
        // Accept unused non-floating tensors too (e.g. integer buffers), but check their byte sizes.
        size_t item = t.dtype == "F64" || t.dtype == "I64" || t.dtype == "U64" ? 8 :
                      t.dtype == "F32" || t.dtype == "I32" || t.dtype == "U32" ? 4 :
                      t.dtype == "BF16" || t.dtype == "F16" || t.dtype == "I16" || t.dtype == "U16" ? 2 :
                      t.dtype == "I8" || t.dtype == "U8" || t.dtype == "BOOL" ||
                      t.dtype == "F8_E4M3" || t.dtype == "F8_E5M2" ? 1 : 0;
        if (!item) die("unsupported safetensors dtype %s", t.dtype.c_str());
        if (file_product(count, item, "safetensors tensor") != b - a) die("tensor %s byte size does not match shape", kv.first.c_str());
        t.data = data + a;
        t.nbytes = b - a;
        if (!tensors_.emplace(kv.first, std::move(t)).second) die("duplicate tensor %s", kv.first.c_str());
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
    size_t item = dtype == "F32" ? 4 : 2;
    if (start < 0 || count < 0 || !file_range(start, count, nbytes / item))
        die("tensor read out of range (start %lld, count %lld)", (long long)start, (long long)count);
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

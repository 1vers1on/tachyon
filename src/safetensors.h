// Tachyon - safetensors reader (mmap, supports sharded checkpoints)
#pragma once
#include "common.h"
#include "json.h"
#include <map>
#include <memory>

struct STTensor {
    std::string dtype;  // BF16 F16 F32
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
    // convert element i to float
    void to_float(float* out, int64_t start, int64_t count) const;
};

class SafeTensors {
public:
    ~SafeTensors();
    void open_dir(const std::string& dir);  // reads model.safetensors or *.index.json
    const STTensor* get(const std::string& name) const;
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    std::vector<std::string> names() const;
private:
    void open_file(const std::string& path);
    std::map<std::string, STTensor> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
};

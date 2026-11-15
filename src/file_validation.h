// Checked sizes for mmap-backed model metadata. Validate before pointer arithmetic.
#pragma once
#include "common.h"
#include "json.h"
#include <limits>

inline size_t file_size_value(const Json& v, const char* what) {
    // JSON numbers are doubles; larger integers cannot be represented exactly.
    if (!v.is_num() || !std::isfinite(v.num) || v.num < 0 ||
        v.num > 9007199254740991.0 || floor(v.num) != v.num ||
        (long double)v.num > std::numeric_limits<size_t>::max())
        die("invalid %s in model metadata", what);
    return (size_t)v.num;
}

inline bool file_range(size_t offset, size_t bytes, size_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

inline size_t file_product(size_t a, size_t b, const char* what) {
    if (b && a > std::numeric_limits<size_t>::max() / b) die("%s size overflow", what);
    return a * b;
}

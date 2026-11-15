#include "quant.h"
#include <thread>
#include <atomic>
#include <algorithm>

// Asymmetric 4-bit quantization of one group with least-squares refinement of (scale, min).
static void quant_group_q4(const float* w, int n, uint8_t* q, float& s_out, float& m_out, const float* a, int qmax = 15) {
    float mn = w[0], mx = w[0];
    for (int i = 1; i < n; i++) { mn = std::min(mn, w[i]); mx = std::max(mx, w[i]); }
    if (mx - mn < 1e-20f) {
        for (int i = 0; i < n; i++) q[i] = 0;
        s_out = 0; m_out = mn;
        return;
    }
    float best_err = INFINITY, best_s = 0, best_m = 0;
    // candidate ranges: shrink min/max a bit, then refine with (importance-weighted) least squares
    const int ncand = 20;
    for (int c = 0; c < ncand; c++) {
        float shrink = 0.015f * c;
        float lo = mn + (mx - mn) * shrink * 0.5f, hi = mx - (mx - mn) * shrink * 0.5f;
        float s = (hi - lo) / (float)qmax, m = lo;
        for (int it = 0; it < 4; it++) {
            float inv = s > 0 ? 1.f / s : 0;
            double S0 = 0, Sq = 0, Sqq = 0, Sw = 0, Sqw = 0;
            for (int i = 0; i < n; i++) {
                int v = (int)lrintf((w[i] - m) * inv);
                v = v < 0 ? 0 : (v > qmax ? qmax : v);
                double ai = a ? a[i] : 1.0;
                S0 += ai; Sq += ai * v; Sqq += ai * v * v; Sw += ai * w[i]; Sqw += ai * v * (double)w[i];
            }
            double det = S0 * Sqq - Sq * Sq;
            if (det <= 0) break;
            float ns = (float)((S0 * Sqw - Sq * Sw) / det);
            float nm = (float)((Sw - ns * Sq) / S0);
            if (!(ns > 0)) break;
            s = ns; m = nm;
        }
        // evaluate with fp16-rounded params
        float sr = h2f(f2h(s)), mr = h2f(f2h(m));
        float inv = sr > 0 ? 1.f / sr : 0;
        float err = 0;
        for (int i = 0; i < n; i++) {
            int v = (int)lrintf((w[i] - mr) * inv);
            v = v < 0 ? 0 : (v > qmax ? qmax : v);
            float d = w[i] - (sr * v + mr);
            err += (a ? a[i] : 1.f) * d * d;
        }
        if (err < best_err) { best_err = err; best_s = sr; best_m = mr; }
    }
    float inv = best_s > 0 ? 1.f / best_s : 0;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf((w[i] - best_m) * inv);
        q[i] = (uint8_t)(v < 0 ? 0 : (v > qmax ? qmax : v));
    }
    s_out = best_s; m_out = best_m;
}

void quantize_row(QType t, const float* src, int64_t cols, uint8_t* dst, uint8_t* scales, const float* imp) {
    switch (t) {
        case QType::F32: memcpy(dst, src, cols * 4); break;
        case QType::F16: {
            uint16_t* d = (uint16_t*)dst;
            for (int64_t i = 0; i < cols; i++) d[i] = f2h(src[i]);
            break;
        }
        case QType::Q8: {
            uint16_t* sc = (uint16_t*)scales;
            int8_t* d = (int8_t*)dst;
            for (int64_t g = 0; g < cols / Q8_GROUP; g++) {
                const float* w = src + g * Q8_GROUP;
                float amax = 0;
                for (int i = 0; i < Q8_GROUP; i++) amax = std::max(amax, fabsf(w[i]));
                float s = h2f(f2h(amax / 127.f));
                float inv = s > 0 ? 1.f / s : 0;
                for (int i = 0; i < Q8_GROUP; i++) {
                    int v = (int)lrintf(w[i] * inv);
                    d[g * Q8_GROUP + i] = (int8_t)std::clamp(v, -127, 127);
                }
                sc[g] = f2h(s);
            }
            break;
        }
        case QType::Q3: {
            uint16_t* sc = (uint16_t*)scales;
            for (int64_t g = 0; g < cols / Q4_GROUP; g++) {
                uint8_t q[Q4_GROUP];
                float s, m;
                quant_group_q4(src + g * Q4_GROUP, Q4_GROUP, q, s, m, imp ? imp + g * Q4_GROUP : nullptr, 7);
                sc[2 * g] = f2h(s);
                sc[2 * g + 1] = f2h(m);
                for (int b = 0; b < Q4_GROUP / 32; b++) {
                    uint32_t L[2] = {0, 0}, H = 0;
                    for (int e = 0; e < 32; e++) {
                        uint32_t v = q[b * 32 + e];
                        int w = e / 16, k = (e % 16) / 4, bb = e % 4;
                        L[w] |= (v & 3u) << (8 * bb + 2 * k);
                        H |= ((v >> 2) & 1u) << (8 * bb + e / 4);
                    }
                    uint32_t* d = (uint32_t*)(dst + (g * Q4_GROUP + b * 32) / 32 * 12);
                    d[0] = L[0]; d[1] = L[1]; d[2] = H;
                }
            }
            break;
        }
        case QType::Q4: {
            uint16_t* sc = (uint16_t*)scales;
            for (int64_t g = 0; g < cols / Q4_GROUP; g++) {
                uint8_t q[Q4_GROUP];
                float s, m;
                quant_group_q4(src + g * Q4_GROUP, Q4_GROUP, q, s, m, imp ? imp + g * Q4_GROUP : nullptr);
                sc[2 * g] = f2h(s);
                sc[2 * g + 1] = f2h(m);
                for (int b = 0; b < Q4_GROUP / 32; b++) {
                    uint8_t* d = dst + (g * Q4_GROUP + b * 32) / 2;
                    for (int j = 0; j < 16; j++) d[j] = q[b * 32 + j] | (q[b * 32 + j + 16] << 4);
                }
            }
            break;
        }
    }
}

void dequantize_row(QType t, const uint8_t* data, const uint8_t* scales, int64_t cols, float* dst) {
    switch (t) {
        case QType::F32: memcpy(dst, data, cols * 4); break;
        case QType::F16: for (int64_t i = 0; i < cols; i++) dst[i] = h2f(((const uint16_t*)data)[i]); break;
        case QType::Q8: {
            const uint16_t* sc = (const uint16_t*)scales;
            for (int64_t i = 0; i < cols; i++) dst[i] = h2f(sc[i / Q8_GROUP]) * ((const int8_t*)data)[i];
            break;
        }
        case QType::Q3: {
            const uint16_t* sc = (const uint16_t*)scales;
            for (int64_t i = 0; i < cols; i++) {
                const uint32_t* blk = (const uint32_t*)(data + (i / 32) * 12);
                int e = i % 32, w = e / 16, k = (e % 16) / 4, bb = e % 4;
                int q = ((blk[w] >> (8 * bb + 2 * k)) & 3) | (((blk[2] >> (8 * bb + e / 4)) & 1) << 2);
                int64_t g = i / Q4_GROUP;
                dst[i] = h2f(sc[2 * g]) * q + h2f(sc[2 * g + 1]);
            }
            break;
        }
        case QType::Q4: {
            const uint16_t* sc = (const uint16_t*)scales;
            for (int64_t i = 0; i < cols; i++) {
                int64_t blk = i / 32, j = i % 32;
                uint8_t byte = data[blk * 16 + (j & 15)];
                int q = j < 16 ? (byte & 15) : (byte >> 4);
                int64_t g = i / Q4_GROUP;
                dst[i] = h2f(sc[2 * g]) * q + h2f(sc[2 * g + 1]);
            }
            break;
        }
    }
}

void quantize_matrix(QType t, int64_t rows, int64_t cols,
                     const std::function<void(int64_t, float*)>& src_row,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& scales, int nthreads, const float* imp) {
    if ((t == QType::Q4 || t == QType::Q3) && cols % Q4_GROUP) die("q4 needs cols %% %d == 0 (got %lld)", Q4_GROUP, (long long)cols);
    if (t == QType::Q8 && cols % Q8_GROUP) die("q8 needs cols %% %d == 0", Q8_GROUP);
    size_t row_data = qtype_data_bytes(t, 1, cols), row_scale = qtype_scale_bytes(t, 1, cols);
    data.assign(row_data * rows, 0);
    scales.assign(row_scale * rows, 0);
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        std::vector<float> buf(cols);
        while (true) {
            int64_t r0 = next.fetch_add(16);
            if (r0 >= rows) break;
            for (int64_t r = r0; r < std::min(rows, r0 + 16); r++) {
                src_row(r, buf.data());
                quantize_row(t, buf.data(), cols, data.data() + r * row_data, scales.data() + r * row_scale, imp);
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
    for (auto& x : th) x.join();
}

IMatrix* g_imatrix = nullptr;

void IMatrix::add(const std::string& name, const float* x, int M, int K) {
    std::lock_guard<std::mutex> lk(mu);
    auto& e = acc[name];
    if (e.first.empty()) e.first.assign(K, 0.0);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) e.first[k] += (double)x[(size_t)m * K + k] * x[(size_t)m * K + k];
    e.second += M;
}

void IMatrix::save(const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) die("cannot write %s", path.c_str());
    uint32_t n = acc.size();
    fwrite("TIMX", 1, 4, f);
    fwrite(&n, 4, 1, f);
    for (auto& kv : acc) {
        uint32_t nl = kv.first.size(), k = kv.second.first.size();
        fwrite(&nl, 4, 1, f); fwrite(kv.first.data(), 1, nl, f);
        fwrite(&k, 4, 1, f);
        std::vector<float> mean(k);
        for (uint32_t i = 0; i < k; i++) mean[i] = (float)(kv.second.first[i] / std::max<int64_t>(1, kv.second.second));
        fwrite(mean.data(), 4, k, f);
    }
    fclose(f);
}

std::map<std::string, std::vector<float>> IMatrix::load(const std::string& path) {
    std::string s = read_file(path);
    if (s.size() < 8 || s.compare(0, 4, "TIMX") != 0) die("bad imatrix file %s", path.c_str());
    std::map<std::string, std::vector<float>> r;
    size_t p = 4;
    uint32_t n; memcpy(&n, &s[p], 4); p += 4;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t nl; memcpy(&nl, &s[p], 4); p += 4;
        std::string name = s.substr(p, nl); p += nl;
        uint32_t k; memcpy(&k, &s[p], 4); p += 4;
        std::vector<float> v(k);
        memcpy(v.data(), &s[p], 4 * k); p += 4 * k;
        r[name] = v;
    }
    return r;
}

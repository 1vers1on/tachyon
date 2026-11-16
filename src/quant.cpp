#include "quant.h"
#include "threadpool.h"
#include <atomic>
#include <algorithm>

// Asymmetric 4-bit quantization of one group with least-squares refinement of (scale, min).
template<bool weighted>
static void quant_group_q4_impl(const float* w, int n, uint8_t* q, float& s_out, float& m_out, const float* a, int qmax) {
    float mn = w[0], mx = w[0];
    for (int i = 1; i < n; i++) { mn = std::min(mn, w[i]); mx = std::max(mx, w[i]); }
    if (mx - mn < 1e-20f) {
        for (int i = 0; i < n; i++) q[i] = 0;
        s_out = 0; m_out = mn;
        return;
    }
    float best_err = INFINITY, best_s = 0, best_m = 0;
    // These moments do not depend on the candidate range or its quantized values.
    double S0 = 0, Sw = 0;
    for (int i = 0; i < n; i++) {
        double ai = weighted ? a[i] : 1.0;
        S0 += ai;
        Sw += ai * w[i];
    }
    // candidate ranges: shrink min/max a bit, then refine with (importance-weighted) least squares
    const int ncand = 20;
#ifdef __AVX2__
    // SIMD lanes hold independent candidates. Each lane sums weights in the original
    // order, avoiding the numerical changes of a parallel reduction over the group.
    constexpr int lanes = 4;
#else
    constexpr int lanes = 1;
#endif
    for (int c = 0; c < ncand; c += lanes) {
        float s[lanes], m[lanes];
        bool active[lanes];
        for (int j = 0; j < lanes; j++) {
            float shrink = 0.015f * (c + j);
            float lo = mn + (mx - mn) * shrink * 0.5f, hi = mx - (mx - mn) * shrink * 0.5f;
            s[j] = (hi - lo) / (float)qmax; m[j] = lo;
            active[j] = true;
        }
        for (int it = 0; it < 4; it++) {
            float inv[lanes];
            for (int j = 0; j < lanes; j++) inv[j] = s[j] > 0 ? 1.f / s[j] : 0;
            double Sq[lanes] = {}, Sqq[lanes] = {}, Sqw[lanes] = {};
#ifdef __AVX2__
            const __m128 vm = _mm_loadu_ps(m), vi = _mm_loadu_ps(inv);
            const __m128i vmax = _mm_set1_epi32(qmax), zero = _mm_setzero_si128();
            __m256d sq = _mm256_setzero_pd(), sqq = sq, sqw = sq;
            for (int i = 0; i < n; i++) {
                __m128i qi = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_set1_ps(w[i]), vm), vi));
                qi = _mm_min_epi32(vmax, _mm_max_epi32(zero, qi));
                __m256d v = _mm256_cvtepi32_pd(qi);
                __m256d av = weighted ? _mm256_mul_pd(_mm256_set1_pd(a[i]), v) : v;
                sq = _mm256_add_pd(sq, av);
                sqq = _mm256_add_pd(sqq, _mm256_mul_pd(av, v));
                sqw = _mm256_add_pd(sqw, _mm256_mul_pd(av, _mm256_set1_pd(w[i])));
            }
            _mm256_storeu_pd(Sq, sq);
            _mm256_storeu_pd(Sqq, sqq);
            _mm256_storeu_pd(Sqw, sqw);
#else
            for (int i = 0; i < n; i++) {
                int v = (int)lrintf((w[i] - m[0]) * inv[0]);
                v = v < 0 ? 0 : (v > qmax ? qmax : v);
                double ai = weighted ? a[i] : 1.0;
                Sq[0] += ai * v; Sqq[0] += ai * v * v; Sqw[0] += ai * v * (double)w[i];
            }
#endif
            bool any_active = false;
            for (int j = 0; j < lanes; j++) {
                if (!active[j]) continue;
                double det = S0 * Sqq[j] - Sq[j] * Sq[j];
                if (det <= 0) { active[j] = false; continue; }
                float ns = (float)((S0 * Sqw[j] - Sq[j] * Sw) / det);
                float nm = (float)((Sw - ns * Sq[j]) / S0);
                if (!(ns > 0)) { active[j] = false; continue; }
                // A fixed point cannot change in the remaining refinement iterations.
                active[j] = !(ns == s[j] && nm == m[j]);
                s[j] = ns; m[j] = nm;
                any_active |= active[j];
            }
            if (!any_active) break;
        }
        // evaluate with fp16-rounded params
        float sr[lanes], mr[lanes], inv[lanes], err[lanes] = {};
        for (int j = 0; j < lanes; j++) {
            sr[j] = h2f(f2h(s[j])); mr[j] = h2f(f2h(m[j]));
            inv[j] = sr[j] > 0 ? 1.f / sr[j] : 0;
        }
#ifdef __AVX2__
        const __m128 vs = _mm_loadu_ps(sr), vm = _mm_loadu_ps(mr), vi = _mm_loadu_ps(inv);
        const __m128i vmax = _mm_set1_epi32(qmax), zero = _mm_setzero_si128();
        __m128 ve = _mm_setzero_ps();
        for (int i = 0; i < n; i++) {
            __m128 vw = _mm_set1_ps(w[i]);
            __m128i qi = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(vw, vm), vi));
            qi = _mm_min_epi32(vmax, _mm_max_epi32(zero, qi));
            __m128 d = _mm_sub_ps(vw, _mm_add_ps(_mm_mul_ps(vs, _mm_cvtepi32_ps(qi)), vm));
            __m128 ad = weighted ? _mm_mul_ps(_mm_set1_ps(a[i]), d) : d;
            ve = _mm_add_ps(ve, _mm_mul_ps(ad, d));
        }
        _mm_storeu_ps(err, ve);
#else
        for (int i = 0; i < n; i++) {
            int v = (int)lrintf((w[i] - mr[0]) * inv[0]);
            v = v < 0 ? 0 : (v > qmax ? qmax : v);
            float d = w[i] - (sr[0] * v + mr[0]);
            err[0] += (weighted ? a[i] : 1.f) * d * d;
        }
#endif
        for (int j = 0; j < lanes; j++) {
            if (err[j] < best_err) { best_err = err[j]; best_s = sr[j]; best_m = mr[j]; }
        }
    }
    float inv = best_s > 0 ? 1.f / best_s : 0;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf((w[i] - best_m) * inv);
        q[i] = (uint8_t)(v < 0 ? 0 : (v > qmax ? qmax : v));
    }
    s_out = best_s; m_out = best_m;
}

static void quant_group_q4(const float* w, int n, uint8_t* q, float& s_out, float& m_out, const float* a, int qmax = 15) {
    if (a) quant_group_q4_impl<true>(w, n, q, s_out, m_out, a, qmax);
    else quant_group_q4_impl<false>(w, n, q, s_out, m_out, a, qmax);
}

// GGUF K-quant block encoders. Tensor metadata records the exact format; mixed recipes live in the converter.
static void quantize_k(QType type,const float* src,int64_t cols,uint8_t* dst,const float* importance) {
    if(cols%256)die("K-quants require a row width divisible by 256");
    for(int64_t off=0;off<cols;off+=256){
        uint8_t* p=dst+(off/256)*packed_bytes((int)type);memset(p,0,packed_bytes((int)type));const float* w=src+off;
        if(type==QType::Q6_K){
            float scales[16],largest=0;
            for(int g=0;g<16;g++){float lo=0,hi=0;for(int i=0;i<16;i++){lo=std::min(lo,w[16*g+i]);hi=std::max(hi,w[16*g+i]);}scales[g]=std::max(-lo/32.f,hi/31.f);largest=std::max(largest,scales[g]);}
            uint16_t dh=f2h(largest/127.f);memcpy(p+208,&dh,2);float d=h2f(dh);
            for(int g=0;g<16;g++)p[192+g]=(uint8_t)std::clamp(d>0?(int)lrintf(scales[g]/d):0,0,127);
            for(int i=0;i<256;i++){float s=d*p[192+i/16];int q=std::clamp(s>0?(int)lrintf(w[i]/s):0,-32,31)+32;int h=i/128,t=(i%128)/32,l=i%32;p[h*64+(t&1)*32+l]|=(q&15)<<(t<2?0:4);p[128+h*32+l]|=(q>>4)<<(2*t);}
        }else if(type==QType::Q4_K){
            float scales[8],mins[8],sd=0,md=0;
            for(int g=0;g<8;g++){uint8_t q[32];float s,m;quant_group_q4(w+g*32,32,q,s,m,importance?importance+off+g*32:nullptr);scales[g]=s;mins[g]=std::max(0.f,-m);sd=std::max(sd,s);md=std::max(md,mins[g]);}
            uint16_t h=f2h(sd/63.f);memcpy(p,&h,2);float d=h2f(h);h=f2h(md/63.f);memcpy(p+2,&h,2);float dm=h2f(h);uint8_t* codes=p+4;
            for(int g=0;g<8;g++){int sc=std::clamp(d>0?(int)lrintf(scales[g]/d):0,0,63),mn=std::clamp(dm>0?(int)lrintf(mins[g]/dm):0,0,63);
                if(g<4){codes[g]|=sc;codes[g+4]|=mn;}else{codes[g+4]|=(sc&15)|((mn&15)<<4);codes[g-4]|=(sc>>4)<<6;codes[g]|=(mn>>4)<<6;}
                float s=d*sc,m=dm*mn;for(int j=0;j<32;j++){int i=32*g+j,q=std::clamp(s>0?(int)lrintf((w[i]+m)/s):0,0,15);p[16+(i/64)*32+i%32]|=q<<(i%64<32?0:4);}}
        }
    }
}

void quantize_row(QType t, const float* src, int64_t cols, uint8_t* dst, uint8_t* scales, const float* imp) {
    if(t==QType::Q4_K||t==QType::Q6_K){quantize_k(t,src,cols,dst,imp);return;}
    if(t==QType::Q8_0){for(int64_t off=0;off<cols;off+=32){uint8_t* p=dst+off/32*34;float peak=0;for(int i=0;i<32;i++)peak=std::max(peak,fabsf(src[off+i]));uint16_t h=f2h(peak/127.f);memcpy(p,&h,2);float d=h2f(h);for(int i=0;i<32;i++)p[2+i]=(uint8_t)(int8_t)std::clamp(d>0?(int)lrintf(src[off+i]/d):0,-127,127);}return;}
    if(qtype_packed(t))die("format %s is import-only; do not requantize ternary or QAT checkpoints",qtype_name(t));
    switch (t) {
        case QType::BF16: for(int64_t i=0;i<cols;i++){uint32_t u;memcpy(&u,src+i,4);((uint16_t*)dst)[i]=(u+0x7fff+((u>>16)&1))>>16;}break;
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
    if (qtype_packed(t)) { for (int64_t i = 0; i < cols; i++) dst[i] = packed_value((int)t, data, i); return; }
    switch (t) {
        case QType::F32: memcpy(dst, data, cols * 4); break;
        case QType::BF16: for(int64_t i=0;i<cols;i++)dst[i]=bf2f(((const uint16_t*)data)[i]);break;
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
                     std::vector<uint8_t>& data, std::vector<uint8_t>& scales, ThreadPool& pool, const float* imp) {
    if(qtype_packed(t)&&cols%packed_group((int)t))die("invalid row width for %s",qtype_name(t));
    if ((t == QType::Q4 || t == QType::Q3) && cols % Q4_GROUP) die("q4 needs cols %% %d == 0 (got %lld)", Q4_GROUP, (long long)cols);
    if (t == QType::Q8 && cols % Q8_GROUP) die("q8 needs cols %% %d == 0", Q8_GROUP);
    size_t row_data = qtype_data_bytes(t, 1, cols), row_scale = qtype_scale_bytes(t, 1, cols);
    data.assign(row_data * rows, 0);
    scales.assign(row_scale * rows, 0);
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        std::vector<float> buf(cols);
        while (true) {
            int64_t r0 = next.fetch_add(16, std::memory_order_relaxed);
            if (r0 >= rows) break;
            for (int64_t r = r0; r < std::min(rows, r0 + 16); r++) {
                src_row(r, buf.data());
                quantize_row(t, buf.data(), cols, data.data() + r * row_data,
                             row_scale ? scales.data() + r * row_scale : nullptr, imp);
            }
        }
    };
    // Each claim covers 16 rows; small vectors need no worker threads.
    if (rows <= 16) worker();
    else pool.run([&](int tid, int) {
        if (tid < (rows + 15) / 16) worker();
    });
}

void quantize_matrix(QType t, int64_t rows, int64_t cols,
                     const std::function<void(int64_t, float*)>& src_row,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& scales, int nthreads, const float* imp) {
    nthreads = (int)std::max<int64_t>(1, std::min<int64_t>(nthreads, (rows + 15) / 16));
    ThreadPool pool(nthreads);
    quantize_matrix(t, rows, cols, src_row, data, scales, pool, imp);
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
    auto take_u32 = [&]() {
        if (s.size() - p < 4) die("truncated imatrix file %s", path.c_str());
        uint32_t value; memcpy(&value, s.data() + p, 4); p += 4;
        return value;
    };
    uint32_t n = take_u32();
    for (uint32_t i = 0; i < n; i++) {
        uint32_t nl = take_u32();
        if (nl > s.size() - p) die("truncated imatrix file %s", path.c_str());
        std::string name = s.substr(p, nl); p += nl;
        uint32_t k = take_u32();
        if (k > (s.size() - p) / sizeof(float)) die("truncated imatrix file %s", path.c_str());
        std::vector<float> v(k);
        if (k) memcpy(v.data(), s.data() + p, sizeof(float) * k);
        p += sizeof(float) * k;
        r[name] = v;
    }
    return r;
}

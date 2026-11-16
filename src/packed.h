// GGUF block codecs. Layout semantics: ggml/Prism (MIT), see docs/model-formats.md.
// Shared between host reference decoding and the runtime CUDA source; numeric tags are Tachyon QType values.
#ifdef __CUDACC_RTC__
#define TACH_PACKED_INLINE __device__ __forceinline__
TACH_PACKED_INLINE float packed_half(const unsigned char* p) { return __half2float(*(const half*)p); }
#else
#pragma once
#define TACH_PACKED_INLINE inline
TACH_PACKED_INLINE float packed_half(const unsigned char* p) { uint16_t h; memcpy(&h, p, 2); return h2f(h); }
#endif
TACH_PACKED_INLINE int packed_group(int qt) { return qt == 5 || qt == 6 ? 128 : qt >= 9 && qt <= 11 ? 256 : 32; }
TACH_PACKED_INLINE int packed_bytes(int qt) {
    switch (qt) { case 5: return 34; case 6: return 28; case 7: return 18; case 8: return 34;
                  case 9: return 144; case 10: return 176; case 11: return 210; default: return 0; }
}
TACH_PACKED_INLINE float packed_value(int qt, const unsigned char* row, int index) {
    int block = packed_group(qt), i = index % block;
    const unsigned char* p = row + (index / block) * packed_bytes(qt);
    if (qt == 5) return packed_half(p) * (int((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    if (qt == 6) {
        int byte, power;
        if (i < 80) { byte = i % 16; power = i / 16; }
        else if (i < 120) { byte = 16 + (i - 80) % 8; power = (i - 80) / 8; }
        else { byte = 24 + (i - 120) % 2; power = (i - 120) / 2; }
        int factor = power == 0 ? 1 : power == 1 ? 3 : power == 2 ? 9 : power == 3 ? 27 : 81;
        int trit = (((p[byte] * factor) & 255) * 3) >> 8;
        return packed_half(p + 26) * (trit - 1);
    }
    if (qt == 7) return packed_half(p) * (int((p[2 + i % 16] >> (i < 16 ? 0 : 4)) & 15) - 8);
    if (qt == 8) return packed_half(p) * (signed char)p[2 + i];
    if (qt == 9 || qt == 10) {
        int g = i / 32, s, m;
        const unsigned char* sc = p + 4;
        if (g < 4) { s = sc[g] & 63; m = sc[g + 4] & 63; }
        else { s = (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4); m = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4); }
        int q = (p[(qt == 9 ? 16 : 48) + (i / 64) * 32 + i % 32] >> (i % 64 < 32 ? 0 : 4)) & 15;
        if (qt == 10) q |= ((p[16 + i % 32] >> (i / 32)) & 1) << 4;
        return packed_half(p) * s * q - packed_half(p + 2) * m;
    }
    if (qt == 11) {
        int h = i / 128, qtr = (i % 128) / 32, lane = i % 32;
        int lo = (p[h * 64 + (qtr & 1) * 32 + lane] >> (qtr < 2 ? 0 : 4)) & 15;
        int hi = (p[128 + h * 32 + lane] >> (2 * qtr)) & 3;
        return packed_half(p + 208) * (signed char)p[192 + i / 16] * ((lo | (hi << 4)) - 32);
    }
    return 0;
}
#undef TACH_PACKED_INLINE

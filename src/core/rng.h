#pragma once

#include "core/math.h"

#include <cstdint>
#include <cstring>

namespace pr {

// PCG32 random number generator (O'Neill), as used by pbrt.
class RNG {
  public:
    static constexpr uint64_t DefaultState = 0x853c49e6748fea9bULL;
    static constexpr uint64_t DefaultStream = 0xda3e39cb94b95bdbULL;
    static constexpr uint64_t Mult = 0x5851f42d4c957f2dULL;

    RNG() : state_(DefaultState), inc_(DefaultStream) {}
    RNG(uint64_t seqIndex, uint64_t offset) { SetSequence(seqIndex, offset); }
    explicit RNG(uint64_t seqIndex) { SetSequence(seqIndex, MixBits(seqIndex)); }

    static uint64_t MixBits(uint64_t v) {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185ULL;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44dULL;
        v ^= (v >> 33);
        return v;
    }

    void SetSequence(uint64_t sequenceIndex, uint64_t offset) {
        state_ = 0u;
        inc_ = (sequenceIndex << 1u) | 1u;
        Uniform32();
        state_ += offset;
        Uniform32();
    }
    void SetSequence(uint64_t sequenceIndex) { SetSequence(sequenceIndex, MixBits(sequenceIndex)); }

    uint32_t Uniform32() {
        uint64_t oldstate = state_;
        state_ = oldstate * Mult + inc_;
        uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = uint32_t(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    float UniformFloat() { return std::min(OneMinusEpsilon, float(Uniform32()) * 0x1p-32f); }
    Vec2f Uniform2D() {
        float a = UniformFloat();
        float b = UniformFloat();
        return {a, b};
    }

    void GetState(uint64_t *state, uint64_t *inc) const {
        *state = state_;
        *inc = inc_;
    }
    void SetState(uint64_t state, uint64_t inc) {
        state_ = state;
        inc_ = inc;
    }

  private:
    uint64_t state_, inc_;
};

// MurmurHash64A over raw bytes.
inline uint64_t MurmurHash64A(const unsigned char *key, size_t len, uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47;
    uint64_t h = seed ^ (len * m);
    const unsigned char *end = key + 8 * (len / 8);
    while (key != end) {
        uint64_t k;
        std::memcpy(&k, key, sizeof(uint64_t));
        key += 8;
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    switch (len & 7) {
    case 7: h ^= uint64_t(key[6]) << 48; [[fallthrough]];
    case 6: h ^= uint64_t(key[5]) << 40; [[fallthrough]];
    case 5: h ^= uint64_t(key[4]) << 32; [[fallthrough]];
    case 4: h ^= uint64_t(key[3]) << 24; [[fallthrough]];
    case 3: h ^= uint64_t(key[2]) << 16; [[fallthrough]];
    case 2: h ^= uint64_t(key[1]) << 8; [[fallthrough]];
    case 1:
        h ^= uint64_t(key[0]);
        h *= m;
    }
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

template <typename... Args> inline uint64_t Hash(Args... args) {
    constexpr size_t sz = (sizeof(Args) + ... + 0);
    unsigned char buf[sz];
    size_t off = 0;
    ((std::memcpy(buf + off, &args, sizeof(Args)), off += sizeof(Args)), ...);
    return MurmurHash64A(buf, sz, 0);
}

// Inverse error function (Giles 2010 single precision approximation).
inline float ErfInv(float x) {
    x = Clamp(x, -0.99999f, 0.99999f);
    float w = -std::log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = std::sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

} // namespace pr

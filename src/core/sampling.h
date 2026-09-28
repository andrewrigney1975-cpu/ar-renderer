#pragma once

#include "core/math.h"

#include <span>
#include <vector>

namespace pr {

inline Vec2f SampleUniformDiskConcentric(Vec2f u) {
    Vec2f uOffset(2 * u.x - 1, 2 * u.y - 1);
    if (uOffset.x == 0 && uOffset.y == 0) return {0, 0};
    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return {r * std::cos(theta), r * std::sin(theta)};
}

inline Vec3f SampleCosineHemisphere(Vec2f u) {
    Vec2f d = SampleUniformDiskConcentric(u);
    float z = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return {d.x, d.y, z};
}
inline float CosineHemispherePDF(float cosTheta) { return cosTheta * InvPi; }

inline Vec3f SampleUniformSphere(Vec2f u) {
    float z = 1 - 2 * u.x;
    float r = SafeSqrt(1 - z * z);
    float phi = 2 * Pi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), z};
}
inline float UniformSpherePDF() { return Inv4Pi; }

inline Vec3f SampleUniformHemisphere(Vec2f u) {
    float z = u.x;
    float r = SafeSqrt(1 - z * z);
    float phi = 2 * Pi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), z};
}
inline float UniformHemispherePDF() { return Inv2Pi; }

inline Vec3f SampleUniformCone(Vec2f u, float cosThetaMax) {
    float cosTheta = (1 - u.x) + u.x * cosThetaMax;
    float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    float phi = u.y * 2 * Pi;
    return SphericalDirection(sinTheta, cosTheta, phi);
}
inline float UniformConePDF(float cosThetaMax) { return 1 / (2 * Pi * (1 - cosThetaMax)); }

// Cosine-power lobe around +z: pdf = (n+1)/(2 pi) cos^n(theta).
inline Vec3f SampleCosPowerLobe(Vec2f u, float n) {
    float cosTheta = std::pow(u.x, 1 / (n + 1));
    float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    return SphericalDirection(sinTheta, cosTheta, 2 * Pi * u.y);
}
inline float CosPowerLobePDF(float cosTheta, float n) {
    if (cosTheta <= 0) return 0;
    return (n + 1) * Inv2Pi * std::pow(cosTheta, n);
}

// Uniform barycentrics on a triangle (pbrt-v4 / Heitz).
inline void SampleUniformTriangle(Vec2f u, float b[3]) {
    float b0, b1;
    if (u.x < u.y) {
        b0 = u.x / 2;
        b1 = u.y - b0;
    } else {
        b1 = u.y / 2;
        b0 = u.x - b1;
    }
    b[0] = b0;
    b[1] = b1;
    b[2] = 1 - b0 - b1;
}

inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    if (std::isinf(Sqr(f))) return 1;
    if (f == 0 && g == 0) return 0;
    return Sqr(f) / (Sqr(f) + Sqr(g));
}

inline float SampleExponential(float u, float a) { return -std::log(1 - u) / a; }

// Piecewise-constant 1D distribution over [0,1].
class Distribution1D {
  public:
    Distribution1D() = default;
    explicit Distribution1D(std::span<const float> f);

    int Count() const { return int(func_.size()); }
    float Integral() const { return funcInt_; }
    float Func(int i) const { return func_[i]; }
    // Continuous sample in [0,1); pdf is w.r.t. the unit interval.
    float SampleContinuous(float u, float *pdf, int *offset = nullptr) const;
    int SampleDiscrete(float u, float *pmf = nullptr, float *uRemapped = nullptr) const;
    float DiscretePMF(int i) const {
        return funcInt_ > 0 ? func_[i] / (funcInt_ * Count()) : 1.f / Count();
    }

  private:
    std::vector<float> func_, cdf_;
    float funcInt_ = 0;
};

// Piecewise-constant 2D distribution over [0,1]^2 (rows = v, columns = u).
class Distribution2D {
  public:
    Distribution2D() = default;
    Distribution2D(const float *f, int nu, int nv);
    Vec2f SampleContinuous(Vec2f u, float *pdf) const;
    float PDF(Vec2f p) const;

  private:
    std::vector<Distribution1D> conditional_;
    Distribution1D marginal_;
};

} // namespace pr

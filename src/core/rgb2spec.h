#pragma once

#include "core/color.h"
#include "core/spectrum.h"

#include <string>
#include <vector>

namespace pr {

// Smooth spectrum s(lambda) = S(c0 lambda^2 + c1 lambda + c2) with S the sigmoid
// S(x) = 1/2 + x / (2 sqrt(1 + x^2)) (Jakob & Hanika 2019).
class RGBSigmoidPolynomial {
  public:
    RGBSigmoidPolynomial() = default;
    RGBSigmoidPolynomial(float c0, float c1, float c2) : c0_(c0), c1_(c1), c2_(c2) {}
    float operator()(float lambda) const { return S(EvaluatePolynomial(lambda, c2_, c1_, c0_)); }
    float MaxValue() const;

  private:
    static float S(float x) {
        if (std::isinf(x)) return x > 0 ? 1.f : 0.f;
        return 0.5f + x / (2 * std::sqrt(1 + x * x));
    }
    float c0_ = 0, c1_ = 0, c2_ = 0;
};

// Lookup table mapping linear sRGB in [0,1]^3 to sigmoid-polynomial coefficients. Built once
// with a Gauss-Newton fit (port of pbrt's rgb2spec_opt) and cached on disk.
class RGBToSpectrumTable {
  public:
    static constexpr int Res = 64;
    // Returns the process-wide table, building (or loading) it on first use.
    static const RGBToSpectrumTable &Get();
    // Optional: directory for the cache file (default: next to the executable, then %LOCALAPPDATA%).
    static void SetCacheDirectory(const std::string &dir);

    RGBSigmoidPolynomial operator()(RGB rgb) const;

  private:
    bool Load(const std::string &path);
    bool Save(const std::string &path) const;
    void Build();
    std::vector<float> zNodes_;
    std::vector<float> coeffs_;  // [3][Res z][Res y][Res x][3]
};

// Reflectance in [0,1] derived from an sRGB triple.
class RGBAlbedoSpectrum : public Spectrum {
  public:
    explicit RGBAlbedoSpectrum(RGB rgb);
    float operator()(float lambda) const override { return rsp_(lambda); }
    float MaxValue() const override { return rsp_.MaxValue(); }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const override {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) s[i] = rsp_(lambda[i]);
        return s;
    }

  private:
    RGBSigmoidPolynomial rsp_;
};

// Unbounded positive spectrum (e.g. scattering coefficients) derived from an sRGB triple.
class RGBUnboundedSpectrum : public Spectrum {
  public:
    explicit RGBUnboundedSpectrum(RGB rgb);
    float operator()(float lambda) const override { return scale_ * rsp_(lambda); }
    float MaxValue() const override { return scale_ * rsp_.MaxValue(); }

  private:
    float scale_ = 1;
    RGBSigmoidPolynomial rsp_;
};

// Emission spectrum: the sRGB colour of a D65-relative illuminant (rgb(1,1,1) -> D65 with Y = 1).
class RGBIlluminantSpectrum : public Spectrum {
  public:
    explicit RGBIlluminantSpectrum(RGB rgb);
    float operator()(float lambda) const override { return scale_ * rsp_(lambda) * StdIlluminantD65()(lambda); }
    float MaxValue() const override { return scale_ * rsp_.MaxValue() * StdIlluminantD65().MaxValue(); }

  private:
    float scale_ = 1;
    RGBSigmoidPolynomial rsp_;
};

// Fast per-sample evaluation helpers used by textures (no heap allocation).
SampledSpectrum RGBAlbedoSample(RGB rgb, const SampledWavelengths &lambda);
SampledSpectrum RGBUnboundedSample(RGB rgb, const SampledWavelengths &lambda);
SampledSpectrum RGBIlluminantSample(RGB rgb, const SampledWavelengths &lambda);

} // namespace pr

#pragma once

#include "core/math.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace pr {

constexpr int NSpectrumSamples = 4;
constexpr float LambdaMin = 360.f, LambdaMax = 830.f;

// ---------------------------------------------------------------------------------------------
// A spectral quantity evaluated at NSpectrumSamples wavelengths.
class SampledSpectrum {
  public:
    SampledSpectrum() { v_.fill(0.f); }
    explicit SampledSpectrum(float c) { v_.fill(c); }

    float operator[](int i) const { return v_[i]; }
    float &operator[](int i) { return v_[i]; }

#define PR_SS_BINOP(op)                                                                          \
    SampledSpectrum operator op(const SampledSpectrum &o) const {                                \
        SampledSpectrum r;                                                                       \
        for (int i = 0; i < NSpectrumSamples; ++i) r.v_[i] = v_[i] op o.v_[i];                  \
        return r;                                                                                \
    }                                                                                            \
    SampledSpectrum &operator op##=(const SampledSpectrum &o) {                                  \
        for (int i = 0; i < NSpectrumSamples; ++i) v_[i] op## = o.v_[i];                        \
        return *this;                                                                            \
    }                                                                                            \
    SampledSpectrum operator op(float s) const {                                                 \
        SampledSpectrum r;                                                                       \
        for (int i = 0; i < NSpectrumSamples; ++i) r.v_[i] = v_[i] op s;                        \
        return r;                                                                                \
    }                                                                                            \
    SampledSpectrum &operator op##=(float s) {                                                   \
        for (int i = 0; i < NSpectrumSamples; ++i) v_[i] op## = s;                              \
        return *this;                                                                            \
    }
    PR_SS_BINOP(+)
    PR_SS_BINOP(-)
    PR_SS_BINOP(*)
#undef PR_SS_BINOP

    // Division treats x/0 as 0 (used for spectral pdf ratios with terminated wavelengths).
    SampledSpectrum operator/(const SampledSpectrum &o) const {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r.v_[i] = o.v_[i] != 0 ? v_[i] / o.v_[i] : 0.f;
        return r;
    }
    SampledSpectrum &operator/=(const SampledSpectrum &o) { return *this = *this / o; }
    SampledSpectrum operator/(float s) const { return *this * (1.f / s); }
    SampledSpectrum &operator/=(float s) { return *this *= (1.f / s); }
    SampledSpectrum operator-() const { return *this * -1.f; }

    explicit operator bool() const {
        for (float x : v_)
            if (x != 0) return true;
        return false;
    }
    bool IsZero() const { return !bool(*this); }
    float MaxComponentValue() const { return std::max(std::max(v_[0], v_[1]), std::max(v_[2], v_[3])); }
    float MinComponentValue() const { return std::min(std::min(v_[0], v_[1]), std::min(v_[2], v_[3])); }
    float Average() const { return (v_[0] + v_[1] + v_[2] + v_[3]) / NSpectrumSamples; }
    bool HasNaNs() const {
        for (float x : v_)
            if (std::isnan(x) || std::isinf(x)) return true;
        return false;
    }

  private:
    std::array<float, NSpectrumSamples> v_;
};
inline SampledSpectrum operator*(float s, const SampledSpectrum &v) { return v * s; }
inline SampledSpectrum Exp(const SampledSpectrum &s) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = std::exp(s[i]);
    return r;
}
inline SampledSpectrum Sqrt(const SampledSpectrum &s) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = SafeSqrt(s[i]);
    return r;
}
inline SampledSpectrum ClampZero(const SampledSpectrum &s) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = std::max(0.f, s[i]);
    return r;
}

// ---------------------------------------------------------------------------------------------
// Hero-wavelength sampled wavelengths (Wilkie et al. 2014).
class SampledWavelengths {
  public:
    static SampledWavelengths SampleUniform(float u);
    // Importance sample the visual response (pbrt-v4 distribution).
    static SampledWavelengths SampleVisible(float u);

    float operator[](int i) const { return lambda_[i]; }
    SampledSpectrum PDF() const {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = pdf_[i];
        return r;
    }
    // Called when a wavelength-dependent (dispersive) event happens: only the hero wavelength
    // survives, and its pdf is scaled so that the estimator remains unbiased.
    void TerminateSecondary() {
        if (SecondaryTerminated()) return;
        for (int i = 1; i < NSpectrumSamples; ++i) pdf_[i] = 0;
        pdf_[0] /= NSpectrumSamples;
    }
    bool SecondaryTerminated() const {
        for (int i = 1; i < NSpectrumSamples; ++i)
            if (pdf_[i] != 0) return false;
        return true;
    }

  private:
    std::array<float, NSpectrumSamples> lambda_{}, pdf_{};
};

float SampleVisibleWavelength(float u);
float VisibleWavelengthPDF(float lambda);

// ---------------------------------------------------------------------------------------------
// CIE 1931 2-degree colour matching functions (multi-lobe Gaussian fit, Wyman et al. 2013).
float CIE_X(float lambda);
float CIE_Y(float lambda);
float CIE_Z(float lambda);
// Integral of CIE_Y over [LambdaMin, LambdaMax] (1nm Riemann sum).
float CIE_Y_Integral();

struct XYZ {
    float x = 0, y = 0, z = 0;
    XYZ operator+(const XYZ &o) const { return {x + o.x, y + o.y, z + o.z}; }
    XYZ operator*(float s) const { return {x * s, y * s, z * s}; }
};

// Convert a sampled spectral radiance estimate to XYZ (Monte Carlo over wavelengths).
XYZ ToXYZ(const SampledSpectrum &s, const SampledWavelengths &lambda);
inline float ToY(const SampledSpectrum &s, const SampledWavelengths &lambda) {
    return ToXYZ(s, lambda).y;
}

// ---------------------------------------------------------------------------------------------
// Continuous spectra.
class Spectrum {
  public:
    virtual ~Spectrum() = default;
    virtual float operator()(float lambda) const = 0;
    virtual float MaxValue() const = 0;
    virtual bool IsConstant() const { return false; }
    virtual SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) s[i] = (*this)(lambda[i]);
        return s;
    }
};
using SpectrumPtr = std::shared_ptr<const Spectrum>;

class ConstantSpectrum : public Spectrum {
  public:
    explicit ConstantSpectrum(float c) : c_(c) {}
    float operator()(float) const override { return c_; }
    float MaxValue() const override { return c_; }
    bool IsConstant() const override { return true; }

  private:
    float c_;
};

class PiecewiseLinearSpectrum : public Spectrum {
  public:
    PiecewiseLinearSpectrum(std::vector<float> lambdas, std::vector<float> values);
    float operator()(float lambda) const override;
    float MaxValue() const override;

  private:
    std::vector<float> lambdas_, values_;
};

// Planck blackbody normalized so its peak is 1.
class BlackbodySpectrum : public Spectrum {
  public:
    explicit BlackbodySpectrum(float T);
    float operator()(float lambda) const override;
    float MaxValue() const override { return 1.f; }
    static float Planck(float lambdaNm, float T);

  private:
    float T_, normalizationFactor_;
};

class ScaledSpectrum : public Spectrum {
  public:
    ScaledSpectrum(float scale, SpectrumPtr s) : scale_(scale), s_(std::move(s)) {}
    float operator()(float lambda) const override { return scale_ * (*s_)(lambda); }
    float MaxValue() const override { return scale_ * s_->MaxValue(); }
    bool IsConstant() const override { return s_->IsConstant(); }

  private:
    float scale_;
    SpectrumPtr s_;
};

// Sellmeier dispersion formula n^2 = 1 + sum B_i l^2 / (l^2 - C_i), l in micrometres.
class SellmeierIOR : public Spectrum {
  public:
    SellmeierIOR(std::vector<float> B, std::vector<float> C) : B_(std::move(B)), C_(std::move(C)) {}
    float operator()(float lambda) const override;
    float MaxValue() const override { return (*this)(LambdaMin); }

  private:
    std::vector<float> B_, C_;
};

// Cauchy dispersion n = A + B / lambda^2 (lambda in nm, B in nm^2).
class CauchyIOR : public Spectrum {
  public:
    CauchyIOR(float A, float B) : A_(A), B_(B) {}
    float operator()(float lambda) const override { return A_ + B_ / (lambda * lambda); }
    float MaxValue() const override { return (*this)(LambdaMin); }
    bool IsConstant() const override { return B_ == 0; }

  private:
    float A_, B_;
};

// Luminance-like integral of a spectrum against CIE Y, normalized by CIE_Y_Integral.
float SpectrumToY(const Spectrum &s);
XYZ SpectrumToXYZ(const Spectrum &s);

// The CIE D65 standard illuminant, normalized so that its Y = 1.
const Spectrum &StdIlluminantD65();

// Named spectral data: "metal-Au-eta", "metal-Au-k", "glass-BK7", ... Returns nullptr if unknown.
SpectrumPtr GetNamedSpectrum(const std::string &name);
// Named IOR shortcuts: BK7, SF11, diamond, water, fused_silica, ... Returns nullptr if unknown.
SpectrumPtr GetNamedIOR(const std::string &name);
// Named conductor: Au, Ag, Cu, Al. Returns false if unknown.
bool GetNamedMetal(const std::string &name, SpectrumPtr *eta, SpectrumPtr *k);

} // namespace pr

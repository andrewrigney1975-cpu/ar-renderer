#pragma once

#include "core/math.h"
#include "core/spectrum.h"

#include <optional>

namespace pr {

enum BxDFFlags : uint32_t {
    BxDF_Unset = 0,
    BxDF_Reflection = 1 << 0,
    BxDF_Transmission = 1 << 1,
    BxDF_Diffuse = 1 << 2,
    BxDF_Glossy = 1 << 3,
    BxDF_Specular = 1 << 4,
    BxDF_DiffuseReflection = BxDF_Diffuse | BxDF_Reflection,
    BxDF_DiffuseTransmission = BxDF_Diffuse | BxDF_Transmission,
    BxDF_GlossyReflection = BxDF_Glossy | BxDF_Reflection,
    BxDF_GlossyTransmission = BxDF_Glossy | BxDF_Transmission,
    BxDF_SpecularReflection = BxDF_Specular | BxDF_Reflection,
    BxDF_SpecularTransmission = BxDF_Specular | BxDF_Transmission,
    BxDF_All = BxDF_Diffuse | BxDF_Glossy | BxDF_Specular | BxDF_Reflection | BxDF_Transmission
};
inline BxDFFlags operator|(BxDFFlags a, BxDFFlags b) { return BxDFFlags(uint32_t(a) | uint32_t(b)); }
inline BxDFFlags &operator|=(BxDFFlags &a, BxDFFlags b) { return a = a | b; }
inline bool IsReflective(BxDFFlags f) { return f & BxDF_Reflection; }
inline bool IsTransmissive(BxDFFlags f) { return f & BxDF_Transmission; }
inline bool IsDiffuse(BxDFFlags f) { return f & BxDF_Diffuse; }
inline bool IsGlossy(BxDFFlags f) { return f & BxDF_Glossy; }
inline bool IsSpecular(BxDFFlags f) { return f & BxDF_Specular; }
inline bool IsNonSpecular(BxDFFlags f) { return f & (BxDF_Diffuse | BxDF_Glossy); }

enum class TransportMode { Radiance, Importance };
inline TransportMode operator!(TransportMode m) {
    return m == TransportMode::Radiance ? TransportMode::Importance : TransportMode::Radiance;
}

enum class BxDFReflTransFlags : uint32_t { Unset = 0, Reflection = 1, Transmission = 2, All = 3 };
inline bool operator&(BxDFReflTransFlags a, BxDFReflTransFlags b) { return (uint32_t(a) & uint32_t(b)) != 0; }

struct BSDFSample {
    SampledSpectrum f;
    Vec3f wi;
    float pdf = 0;
    BxDFFlags flags = BxDF_Unset;
    float eta = 1;
    // True when pdf is only proportional to the real density (stochastic BSDFs); the integrator
    // must then use BSDF::PDF() for MIS, but f/pdf is still an unbiased throughput estimate.
    bool pdfIsProportional = false;

    BSDFSample() = default;
    BSDFSample(const SampledSpectrum &f, const Vec3f &wi, float pdf, BxDFFlags flags, float eta = 1,
               bool pdfIsProportional = false)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta), pdfIsProportional(pdfIsProportional) {}
    bool IsReflection() const { return pr::IsReflective(flags); }
    bool IsTransmission() const { return pr::IsTransmissive(flags); }
    bool IsSpecular() const { return pr::IsSpecular(flags); }
};

// BxDFs live in per-thread scratch memory and are never destroyed individually, so they must not
// own resources. All directions are in the local shading frame (z = normal).
class BxDF {
  public:
    virtual BxDFFlags Flags() const = 0;
    virtual SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode mode) const = 0;
    virtual std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                               BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const = 0;
    virtual float PDF(Vec3f wo, Vec3f wi, TransportMode mode,
                      BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const = 0;
    // False when PDF() is itself a stochastic estimate (layered BxDFs).
    virtual bool PDFIsExact() const { return true; }
};

// ---------------------------------------------------------------------------------------------
class TrowbridgeReitzDistribution {
  public:
    TrowbridgeReitzDistribution() = default;
    TrowbridgeReitzDistribution(float ax, float ay) : alpha_x(ax), alpha_y(ay) {
        if (!EffectivelySmooth()) {
            // Very small but nonzero alphas cause numerical trouble; clamp them.
            alpha_x = std::max(alpha_x, 1e-4f);
            alpha_y = std::max(alpha_y, 1e-4f);
        }
    }
    // Artist roughness in [0,1] -> alpha (roughness^2, as in OpenPBR/Disney).
    static float RoughnessToAlpha(float r) { return Sqr(Clamp(r, 0.f, 1.f)); }

    bool EffectivelySmooth() const { return std::max(alpha_x, alpha_y) < 1e-3f; }
    float D(const Vec3f &wm) const {
        float tan2Theta = Tan2Theta(wm);
        if (std::isinf(tan2Theta)) return 0;
        float cos4Theta = Sqr(Cos2Theta(wm));
        if (cos4Theta < 1e-16f) return 0;
        float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }
    float Lambda(const Vec3f &w) const {
        float tan2Theta = Tan2Theta(w);
        if (std::isinf(tan2Theta)) return 0;
        float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
        return (std::sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }
    float G1(const Vec3f &w) const { return 1 / (1 + Lambda(w)); }
    float G(const Vec3f &wo, const Vec3f &wi) const { return 1 / (1 + Lambda(wo) + Lambda(wi)); }
    float D(const Vec3f &w, const Vec3f &wm) const {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    float PDF(const Vec3f &w, const Vec3f &wm) const { return D(w, wm); }
    // Sample a visible normal (Heitz 2018), as in pbrt-v4.
    Vec3f Sample_wm(const Vec3f &w, Vec2f u) const;

  private:
    float alpha_x = 0, alpha_y = 0;
};

float FrDielectric(float cosTheta_i, float eta);
SampledSpectrum FrComplex(float cosTheta_i, const SampledSpectrum &eta, const SampledSpectrum &k);

float HenyeyGreenstein(float cosTheta, float g);
Vec3f SampleHenyeyGreenstein(const Vec3f &wo, float g, Vec2f u, float *pdf);

// ---------------------------------------------------------------------------------------------
class DiffuseBxDF : public BxDF {
  public:
    explicit DiffuseBxDF(const SampledSpectrum &R) : R_(R) {}
    BxDFFlags Flags() const override { return R_ ? BxDF_DiffuseReflection : BxDF_Unset; }
    SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode) const override;
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const override;

  private:
    SampledSpectrum R_;
};

class ConductorBxDF : public BxDF {
  public:
    ConductorBxDF(const TrowbridgeReitzDistribution &d, const SampledSpectrum &eta, const SampledSpectrum &k)
        : distrib_(d), eta_(eta), k_(k) {}
    BxDFFlags Flags() const override {
        return distrib_.EffectivelySmooth() ? BxDF_SpecularReflection : BxDF_GlossyReflection;
    }
    SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode) const override;
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const override;

  private:
    TrowbridgeReitzDistribution distrib_;
    SampledSpectrum eta_, k_;
};

class DielectricBxDF : public BxDF {
  public:
    DielectricBxDF(float eta, const TrowbridgeReitzDistribution &d) : eta_(eta), distrib_(d) {}
    BxDFFlags Flags() const override {
        BxDFFlags flags = (eta_ == 1) ? BxDF_Transmission : (BxDF_Reflection | BxDF_Transmission);
        return flags | (distrib_.EffectivelySmooth() ? BxDF_Specular : BxDF_Glossy);
    }
    SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode mode) const override;
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const override;

  private:
    float eta_;
    TrowbridgeReitzDistribution distrib_;
};

class ThinDielectricBxDF : public BxDF {
  public:
    explicit ThinDielectricBxDF(float eta) : eta_(eta) {}
    BxDFFlags Flags() const override { return BxDF_Reflection | BxDF_Transmission | BxDF_Specular; }
    SampledSpectrum f(Vec3f, Vec3f, TransportMode) const override { return SampledSpectrum(0.f); }
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f, Vec3f, TransportMode, BxDFReflTransFlags) const override { return 0; }

  private:
    float eta_;
};

// Stochastic layered BxDF (Guo et al. 2018 position-free Monte Carlo, as in pbrt-v4): a top
// interface over a bottom BxDF with an optional scattering medium of the given albedo between.
// Russian roulette keeps the random walk unbiased; maxDepth is only a safety limit.
class LayeredBxDF : public BxDF {
  public:
    LayeredBxDF(const BxDF *top, const BxDF *bottom, float thickness, const SampledSpectrum &albedo,
                float g, bool twoSided, int maxDepth = 64, int nSamples = 1)
        : top_(top), bottom_(bottom), thickness_(std::max(thickness, std::numeric_limits<float>::min())),
          g_(g), albedo_(albedo), twoSided_(twoSided), maxDepth_(maxDepth), nSamples_(nSamples) {}
    BxDFFlags Flags() const override;
    SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode mode) const override;
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const override;
    bool PDFIsExact() const override { return false; }

  private:
    static float Tr(float dz, const Vec3f &w) {
        if (std::abs(dz) <= std::numeric_limits<float>::min()) return 1;
        return std::exp(-std::abs(dz / w.z));
    }
    const BxDF *top_;
    const BxDF *bottom_;
    float thickness_, g_;
    SampledSpectrum albedo_;
    bool twoSided_;
    int maxDepth_, nSamples_;
};

// Adds a "Charlie" sheen lobe (Estevez & Kulla 2017, Neubelt-Pettineo visibility) on top of a
// base BxDF. The base is attenuated by the sheen's directional albedo to conserve energy.
class SheenBxDF : public BxDF {
  public:
    SheenBxDF(const BxDF *base, const SampledSpectrum &color, float roughness, float weight);
    BxDFFlags Flags() const override { return base_->Flags() | BxDF_GlossyReflection; }
    SampledSpectrum f(Vec3f wo, Vec3f wi, TransportMode mode) const override;
    std::optional<BSDFSample> Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                       BxDFReflTransFlags sampleFlags) const override;
    float PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const override;
    bool PDFIsExact() const override { return base_->PDFIsExact(); }

    static float Albedo(float cosTheta, float alpha);

  private:
    SampledSpectrum SheenF(const Vec3f &wo, const Vec3f &wi) const;
    float BaseScale(const Vec3f &wo, const Vec3f &wi) const;
    float SheenProb(const Vec3f &wo) const;
    const BxDF *base_;
    SampledSpectrum color_;
    float alpha_, weight_, colorMax_;
};

} // namespace pr

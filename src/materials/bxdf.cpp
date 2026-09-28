#include "materials/bxdf.h"

#include "core/rng.h"
#include "core/sampling.h"

#include <complex>

namespace pr {

// ---------------------------------------------------------------------------------------------
// Microfacet distribution

static Vec2f SampleUniformDiskPolar(Vec2f u) {
    float r = std::sqrt(u.x);
    float theta = 2 * Pi * u.y;
    return {r * std::cos(theta), r * std::sin(theta)};
}

Vec3f TrowbridgeReitzDistribution::Sample_wm(const Vec3f &w, Vec2f u) const {
    Vec3f wh = Normalize(Vec3f(alpha_x * w.x, alpha_y * w.y, w.z));
    if (wh.z < 0) wh = -wh;
    Vec3f T1 = (wh.z < 0.99999f) ? Normalize(Cross(Vec3f(0, 0, 1), wh)) : Vec3f(1, 0, 0);
    Vec3f T2 = Cross(wh, T1);
    Vec2f p = SampleUniformDiskPolar(u);
    float h = std::sqrt(1 - Sqr(p.x));
    p.y = Lerp((1 + wh.z) / 2, h, p.y);
    float pz = std::sqrt(std::max(0.f, 1 - p.x * p.x - p.y * p.y));
    Vec3f nh = T1 * p.x + T2 * p.y + wh * pz;
    return Normalize(Vec3f(alpha_x * nh.x, alpha_y * nh.y, std::max(1e-6f, nh.z)));
}

float FrDielectric(float cosTheta_i, float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1.f, 1.f);
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1) return 1.f;
    float cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

static float FrComplex1(float cosTheta_i, std::complex<float> eta) {
    using C = std::complex<float>;
    cosTheta_i = Clamp(cosTheta_i, 0.f, 1.f);
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    C sin2Theta_t = sin2Theta_i / (eta * eta);
    C cosTheta_t = std::sqrt(C(1.f) - sin2Theta_t);
    C r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    C r_perp = (C(cosTheta_i) - eta * cosTheta_t) / (C(cosTheta_i) + eta * cosTheta_t);
    return (std::norm(r_parl) + std::norm(r_perp)) / 2;
}

SampledSpectrum FrComplex(float cosTheta_i, const SampledSpectrum &eta, const SampledSpectrum &k) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = FrComplex1(cosTheta_i, std::complex<float>(eta[i], k[i]));
    return r;
}

float HenyeyGreenstein(float cosTheta, float g) {
    float denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

Vec3f SampleHenyeyGreenstein(const Vec3f &wo, float g, Vec2f u, float *pdf) {
    float cosTheta;
    if (std::abs(g) < 1e-3f) cosTheta = 1 - 2 * u.x;
    else cosTheta = -1 / (2 * g) * (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u.x)));
    cosTheta = Clamp(cosTheta, -1.f, 1.f);
    float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    float phi = 2 * Pi * u.y;
    Frame f = Frame::FromZ(wo);
    Vec3f wi = f.FromLocal(SphericalDirection(sinTheta, cosTheta, phi));
    if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
}

// ---------------------------------------------------------------------------------------------
// Diffuse

SampledSpectrum DiffuseBxDF::f(Vec3f wo, Vec3f wi, TransportMode) const {
    if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);
    return R_ * InvPi;
}

std::optional<BSDFSample> DiffuseBxDF::Sample_f(Vec3f wo, float, Vec2f u, TransportMode,
                                                BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return std::nullopt;
    Vec3f wi = SampleCosineHemisphere(u);
    if (wo.z < 0) wi.z *= -1;
    float pdf = CosineHemispherePDF(AbsCosTheta(wi));
    return BSDFSample(R_ * InvPi, wi, pdf, BxDF_DiffuseReflection);
}

float DiffuseBxDF::PDF(Vec3f wo, Vec3f wi, TransportMode, BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection) || !SameHemisphere(wo, wi)) return 0;
    return CosineHemispherePDF(AbsCosTheta(wi));
}

// ---------------------------------------------------------------------------------------------
// Conductor

SampledSpectrum ConductorBxDF::f(Vec3f wo, Vec3f wi, TransportMode) const {
    if (!SameHemisphere(wo, wi) || distrib_.EffectivelySmooth()) return SampledSpectrum(0.f);
    float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
    if (cosTheta_i == 0 || cosTheta_o == 0) return SampledSpectrum(0.f);
    Vec3f wm = wi + wo;
    if (LengthSquared(wm) == 0) return SampledSpectrum(0.f);
    wm = Normalize(wm);
    SampledSpectrum F = FrComplex(AbsDot(wo, wm), eta_, k_);
    return F * (distrib_.D(wm) * distrib_.G(wo, wi) / (4 * cosTheta_i * cosTheta_o));
}

std::optional<BSDFSample> ConductorBxDF::Sample_f(Vec3f wo, float, Vec2f u, TransportMode,
                                                  BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return std::nullopt;
    if (distrib_.EffectivelySmooth()) {
        Vec3f wi(-wo.x, -wo.y, wo.z);
        SampledSpectrum f = FrComplex(AbsCosTheta(wi), eta_, k_) / AbsCosTheta(wi);
        return BSDFSample(f, wi, 1, BxDF_SpecularReflection);
    }
    if (wo.z == 0) return std::nullopt;
    Vec3f wm = distrib_.Sample_wm(wo, u);
    Vec3f wi = Reflect(wo, wm);
    if (!SameHemisphere(wo, wi)) return std::nullopt;
    float pdf = distrib_.PDF(wo, wm) / (4 * AbsDot(wo, wm));
    float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
    if (cosTheta_i == 0 || cosTheta_o == 0) return std::nullopt;
    SampledSpectrum F = FrComplex(AbsDot(wo, wm), eta_, k_);
    SampledSpectrum f = F * (distrib_.D(wm) * distrib_.G(wo, wi) / (4 * cosTheta_i * cosTheta_o));
    return BSDFSample(f, wi, pdf, BxDF_GlossyReflection);
}

float ConductorBxDF::PDF(Vec3f wo, Vec3f wi, TransportMode, BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;
    if (!SameHemisphere(wo, wi) || distrib_.EffectivelySmooth()) return 0;
    Vec3f wm = wo + wi;
    if (LengthSquared(wm) == 0) return 0;
    wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
    return distrib_.PDF(wo, wm) / (4 * AbsDot(wo, wm));
}

// ---------------------------------------------------------------------------------------------
// Dielectric

std::optional<BSDFSample> DielectricBxDF::Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                                   BxDFReflTransFlags sampleFlags) const {
    if (eta_ == 1 || distrib_.EffectivelySmooth()) {
        float R = FrDielectric(CosTheta(wo), eta_), T = 1 - R;
        float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
        if (pr == 0 && pt == 0) return std::nullopt;
        if (uc < pr / (pr + pt)) {
            Vec3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDF_SpecularReflection);
        }
        Vec3f wi;
        float etap;
        if (!Refract(wo, Vec3f(0, 0, 1), eta_, &etap, &wi)) return std::nullopt;
        SampledSpectrum ft(T / AbsCosTheta(wi));
        if (mode == TransportMode::Radiance) ft /= Sqr(etap);
        return BSDFSample(ft, wi, pt / (pr + pt), BxDF_SpecularTransmission, etap);
    }
    Vec3f wm = distrib_.Sample_wm(wo, u);
    float R = FrDielectric(Dot(wo, wm), eta_), T = 1 - R;
    float pr = R, pt = T;
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
    if (pr == 0 && pt == 0) return std::nullopt;
    if (uc < pr / (pr + pt)) {
        Vec3f wi = Reflect(wo, wm);
        if (!SameHemisphere(wo, wi)) return std::nullopt;
        float pdf = distrib_.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
        SampledSpectrum f(distrib_.D(wm) * distrib_.G(wo, wi) * R / (4 * CosTheta(wi) * CosTheta(wo)));
        return BSDFSample(f, wi, pdf, BxDF_GlossyReflection);
    }
    float etap;
    Vec3f wi;
    bool tir = !Refract(wo, wm, eta_, &etap, &wi);
    if (tir || SameHemisphere(wo, wi) || wi.z == 0) return std::nullopt;
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
    float dwm_dwi = AbsDot(wi, wm) / denom;
    float pdf = distrib_.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
    SampledSpectrum ft(T * distrib_.D(wm) * distrib_.G(wo, wi) *
                       std::abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
    if (mode == TransportMode::Radiance) ft /= Sqr(etap);
    return BSDFSample(ft, wi, pdf, BxDF_GlossyTransmission, etap);
}

SampledSpectrum DielectricBxDF::f(Vec3f wo, Vec3f wi, TransportMode mode) const {
    if (eta_ == 1 || distrib_.EffectivelySmooth()) return SampledSpectrum(0.f);
    float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
    bool reflect = cosTheta_i * cosTheta_o > 0;
    float etap = 1;
    if (!reflect) etap = cosTheta_o > 0 ? eta_ : (1 / eta_);
    Vec3f wm = wi * etap + wo;
    if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0) return SampledSpectrum(0.f);
    wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
    if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return SampledSpectrum(0.f);
    float F = FrDielectric(Dot(wo, wm), eta_);
    if (reflect)
        return SampledSpectrum(distrib_.D(wm) * distrib_.G(wo, wi) * F / std::abs(4 * cosTheta_i * cosTheta_o));
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
    float ft = distrib_.D(wm) * (1 - F) * distrib_.G(wo, wi) * std::abs(Dot(wi, wm) * Dot(wo, wm) / denom);
    if (mode == TransportMode::Radiance) ft /= Sqr(etap);
    return SampledSpectrum(ft);
}

float DielectricBxDF::PDF(Vec3f wo, Vec3f wi, TransportMode, BxDFReflTransFlags sampleFlags) const {
    if (eta_ == 1 || distrib_.EffectivelySmooth()) return 0;
    float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
    bool reflect = cosTheta_i * cosTheta_o > 0;
    float etap = 1;
    if (!reflect) etap = cosTheta_o > 0 ? eta_ : (1 / eta_);
    Vec3f wm = wi * etap + wo;
    if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0) return 0;
    wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
    if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return 0;
    float R = FrDielectric(Dot(wo, wm), eta_), T = 1 - R;
    float pr = R, pt = T;
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
    if (pr == 0 && pt == 0) return 0;
    if (reflect) return distrib_.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
    float dwm_dwi = AbsDot(wi, wm) / denom;
    return distrib_.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
}

// ---------------------------------------------------------------------------------------------
// Thin dielectric

std::optional<BSDFSample> ThinDielectricBxDF::Sample_f(Vec3f wo, float uc, Vec2f, TransportMode,
                                                       BxDFReflTransFlags sampleFlags) const {
    float R = FrDielectric(AbsCosTheta(wo), eta_), T = 1 - R;
    if (R < 1) {
        R += Sqr(T) * R / (1 - Sqr(R));
        T = 1 - R;
    }
    float pr = R, pt = T;
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
    if (pr == 0 && pt == 0) return std::nullopt;
    if (uc < pr / (pr + pt)) {
        Vec3f wi(-wo.x, -wo.y, wo.z);
        return BSDFSample(SampledSpectrum(R / AbsCosTheta(wi)), wi, pr / (pr + pt), BxDF_SpecularReflection);
    }
    Vec3f wi = -wo;
    return BSDFSample(SampledSpectrum(T / AbsCosTheta(wi)), wi, pt / (pr + pt), BxDF_SpecularTransmission);
}

// ---------------------------------------------------------------------------------------------
// Layered

BxDFFlags LayeredBxDF::Flags() const {
    BxDFFlags topFlags = top_->Flags(), bottomFlags = bottom_->Flags();
    BxDFFlags flags = BxDF_Reflection;
    if (IsSpecular(topFlags)) flags |= BxDF_Specular;
    if (IsDiffuse(topFlags) || IsDiffuse(bottomFlags) || albedo_) flags |= BxDF_Diffuse;
    else if (IsGlossy(topFlags) || IsGlossy(bottomFlags)) flags |= BxDF_Glossy;
    if (IsTransmissive(topFlags) && IsTransmissive(bottomFlags)) flags |= BxDF_Transmission;
    return flags;
}

SampledSpectrum LayeredBxDF::f(Vec3f wo, Vec3f wi, TransportMode mode) const {
    SampledSpectrum f(0.f);
    if (twoSided_ && wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }
    bool enteredTop = twoSided_ || wo.z > 0;
    const BxDF *enterInterface = enteredTop ? top_ : bottom_;
    const BxDF *exitInterface, *nonExitInterface;
    if (SameHemisphere(wo, wi) ^ enteredTop) {
        exitInterface = bottom_;
        nonExitInterface = top_;
    } else {
        exitInterface = top_;
        nonExitInterface = bottom_;
    }
    float exitZ = (SameHemisphere(wo, wi) ^ enteredTop) ? 0 : thickness_;
    if (SameHemisphere(wo, wi)) f = enterInterface->f(wo, wi, mode) * float(nSamples_);

    RNG rng(Hash(0ull, wo), Hash(wi));
    auto r = [&rng]() { return std::min(rng.UniformFloat(), OneMinusEpsilon); };

    for (int s = 0; s < nSamples_; ++s) {
        float uc = r();
        Vec2f u0(r(), r());
        auto wos = enterInterface->Sample_f(wo, uc, u0, mode, BxDFReflTransFlags::Transmission);
        if (!wos || !wos->f || wos->pdf == 0 || wos->wi.z == 0) continue;
        uc = r();
        Vec2f u1(r(), r());
        auto wis = exitInterface->Sample_f(wi, uc, u1, !mode, BxDFReflTransFlags::Transmission);
        if (!wis || !wis->f || wis->pdf == 0 || wis->wi.z == 0) continue;

        SampledSpectrum beta = wos->f * AbsCosTheta(wos->wi) / wos->pdf;
        float z = enteredTop ? thickness_ : 0;
        Vec3f w = wos->wi;

        for (int depth = 0; depth < maxDepth_; ++depth) {
            if (depth > 3 && beta.MaxComponentValue() < 0.25f) {
                float q = std::max(0.f, 1 - beta.MaxComponentValue());
                if (r() < q) break;
                beta /= 1 - q;
            }
            if (!albedo_) {
                z = (z == thickness_) ? 0 : thickness_;
                beta *= Tr(thickness_, w);
            } else {
                float sigma_t = 1;
                float dz = SampleExponential(r(), sigma_t / std::abs(w.z));
                float zp = w.z > 0 ? (z + dz) : (z - dz);
                if (0 < zp && zp < thickness_) {
                    float wt = 1;
                    float phasePdfExit = HenyeyGreenstein(Dot(-w, -wis->wi), g_);
                    if (!IsSpecular(exitInterface->Flags())) wt = PowerHeuristic(1, wis->pdf, 1, phasePdfExit);
                    f += beta * albedo_ * phasePdfExit * wt * Tr(zp - exitZ, wis->wi) * wis->f / wis->pdf;
                    float phasePdf;
                    Vec3f wiPhase = SampleHenyeyGreenstein(-w, g_, Vec2f(r(), r()), &phasePdf);
                    if (phasePdf == 0 || wiPhase.z == 0) continue;
                    // p / pdf == 1 for HG importance sampling
                    beta *= albedo_;
                    w = wiPhase;
                    z = zp;
                    if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) && !IsSpecular(exitInterface->Flags())) {
                        SampledSpectrum fExit = exitInterface->f(-w, wi, mode);
                        if (fExit) {
                            float exitPDF = exitInterface->PDF(-w, wi, mode, BxDFReflTransFlags::Transmission);
                            float wt2 = PowerHeuristic(1, phasePdf, 1, exitPDF);
                            f += beta * Tr(zp - exitZ, wiPhase) * fExit * wt2;
                        }
                    }
                    continue;
                }
                z = Clamp(zp, 0.f, thickness_);
            }
            if (z == exitZ) {
                float uc2 = r();
                Vec2f u2(r(), r());
                auto bs = exitInterface->Sample_f(-w, uc2, u2, mode, BxDFReflTransFlags::Reflection);
                if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0) break;
                beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                w = bs->wi;
            } else {
                if (!IsSpecular(nonExitInterface->Flags())) {
                    float wt = 1;
                    if (!IsSpecular(exitInterface->Flags()))
                        wt = PowerHeuristic(1, wis->pdf, 1, nonExitInterface->PDF(-w, -wis->wi, mode));
                    f += beta * nonExitInterface->f(-w, -wis->wi, mode) * AbsCosTheta(wis->wi) * wt *
                         Tr(thickness_, wis->wi) * wis->f / wis->pdf;
                }
                float uc2 = r();
                Vec2f u2(r(), r());
                auto bs = nonExitInterface->Sample_f(-w, uc2, u2, mode, BxDFReflTransFlags::Reflection);
                if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0) break;
                beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                w = bs->wi;
                if (!IsSpecular(exitInterface->Flags())) {
                    SampledSpectrum fExit = exitInterface->f(-w, wi, mode);
                    if (fExit) {
                        float wt = 1;
                        if (!IsSpecular(nonExitInterface->Flags())) {
                            float exitPDF = exitInterface->PDF(-w, wi, mode, BxDFReflTransFlags::Transmission);
                            wt = PowerHeuristic(1, bs->pdf, 1, exitPDF);
                        }
                        f += beta * Tr(thickness_, bs->wi) * fExit * wt;
                    }
                }
            }
        }
    }
    return f / float(nSamples_);
}

std::optional<BSDFSample> LayeredBxDF::Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                                BxDFReflTransFlags) const {
    bool flipWi = false;
    if (twoSided_ && wo.z < 0) {
        wo = -wo;
        flipWi = true;
    }
    bool enteredTop = twoSided_ || wo.z > 0;
    auto bs = enteredTop ? top_->Sample_f(wo, uc, u, mode) : bottom_->Sample_f(wo, uc, u, mode);
    if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0) return std::nullopt;
    if (bs->IsReflection()) {
        if (flipWi) bs->wi = -bs->wi;
        bs->pdfIsProportional = true;
        return bs;
    }
    Vec3f w = bs->wi;
    bool specularPath = bs->IsSpecular();

    RNG rng(Hash(0ull, wo), Hash(uc, u));
    auto r = [&rng]() { return std::min(rng.UniformFloat(), OneMinusEpsilon); };

    SampledSpectrum f = bs->f * AbsCosTheta(bs->wi);
    float pdf = bs->pdf;
    float z = enteredTop ? thickness_ : 0;

    for (int depth = 0; depth < maxDepth_; ++depth) {
        float rrBeta = f.MaxComponentValue() / pdf;
        if (depth > 3 && rrBeta < 0.25f) {
            float q = std::max(0.f, 1 - rrBeta);
            if (r() < q) return std::nullopt;
            pdf *= 1 - q;
        }
        if (w.z == 0) return std::nullopt;
        if (albedo_) {
            float sigma_t = 1;
            float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
            float zp = w.z > 0 ? (z + dz) : (z - dz);
            if (zp == z) return std::nullopt;
            if (0 < zp && zp < thickness_) {
                float phasePdf;
                Vec3f wiPhase = SampleHenyeyGreenstein(-w, g_, Vec2f(r(), r()), &phasePdf);
                if (phasePdf == 0 || wiPhase.z == 0) return std::nullopt;
                f *= albedo_ * phasePdf;
                pdf *= phasePdf;
                specularPath = false;
                w = wiPhase;
                z = zp;
                continue;
            }
            z = Clamp(zp, 0.f, thickness_);
        } else {
            z = (z == thickness_) ? 0 : thickness_;
            f *= Tr(thickness_, w);
        }
        const BxDF *iface = (z == 0) ? bottom_ : top_;
        float uc2 = r();
        Vec2f u2(r(), r());
        auto bs2 = iface->Sample_f(-w, uc2, u2, mode);
        if (!bs2 || !bs2->f || bs2->pdf == 0 || bs2->wi.z == 0) return std::nullopt;
        f *= bs2->f;
        pdf *= bs2->pdf;
        specularPath &= bs2->IsSpecular();
        w = bs2->wi;
        if (bs2->IsTransmission()) {
            BxDFFlags flags = SameHemisphere(wo, w) ? BxDF_Reflection : BxDF_Transmission;
            flags |= specularPath ? BxDF_Specular : BxDF_Glossy;
            if (flipWi) w = -w;
            return BSDFSample(f, w, pdf, flags, 1.f, true);
        }
        f *= AbsCosTheta(bs2->wi);
    }
    return std::nullopt;
}

float LayeredBxDF::PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags) const {
    if (twoSided_ && wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }
    RNG rng(Hash(0ull, wi), Hash(wo));
    auto r = [&rng]() { return std::min(rng.UniformFloat(), OneMinusEpsilon); };
    bool enteredTop = twoSided_ || wo.z > 0;
    float pdfSum = 0;
    if (SameHemisphere(wo, wi)) {
        auto reflFlag = BxDFReflTransFlags::Reflection;
        pdfSum += nSamples_ * (enteredTop ? top_->PDF(wo, wi, mode, reflFlag) : bottom_->PDF(wo, wi, mode, reflFlag));
    }
    for (int s = 0; s < nSamples_; ++s) {
        if (SameHemisphere(wo, wi)) {
            const BxDF *rInterface = enteredTop ? bottom_ : top_;
            const BxDF *tInterface = enteredTop ? top_ : bottom_;
            auto trans = BxDFReflTransFlags::Transmission;
            float a = r();
            Vec2f ua(r(), r());
            auto wos = tInterface->Sample_f(wo, a, ua, mode, trans);
            float b = r();
            Vec2f ub(r(), r());
            auto wis = tInterface->Sample_f(wi, b, ub, !mode, trans);
            if (wos && wos->f && wos->pdf > 0 && wis && wis->f && wis->pdf > 0) {
                if (!IsNonSpecular(tInterface->Flags())) {
                    pdfSum += rInterface->PDF(-wos->wi, -wis->wi, mode);
                } else {
                    float c = r();
                    Vec2f uc2(r(), r());
                    auto rs = rInterface->Sample_f(-wos->wi, c, uc2, mode);
                    if (rs && rs->f && rs->pdf > 0) {
                        if (!IsNonSpecular(rInterface->Flags())) {
                            pdfSum += tInterface->PDF(-rs->wi, wi, mode);
                        } else {
                            float rPDF = rInterface->PDF(-wos->wi, -wis->wi, mode);
                            float wt = PowerHeuristic(1, wis->pdf, 1, rPDF);
                            pdfSum += wt * rPDF;
                            float tPDF = tInterface->PDF(-rs->wi, wi, mode);
                            wt = PowerHeuristic(1, rs->pdf, 1, tPDF);
                            pdfSum += wt * tPDF;
                        }
                    }
                }
            }
        } else {
            const BxDF *toInterface = enteredTop ? top_ : bottom_;
            const BxDF *tiInterface = enteredTop ? bottom_ : top_;
            float a = r();
            Vec2f ua(r(), r());
            auto wos = toInterface->Sample_f(wo, a, ua, mode);
            if (!wos || !wos->f || wos->pdf == 0 || wos->wi.z == 0 || wos->IsReflection()) continue;
            float b = r();
            Vec2f ub(r(), r());
            auto wis = tiInterface->Sample_f(wi, b, ub, !mode);
            if (!wis || !wis->f || wis->pdf == 0 || wis->wi.z == 0 || wis->IsReflection()) continue;
            if (IsSpecular(toInterface->Flags())) pdfSum += tiInterface->PDF(-wos->wi, wi, mode);
            else if (IsSpecular(tiInterface->Flags())) pdfSum += toInterface->PDF(wo, -wis->wi, mode);
            else pdfSum += (toInterface->PDF(wo, -wis->wi, mode) + tiInterface->PDF(-wos->wi, wi, mode)) / 2;
        }
    }
    return Lerp(0.9f, 1 / (4 * Pi), pdfSum / nSamples_);
}

// ---------------------------------------------------------------------------------------------
// Sheen

static float CharlieD(float cosThetaH, float alpha) {
    float invAlpha = 1 / alpha;
    float sin2 = std::max(0.f, 1 - cosThetaH * cosThetaH);
    return (2 + invAlpha) * std::pow(sin2, 0.5f * invAlpha) * Inv2Pi;
}

static float SheenValue(const Vec3f &wo, const Vec3f &wi, float alpha) {
    if (!SameHemisphere(wo, wi)) return 0;
    float co = AbsCosTheta(wo), ci = AbsCosTheta(wi);
    if (co == 0 || ci == 0) return 0;
    Vec3f wh = Normalize(wo + wi);
    float D = CharlieD(AbsCosTheta(wh), alpha);
    return D / (4 * (ci + co - ci * co));
}

float SheenBxDF::Albedo(float cosTheta, float alpha) {
    constexpr int NMu = 32, NAlpha = 16, NSamples = 32;
    static const std::vector<float> table = [] {
        std::vector<float> t(NMu * NAlpha);
        for (int ia = 0; ia < NAlpha; ++ia) {
            float a = Lerp(float(ia) / (NAlpha - 1), 0.05f, 1.f);
            for (int im = 0; im < NMu; ++im) {
                float mu = std::max(1e-3f, float(im) / (NMu - 1));
                Vec3f wo(SafeSqrt(1 - mu * mu), 0, mu);
                double sum = 0;
                for (int i = 0; i < NSamples; ++i)
                    for (int j = 0; j < NSamples; ++j) {
                        Vec2f u((i + 0.5f) / NSamples, (j + 0.5f) / NSamples);
                        Vec3f wi = SampleCosineHemisphere(u);
                        // f * cos / pdf with pdf = cos/pi
                        sum += SheenValue(wo, wi, a) * Pi;
                    }
                t[ia * NMu + im] = float(sum / (NSamples * NSamples));
            }
        }
        return t;
    }();
    float fa = (Clamp(alpha, 0.05f, 1.f) - 0.05f) / 0.95f * (NAlpha - 1);
    float fm = Clamp(std::abs(cosTheta), 0.f, 1.f) * (NMu - 1);
    int ia = std::min(int(fa), NAlpha - 2), im = std::min(int(fm), NMu - 2);
    float ta = fa - ia, tm = fm - im;
    auto at = [&](int a, int m) { return table[a * NMu + m]; };
    return Lerp(ta, Lerp(tm, at(ia, im), at(ia, im + 1)), Lerp(tm, at(ia + 1, im), at(ia + 1, im + 1)));
}

SheenBxDF::SheenBxDF(const BxDF *base, const SampledSpectrum &color, float roughness, float weight)
    : base_(base), color_(color), alpha_(Clamp(roughness, 0.05f, 1.f)), weight_(Clamp(weight, 0.f, 1.f)) {
    colorMax_ = Clamp(color.MaxComponentValue(), 0.f, 1.f);
}

SampledSpectrum SheenBxDF::SheenF(const Vec3f &wo, const Vec3f &wi) const {
    return color_ * (weight_ * SheenValue(wo, wi, alpha_));
}

float SheenBxDF::BaseScale(const Vec3f &wo, const Vec3f &wi) const {
    float e = std::max(Albedo(CosTheta(wo), alpha_), Albedo(CosTheta(wi), alpha_));
    return std::max(0.f, 1 - weight_ * colorMax_ * e);
}

float SheenBxDF::SheenProb(const Vec3f &wo) const {
    float e = weight_ * colorMax_ * Albedo(CosTheta(wo), alpha_);
    if (!base_->Flags()) return 1;
    return Clamp(e, 0.f, 0.9f);
}

SampledSpectrum SheenBxDF::f(Vec3f wo, Vec3f wi, TransportMode mode) const {
    return SheenF(wo, wi) + base_->f(wo, wi, mode) * BaseScale(wo, wi);
}

float SheenBxDF::PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFReflTransFlags sampleFlags) const {
    float ps = SheenProb(wo);
    float pdfSheen = (sampleFlags & BxDFReflTransFlags::Reflection) && SameHemisphere(wo, wi)
                         ? CosineHemispherePDF(AbsCosTheta(wi))
                         : 0.f;
    return ps * pdfSheen + (1 - ps) * base_->PDF(wo, wi, mode, sampleFlags);
}

std::optional<BSDFSample> SheenBxDF::Sample_f(Vec3f wo, float uc, Vec2f u, TransportMode mode,
                                              BxDFReflTransFlags sampleFlags) const {
    float ps = SheenProb(wo);
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) ps = 0;
    bool exact = base_->PDFIsExact();
    if (uc < ps) {
        Vec3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0) wi.z = -wi.z;
        if (exact) {
            float pdf = PDF(wo, wi, mode, sampleFlags);
            if (pdf == 0) return std::nullopt;
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDF_GlossyReflection);
        }
        // One-sample lobe-selection estimator (pdf only proportional).
        float pdf = ps * CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(SheenF(wo, wi), wi, pdf, BxDF_GlossyReflection, 1.f, true);
    }
    float uc2 = ps < 1 ? std::min((uc - ps) / (1 - ps), OneMinusEpsilon) : 0.f;
    auto bs = base_->Sample_f(wo, uc2, u, mode, sampleFlags);
    if (!bs || bs->pdf == 0) return std::nullopt;
    if (bs->IsSpecular() || !exact || bs->pdfIsProportional) {
        bs->f *= BaseScale(wo, bs->wi);
        bs->pdf *= (1 - ps);
        if (!bs->IsSpecular()) bs->pdfIsProportional = true;
        return bs;
    }
    float pdf = PDF(wo, bs->wi, mode, sampleFlags);
    if (pdf == 0) return std::nullopt;
    bs->f = f(wo, bs->wi, mode);
    bs->pdf = pdf;
    return bs;
}

} // namespace pr

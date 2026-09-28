#include "materials/material.h"

namespace pr {

BSDF Material::GetBSDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const {
    const BxDF *bxdf = GetBxDF(ctx, lambda, buf);
    if (!bxdf) return {};
    if (sheen_.weight > 0 && sheen_.color) {
        SampledSpectrum c = ClampZero(sheen_.color->Evaluate(ctx, lambda));
        float rough = sheen_.roughness ? sheen_.roughness->Evaluate(ctx) : 0.3f;
        bxdf = buf.New<SheenBxDF>(bxdf, c, rough, sheen_.weight);
    }
    return BSDF(ctx.ns, ctx.dpdus, bxdf);
}

const BxDF *DiffuseMaterial::GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                     ScratchBuffer &buf) const {
    SampledSpectrum r = reflectance_->Evaluate(ctx, lambda);
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = Clamp(r[i], 0.f, 1.f);
    return buf.New<DiffuseBxDF>(r);
}

const BxDF *ConductorMaterial::MakeBxDF(const ConductorParams &p, const MaterialEvalContext &ctx,
                                        const SampledWavelengths &lambda, ScratchBuffer &buf) {
    float ur = p.uRoughness ? p.uRoughness->Evaluate(ctx) : 0.f;
    float vr = p.vRoughness ? p.vRoughness->Evaluate(ctx) : ur;
    TrowbridgeReitzDistribution distrib(TrowbridgeReitzDistribution::RoughnessToAlpha(ur),
                                        TrowbridgeReitzDistribution::RoughnessToAlpha(vr));
    SampledSpectrum etas, ks;
    if (p.eta && p.k) {
        etas = p.eta->Evaluate(ctx, lambda);
        ks = p.k->Evaluate(ctx, lambda);
    } else {
        // Map artist reflectance r to a conductor with eta = 1 and k = 2 sqrt(r) / sqrt(1 - r),
        // which yields exactly r at normal incidence (pbrt-v4).
        SampledSpectrum r = p.reflectance->Evaluate(ctx, lambda);
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = Clamp(r[i], 0.f, 0.9999f);
        etas = SampledSpectrum(1.f);
        ks = Sqrt(r) * 2.f / Sqrt(ClampZero(SampledSpectrum(1.f) - r));
    }
    return buf.New<ConductorBxDF>(distrib, etas, ks);
}

const BxDF *ConductorMaterial::GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                       ScratchBuffer &buf) const {
    return MakeBxDF(p_, ctx, lambda, buf);
}

const BxDF *DielectricMaterial::GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                        ScratchBuffer &buf) const {
    float sampledEta = (*eta_)(lambda[0]);
    if (!eta_->IsConstant()) lambda.TerminateSecondary();
    if (sampledEta == 0) sampledEta = 1;
    if (thin_) return buf.New<ThinDielectricBxDF>(sampledEta);
    float ur = uRough_ ? uRough_->Evaluate(ctx) : 0.f;
    float vr = vRough_ ? vRough_->Evaluate(ctx) : ur;
    TrowbridgeReitzDistribution distrib(TrowbridgeReitzDistribution::RoughnessToAlpha(ur),
                                        TrowbridgeReitzDistribution::RoughnessToAlpha(vr));
    return buf.New<DielectricBxDF>(sampledEta, distrib);
}

static const BxDF *MakeCoatTop(const CoatParams &c, const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                               ScratchBuffer &buf, float *thickness, SampledSpectrum *albedo) {
    float eta = (*c.eta)(lambda[0]);
    if (!c.eta->IsConstant()) lambda.TerminateSecondary();
    if (eta == 0) eta = 1;
    float rough = c.roughness ? c.roughness->Evaluate(ctx) : 0.f;
    float a = TrowbridgeReitzDistribution::RoughnessToAlpha(rough);
    *thickness = c.thickness ? c.thickness->Evaluate(ctx) : 0.01f;
    *albedo = c.albedo ? ClampZero(c.albedo->Evaluate(ctx, lambda)) : SampledSpectrum(0.f);
    return buf.New<DielectricBxDF>(eta, TrowbridgeReitzDistribution(a, a));
}

const BxDF *CoatedDiffuseMaterial::GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                           ScratchBuffer &buf) const {
    float thickness;
    SampledSpectrum albedo;
    const BxDF *top = MakeCoatTop(coat_, ctx, lambda, buf, &thickness, &albedo);
    SampledSpectrum r = reflectance_->Evaluate(ctx, lambda);
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = Clamp(r[i], 0.f, 1.f);
    const BxDF *bottom = buf.New<DiffuseBxDF>(r);
    return buf.New<LayeredBxDF>(top, bottom, thickness, albedo, coat_.g, true, coat_.maxDepth, coat_.nSamples);
}

const BxDF *CoatedConductorMaterial::GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                             ScratchBuffer &buf) const {
    float thickness;
    SampledSpectrum albedo;
    const BxDF *top = MakeCoatTop(coat_, ctx, lambda, buf, &thickness, &albedo);
    // The conductor sits below the coat: its IOR is relative to the coat medium.
    float coatEta = (*coat_.eta)(lambda[0]);
    if (coatEta == 0) coatEta = 1;
    const ConductorParams &p = conductor_;
    float ur = p.uRoughness ? p.uRoughness->Evaluate(ctx) : 0.f;
    float vr = p.vRoughness ? p.vRoughness->Evaluate(ctx) : ur;
    TrowbridgeReitzDistribution distrib(TrowbridgeReitzDistribution::RoughnessToAlpha(ur),
                                        TrowbridgeReitzDistribution::RoughnessToAlpha(vr));
    SampledSpectrum etas, ks;
    if (p.eta && p.k) {
        etas = p.eta->Evaluate(ctx, lambda) / coatEta;
        ks = p.k->Evaluate(ctx, lambda) / coatEta;
    } else {
        SampledSpectrum r = p.reflectance->Evaluate(ctx, lambda);
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = Clamp(r[i], 0.f, 0.9999f);
        etas = SampledSpectrum(1.f);
        ks = Sqrt(r) * 2.f / Sqrt(ClampZero(SampledSpectrum(1.f) - r));
    }
    const BxDF *bottom = buf.New<ConductorBxDF>(distrib, etas, ks);
    return buf.New<LayeredBxDF>(top, bottom, thickness, albedo, coat_.g, true, coat_.maxDepth, coat_.nSamples);
}

} // namespace pr

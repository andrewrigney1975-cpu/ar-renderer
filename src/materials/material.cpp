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

bool Material::PerturbShading(MaterialEvalContext &ctx) const {
    Vec3f ns = ctx.ns;
    bool changed = false;
    if (bumpMap_.height) {
        // Finite-difference gradient of the height field in uv space.
        const float du = 1.f / 2048, dv = 1.f / 2048;
        TextureEvalContext c0 = ctx, cu = ctx, cv = ctx;
        cu.uv.x += du;
        cu.p = ctx.p + ctx.dpdu * du;
        cv.uv.y += dv;
        cv.p = ctx.p + ctx.dpdv * dv;
        float h0 = bumpMap_.height->Evaluate(c0);
        float dhdu = (bumpMap_.height->Evaluate(cu) - h0) / du * bumpMap_.scale;
        float dhdv = (bumpMap_.height->Evaluate(cv) - h0) / dv * bumpMap_.scale;
        Vec3f dpdu = ctx.dpdu + ns * dhdu, dpdv = ctx.dpdv + ns * dhdv;
        Vec3f n = Cross(dpdu, dpdv);
        float l = Length(n);
        if (l > 0 && std::isfinite(l)) {
            n = n / l;
            if (Dot(n, ns) < 0) n = -n;
            ns = n;
            ctx.dpdus = dpdu;
            changed = true;
        }
    }
    if (normalMap_.image) {
        RGB c = normalMap_.image->Lookup(normalMap_.mapping.Map(ctx.uv));
        Vec3f t(2 * c.r - 1, 2 * c.g - 1, 2 * c.b - 1);
        t.x *= normalMap_.strength;
        t.y *= normalMap_.flipGreen ? -normalMap_.strength : normalMap_.strength;
        Frame f = Frame::FromXZ(ctx.dpdus, ns);
        // Bitangent follows the v direction of the parameterization.
        if (Dot(f.y, ctx.dpdv) < 0) f.y = -f.y;
        Vec3f n = f.x * t.x + f.y * t.y + f.z * std::max(t.z, 1e-3f);
        float l = Length(n);
        if (l > 0 && std::isfinite(l)) {
            ns = n / l;
            changed = true;
        }
    }
    if (!changed) return false;
    // Keep the shading normal in the geometric hemisphere.
    if (Dot(ns, ctx.n) < 0) ns = -ns;
    ctx.ns = ns;
    ctx.dpdus = ctx.dpdus - ns * Dot(ctx.dpdus, ns);
    return true;
}

BSDF EvaluateSurface(const Material &m, SurfaceInteraction &si, SampledWavelengths &lambda, ScratchBuffer &buf) {
    MaterialEvalContext ctx = MaterialEvalContext::From(si);
    if (m.PerturbShading(ctx)) {
        si.shading.n = ctx.ns;
        si.shading.dpdu = ctx.dpdus;
    }
    return m.GetBSDF(ctx, lambda, buf);
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

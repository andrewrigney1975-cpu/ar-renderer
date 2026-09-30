#include "integrators/aov.h"

#include "core/memory.h"
#include "core/parallel.h"
#include "integrators/integrator.h"
#include "media/medium.h"

#include <optional>

namespace pr {

bool ParseAOV(const std::string &name, AOVType *out) {
    if (name == "albedo") *out = AOVType::Albedo;
    else if (name == "normal") *out = AOVType::Normal;
    else if (name == "depth" || name == "z") *out = AOVType::Depth;
    else if (name == "position" || name == "p") *out = AOVType::Position;
    else return false;
    return true;
}

namespace {

struct AOVSample {
    RGB value;
    bool hit = false;
};

// Nearest non-interface hit along ray (interfaces only bound media).
std::optional<SurfaceInteraction> NextSurface(const Scene &scene, Ray ray, float *tHit) {
    for (int crossings = 0; crossings < 64; ++crossings) {
        auto si = scene.Intersect(ray, Infinity, tHit);
        if (!si || !si->primitive->IsInterface()) return si;
        ray = si->SpawnRay(ray.d);
    }
    return std::nullopt;
}

// End of a guide chain: the surface where the albedo/normal is taken, and what to use there.
struct ChainEnd {
    SurfaceInteraction si;
    SampledSpectrum throughput{1.f};
    const Medium *enteredMedium = nullptr;  // chain entered a scattering medium at si
    bool valid = false;
};

// Follows perfectly specular scattering from the first hit (when throughSpecular) until a
// non-specular surface, a scattering medium, or a miss.
ChainEnd FollowChain(const Scene &scene, SurfaceInteraction si, bool throughSpecular, SampledWavelengths &lambda,
                     Sampler &sampler, ScratchBuffer &buf) {
    ChainEnd end;
    for (int bounce = 0;; ++bounce) {
        const Primitive *prim = si.primitive;
        if (!throughSpecular || bounce == 8 || !prim->material || prim->areaLight) {
            end.si = si;
            end.valid = true;
            return end;
        }
        SurfaceInteraction sc = si;
        BSDF bsdf = EvaluateSurface(*prim->material, sc, lambda, buf);
        if (!bsdf || bsdf.HasNonSpecular()) {
            end.si = si;
            end.valid = true;
            return end;
        }
        float uc = sampler.Get1D();
        Vec2f u = sampler.Get2D();
        auto bs = bsdf.Sample_f(sc.wo, uc, u);
        if (!bs || bs->pdf == 0) return end;
        end.throughput *= bs->f * (AbsDot(bs->wi, sc.shading.n) / bs->pdf);
        const Medium *m = bs->IsTransmission() ? sc.GetMedium(bs->wi) : nullptr;
        if (m && m->ScatteringAlbedo(lambda)) {
            end.si = sc;
            end.enteredMedium = m;
            end.valid = true;
            return end;
        }
        float t;
        auto next = NextSurface(scene, sc.SpawnRay(bs->wi), &t);
        if (!next) return end;
        si = *next;
    }
}

AOVSample EvalAOV(const Scene &scene, Ray ray, AOVType type, bool throughSpecular, Sampler &sampler, ScratchBuffer &buf) {
    AOVSample out;
    float tHit;
    auto first = NextSurface(scene, ray, &tHit);
    if (!first) return out;
    out.hit = true;
    switch (type) {
    case AOVType::Depth: out.value = RGB(tHit, tHit, tHit); return out;
    case AOVType::Position: out.value = RGB(first->p.x, first->p.y, first->p.z); return out;
    case AOVType::Normal:
    case AOVType::Albedo: break;
    }
    if (type == AOVType::Normal) {
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
        ChainEnd end = FollowChain(scene, *first, throughSpecular, lambda, sampler, buf);
        if (!end.valid) return out;  // escaped through glass: no normal
        SurfaceInteraction &si = end.si;
        if (si.primitive->material) EvaluateSurface(*si.primitive->material, si, lambda, buf);  // normal/bump maps
        Vec3f n = si.shading.n;
        if (Dot(n, si.wo) < 0) n = -n;
        out.value = RGB(n.x, n.y, n.z);
        return out;
    }
    // Directional albedo, estimated with the BSDF's own sampling and expressed as a reflectance
    // colour (under D65 so white stays white). Wavelengths are resampled per estimate so
    // dispersive materials do not produce colour noise.
    const Spectrum &d65 = StdIlluminantD65();
    XYZ xyz;
    const int n = 8;
    for (int i = 0; i < n; ++i) {
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
        ChainEnd end = FollowChain(scene, *first, throughSpecular, lambda, sampler, buf);
        SampledSpectrum a(0.f);
        if (end.valid) {
            const Primitive *prim = end.si.primitive;
            if (end.enteredMedium) {
                a = end.throughput * end.enteredMedium->ScatteringAlbedo(lambda);
            } else if (prim->areaLight) {
                a = end.throughput;
            } else if (prim->material) {
                SurfaceInteraction sc = end.si;
                BSDF bsdf = EvaluateSurface(*prim->material, sc, lambda, buf);
                if (bsdf) {
                    float uc = sampler.Get1D();
                    Vec2f u = sampler.Get2D();
                    auto bs = bsdf.Sample_f(sc.wo, uc, u);
                    if (bs) a = end.throughput * bs->f * (AbsDot(bs->wi, sc.shading.n) / bs->pdf);
                }
            }
        }
        SampledSpectrum lit;
        for (int k = 0; k < NSpectrumSamples; ++k) lit[k] = a[k] * d65(lambda[k]);
        xyz = xyz + ToXYZ(lit, lambda) * (1.f / n);
    }
    RGB ad = SpectralWhiteAdaptation().Apply(xyz.x, xyz.y, xyz.z);
    out.value = XYZToRGBMatrix(ColorSpaceId::sRGB).Apply(ad.r, ad.g, ad.b);
    if (throughSpecular)  // denoisers expect reflectances in [0, 1]
        out.value = RGB(Clamp(out.value.r, 0.f, 1.f), Clamp(out.value.g, 0.f, 1.f), Clamp(out.value.b, 0.f, 1.f));
    return out;
}

} // namespace

Image RenderAOV(const Scene &scene, AOVType type, int spp, uint64_t seed, bool throughSpecular) {
    const FilmSettings &fs = scene.settings.film;
    Image img(fs.width, fs.height);
    int nThreads = ThreadCount();
    std::vector<std::unique_ptr<ScratchBuffer>> bufs(nThreads);
    for (auto &b : bufs) b = std::make_unique<ScratchBuffer>();
    ParallelFor(int64_t(fs.height), [&](int64_t y, int ti) {
        IndependentSampler sampler;
        for (int x = 0; x < fs.width; ++x) {
            double sum[3] = {0, 0, 0};
            int hits = 0;
            float minDepth = Infinity;
            for (int s = 0; s < spp; ++s) {
                sampler.StartPixelSample(uint64_t(y) * fs.width + x, uint64_t(s), seed ^ 0xa0f5ull);
                Vec2f u = sampler.Get2D();
                CameraSample cs{Vec2f(x + u.x, float(y) + u.y), sampler.Get2D()};
                SampledWavelengths lw = SampledWavelengths::SampleVisible(0.5f);
                float cw;
                auto ray = scene.camera->GenerateRay(cs, lw, &cw);
                if (!ray) continue;
                AOVSample a = EvalAOV(scene, *ray, type, throughSpecular, sampler, *bufs[ti]);
                bufs[ti]->Reset();
                if (!a.hit) continue;
                ++hits;
                if (type == AOVType::Depth) minDepth = std::min(minDepth, a.value.r);
                for (int c = 0; c < 3; ++c) sum[c] += a.value[c];
            }
            RGB v;
            if (type == AOVType::Depth) {
                // Nearest depth avoids meaningless averages across silhouettes; misses are +inf.
                float d = hits > 0 ? minDepth : 1e10f;
                v = RGB(d, d, d);
            } else if (type == AOVType::Normal) {
                Vec3f n{float(sum[0]), float(sum[1]), float(sum[2])};
                float l = Length(n);
                if (l > 0) n = n / l;
                v = RGB(n.x, n.y, n.z);
            } else if (type == AOVType::Position) {
                v = hits > 0 ? RGB(float(sum[0] / hits), float(sum[1] / hits), float(sum[2] / hits)) : RGB();
            } else {
                // Coverage-weighted average: background contributes zero.
                v = RGB(float(sum[0] / spp), float(sum[1] / spp), float(sum[2] / spp));
            }
            img.Set(x, int(y), v);
        }
    });
    return img;
}

} // namespace pr

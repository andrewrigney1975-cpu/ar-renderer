#include "integrators/aov.h"

#include "core/memory.h"
#include "core/parallel.h"
#include "integrators/integrator.h"

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

AOVSample EvalAOV(const Scene &scene, Ray ray, AOVType type, Sampler &sampler, ScratchBuffer &buf) {
    AOVSample out;
    for (int crossings = 0; crossings < 64; ++crossings) {
        float tHit;
        auto si = scene.Intersect(ray, Infinity, &tHit);
        if (!si) return out;
        const Primitive *prim = si->primitive;
        if (prim->IsInterface()) {
            ray = si->SpawnRay(ray.d);
            continue;
        }
        out.hit = true;
        switch (type) {
        case AOVType::Depth: out.value = RGB(tHit, tHit, tHit); return out;
        case AOVType::Position: out.value = RGB(si->p.x, si->p.y, si->p.z); return out;
        case AOVType::Normal:
        case AOVType::Albedo: break;
        }
        if (type == AOVType::Normal) {
            SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
            if (prim->material) EvaluateSurface(*prim->material, *si, lambda, buf);  // applies normal/bump maps
            Vec3f n = si->shading.n;
            if (Dot(n, -ray.d) < 0) n = -n;
            out.value = RGB(n.x, n.y, n.z);
            return out;
        }
        if (!prim->material) return out;
        // Directional albedo of the first surface, estimated with the BSDF's own sampling and
        // expressed as a reflectance colour (under D65 so white stays white). Wavelengths are
        // resampled per estimate so dispersive materials do not produce colour noise.
        const Spectrum &d65 = StdIlluminantD65();
        XYZ xyz;
        const int n = 8;
        for (int i = 0; i < n; ++i) {
            SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
            SurfaceInteraction sc = *si;
            BSDF bsdf = EvaluateSurface(*prim->material, sc, lambda, buf);
            SampledSpectrum a(prim->areaLight ? 1.f : 0.f);
            if (bsdf && !prim->areaLight) {
                float uc = sampler.Get1D();
                Vec2f u = sampler.Get2D();
                auto bs = bsdf.Sample_f(sc.wo, uc, u);
                if (bs) a = bs->f * (AbsDot(bs->wi, sc.shading.n) / bs->pdf);
            }
            SampledSpectrum lit;
            for (int k = 0; k < NSpectrumSamples; ++k) lit[k] = a[k] * d65(lambda[k]);
            xyz = xyz + ToXYZ(lit, lambda) * (1.f / n);
        }
        RGB ad = SpectralWhiteAdaptation().Apply(xyz.x, xyz.y, xyz.z);
        out.value = XYZToRGBMatrix(ColorSpaceId::sRGB).Apply(ad.r, ad.g, ad.b);
        return out;
    }
    return out;
}

} // namespace

Image RenderAOV(const Scene &scene, AOVType type, int spp, uint64_t seed) {
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
                AOVSample a = EvalAOV(scene, *ray, type, sampler, *bufs[ti]);
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

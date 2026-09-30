#include "integrators/path.h"

#include "core/log.h"
#include "core/parallel.h"

namespace pr {

int RunProgressivePasses(const Scene &scene, Film &film, RenderControl &control, const IntegratorSettings &s,
                         const char *stage, const PixelSampleFn &fn, std::string *err) {
    (void)scene;
    const int sw = film.SampleWidth(), sh = film.SampleHeight();
    const int tile = 16;
    const int tilesX = (sw + tile - 1) / tile, tilesY = (sh + tile - 1) / tile;
    const int targetSpp = s.timeLimit > 0 ? std::numeric_limits<int>::max() : std::max(1, s.spp);
    int done = 0;
    // Resuming continues the per-pixel sample sequence where the checkpoint stopped, so a resumed
    // render is statistically identical to an uninterrupted one.
    if (!control.resumePath.empty()) {
        int32_t passes = 0;
        if (!ReadCheckpoint(control, s, film, [&](BinaryReader &r) { return r.Get(&passes); }, err)) return -1;
        done = passes;
        LogInfo("resumed at {} samples per pixel", done);
    }
    auto checkpoint = [&] {
        std::string e;
        if (!WriteCheckpoint(control, s, film, [&](BinaryWriter &w) { w.Put(int32_t(done)); }, &e)) LogWarning("{}", e);
    };
    while (done < targetSpp && !control.cancel) {
        int sampleIndex = done;
        ParallelFor(int64_t(tilesX) * tilesY, [&](int64_t t, int threadIndex) {
            int tx = int(t % tilesX), ty = int(t / tilesX);
            for (int py = ty * tile; py < std::min(sh, (ty + 1) * tile); ++py)
                for (int px = tx * tile; px < std::min(sw, (tx + 1) * tile); ++px) fn(px, py, sampleIndex, threadIndex);
        });
        ++done;
        double elapsed = control.Elapsed();
        ProgressInfo pi;
        pi.stage = stage;
        pi.elapsed = elapsed;
        pi.samplesPerPixel = done;
        pi.fraction = s.timeLimit > 0 ? std::min(1.0, elapsed / s.timeLimit) : double(done) / targetSpp;
        control.Progress(pi, done == targetSpp);
        control.MaybePreview(film, 1.0 / done);
        if (control.CheckpointDue()) checkpoint();
        if (s.timeLimit > 0 && elapsed >= s.timeLimit) break;
    }
    checkpoint();
    return done;
}

bool PathIntegrator::Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) {
    if (!scene.camera) {
        *err = "no camera";
        return false;
    }
    int nThreads = ThreadCount();
    std::vector<std::unique_ptr<ScratchBuffer>> scratch(nThreads);
    for (auto &b : scratch) b = std::make_unique<ScratchBuffer>();
    const int m = film.Margin();
    const int sw = film.SampleWidth();
    int spp = RunProgressivePasses(scene, film, control, settings_, "render", [&](int px, int py, int si, int ti) {
        ScratchBuffer &buf = *scratch[ti];
        IndependentSampler sampler;
        sampler.StartPixelSample(uint64_t(py) * sw + px, uint64_t(si), seed_);
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
        Vec2f u = sampler.Get2D();
        Vec2f pFilm(px - m + u.x, py - m + u.y);
        CameraSample cs{pFilm, sampler.Get2D()};
        float weight = 1;
        auto ray = scene.camera->GenerateRay(cs, lambda, &weight);
        if (ray && weight > 0) {
            SampledSpectrum L = Li(scene, *ray, lambda, sampler, buf) * weight;
            if (!L.HasNaNs()) film.AddSample(pFilm, L, lambda);
        }
        buf.Reset();
    }, err);
    if (spp < 0) return false;
    *scale = spp > 0 ? 1.0 / spp : 0.0;
    return true;
}

SampledSpectrum PathIntegrator::SampleLd(const Scene &scene, const Interaction &intr, const BSDF *bsdf, float g,
                                         const Vec3f &ns, SampledWavelengths &lambda, Sampler &sampler) const {
    float uLight = sampler.Get1D();
    Vec2f uL = sampler.Get2D();
    float pmf;
    LightSampleContext ctx(intr);
    const Light *light = settings_.lightBVH ? scene.lightBVH.Sample(ctx.p, ctx.n, uLight, &pmf)
                                            : scene.lightSampler.Sample(uLight, &pmf);
    if (!light || pmf == 0) return SampledSpectrum(0.f);
    auto ls = light->SampleLi(ctx, uL, lambda);
    if (!ls || !ls->L || ls->pdf == 0) return SampledSpectrum(0.f);
    Vec3f wo = intr.wo, wi = ls->wi;
    SampledSpectrum f;
    float scatterPdf;
    if (bsdf) {
        f = bsdf->f(wo, wi) * AbsDot(wi, ns);
        if (!f) return SampledSpectrum(0.f);
        scatterPdf = bsdf->PDF(wo, wi);
    } else {
        float p = HenyeyGreenstein(Dot(wo, wi), g);
        f = SampledSpectrum(p);
        scatterPdf = p;
    }
    SampledSpectrum Tr = scene.Tr(intr, ls->pLight, lambda);
    if (!Tr) return SampledSpectrum(0.f);
    float lightPdf = pmf * ls->pdf;
    if (IsDeltaLight(light->Type())) return f * Tr * ls->L / lightPdf;
    float w = PowerHeuristic(1, lightPdf, 1, scatterPdf);
    return f * Tr * ls->L * (w / lightPdf);
}

SampledSpectrum PathIntegrator::Li(const Scene &scene, Ray ray, SampledWavelengths &lambda, Sampler &sampler,
                                   ScratchBuffer &buf) const {
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = false;
    int depth = 0;
    float etaScale = 1;
    float prevPdf = 1;
    LightSampleContext prevCtx;
    const Medium *medium = scene.camera->medium;
    const int maxDepth = settings_.maxDepth > 0 ? settings_.maxDepth : std::numeric_limits<int>::max();
    int interfaceCrossings = 0;

    // Path-level spectral MIS for media: distances are sampled with the hero wavelength and
    // rPath[i] = (path pdf had wavelength i been the hero) / (hero path pdf). Each contribution is
    // divided by rPath.Average(), which keeps per-wavelength weights bounded however many
    // scattering events a chromatic medium adds. If dispersion later collapses the path to its
    // hero wavelength, only the hero estimate is valid, so the unweighted sum (Lhero) is returned.
    SampledSpectrum rPath(1.f), Lhero(0.f);
    auto add = [&](const SampledSpectrum &c) {
        Lhero += c;
        float a = rPath.Average();
        if (a > 0 && std::isfinite(a)) L += c / a;
    };
    // Throughput as it will actually be weighted (for Russian roulette).
    auto effectiveBeta = [&] {
        if (lambda.SecondaryTerminated()) return beta;
        float a = rPath.Average();
        return a > 0 && std::isfinite(a) ? beta / a : SampledSpectrum(0.f);
    };

    auto addEmitted = [&](const Light *light, const SampledSpectrum &Le, const Vec3f &dir) {
        if (!Le) return;
        if (depth == 0 || specularBounce) {
            add(beta * Le);
        } else {
            float pmf = settings_.lightBVH ? scene.lightBVH.PMF(prevCtx.p, prevCtx.n, light) : scene.lightSampler.PMF(light);
            float lightPdf = pmf * light->PDF_Li(prevCtx, dir);
            float w = PowerHeuristic(1, prevPdf, 1, lightPdf);
            add(beta * Le * w);
        }
    };

    while (true) {
        float tHit = Infinity;
        auto si = scene.Intersect(ray, Infinity, &tHit);

        if (medium) {
            auto ds = medium->SampleDistanceHero(ray, si ? tHit : Infinity, sampler.Get1D(), lambda);
            beta *= ds.weight;
            rPath *= ds.pdfRatio;
            if (!beta) break;
            if (ds.scattered) {
                if (depth++ >= maxDepth) break;
                MediumInteraction mi;
                mi.p = ray(ds.t);
                mi.wo = -ray.d;
                mi.medium = medium;
                mi.g = medium->G();
                add(beta * SampleLd(scene, mi, nullptr, mi.g, Vec3f(), lambda, sampler));
                float phasePdf;
                Vec3f wi = SampleHenyeyGreenstein(mi.wo, mi.g, sampler.Get2D(), &phasePdf);
                if (phasePdf == 0) break;
                // Phase function importance sampling: p / pdf == 1.
                prevPdf = phasePdf;
                prevCtx = LightSampleContext(mi);
                specularBounce = false;
                ray = Ray(mi.p, wi);
                // Russian roulette.
                if (settings_.russianRoulette && depth > 1) {
                    float m = effectiveBeta().MaxComponentValue() * etaScale;
                    if (m < 1) {
                        float q = std::max(0.f, 1 - m);
                        if (sampler.Get1D() < q) break;
                        beta /= 1 - q;
                    }
                }
                continue;
            }
        }

        if (!si) {
            for (const Light *light : scene.infiniteLights) addEmitted(light, light->Le(ray, lambda), ray.d);
            break;
        }

        const Primitive *prim = si->primitive;
        if (prim->areaLight) addEmitted(prim->areaLight, prim->areaLight->L(si->p, si->n, si->uv, -ray.d, lambda), ray.d);

        if (prim->IsInterface()) {
            if (++interfaceCrossings > 1000) break;
            medium = si->GetMedium(ray.d);
            ray = si->SpawnRay(ray.d);
            continue;
        }

        BSDF bsdf = prim->material ? EvaluateSurface(*prim->material, *si, lambda, buf) : BSDF();
        if (!bsdf) break;
        if (depth++ >= maxDepth) break;

        if (bsdf.HasNonSpecular()) add(beta * SampleLd(scene, *si, &bsdf, 0, si->shading.n, lambda, sampler));

        Vec3f wo = si->wo;
        float uc = sampler.Get1D();
        Vec2f u2 = sampler.Get2D();
        auto bs = bsdf.Sample_f(wo, uc, u2);
        if (!bs) break;
        beta *= bs->f * (AbsDot(bs->wi, si->shading.n) / bs->pdf);
        prevPdf = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        specularBounce = bs->IsSpecular();
        if (bs->IsTransmission()) etaScale *= Sqr(bs->eta);
        prevCtx = LightSampleContext(*si);
        ray = si->SpawnRay(bs->wi);
        medium = si->GetMedium(bs->wi);

        if (!beta || beta.HasNaNs()) break;
        if (settings_.russianRoulette && depth > 1) {
            float m = effectiveBeta().MaxComponentValue() * etaScale;
            if (m < 1) {
                float q = std::max(0.f, 1 - m);
                if (sampler.Get1D() < q) break;
                beta /= 1 - q;
            }
        }
    }
    return lambda.SecondaryTerminated() ? Lhero : L;
}

} // namespace pr

#include "integrators/bdpt.h"

#include "core/log.h"
#include "core/parallel.h"
#include "integrators/path.h"

namespace pr {

// ---------------------------------------------------------------------------------------------
// Vertex helpers

static float CorrectShadingNormal(const SurfaceInteraction &si, const Vec3f &wo, const Vec3f &wi, TransportMode mode) {
    if (mode == TransportMode::Importance) {
        float num = AbsDot(wo, si.shading.n) * AbsDot(wi, si.n);
        float denom = AbsDot(wo, si.n) * AbsDot(wi, si.shading.n);
        if (denom == 0) return 0;
        return num / denom;
    }
    return 1;
}

static float InfiniteLightDensity(const Scene &scene, const Vec3f &w) {
    float pdf = 0;
    for (const Light *light : scene.infiniteLights)
        pdf += light->PDF_Li(LightSampleContext(), -w) * scene.lightSampler.PMF(light);
    return pdf;
}

bool Vertex::IsConnectible() const {
    switch (type) {
    case VertexType::Medium: return true;
    case VertexType::Light: return !light || light->Type() != LightType::DeltaDirection;
    case VertexType::Camera: return true;
    case VertexType::Surface: return bool(bsdf) && bsdf.HasNonSpecular();
    }
    return false;
}

SampledSpectrum Vertex::f(const Vertex &next, TransportMode mode) const {
    Vec3f wi = next.p() - p();
    if (LengthSquared(wi) == 0) return SampledSpectrum(0.f);
    wi = Normalize(wi);
    switch (type) {
    case VertexType::Surface:
        return bsdf.f(si.wo, wi, mode) * CorrectShadingNormal(si, si.wo, wi, mode);
    case VertexType::Medium: return SampledSpectrum(HenyeyGreenstein(Dot(si.wo, wi), g));
    default: return SampledSpectrum(0.f);
    }
}

float Vertex::ConvertDensity(float pdf, const Vertex &next) const {
    if (next.IsInfiniteLight()) return pdf;
    Vec3f w = next.p() - p();
    float d2 = LengthSquared(w);
    if (d2 == 0) return 0;
    float invDist2 = 1 / d2;
    if (next.IsOnSurface()) pdf *= AbsDot(next.ng(), w * std::sqrt(invDist2));
    return pdf * invDist2;
}

float Vertex::Pdf(const Scene &scene, const Vertex *prev, const Vertex &next) const {
    if (type == VertexType::Light) return PdfLight(scene, next);
    Vec3f wn = next.p() - p();
    if (LengthSquared(wn) == 0) return 0;
    wn = Normalize(wn);
    Vec3f wp;
    if (prev) {
        wp = prev->p() - p();
        if (LengthSquared(wp) == 0) return 0;
        wp = Normalize(wp);
    }
    float pdf = 0, unused;
    if (type == VertexType::Camera) camera->PDF_We(Ray(p(), wn), &unused, &pdf);
    else if (type == VertexType::Surface) pdf = bsdf.PDF(wp, wn);
    else if (type == VertexType::Medium) pdf = HenyeyGreenstein(Dot(wp, wn), g);
    return ConvertDensity(pdf, next);
}

float Vertex::PdfLight(const Scene &scene, const Vertex &v) const {
    Vec3f w = v.p() - p();
    float d2 = LengthSquared(w);
    if (d2 == 0) return 0;
    float invDist2 = 1 / d2;
    w = w * std::sqrt(invDist2);
    float pdf;
    if (IsInfiniteLight()) {
        pdf = 1 / (Pi * scene.worldRadius * scene.worldRadius);
    } else {
        const Light *l = AreaLight();
        if (!l) return 0;
        float pdfPos, pdfDir;
        l->PDF_Le(Ray(p(), w), ng(), &pdfPos, &pdfDir);
        pdf = pdfDir * invDist2;
    }
    if (v.IsOnSurface()) pdf *= AbsDot(v.ng(), w);
    return pdf;
}

float Vertex::PdfLightOrigin(const Scene &scene, const Vertex &v) const {
    Vec3f w = v.p() - p();
    if (LengthSquared(w) == 0) return 0;
    w = Normalize(w);
    if (IsInfiniteLight()) return InfiniteLightDensity(scene, w);
    const Light *l = AreaLight();
    if (!l) return 0;
    float pdfChoice = scene.lightSampler.PMF(l);
    float pdfPos, pdfDir;
    l->PDF_Le(Ray(p(), w), ng(), &pdfPos, &pdfDir);
    return pdfPos * pdfChoice;
}

SampledSpectrum Vertex::Le(const Scene &scene, const Vertex &v, const SampledWavelengths &lambda) const {
    if (!IsLight()) return SampledSpectrum(0.f);
    Vec3f w = v.p() - p();
    if (LengthSquared(w) == 0) return SampledSpectrum(0.f);
    w = Normalize(w);
    if (IsInfiniteLight()) {
        SampledSpectrum Le(0.f);
        for (const Light *inf : scene.infiniteLights) Le += inf->Le(Ray(p(), -w), lambda);
        return Le;
    }
    const Light *l = AreaLight();
    return l ? l->L(p(), ng(), si.uv, w, lambda) : SampledSpectrum(0.f);
}

static Vertex CreateCamera(const Camera *camera, const Ray &ray, const SampledSpectrum &beta) {
    Vertex v;
    v.type = VertexType::Camera;
    v.camera = camera;
    v.si = SurfaceInteraction();
    v.si.p = ray.o;
    v.si.medium = camera->medium;
    v.beta = beta;
    return v;
}

static Vertex CreateCamera(const Camera *camera, const Interaction &it, const SampledSpectrum &beta) {
    Vertex v;
    v.type = VertexType::Camera;
    v.camera = camera;
    v.si = SurfaceInteraction();
    static_cast<Interaction &>(v.si) = it;
    v.beta = beta;
    return v;
}

static Vertex CreateLight(const Light *light, const Interaction &it, const SampledSpectrum &Le, float pdf) {
    Vertex v;
    v.type = VertexType::Light;
    v.light = light;
    v.si = SurfaceInteraction();
    static_cast<Interaction &>(v.si) = it;
    v.beta = Le;
    v.pdfFwd = pdf;
    return v;
}

// ---------------------------------------------------------------------------------------------
// Random walks

static int RandomWalk(BDPTContext &ctx, Ray ray, SampledSpectrum beta, float pdf, int maxDepth, TransportMode mode,
                      Vertex *path, const Medium *medium) {
    if (maxDepth == 0) return 0;
    int bounces = 0;
    float pdfFwd = pdf, pdfRev = 0;
    int interfaceCrossings = 0;
    const Scene &scene = ctx.scene;
    while (true) {
        float tHit = Infinity;
        auto si = scene.Intersect(ray, Infinity, &tHit);
        bool scattered = false;
        float tScatter = 0;
        if (medium) {
            auto ds = medium->SampleDistance(ray, si ? tHit : Infinity, ctx.sampler.Get1D(), ctx.lambda);
            beta *= ds.weight;
            scattered = ds.scattered;
            tScatter = ds.t;
        }
        if (!beta || beta.HasNaNs()) break;
        Vertex &vertex = path[bounces];
        Vertex &prev = path[bounces - 1];
        if (scattered) {
            vertex = Vertex();
            vertex.type = VertexType::Medium;
            vertex.si.p = ray(tScatter);
            vertex.si.wo = -ray.d;
            vertex.si.medium = medium;
            vertex.g = medium->G();
            vertex.beta = beta;
            vertex.pdfFwd = prev.ConvertDensity(pdfFwd, vertex);
            if (++bounces >= maxDepth) break;
            float phasePdf;
            Vec3f wi = SampleHenyeyGreenstein(vertex.si.wo, vertex.g, ctx.sampler.Get2D(), &phasePdf);
            if (phasePdf == 0) break;
            pdfFwd = pdfRev = phasePdf;
            ray = Ray(vertex.si.p, wi);
        } else {
            if (!si) {
                if (mode == TransportMode::Radiance) {
                    vertex = Vertex();
                    vertex.type = VertexType::Light;
                    vertex.infinite = true;
                    vertex.si.p = ray.o + ray.d;
                    vertex.beta = beta;
                    vertex.pdfFwd = pdfFwd;
                    ++bounces;
                }
                break;
            }
            if (si->primitive->IsInterface()) {
                if (++interfaceCrossings > 1000) break;
                medium = si->GetMedium(ray.d);
                ray = si->SpawnRay(ray.d);
                continue;
            }
            vertex = Vertex();
            vertex.type = VertexType::Surface;
            vertex.si = *si;
            vertex.bsdf = si->primitive->material
                              ? EvaluateSurface(*si->primitive->material, vertex.si, ctx.lambda, ctx.buf)
                              : BSDF();
            vertex.beta = beta;
            vertex.pdfFwd = prev.ConvertDensity(pdfFwd, vertex);
            if (++bounces >= maxDepth) break;
            if (!vertex.bsdf) break;
            Vec3f wo = si->wo;
            float uc = ctx.sampler.Get1D();
            Vec2f u2 = ctx.sampler.Get2D();
            auto bs = vertex.bsdf.Sample_f(wo, uc, u2, mode);
            if (!bs) break;
            pdfFwd = bs->pdfIsProportional ? vertex.bsdf.PDF(wo, bs->wi, mode) : bs->pdf;
            beta *= bs->f * (AbsDot(bs->wi, vertex.si.shading.n) / bs->pdf);
            pdfRev = vertex.bsdf.PDF(bs->wi, wo, !mode);
            if (bs->IsSpecular()) {
                vertex.delta = true;
                pdfRev = pdfFwd = 0;
            }
            beta *= CorrectShadingNormal(vertex.si, wo, bs->wi, mode);
            ray = si->SpawnRay(bs->wi);
            medium = si->GetMedium(bs->wi);
        }
        prev.pdfRev = vertex.ConvertDensity(pdfRev, prev);
        if (ctx.russianRoulette && bounces > 3) {
            float m = beta.MaxComponentValue();
            if (m < 1) {
                float q = std::max(0.f, 1 - m);
                if (ctx.sampler.Get1D() < q) break;
                beta /= 1 - q;
            }
        }
    }
    return bounces;
}

int GenerateCameraSubpath(BDPTContext &ctx, int maxDepth, Vec2f pFilm, Vertex *path) {
    if (maxDepth == 0) return 0;
    const Camera *camera = ctx.scene.camera.get();
    CameraSample cs{pFilm, ctx.sampler.Get2D()};
    auto ray = camera->GenerateRay(cs);
    if (!ray) return 0;
    SampledSpectrum beta(1.f);
    path[0] = CreateCamera(camera, *ray, beta);
    float pdfPos, pdfDir;
    camera->PDF_We(*ray, &pdfPos, &pdfDir);
    if (pdfDir == 0) return 0;
    return RandomWalk(ctx, *ray, beta, pdfDir, maxDepth - 1, TransportMode::Radiance, path + 1, camera->medium) + 1;
}

int GenerateLightSubpath(BDPTContext &ctx, int maxDepth, Vertex *path) {
    if (maxDepth == 0) return 0;
    const Scene &scene = ctx.scene;
    float lightPmf;
    const Light *light = scene.lightSampler.Sample(ctx.sampler.Get1D(), &lightPmf);
    if (!light || lightPmf == 0) return 0;
    Vec2f u1 = ctx.sampler.Get2D();
    Vec2f u2 = ctx.sampler.Get2D();
    auto les = light->SampleLe(u1, u2, ctx.lambda);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L) return 0;
    const Ray &ray = les->ray;
    path[0] = CreateLight(light, les->intr, les->L, les->pdfPos * lightPmf);
    if (light->Type() == LightType::Infinite) path[0].infinite = true;
    float cosLight = les->intr.n.IsZero() ? 1.f : AbsDot(les->intr.n, ray.d);
    SampledSpectrum beta = les->L * (cosLight / (lightPmf * les->pdfPos * les->pdfDir));
    const Medium *medium = les->intr.GetMedium(ray.d);
    int nVertices = RandomWalk(ctx, ray, beta, les->pdfDir, maxDepth - 1, TransportMode::Importance, path + 1, medium);
    if (path[0].IsInfiniteLight()) {
        if (nVertices > 0) {
            path[1].pdfFwd = les->pdfPos;
            if (path[1].IsOnSurface()) path[1].pdfFwd *= AbsDot(ray.d, path[1].ng());
        }
        path[0].pdfFwd = InfiniteLightDensity(scene, ray.d);
    }
    return nVertices + 1;
}

// ---------------------------------------------------------------------------------------------
// Connection and MIS

static SampledSpectrum G(BDPTContext &ctx, const Vertex &v0, const Vertex &v1) {
    Vec3f d = v0.p() - v1.p();
    float g = 1 / LengthSquared(d);
    d = d * std::sqrt(g);
    if (v0.IsOnSurface()) g *= AbsDot(v0.ns(), d);
    if (v1.IsOnSurface()) g *= AbsDot(v1.ns(), d);
    return ctx.scene.Tr(v0.si, v1.si, ctx.lambda) * g;
}

static float MISWeight(BDPTContext &ctx, Vertex *lightVertices, Vertex *cameraVertices, Vertex &sampled, int s, int t) {
    if (s + t == 2) return 1;
    const Scene &scene = ctx.scene;
    float sumRi = 0;
    auto remap0 = [](float f) { return f != 0 ? f : 1.f; };

    Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr;
    Vertex *pt = t > 0 ? &cameraVertices[t - 1] : nullptr;
    Vertex *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr;
    Vertex *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

    // Temporarily modify vertex attributes for this connection strategy (restored on exit).
    Vertex saved1;
    Vertex *a1 = nullptr;
    if (s == 1) a1 = qs;
    else if (t == 1) a1 = pt;
    if (a1) {
        saved1 = *a1;
        *a1 = sampled;
    }
    bool savedPtDelta = pt ? pt->delta : false, savedQsDelta = qs ? qs->delta : false;
    if (pt) pt->delta = false;
    if (qs) qs->delta = false;
    float savedPtPdfRev = 0, savedPtMinusPdfRev = 0, savedQsPdfRev = 0, savedQsMinusPdfRev = 0;
    if (pt) {
        savedPtPdfRev = pt->pdfRev;
        pt->pdfRev = s > 0 ? qs->Pdf(scene, qsMinus, *pt) : pt->PdfLightOrigin(scene, *ptMinus);
    }
    if (ptMinus) {
        savedPtMinusPdfRev = ptMinus->pdfRev;
        ptMinus->pdfRev = s > 0 ? pt->Pdf(scene, qs, *ptMinus) : pt->PdfLight(scene, *ptMinus);
    }
    if (qs) {
        savedQsPdfRev = qs->pdfRev;
        qs->pdfRev = pt->Pdf(scene, ptMinus, *qs);
    }
    if (qsMinus) {
        savedQsMinusPdfRev = qsMinus->pdfRev;
        qsMinus->pdfRev = qs->Pdf(scene, pt, *qsMinus);
    }

    float ri = 1;
    for (int i = t - 1; i > 0; --i) {
        ri *= remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta) sumRi += ri;
    }
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex = i > 0 ? lightVertices[i - 1].delta : lightVertices[0].IsDeltaLight();
        if (!lightVertices[i].delta && !deltaLightvertex) sumRi += ri;
    }

    // Restore in reverse order.
    if (qsMinus) qsMinus->pdfRev = savedQsMinusPdfRev;
    if (qs) qs->pdfRev = savedQsPdfRev;
    if (ptMinus) ptMinus->pdfRev = savedPtMinusPdfRev;
    if (pt) pt->pdfRev = savedPtPdfRev;
    if (qs) qs->delta = savedQsDelta;
    if (pt) pt->delta = savedPtDelta;
    if (a1) *a1 = saved1;
    float w = 1 / (1 + sumRi);
    return std::isfinite(w) ? w : 0.f;
}

SampledSpectrum ConnectBDPT(BDPTContext &ctx, Vertex *lightVertices, Vertex *cameraVertices, int s, int t,
                            Vec2f *pRaster, float *misWeightPtr) {
    const Scene &scene = ctx.scene;
    SampledSpectrum L(0.f);
    if (t > 1 && s != 0 && cameraVertices[t - 1].type == VertexType::Light) return L;
    Vertex sampled;
    if (s == 0) {
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsLight()) L = pt.Le(scene, cameraVertices[t - 2], ctx.lambda) * pt.beta;
    } else if (t == 1) {
        const Vertex &qs = lightVertices[s - 1];
        if (qs.IsConnectible()) {
            auto cs = scene.camera->SampleWi(qs.si, ctx.sampler.Get2D());
            if (cs && cs->pdf > 0 && cs->Wi) {
                *pRaster = cs->pRaster;
                sampled = CreateCamera(scene.camera.get(), cs->pLens, cs->Wi / cs->pdf);
                L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
                if (qs.IsOnSurface()) L *= AbsDot(cs->wi, qs.ns());
                if (L) L *= scene.Tr(qs.si, cs->pLens, ctx.lambda);
            }
        }
    } else if (s == 1) {
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsConnectible()) {
            float lightPmf;
            const Light *light = scene.lightSampler.Sample(ctx.sampler.Get1D(), &lightPmf);
            Vec2f u = ctx.sampler.Get2D();
            if (light && lightPmf > 0) {
                LightSampleContext lctx(pt.si);
                lctx.ns = pt.ns();
                auto ls = light->SampleLi(lctx, u, ctx.lambda);
                if (ls && ls->pdf > 0 && ls->L) {
                    sampled = CreateLight(light, ls->pLight, ls->L / (ls->pdf * lightPmf), 0);
                    if (light->Type() == LightType::Infinite) sampled.infinite = true;
                    sampled.pdfFwd = sampled.PdfLightOrigin(scene, pt);
                    L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;
                    if (pt.IsOnSurface()) L *= AbsDot(ls->wi, pt.ns());
                    if (L) L *= scene.Tr(pt.si, ls->pLight, ctx.lambda);
                }
            }
        }
    } else {
        const Vertex &qs = lightVertices[s - 1], &pt = cameraVertices[t - 1];
        if (qs.IsConnectible() && pt.IsConnectible()) {
            L = qs.beta * qs.f(pt, TransportMode::Importance) * pt.f(qs, TransportMode::Radiance) * pt.beta;
            if (L) L *= G(ctx, qs, pt);
        }
    }
    float misWeight = L ? MISWeight(ctx, lightVertices, cameraVertices, sampled, s, t) : 0.f;
    L *= misWeight;
    if (misWeightPtr) *misWeightPtr = misWeight;
    if (L.HasNaNs()) return SampledSpectrum(0.f);
    return L;
}

// ---------------------------------------------------------------------------------------------

bool BDPTIntegrator::Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) {
    if (!scene.camera || !scene.camera->SupportsLightTracing()) {
        *err = "BDPT requires a perspective or thin-lens camera";
        return false;
    }
    int nThreads = ThreadCount();
    const int maxDepth = settings_.maxDepth;
    struct ThreadData {
        ScratchBuffer buf;
        std::vector<Vertex> cam, light;
    };
    std::vector<std::unique_ptr<ThreadData>> td(nThreads);
    for (auto &t : td) {
        t = std::make_unique<ThreadData>();
        t->cam.resize(maxDepth + 2);
        t->light.resize(maxDepth + 1);
    }
    const int m = film.Margin();
    const int sw = film.SampleWidth();
    int spp = RunProgressivePasses(scene, film, control, settings_, "render", [&](int px, int py, int si, int ti) {
        ThreadData &d = *td[ti];
        IndependentSampler sampler;
        sampler.StartPixelSample(uint64_t(py) * sw + px, uint64_t(si), seed_);
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
        Vec2f u = sampler.Get2D();
        Vec2f pFilm(px - m + u.x, py - m + u.y);
        BDPTContext ctx{scene, lambda, sampler, d.buf, settings_.russianRoulette};
        int nCamera = GenerateCameraSubpath(ctx, maxDepth + 2, pFilm, d.cam.data());
        int nLight = GenerateLightSubpath(ctx, maxDepth + 1, d.light.data());
        SampledSpectrum L(0.f);
        for (int t = 1; t <= nCamera; ++t) {
            for (int s = 0; s <= nLight; ++s) {
                int depth = t + s - 2;
                if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth) continue;
                Vec2f pFilmNew = pFilm;
                SampledSpectrum Lpath = ConnectBDPT(ctx, d.light.data(), d.cam.data(), s, t, &pFilmNew);
                if (t != 1) L += Lpath;
                else if (Lpath) film.AddSample(pFilmNew, Lpath, lambda);
            }
        }
        film.AddSample(pFilm, L, lambda);
        d.buf.Reset();
    }, err);
    if (spp < 0) return false;
    *scale = spp > 0 ? 1.0 / spp : 0.0;
    return true;
}

} // namespace pr

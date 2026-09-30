#include "integrators/gpu_path.h"
#include "integrators/path.h"

#include "cameras/camera.h"
#include "core/log.h"
#include "core/parallel.h"
#include "core/rgb2spec.h"
#include "gpu/gpu_api.h"
#include "materials/bxdf.h"
#include "materials/material.h"
#include "media/medium.h"

#include <map>
#include <mutex>
#include <set>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef RGB
#endif

#include <chrono>
#include <filesystem>

namespace pr {

// ---------------------------------------------------------------------------------------------
// Module loading

namespace {

struct GpuModule {
    bool loaded = false;
    std::string status;
    int (*listDevices)(prgpu::DeviceInfo *, int) = nullptr;
    int (*render)(int, const prgpu::SceneDesc *, const prgpu::RenderParams *, float *, char *, int) = nullptr;
};

GpuModule &Module() {
    static GpuModule m = [] {
        GpuModule g;
#ifdef _WIN32
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        std::filesystem::path dir = std::filesystem::path(std::wstring(buf, n)).parent_path();
        // The SYCL runtime DLLs are expected next to the module (or on PATH).
        SetDllDirectoryW(dir.wstring().c_str());
        HMODULE h = LoadLibraryW((dir / L"prender_gpu.dll").wstring().c_str());
        if (!h) {
            g.status = "prender_gpu.dll not found or its oneAPI runtime is missing";
            return g;
        }
        auto version = reinterpret_cast<int (*)()>(GetProcAddress(h, "prgpu_api_version"));
        g.listDevices = reinterpret_cast<decltype(g.listDevices)>(GetProcAddress(h, "prgpu_list_devices"));
        g.render = reinterpret_cast<decltype(g.render)>(GetProcAddress(h, "prgpu_render"));
        if (!version || !g.listDevices || !g.render || version() != prgpu::kApiVersion) {
            g.status = "prender_gpu.dll has an incompatible interface";
            return g;
        }
        g.loaded = true;
        g.status = "ok";
#else
        g.status = "GPU rendering is only available on Windows builds";
#endif
        return g;
    }();
    return m;
}

} // namespace

std::vector<RenderDeviceInfo> ListRenderDevices(std::string *gpuStatus) {
    std::vector<RenderDeviceInfo> out;
    RenderDeviceInfo cpu;
    cpu.name = "CPU";
    cpu.backend = "native";
    cpu.computeUnits = ThreadCount();
    out.push_back(cpu);
    GpuModule &m = Module();
    if (gpuStatus) *gpuStatus = m.status;
    if (!m.loaded) return out;
    prgpu::DeviceInfo devs[16];
    int n = m.listDevices(devs, 16);
    for (int i = 0; i < n; ++i) {
        RenderDeviceInfo d;
        d.index = i;
        d.name = devs[i].name;
        d.backend = devs[i].backend;
        d.memory = devs[i].globalMemory;
        d.computeUnits = devs[i].computeUnits;
        d.isGpu = devs[i].isGpu != 0;
        out.push_back(d);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Scene export: flatten the object-oriented scene into POD buffers.

namespace {

class Exporter {
  public:
    std::vector<prgpu::Spectrum> spectra;
    std::vector<prgpu::Material> materials;
    std::vector<prgpu::Primitive> prims;
    std::vector<prgpu::Light> lights;
    std::vector<prgpu::Medium> media;
    std::set<std::string> warnings;

    // Homogeneous media map directly; other media are not supported on the GPU yet.
    int MediumIndex(const pr::Medium *md) {
        if (!md) return -1;
        auto it = mediumCache_.find(md);
        if (it != mediumCache_.end()) return it->second;
        int id = -1;
        if (auto *h = dynamic_cast<const HomogeneousMedium *>(md)) {
            prgpu::Medium g{};
            g.sigmaA = Spec(h->SigmaAPtr());
            g.sigmaS = Spec(h->SigmaSPtr());
            g.scale = h->Scale();
            g.g = h->G();
            media.push_back(g);
            id = int(media.size() - 1);
        } else {
            warnings.insert("heterogeneous media are ignored on the GPU");
        }
        mediumCache_[md] = id;
        return id;
    }
    int InterfaceMat() {
        if (interfaceMat_ < 0) {
            prgpu::Material m{};
            m.albedo = m.albedoB = m.eta = m.k = -1;
            m.type = prgpu::MatInterface;
            materials.push_back(m);
            interfaceMat_ = int(materials.size() - 1);
        }
        return interfaceMat_;
    }

    int Spec(const Spectrum &s) {
        prgpu::Spectrum t;
        for (int i = 0; i < prgpu::kSpectrumSamples; ++i) t.v[i] = s(prgpu::kSpectrumMin + prgpu::kSpectrumStep * i);
        spectra.push_back(t);
        return int(spectra.size() - 1);
    }
    int Spec(const SpectrumPtr &s) {
        auto it = cache_.find(s.get());
        if (it != cache_.end()) return it->second;
        int id = Spec(*s);
        cache_[s.get()] = id;
        return id;
    }
    // Spectrum textures: constants exactly, checkers as two spectra, anything else by its average.
    int SpecTex(const SpectrumTexturePtr &t, int *second, float *scale) {
        *second = -1;
        *scale = 0;
        if (auto *c = dynamic_cast<const ConstantSpectrumTexture *>(t.get())) return Spec(c->GetSpectrum());
        if (auto *ch = dynamic_cast<const CheckerSpectrumTexture *>(t.get())) {
            int dummy;
            float s2;
            int a = SpecTex(ch->A(), &dummy, &s2), b = SpecTex(ch->B(), &dummy, &s2);
            *second = b;
            *scale = ch->Mapping().scale.x;
            return a;
        }
        warnings.insert("image/procedural textures are approximated by their average colour");
        RGB avg = t->AverageRGB();
        RGBAlbedoSpectrum s(RGB(Clamp(avg.r, 0.f, 1.f), Clamp(avg.g, 0.f, 1.f), Clamp(avg.b, 0.f, 1.f)));
        return Spec(s);
    }
    float FloatTex(const FloatTexturePtr &t, float def) {
        if (!t) return def;
        if (auto *c = dynamic_cast<const ConstantFloatTexture *>(t.get())) return c->Value();
        warnings.insert("roughness textures are approximated by a constant");
        return def;
    }
    void Conductor(const ConductorParams &p, prgpu::Material *m) {
        m->type = prgpu::MatConductor;
        float ur = FloatTex(p.uRoughness, 0.f), vr = p.vRoughness ? FloatTex(p.vRoughness, ur) : ur;
        m->alphaX = TrowbridgeReitzDistribution::RoughnessToAlpha(ur);
        m->alphaY = TrowbridgeReitzDistribution::RoughnessToAlpha(vr);
        if (p.eta && p.k) {
            int d;
            float s;
            m->eta = SpecTex(p.eta, &d, &s);
            m->k = SpecTex(p.k, &d, &s);
        } else {
            m->albedo = SpecTex(p.reflectance, &m->albedoB, &m->checkerScale);
        }
    }
    void Coat(const CoatParams &c, prgpu::Material *m) {
        m->coatEta = Spec(c.eta);
        m->coatDispersive = c.eta->IsConstant() ? 0 : 1;
        m->coatAlpha = TrowbridgeReitzDistribution::RoughnessToAlpha(FloatTex(c.roughness, 0.f));
        m->thickness = FloatTex(c.thickness, 0.01f);
        m->coatAlbedo = -1;
        if (c.albedo) {
            int d;
            float sc;
            m->coatAlbedo = SpecTex(c.albedo, &d, &sc);
        }
        m->coatG = c.g;
        m->coatMaxDepth = c.maxDepth;
        m->coatSamples = c.nSamples;
    }
    int Mat(const Material *mat) {
        auto it = matCache_.find(mat);
        if (it != matCache_.end()) return it->second;
        prgpu::Material m{};
        m.albedo = m.albedoB = m.eta = m.k = -1;
        m.coatEta = m.coatAlbedo = m.sheenColor = -1;
        if (!mat) {
            m.type = prgpu::MatBlack;
        } else if (auto *d = dynamic_cast<const DiffuseMaterial *>(mat)) {
            m.type = prgpu::MatDiffuse;
            m.albedo = SpecTex(d->Reflectance(), &m.albedoB, &m.checkerScale);
        } else if (auto *c = dynamic_cast<const ConductorMaterial *>(mat)) {
            Conductor(c->Params(), &m);
        } else if (auto *g = dynamic_cast<const DielectricMaterial *>(mat)) {
            m.type = prgpu::MatDielectric;
            m.eta = Spec(g->Eta());
            m.dispersive = g->IsDispersive() ? 1 : 0;
            m.thin = g->Thin() ? 1 : 0;
            float ur = FloatTex(g->URoughness(), 0.f), vr = g->VRoughness() ? FloatTex(g->VRoughness(), ur) : ur;
            m.alphaX = TrowbridgeReitzDistribution::RoughnessToAlpha(ur);
            m.alphaY = TrowbridgeReitzDistribution::RoughnessToAlpha(vr);
        } else if (auto *cd = dynamic_cast<const CoatedDiffuseMaterial *>(mat)) {
            m.albedo = SpecTex(cd->Reflectance(), &m.albedoB, &m.checkerScale);
            Coat(cd->Coat(), &m);
            m.type = prgpu::MatCoatedDiffuse;
        } else if (auto *cc = dynamic_cast<const CoatedConductorMaterial *>(mat)) {
            Conductor(cc->Conductor(), &m);
            Coat(cc->Coat(), &m);
            m.type = prgpu::MatCoatedConductor;
        } else {
            warnings.insert("unsupported material rendered as grey diffuse");
            m.type = prgpu::MatDiffuse;
            m.albedo = Spec(ConstantSpectrum(0.5f));
        }
        if (mat && mat->HasSheen() && mat->Sheen().color) {
            int d;
            float sc;
            m.sheenColor = SpecTex(mat->Sheen().color, &d, &sc);
            if (d >= 0) warnings.insert("checker sheen colours use their first colour on the GPU");
            m.sheenRoughness = FloatTex(mat->Sheen().roughness, 0.3f);
            m.sheenWeight = mat->Sheen().weight;
        }
        if (mat && mat->HasShadingPerturbation()) warnings.insert("normal/bump maps are ignored on the GPU");
        materials.push_back(m);
        int id = int(materials.size() - 1);
        matCache_[mat] = id;
        return id;
    }

  private:
    std::map<const Spectrum *, int> cache_;
    std::map<const Material *, int> matCache_;
    std::map<const pr::Medium *, int> mediumCache_;
    int interfaceMat_ = -1;
};

prgpu::Float3 F3(const Vec3f &v) { return {v.x, v.y, v.z}; }

} // namespace

bool GpuPathIntegrator::Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) {
    GpuModule &mod = Module();
    auto *camera = dynamic_cast<const PerspectiveCamera *>(scene.camera.get());
    if (!mod.loaded || !camera) {
        LogWarning("{}; rendering on the CPU", !mod.loaded ? "GPU rendering unavailable: " + mod.status
                                                           : std::string("the GPU supports perspective/thin-lens cameras only"));
        IntegratorSettings cpu = settings_;
        cpu.device = -1;
        return PathIntegrator(cpu, seed_).Render(scene, film, control, scale, err);
    }
    Exporter ex;
    // Primitives
    std::map<const Light *, int> lightOfPrim;
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const Primitive &p = scene.primitives[i];
        prgpu::Primitive g{};
        g.light = -1;
        g.mediumInside = ex.MediumIndex(p.mediumInterface.inside);
        g.mediumOutside = ex.MediumIndex(p.mediumInterface.outside);
        if (auto *sp = dynamic_cast<const Sphere *>(p.shape)) {
            g.type = prgpu::PrimSphere;
            g.p0 = F3(sp->Center());
            g.r = sp->Radius();
            g.flip = sp->Flip() ? 1 : 0;
        } else if (auto *tr = dynamic_cast<const Triangle *>(p.shape)) {
            const TriangleMesh &m = tr->Mesh();
            const int *vi = &m.indices[3 * tr->Index()];
            g.type = prgpu::PrimTriangle;
            g.p0 = F3(m.p[vi[0]]);
            g.p1 = F3(m.p[vi[1]]);
            g.p2 = F3(m.p[vi[2]]);
            if (!m.n.empty()) {
                g.n0 = F3(m.n[vi[0]]);
                g.n1 = F3(m.n[vi[1]]);
                g.n2 = F3(m.n[vi[2]]);
            }
            if (!m.uv.empty()) {
                g.u0 = m.uv[vi[0]].x; g.v0 = m.uv[vi[0]].y;
                g.u1 = m.uv[vi[1]].x; g.v1 = m.uv[vi[1]].y;
                g.u2 = m.uv[vi[2]].x; g.v2 = m.uv[vi[2]].y;
            } else {
                g.u0 = 0; g.v0 = 0; g.u1 = 1; g.v1 = 0; g.u2 = 1; g.v2 = 1;
            }
        } else {
            *err = "unsupported shape for the GPU";
            return false;
        }
        // Interfaces are invisible medium boundaries: the kernel only switches media there.
        g.material = p.IsInterface() ? ex.InterfaceMat() : ex.Mat(p.material);
        ex.prims.push_back(g);
    }
    const int cameraMedium = ex.MediumIndex(scene.camera->medium);
    // Lights (power-based selection, renormalized over the supported ones)
    float envPmf = 0, pmfSum = 0;
    int envSpectrum = -1;
    float envScale = 0;
    for (const auto &lp : scene.lights) {
        const Light *l = lp.get();
        float pmf = scene.lightSampler.PMF(l);
        if (auto *al = dynamic_cast<const DiffuseAreaLight *>(l)) {
            int primIndex = -1;
            for (size_t i = 0; i < scene.primitives.size(); ++i)
                if (scene.primitives[i].areaLight == l) {
                    primIndex = int(i);
                    break;
                }
            if (primIndex < 0) continue;
            prgpu::Light g{};
            g.type = prgpu::LightArea;
            g.primitive = primIndex;
            g.spectrum = ex.Spec(al->Emission());
            g.scale = al->Scale();
            g.twoSided = al->TwoSided() ? 1 : 0;
            g.cosPower = al->CosPower();
            g.area = al->GetShape()->Area();
            g.pmf = pmf;
            ex.prims[size_t(primIndex)].light = int(ex.lights.size());
            ex.lights.push_back(g);
            pmfSum += pmf;
        } else if (auto *pl = dynamic_cast<const PointLight *>(l)) {
            prgpu::Light g{};
            g.type = prgpu::LightPoint;
            g.primitive = -1;
            g.spectrum = ex.Spec(pl->Intensity());
            g.scale = pl->Scale();
            g.position = F3(pl->Position());
            g.pmf = pmf;
            ex.lights.push_back(g);
            pmfSum += pmf;
        } else if (auto *env = dynamic_cast<const EnvironmentLight *>(l); env && env->ConstantRadiance()) {
            envSpectrum = ex.Spec(env->ConstantRadiance());
            envScale = env->Scale();
            envPmf = pmf;
            pmfSum += pmf;
        } else {
            ex.warnings.insert("some light types (IES, spot, distant, sun, image environment) are skipped on the GPU");
        }
    }
    if (pmfSum > 0) {
        for (auto &l : ex.lights) l.pmf /= pmfSum;
        envPmf /= pmfSum;
    }
    for (const auto &w : ex.warnings) LogWarning("GPU: {}", w);

    // BVH (the CPU hierarchy is reused as-is)
    const BVH &bvh = scene.GetBVH();
    std::vector<prgpu::BVHNode> nodes;
    nodes.reserve(bvh.Nodes().size());
    for (const auto &n : bvh.Nodes()) {
        prgpu::BVHNode g{};
        g.bmin = F3(n.bounds.pMin);
        g.bmax = F3(n.bounds.pMax);
        g.offset = n.offset;
        g.nPrims = n.nPrims;
        g.axis = n.axis;
        nodes.push_back(g);
    }

    prgpu::SceneDesc sd{};
    sd.apiVersion = prgpu::kApiVersion;
    sd.nodes = nodes.data();
    sd.nodeCount = int(nodes.size());
    sd.primOrder = bvh.Order().data();
    sd.prims = ex.prims.data();
    sd.primCount = int(ex.prims.size());
    sd.materials = ex.materials.data();
    sd.materialCount = int(ex.materials.size());
    sd.lights = ex.lights.data();
    sd.lightCount = int(ex.lights.size());
    sd.spectra = ex.spectra.data();
    sd.spectrumCount = int(ex.spectra.size());
    sd.media = ex.media.data();
    sd.mediumCount = int(ex.media.size());
    sd.cameraMedium = cameraMedium;
    sd.envSpectrum = envSpectrum;
    sd.envScale = envScale;
    sd.envPmf = envPmf;
    sd.worldCenter = F3(scene.worldCenter);
    sd.worldRadius = scene.worldRadius;
    const Mat4 &m = camera->CameraToWorld().Matrix();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) sd.camera.cameraToWorld[r * 4 + c] = m.m[r][c];
    sd.camera.tanHalfFov = camera->TanHalfFov();
    sd.camera.aspect = float(film.Width()) / film.Height();
    sd.camera.lensRadius = camera->LensRadius();
    sd.camera.focalDistance = camera->FocalDistance();
    sd.camera.width = film.Width();
    sd.camera.height = film.Height();
    sd.camera.margin = 0;
    sd.cieYIntegral = CIE_Y_Integral();
    std::vector<float> sheenTable(size_t(prgpu::kSheenAlpha) * prgpu::kSheenMu);
    for (int ia = 0; ia < prgpu::kSheenAlpha; ++ia)
        for (int im = 0; im < prgpu::kSheenMu; ++im)
            sheenTable[size_t(ia) * prgpu::kSheenMu + im] = SheenBxDF::Albedo(
                float(im) / (prgpu::kSheenMu - 1), 0.05f + 0.95f * float(ia) / (prgpu::kSheenAlpha - 1));
    sd.sheenAlbedo = sheenTable.data();

    // Pixel filter: the kernel importance-samples a tabulated 1D CDF and weights by filter / pdf.
    const Filter &filter = film.GetFilter();
    sd.filter.type = filter.Type() == FilterType::Gaussian        ? prgpu::FilterGaussian
                     : filter.Type() == FilterType::BlackmanHarris ? prgpu::FilterBlackmanHarris
                                                                   : prgpu::FilterBox;
    sd.filter.radius = filter.Radius();
    sd.filter.sigma = filter.Sigma();
    sd.filter.norm = filter.Norm();
    {
        double sum = 0, peak = 0, bins[prgpu::kFilterBins];
        for (int i = 0; i < prgpu::kFilterBins; ++i) {
            float x = -filter.Radius() + (i + 0.5f) * (2 * filter.Radius() / prgpu::kFilterBins);
            bins[i] = std::abs(filter.Eval1D(x));
            peak = std::max(peak, bins[i]);
        }
        // A floor keeps the density positive wherever the filter is.
        for (double &b : bins) b += 1e-3 * peak;
        for (double b : bins) sum += b;
        double acc = 0;
        sd.filter.cdf[0] = 0;
        for (int i = 0; i < prgpu::kFilterBins; ++i) {
            acc += bins[i] / sum;
            sd.filter.cdf[i + 1] = float(acc);
        }
        sd.filter.cdf[prgpu::kFilterBins] = 1;
    }

    // Progressive passes: small sample batches keep progress, previews and cancel responsive.
    const int w = film.Width(), h = film.Height();
    std::vector<float> xyz(size_t(w) * h * 3);
    const int targetSpp = settings_.timeLimit > 0 ? std::numeric_limits<int>::max() : std::max(1, settings_.spp);
    int done = 0, batch = 1;
    char msg[1024] = {0};
    // Checkpoints: the film plus the number of samples per pixel. Sample streams are indexed by
    // sample number, so a resumed render continues exactly where the checkpoint stopped.
    if (!control.resumePath.empty()) {
        int32_t passes = 0;
        if (!ReadCheckpoint(control, settings_, film, [&](BinaryReader &r) { return r.Get(&passes); }, err)) return false;
        done = passes;
        LogInfo("resumed at {} samples per pixel", done);
    }
    auto checkpoint = [&] {
        std::string e;
        if (!WriteCheckpoint(control, settings_, film, [&](BinaryWriter &w) { w.Put(int32_t(done)); }, &e)) LogWarning("{}", e);
    };
    while (done < targetSpp && !control.cancel) {
        int n = std::min(batch, targetSpp - done);
        std::fill(xyz.begin(), xyz.end(), 0.f);
        prgpu::RenderParams rp{};
        rp.spp = n;
        rp.firstSample = done;
        rp.seed = seed_;
        rp.maxDepth = settings_.maxDepth;
        rp.clampLuminance = film.Settings().clampLuminance;
        auto t0 = std::chrono::steady_clock::now();
        if (mod.render(device_, &sd, &rp, xyz.data(), msg, sizeof(msg)) != 0) {
            *err = std::string("GPU render failed: ") + msg;
            return false;
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float *v = &xyz[3 * (size_t(y) * w + x)];
                film.AddPixelXYZ(x, y, XYZ{v[0], v[1], v[2]});
            }
        done += n;
        // Aim for ~0.25 s per batch.
        if (dt < 0.15) batch = std::min(batch * 2, 256);
        else if (dt > 0.5 && batch > 1) batch /= 2;
        double elapsed = control.Elapsed();
        ProgressInfo pi;
        pi.stage = "render";
        pi.elapsed = elapsed;
        pi.samplesPerPixel = done;
        pi.fraction = settings_.timeLimit > 0 ? std::min(1.0, elapsed / settings_.timeLimit) : double(done) / targetSpp;
        control.Progress(pi, done == targetSpp);
        control.MaybePreview(film, 1.0 / done);
        if (control.CheckpointDue()) checkpoint();
        if (settings_.timeLimit > 0 && elapsed >= settings_.timeLimit) break;
    }
    checkpoint();
    *scale = done > 0 ? 1.0 / done : 0.0;
    return true;
}

} // namespace pr

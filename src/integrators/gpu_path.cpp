#include "integrators/gpu_path.h"

#include "cameras/camera.h"
#include "core/log.h"
#include "core/parallel.h"
#include "core/rgb2spec.h"
#include "gpu/gpu_api.h"
#include "materials/material.h"

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
    std::set<std::string> warnings;

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
    int Mat(const Material *mat) {
        auto it = matCache_.find(mat);
        if (it != matCache_.end()) return it->second;
        prgpu::Material m{};
        m.albedo = m.albedoB = m.eta = m.k = -1;
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
            warnings.insert("coated_diffuse is rendered as diffuse on the GPU");
            m.type = prgpu::MatDiffuse;
            m.albedo = SpecTex(cd->Reflectance(), &m.albedoB, &m.checkerScale);
        } else if (auto *cc = dynamic_cast<const CoatedConductorMaterial *>(mat)) {
            warnings.insert("coated_conductor is rendered without its coat on the GPU");
            Conductor(cc->Conductor(), &m);
        } else {
            warnings.insert("unsupported material rendered as grey diffuse");
            m.type = prgpu::MatDiffuse;
            m.albedo = Spec(ConstantSpectrum(0.5f));
        }
        if (mat && mat->HasSheen()) warnings.insert("sheen is ignored on the GPU");
        if (mat && mat->HasShadingPerturbation()) warnings.insert("normal/bump maps are ignored on the GPU");
        materials.push_back(m);
        int id = int(materials.size() - 1);
        matCache_[mat] = id;
        return id;
    }

  private:
    std::map<const Spectrum *, int> cache_;
    std::map<const Material *, int> matCache_;
};

prgpu::Float3 F3(const Vec3f &v) { return {v.x, v.y, v.z}; }

} // namespace

bool GpuPathIntegrator::Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) {
    GpuModule &mod = Module();
    if (!mod.loaded) {
        *err = "GPU rendering unavailable: " + mod.status;
        return false;
    }
    auto *camera = dynamic_cast<const PerspectiveCamera *>(scene.camera.get());
    if (!camera) {
        *err = "the GPU path tracer supports perspective/thin-lens cameras only";
        return false;
    }
    if (!control.resumePath.empty() || !control.checkpointPath.empty())
        LogWarning("checkpointing is not supported by the GPU trial integrator; ignored");

    Exporter ex;
    // Primitives
    std::map<const Light *, int> lightOfPrim;
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const Primitive &p = scene.primitives[i];
        prgpu::Primitive g{};
        g.light = -1;
        if (p.mediumInterface.inside || p.mediumInterface.outside) ex.warnings.insert("participating media are ignored on the GPU");
        if (p.IsInterface()) {
            // Invisible medium boundary: treat as non-existent by making it a pass-through black
            // surface would be wrong; instead skip intersection via an out-of-reach sphere.
            g.type = prgpu::PrimSphere;
            g.p0 = {0, 0, 0};
            g.r = -1;  // never intersects (kernel skips r <= 0)
            g.material = ex.Mat(nullptr);
        } else if (auto *sp = dynamic_cast<const Sphere *>(p.shape)) {
            g.type = prgpu::PrimSphere;
            g.p0 = F3(sp->Center());
            g.r = sp->Radius();
            g.flip = sp->Flip() ? 1 : 0;
            g.material = ex.Mat(p.material);
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
            g.material = ex.Mat(p.material);
        } else {
            *err = "unsupported shape for the GPU";
            return false;
        }
        ex.prims.push_back(g);
    }
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

    if (film.Settings().filter.type != FilterType::Box)
        LogWarning("GPU: the trial integrator uses a box pixel filter");

    // Progressive passes: small sample batches keep progress, previews and cancel responsive.
    const int w = film.Width(), h = film.Height();
    std::vector<float> xyz(size_t(w) * h * 3);
    const int targetSpp = settings_.timeLimit > 0 ? std::numeric_limits<int>::max() : std::max(1, settings_.spp);
    int done = 0, batch = 1;
    char msg[1024] = {0};
    while (done < targetSpp && !control.cancel) {
        int n = std::min(batch, targetSpp - done);
        std::fill(xyz.begin(), xyz.end(), 0.f);
        prgpu::RenderParams rp{};
        rp.spp = n;
        rp.firstSample = done;
        rp.seed = seed_;
        rp.maxDepth = settings_.maxDepth;
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
        if (settings_.timeLimit > 0 && elapsed >= settings_.timeLimit) break;
    }
    *scale = done > 0 ? 1.0 / done : 0.0;
    return true;
}

} // namespace pr

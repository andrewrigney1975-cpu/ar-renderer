#pragma once

#include "core/memory.h"
#include "integrators/integrator.h"
#include "materials/bsdf.h"

#include <vector>

namespace pr {

enum class VertexType { Camera, Light, Surface, Medium };

// A path vertex for bidirectional path construction (after Veach 1997 / pbrt).
struct Vertex {
    VertexType type = VertexType::Surface;
    SampledSpectrum beta;
    SurfaceInteraction si;       // position/normals/medium info for all vertex types
    BSDF bsdf;                   // surface vertices
    float g = 0;                 // medium vertices (HG asymmetry)
    const Camera *camera = nullptr;
    const Light *light = nullptr;
    bool infinite = false;       // light vertex representing an escaped ray / infinite light
    bool delta = false;
    float pdfFwd = 0, pdfRev = 0;

    const Vec3f &p() const { return si.p; }
    const Vec3f &ng() const { return si.n; }
    Vec3f ns() const { return type == VertexType::Surface ? si.shading.n : si.n; }
    bool IsOnSurface() const { return !si.n.IsZero(); }

    const Light *AreaLight() const {
        if (type == VertexType::Light) return light;
        if (type == VertexType::Surface && si.primitive) return si.primitive->areaLight;
        return nullptr;
    }
    bool IsLight() const { return type == VertexType::Light || (type == VertexType::Surface && AreaLight()); }
    bool IsDeltaLight() const { return type == VertexType::Light && light && pr::IsDeltaLight(light->Type()); }
    bool IsInfiniteLight() const {
        return type == VertexType::Light &&
               (infinite || !light || light->Type() == LightType::Infinite || light->Type() == LightType::DeltaDirection);
    }
    bool IsConnectible() const;

    SampledSpectrum f(const Vertex &next, TransportMode mode) const;
    float ConvertDensity(float pdf, const Vertex &next) const;
    float Pdf(const Scene &scene, const Vertex *prev, const Vertex &next) const;
    float PdfLight(const Scene &scene, const Vertex &v) const;
    float PdfLightOrigin(const Scene &scene, const Vertex &v) const;
    SampledSpectrum Le(const Scene &scene, const Vertex &v, const SampledWavelengths &lambda) const;
};

struct BDPTContext {
    const Scene &scene;
    SampledWavelengths &lambda;
    Sampler &sampler;
    ScratchBuffer &buf;
    bool russianRoulette = false;
};

int GenerateCameraSubpath(BDPTContext &ctx, int maxDepth, Vec2f pFilm, Vertex *path);
int GenerateLightSubpath(BDPTContext &ctx, int maxDepth, Vertex *path);
SampledSpectrum ConnectBDPT(BDPTContext &ctx, Vertex *lightVertices, Vertex *cameraVertices, int s, int t,
                            Vec2f *pRaster, float *misWeight = nullptr);

class BDPTIntegrator : public Integrator {
  public:
    BDPTIntegrator(const IntegratorSettings &s, uint64_t seed) : settings_(s), seed_(seed) {}
    bool Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) override;

  private:
    IntegratorSettings settings_;
    uint64_t seed_;
};

} // namespace pr

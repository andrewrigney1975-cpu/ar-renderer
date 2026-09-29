#pragma once

#include "cameras/camera.h"
#include "core/film.h"
#include "geometry/bvh.h"
#include "lights/light.h"
#include "lights/light_bvh.h"
#include "materials/material.h"
#include "media/medium.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pr {

struct Primitive {
    const Shape *shape = nullptr;
    const Material *material = nullptr;  // nullptr => pure medium boundary (invisible interface)
    const Light *areaLight = nullptr;
    MediumInterface mediumInterface;
    int object = -1;
    bool IsInterface() const { return material == nullptr && areaLight == nullptr; }
};

enum class IntegratorType { Path, BDPT, MMLT, PSSMLT };

struct IntegratorSettings {
    IntegratorType type = IntegratorType::MMLT;
    int maxDepth = -1;              // <= 0: integrator default (path: unlimited, bdpt: 256, mlt: 32)
    int spp = 64;                   // path / bdpt: samples per pixel
    double mutationsPerPixel = 256; // mlt
    int64_t bootstrapSamples = 100000;
    int chains = 1024;
    float largeStepProbability = 0.3f;
    float sigma = 0.01f;
    double timeLimit = 0;           // seconds; 0 = none
    bool russianRoulette = true;    // path / bdpt only
    bool lightBVH = true;           // path: importance-based light selection (else by power)
};

struct OutputSpec {
    std::string file;
    ToneMap toneMap = ToneMap::ACES;
    bool half = false;
    bool zip = true;  // EXR: ZIP compression
    std::string aov;  // empty: beauty; else albedo | normal | depth | position
};

struct RenderSettings {
    FilmSettings film;
    IntegratorSettings integrator;
    std::vector<OutputSpec> outputs;
    std::string cameraName;
    uint64_t seed = 0;
};

class Scene {
  public:
    // Owned resources
    std::vector<std::unique_ptr<Shape>> shapes;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Medium>> media;
    std::vector<std::shared_ptr<Light>> lights;
    std::vector<std::shared_ptr<TriangleMesh>> meshes;
    std::vector<Primitive> primitives;  // index == shape index passed to the BVH
    std::vector<std::string> objectNames;

    std::vector<const Light *> infiniteLights;
    std::unique_ptr<Camera> camera;
    RenderSettings settings;
    LightSampler lightSampler;      // power-based (BDPT/MLT light subpaths need a context-free pmf)
    BVHLightSampler lightBVH;       // spatially adaptive (path tracer next-event estimation)
    Vec3f worldCenter;
    float worldRadius = 1;  // bounding sphere of geometry + camera (used by infinite lights)

    // Finalize: build BVH, preprocess lights, set up light sampling.
    void Build();

    Bounds3f Bounds() const { return bvh_.Bounds(); }
    std::optional<SurfaceInteraction> Intersect(const Ray &ray, float tMax, float *tHit) const;
    bool IntersectP(const Ray &ray, float tMax) const { return bvh_.IntersectP(ray, tMax); }

    // Transmittance between two interactions (through media and interface surfaces). Zero if an
    // opaque (non-interface) surface blocks the segment.
    SampledSpectrum Tr(const Interaction &p0, const Interaction &p1, const SampledWavelengths &lambda) const;
    // Innermost medium enclosing a point (via containment tests of medium-bearing closed shapes).
    const Medium *MediumAt(const Vec3f &p) const;
    // Hash of geometry, lights and camera, used to validate checkpoints.
    uint64_t Fingerprint() const;

  private:
    BVH bvh_;
};

} // namespace pr

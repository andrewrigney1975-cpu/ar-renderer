#include "scene/scene.h"

#include "core/rng.h"

#include <map>

namespace pr {

void Scene::Build() {
    std::vector<const Shape *> s;
    s.reserve(primitives.size());
    for (const Primitive &p : primitives) s.push_back(p.shape);
    bvh_.Build(std::move(s));
    Bounds3f b = bvh_.Bounds();
    if (camera) b = Union(b, camera->Position());
    b.BoundingSphere(&worldCenter, &worldRadius);
    worldRadius = std::max(worldRadius, 1e-3f);
    std::vector<const Light *> all;
    infiniteLights.clear();
    for (auto &l : lights) {
        l->Preprocess(b);
        all.push_back(l.get());
        if (l->Type() == LightType::Infinite) infiniteLights.push_back(l.get());
    }
    lightSampler = LightSampler(all);
    lightBVH = BVHLightSampler(all);
}

std::optional<SurfaceInteraction> Scene::Intersect(const Ray &ray, float tMax, float *tHit) const {
    BVH::Hit hit;
    if (!bvh_.Intersect(ray, tMax, &hit)) return std::nullopt;
    const Primitive &prim = primitives[hit.prim];
    SurfaceInteraction si;
    prim.shape->ComputeInteraction(ray, hit.t, hit.info, &si);
    si.primitive = &prim;
    si.mediumInterface = &prim.mediumInterface;
    if (tHit) *tHit = hit.t;
    return si;
}

SampledSpectrum Scene::Tr(const Interaction &p0, const Interaction &p1, const SampledWavelengths &lambda) const {
    Ray ray = p0.SpawnRayTo(p1);
    const Medium *medium = p0.GetMedium(ray.d);
    SampledSpectrum Tr(1.f);
    for (int iter = 0; iter < 256; ++iter) {
        float tHit = 1 - ShadowEpsilon;
        auto si = Intersect(ray, 1 - ShadowEpsilon, &tHit);
        if (si && !si->primitive->IsInterface()) return SampledSpectrum(0.f);
        if (medium) {
            float len = Length(ray.d);
            Tr *= medium->Transmittance(Ray(ray.o, ray.d / len), (si ? tHit : 1.f) * len, lambda);
        }
        if (!Tr) return Tr;
        if (!si) return Tr;
        medium = si->GetMedium(ray.d);
        ray = si->SpawnRayTo(p1);
    }
    return SampledSpectrum(0.f);
}

uint64_t Scene::Fingerprint() const {
    uint64_t h = Hash(uint64_t(primitives.size()), uint64_t(lights.size()));
    for (const Primitive &p : primitives) {
        Bounds3f b = p.shape->Bounds();
        h = Hash(h, b.pMin, b.pMax, p.material != nullptr, p.areaLight != nullptr);
    }
    for (const auto &l : lights) h = Hash(h, l->PowerY());
    if (camera) h = Hash(h, camera->Position());
    return h;
}

const Medium *Scene::MediumAt(const Vec3f &p) const {
    // Collect each object's interior medium and test containment; pick the smallest container.
    const Medium *best = nullptr;
    float bestVolume = Infinity;
    std::map<int, bool> tested;
    for (const Primitive &prim : primitives) {
        if (!prim.mediumInterface.inside || prim.object < 0) continue;
        if (tested.count(prim.object)) continue;
        tested[prim.object] = true;
        if (!prim.shape->Contains(p)) continue;
        Bounds3f b = prim.shape->Bounds();
        // For meshes, the object's bounds: accumulate over all its primitives.
        for (const Primitive &q : primitives)
            if (q.object == prim.object) b = Union(b, q.shape->Bounds());
        Vec3f d = b.Diagonal();
        float vol = d.x * d.y * d.z;
        if (vol < bestVolume) {
            bestVolume = vol;
            best = prim.mediumInterface.inside;
        }
    }
    return best;
}

} // namespace pr

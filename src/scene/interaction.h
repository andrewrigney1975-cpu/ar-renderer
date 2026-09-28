#pragma once

#include "core/math.h"

namespace pr {

class Medium;
class Material;
class Light;
class Shape;
struct Primitive;

struct MediumInterface {
    const Medium *inside = nullptr;
    const Medium *outside = nullptr;
    bool IsTransition() const { return inside != outside; }
};

// Relative epsilon used to offset spawned rays off surfaces.
inline float RayEpsilon(const Vec3f &p) { return 5e-5f * std::max(1.f, MaxComponent(Abs(p))); }
constexpr float ShadowEpsilon = 1e-4f;

struct Interaction {
    Vec3f p;
    Vec3f n;   // geometric normal; zero for medium interactions and non-surface endpoints
    Vec3f wo;  // outgoing direction (towards the previous vertex), normalized
    Vec2f uv;
    const MediumInterface *mediumInterface = nullptr;
    const Medium *medium = nullptr;  // used when mediumInterface == nullptr

    Interaction() = default;
    Interaction(const Vec3f &p, const Vec3f &n) : p(p), n(n) {}

    bool IsSurfaceInteraction() const { return !n.IsZero(); }
    bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }

    // Medium on the side of the surface that direction w points into.
    const Medium *GetMedium(const Vec3f &w) const {
        if (mediumInterface) return Dot(w, n) > 0 ? mediumInterface->outside : mediumInterface->inside;
        return medium;
    }

    Vec3f OffsetRayOrigin(const Vec3f &w) const {
        if (n.IsZero()) return p;
        float eps = RayEpsilon(p);
        Vec3f offset = n * eps;
        if (Dot(w, n) < 0) offset = -offset;
        return p + offset;
    }
    Ray SpawnRay(const Vec3f &d) const { return Ray(OffsetRayOrigin(d), d); }
    // Ray towards point p2 (optionally with its surface normal for offsetting). Direction is not
    // normalized: the segment is t in (0, 1 - ShadowEpsilon).
    Ray SpawnRayTo(const Interaction &it) const {
        Vec3f o = OffsetRayOrigin(it.p - p);
        Vec3f target = it.OffsetRayOrigin(o - it.p);
        return Ray(o, target - o);
    }
};

struct SurfaceInteraction : Interaction {
    struct {
        Vec3f n;     // shading normal (same hemisphere as n)
        Vec3f dpdu;  // shading tangent
    } shading;
    Vec3f dpdu, dpdv;
    const Primitive *primitive = nullptr;
    int faceIndex = 0;

    // Frame used for BSDF evaluation (z = shading normal).
    Frame ShadingFrame() const { return Frame::FromXZ(shading.dpdu, shading.n); }
};

struct MediumInteraction : Interaction {
    float g = 0;  // Henyey-Greenstein asymmetry of the phase function at this point
};

} // namespace pr

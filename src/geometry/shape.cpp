#include "geometry/shape.h"

#include "core/sampling.h"

namespace pr {

static inline Vec3f CrossF(const Vec3f &a, const Vec3f &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Generic solid-angle sampling via area sampling.
std::optional<ShapeSample> Shape::Sample(const ShapeSampleContext &ctx, Vec2f u) const {
    auto ss = Sample(u);
    if (!ss) return std::nullopt;
    Vec3f wi = ss->intr.p - ctx.p;
    float d2 = LengthSquared(wi);
    if (d2 == 0) return std::nullopt;
    wi = wi / std::sqrt(d2);
    float cosTheta = AbsDot(ss->intr.n, -wi);
    if (cosTheta == 0) return std::nullopt;
    ss->pdf *= d2 / cosTheta;
    if (!std::isfinite(ss->pdf)) return std::nullopt;
    return ss;
}

float Shape::PDF(const ShapeSampleContext &ctx, const Vec3f &wi) const {
    Ray ray(ctx.p, wi);
    float tHit;
    HitInfo hit;
    if (!Intersect(ray, Infinity, &tHit, &hit)) return 0;
    SurfaceInteraction si;
    ComputeInteraction(ray, tHit, hit, &si);
    float cosTheta = AbsDot(si.n, -wi);
    if (cosTheta == 0) return 0;
    float pdf = (1 / Area()) * DistanceSquared(ctx.p, si.p) / cosTheta;
    return std::isfinite(pdf) ? pdf : 0.f;
}

// ---------------------------------------------------------------------------------------------
// Sphere

bool Sphere::Intersect(const Ray &ray, float tMax, float *tHit, HitInfo *) const {
    // Robust quadratic (Haines et al., Ray Tracing Gems ch. 7), in double precision.
    double ox = ray.o.x - c_.x, oy = ray.o.y - c_.y, oz = ray.o.z - c_.z;
    double dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;
    double a = dx * dx + dy * dy + dz * dz;
    double b = ox * dx + oy * dy + oz * dz;  // half of the usual b
    double c = ox * ox + oy * oy + oz * oz - double(r_) * r_;
    double k = b / a;
    double lx = ox - k * dx, ly = oy - k * dy, lz = oz - k * dz;
    double disc = double(r_) * r_ - (lx * lx + ly * ly + lz * lz);
    if (disc < 0) return false;
    double q = -b - std::copysign(std::sqrt(a * disc), b);
    double t0 = c / q, t1 = q / a;
    if (t0 > t1) std::swap(t0, t1);
    double t = t0;
    if (t <= 0) t = t1;
    if (t <= 0 || t >= tMax) return false;
    *tHit = float(t);
    return true;
}

void Sphere::ComputeInteraction(const Ray &ray, float tHit, const HitInfo &, SurfaceInteraction *si) const {
    Vec3f pHit = ray(tHit);
    Vec3f rel = pHit - c_;
    float len = Length(rel);
    Vec3f n = len > 0 ? rel / len : Vec3f(0, 1, 0);
    pHit = c_ + n * r_;
    float phi = std::atan2(n.z, n.x);
    if (phi < 0) phi += 2 * Pi;
    float theta = SafeACos(n.y);
    si->p = pHit;
    si->n = flip_ ? -n : n;
    si->uv = {phi * Inv2Pi, theta * InvPi};
    si->wo = -Normalize(ray.d);
    Vec3f dpdu(-rel.z, 0, rel.x);
    if (LengthSquared(dpdu) < 1e-12f) {
        Vec3f t2;
        CoordinateSystem(n, &dpdu, &t2);
    }
    si->dpdu = dpdu;
    si->shading.n = si->n;
    si->shading.dpdu = dpdu;
}

std::optional<ShapeSample> Sphere::Sample(Vec2f u) const {
    Vec3f n = SampleUniformSphere(u);
    ShapeSample ss;
    ss.intr.p = c_ + n * r_;
    ss.intr.n = flip_ ? -n : n;
    float phi = std::atan2(n.z, n.x);
    if (phi < 0) phi += 2 * Pi;
    ss.intr.uv = {phi * Inv2Pi, SafeACos(n.y) * InvPi};
    ss.pdf = 1 / Area();
    return ss;
}

std::optional<ShapeSample> Sphere::Sample(const ShapeSampleContext &ctx, Vec2f u) const {
    float dc2 = DistanceSquared(ctx.p, c_);
    if (dc2 <= r_ * r_ * 1.0001f) return Shape::Sample(ctx, u);
    float sinThetaMax = r_ / std::sqrt(dc2);
    float sin2ThetaMax = Sqr(sinThetaMax);
    float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
    float oneMinusCosThetaMax = 1 - cosThetaMax;
    float cosTheta = (cosThetaMax - 1) * u.x + 1;
    float sin2Theta = 1 - Sqr(cosTheta);
    if (sin2ThetaMax < 0.00068523f) {
        // Small-angle fallback (sin^2(1.5 deg)) for numerical accuracy.
        sin2Theta = sin2ThetaMax * u.x;
        cosTheta = std::sqrt(1 - sin2Theta);
        oneMinusCosThetaMax = sin2ThetaMax / 2;
    }
    float cosAlpha = sin2Theta / sinThetaMax + cosTheta * SafeSqrt(1 - sin2Theta / Sqr(sinThetaMax));
    float sinAlpha = SafeSqrt(1 - Sqr(cosAlpha));
    float phi = u.y * 2 * Pi;
    Vec3f w = SphericalDirection(sinAlpha, cosAlpha, phi);
    Frame frame = Frame::FromZ(Normalize(c_ - ctx.p));
    Vec3f n = frame.FromLocal(-w);
    ShapeSample ss;
    ss.intr.p = c_ + n * r_;
    ss.intr.n = flip_ ? -n : n;
    float uphi = std::atan2(n.z, n.x);
    if (uphi < 0) uphi += 2 * Pi;
    ss.intr.uv = {uphi * Inv2Pi, SafeACos(n.y) * InvPi};
    ss.pdf = 1 / (2 * Pi * oneMinusCosThetaMax);
    return ss;
}

float Sphere::PDF(const ShapeSampleContext &ctx, const Vec3f &wi) const {
    float dc2 = DistanceSquared(ctx.p, c_);
    if (dc2 <= r_ * r_ * 1.0001f) return Shape::PDF(ctx, wi);
    float sin2ThetaMax = r_ * r_ / dc2;
    float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
    float oneMinusCosThetaMax = 1 - cosThetaMax;
    if (sin2ThetaMax < 0.00068523f) oneMinusCosThetaMax = sin2ThetaMax / 2;
    // Direction must fall within the cone subtended by the sphere.
    Vec3f toC = (c_ - ctx.p) / std::sqrt(dc2);
    if (Dot(toC, wi) < cosThetaMax - 1e-6f) return 0;
    return 1 / (2 * Pi * oneMinusCosThetaMax);
}

// ---------------------------------------------------------------------------------------------
// Triangle meshes

void TriangleMesh::ApplyTransform(const Transform &t) {
    for (auto &v : p) v = t.Point(v);
    for (auto &v : n) {
        v = t.Normal(v);
        float l = Length(v);
        if (l > 0) v = v / l;
    }
    // A mirroring transform flips winding; restore outward orientation for consistency.
    const auto &m = t.Matrix().m;
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (det < 0)
        for (size_t i = 0; i + 2 < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
}

Bounds3f TriangleMesh::Bounds() const {
    Bounds3f b;
    for (const auto &v : p) b = Union(b, v);
    return b;
}

bool TriangleMesh::Contains(const Vec3f &pt) const {
    // Count crossings along an irrational-ish direction to avoid edge degeneracies.
    Vec3f d = Normalize(Vec3f(0.5773f, 0.5774f, 0.5771f));
    int crossings = 0;
    for (int t = 0; t < TriangleCount(); ++t) {
        const Vec3f &p0 = p[indices[3 * t]], &p1 = p[indices[3 * t + 1]], &p2 = p[indices[3 * t + 2]];
        Vec3f e1 = p1 - p0, e2 = p2 - p0;
        Vec3f pvec = CrossF(d, e2);
        float det = Dot(e1, pvec);
        if (det == 0) continue;
        float inv = 1 / det;
        Vec3f tvec = pt - p0;
        float u = Dot(tvec, pvec) * inv;
        if (u < 0 || u > 1) continue;
        Vec3f qvec = CrossF(tvec, e1);
        float v = Dot(d, qvec) * inv;
        if (v < 0 || u + v > 1) continue;
        if (Dot(e2, qvec) * inv > 0) ++crossings;
    }
    return (crossings & 1) != 0;
}

Bounds3f Triangle::Bounds() const {
    Vec3f p0, p1, p2;
    Vertices(&p0, &p1, &p2);
    return Union(Bounds3f(p0, p1), p2);
}

float Triangle::Area() const {
    Vec3f p0, p1, p2;
    Vertices(&p0, &p1, &p2);
    return 0.5f * Length(Cross(p1 - p0, p2 - p0));
}

bool Triangle::Intersect(const Ray &ray, float tMax, float *tHit, HitInfo *hit) const {
    Vec3f p0, p1, p2;
    Vertices(&p0, &p1, &p2);
    Vec3f e1 = p1 - p0, e2 = p2 - p0;
    Vec3f pvec = CrossF(ray.d, e2);
    float det = Dot(e1, pvec);
    if (det == 0 || !std::isfinite(det)) return false;
    float inv = 1 / det;
    Vec3f tvec = ray.o - p0;
    float u = Dot(tvec, pvec) * inv;
    if (u < 0 || u > 1) return false;
    Vec3f qvec = CrossF(tvec, e1);
    float v = Dot(ray.d, qvec) * inv;
    if (v < 0 || u + v > 1) return false;
    float t = Dot(e2, qvec) * inv;
    if (t <= 0 || t >= tMax) return false;
    *tHit = t;
    hit->b1 = u;
    hit->b2 = v;
    return true;
}

void Triangle::ComputeInteraction(const Ray &ray, float, const HitInfo &hit, SurfaceInteraction *si) const {
    const int *vi = &mesh_->indices[3 * tri_];
    Vec3f p0 = mesh_->p[vi[0]], p1 = mesh_->p[vi[1]], p2 = mesh_->p[vi[2]];
    float b1 = hit.b1, b2 = hit.b2, b0 = 1 - b1 - b2;
    si->p = p0 * b0 + p1 * b1 + p2 * b2;
    Vec3f n = Normalize(CrossF(p1 - p0, p2 - p0));
    Vec2f uv0(0, 0), uv1(1, 0), uv2(1, 1);
    if (!mesh_->uv.empty()) {
        uv0 = mesh_->uv[vi[0]];
        uv1 = mesh_->uv[vi[1]];
        uv2 = mesh_->uv[vi[2]];
    }
    si->uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
    // dpdu from the uv parameterization.
    Vec2f duv02 = uv0 - uv2, duv12 = uv1 - uv2;
    Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    float det = duv02.x * duv12.y - duv02.y * duv12.x;
    Vec3f dpdu;
    bool degenerate = std::abs(det) < 1e-9f;
    if (!degenerate) {
        float inv = 1 / det;
        dpdu = (dp02 * duv12.y - dp12 * duv02.y) * inv;
        degenerate = LengthSquared(Cross(dpdu, n)) == 0 || !std::isfinite(dpdu.x);
    }
    if (degenerate) {
        Vec3f t2;
        CoordinateSystem(n, &dpdu, &t2);
    }
    si->dpdu = dpdu;
    Vec3f ns = n;
    if (!mesh_->n.empty()) {
        ns = mesh_->n[vi[0]] * b0 + mesh_->n[vi[1]] * b1 + mesh_->n[vi[2]] * b2;
        float l = Length(ns);
        ns = l > 0 ? ns / l : n;
        n = FaceForward(n, ns);  // geometric normal follows the authored orientation
    }
    si->n = n;
    si->shading.n = ns;
    si->shading.dpdu = dpdu;
    si->wo = -Normalize(ray.d);
    si->faceIndex = tri_;
}

std::optional<ShapeSample> Triangle::Sample(Vec2f u) const {
    const int *vi = &mesh_->indices[3 * tri_];
    Vec3f p0 = mesh_->p[vi[0]], p1 = mesh_->p[vi[1]], p2 = mesh_->p[vi[2]];
    float b[3];
    SampleUniformTriangle(u, b);
    ShapeSample ss;
    ss.intr.p = p0 * b[0] + p1 * b[1] + p2 * b[2];
    Vec3f n = Normalize(CrossF(p1 - p0, p2 - p0));
    if (!mesh_->n.empty()) {
        Vec3f ns = mesh_->n[vi[0]] * b[0] + mesh_->n[vi[1]] * b[1] + mesh_->n[vi[2]] * b[2];
        n = FaceForward(n, ns);
    }
    ss.intr.n = n;
    Vec2f uv0(0, 0), uv1(1, 0), uv2(1, 1);
    if (!mesh_->uv.empty()) {
        uv0 = mesh_->uv[vi[0]];
        uv1 = mesh_->uv[vi[1]];
        uv2 = mesh_->uv[vi[2]];
    }
    ss.intr.uv = uv0 * b[0] + uv1 * b[1] + uv2 * b[2];
    ss.pdf = 1 / Area();
    return ss;
}

// ---------------------------------------------------------------------------------------------
// Procedural meshes

std::shared_ptr<TriangleMesh> MakeRectMesh(float w, float d) {
    auto m = std::make_shared<TriangleMesh>();
    float hw = w / 2, hd = d / 2;
    m->p = {{-hw, 0, -hd}, {hw, 0, -hd}, {hw, 0, hd}, {-hw, 0, hd}};
    m->n = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    m->uv = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    // Counter-clockwise when seen from +Y: (p0, p2, p1) gives normal +Y.
    m->indices = {0, 2, 1, 0, 3, 2};
    return m;
}

std::shared_ptr<TriangleMesh> MakeDiskMesh(float radius, int segments) {
    auto m = std::make_shared<TriangleMesh>();
    m->p.push_back({0, 0, 0});
    m->n.push_back({0, 1, 0});
    m->uv.push_back({0.5f, 0.5f});
    for (int i = 0; i < segments; ++i) {
        float a = 2 * Pi * i / segments;
        m->p.push_back({radius * std::cos(a), 0, radius * std::sin(a)});
        m->n.push_back({0, 1, 0});
        m->uv.push_back({0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a)});
    }
    for (int i = 0; i < segments; ++i) {
        int a = 1 + i, b = 1 + (i + 1) % segments;
        m->indices.insert(m->indices.end(), {0, b, a});
    }
    return m;
}

std::shared_ptr<TriangleMesh> MakeBoxMesh(const Vec3f &size) {
    auto m = std::make_shared<TriangleMesh>();
    Vec3f h = size * 0.5f;
    struct Face { Vec3f n, u, v; };
    const Face faces[6] = {{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
                           {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
                           {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}}};
    for (const Face &f : faces) {
        int base = int(m->p.size());
        Vec3f c = f.n * h;
        Vec3f U = f.u * h, V = f.v * h;
        m->p.push_back(c - U - V);
        m->p.push_back(c + U - V);
        m->p.push_back(c + U + V);
        m->p.push_back(c - U + V);
        for (int i = 0; i < 4; ++i) m->n.push_back(f.n);
        m->uv.insert(m->uv.end(), {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
        // u x v == n, so (0,1,2) is counter-clockwise around n.
        m->indices.insert(m->indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    m->closed = true;
    return m;
}

} // namespace pr

#pragma once

#include "core/math.h"
#include "scene/interaction.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pr {

struct HitInfo {
    float b1 = 0, b2 = 0;  // triangle barycentrics (b0 = 1 - b1 - b2)
};

struct ShapeSample {
    Interaction intr;
    float pdf = 0;
};

struct ShapeSampleContext {
    Vec3f p;
    Vec3f n;   // geometric normal at the reference point (zero in media)
    Vec3f ns;  // shading normal
};

class Shape {
  public:
    virtual ~Shape() = default;
    virtual Bounds3f Bounds() const = 0;
    // Closest intersection with t in (0, tMax). Fills tHit and hit parameters.
    virtual bool Intersect(const Ray &ray, float tMax, float *tHit, HitInfo *hit) const = 0;
    virtual void ComputeInteraction(const Ray &ray, float tHit, const HitInfo &hit,
                                    SurfaceInteraction *si) const = 0;
    virtual float Area() const = 0;
    // Uniformly sample a point by area; pdf is with respect to area.
    virtual std::optional<ShapeSample> Sample(Vec2f u) const = 0;
    // Sample a point as seen from ctx; pdf is with respect to solid angle at ctx.p.
    virtual std::optional<ShapeSample> Sample(const ShapeSampleContext &ctx, Vec2f u) const;
    // Solid-angle pdf of sampling direction wi from ctx (assumes the ray hits this shape).
    virtual float PDF(const ShapeSampleContext &ctx, const Vec3f &wi) const;
    // Bound of the surface normals: returns the cone axis and cos(half-angle) (-1 = all directions).
    virtual void NormalBounds(Vec3f *w, float *cosTheta) const {
        *w = Vec3f(0, 0, 1);
        *cosTheta = -1;
    }
    // Point containment for closed shapes (used to auto-detect enclosing media).
    virtual bool Contains(const Vec3f &) const { return false; }
};

class Sphere : public Shape {
  public:
    Sphere(const Vec3f &center, float radius, bool flipNormals = false)
        : c_(center), r_(radius), flip_(flipNormals) {}
    Bounds3f Bounds() const override { return {c_ - Vec3f(r_, r_, r_), c_ + Vec3f(r_, r_, r_)}; }
    bool Intersect(const Ray &ray, float tMax, float *tHit, HitInfo *hit) const override;
    void ComputeInteraction(const Ray &ray, float tHit, const HitInfo &hit, SurfaceInteraction *si) const override;
    float Area() const override { return 4 * Pi * r_ * r_; }
    std::optional<ShapeSample> Sample(Vec2f u) const override;
    std::optional<ShapeSample> Sample(const ShapeSampleContext &ctx, Vec2f u) const override;
    float PDF(const ShapeSampleContext &ctx, const Vec3f &wi) const override;
    bool Contains(const Vec3f &p) const override { return DistanceSquared(p, c_) < r_ * r_; }
    const Vec3f &Center() const { return c_; }
    float Radius() const { return r_; }
    bool Flip() const { return flip_; }

  private:
    Vec3f c_;
    float r_;
    bool flip_;
};

struct TriangleMesh {
    std::vector<Vec3f> p;
    std::vector<Vec3f> n;   // optional per-vertex normals
    std::vector<Vec2f> uv;  // optional per-vertex uvs
    std::vector<int> indices;
    bool closed = false;    // hint: the mesh encloses a volume
    int TriangleCount() const { return int(indices.size() / 3); }
    void ApplyTransform(const Transform &t);
    Bounds3f Bounds() const;
    // Parity test along a fixed direction (valid for closed, consistently oriented meshes).
    bool Contains(const Vec3f &pt) const;
};

class Triangle : public Shape {
  public:
    Triangle(std::shared_ptr<const TriangleMesh> mesh, int index) : mesh_(std::move(mesh)), tri_(index) {}
    Bounds3f Bounds() const override;
    bool Intersect(const Ray &ray, float tMax, float *tHit, HitInfo *hit) const override;
    void ComputeInteraction(const Ray &ray, float tHit, const HitInfo &hit, SurfaceInteraction *si) const override;
    float Area() const override;
    std::optional<ShapeSample> Sample(Vec2f u) const override;
    bool Contains(const Vec3f &p) const override { return mesh_->closed && mesh_->Contains(p); }
    void NormalBounds(Vec3f *w, float *cosTheta) const override;
    const TriangleMesh &Mesh() const { return *mesh_; }
    int Index() const { return tri_; }

  private:
    void Vertices(Vec3f *p0, Vec3f *p1, Vec3f *p2) const {
        const int *v = &mesh_->indices[3 * tri_];
        *p0 = mesh_->p[v[0]];
        *p1 = mesh_->p[v[1]];
        *p2 = mesh_->p[v[2]];
    }
    std::shared_ptr<const TriangleMesh> mesh_;
    int tri_;
};

// Procedural meshes (object space).
std::shared_ptr<TriangleMesh> MakeRectMesh(float width, float depth);          // XZ plane, normal +Y
std::shared_ptr<TriangleMesh> MakeBoxMesh(const Vec3f &size);                  // centered, outward normals
std::shared_ptr<TriangleMesh> MakeDiskMesh(float radius, int segments = 64);   // XZ plane, normal +Y

// Load OBJ or PLY into a triangle mesh (object space).
std::shared_ptr<TriangleMesh> LoadMesh(const std::string &path, std::string *err);

} // namespace pr

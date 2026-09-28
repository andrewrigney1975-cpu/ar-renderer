#pragma once

#include "core/image.h"
#include "core/sampling.h"
#include "core/spectrum.h"
#include "geometry/shape.h"
#include "scene/interaction.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace pr {

enum class LightType { DeltaPosition, DeltaDirection, Area, Infinite };
inline bool IsDeltaLight(LightType t) { return t == LightType::DeltaPosition || t == LightType::DeltaDirection; }

struct LightSampleContext {
    Vec3f p, n, ns;
    LightSampleContext() = default;
    explicit LightSampleContext(const Interaction &it) : p(it.p), n(it.n), ns(it.n) {}
};

struct LightLiSample {
    SampledSpectrum L;
    Vec3f wi;
    float pdf = 0;
    Interaction pLight;
};

struct LightLeSample {
    SampledSpectrum L;
    Ray ray;
    Interaction intr;  // origin on the light (n may be zero)
    float pdfPos = 0, pdfDir = 0;
};

class Light {
  public:
    explicit Light(LightType t) : type_(t) {}
    virtual ~Light() = default;
    LightType Type() const { return type_; }

    // Approximate emitted power (luminance-weighted), used for light selection.
    virtual float PowerY() const = 0;
    virtual void Preprocess(const Bounds3f &) {}

    virtual std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                                  const SampledWavelengths &lambda) const = 0;
    virtual float PDF_Li(const LightSampleContext &ctx, const Vec3f &wi) const = 0;
    // Area lights: radiance leaving point p (normal n) in direction w.
    virtual SampledSpectrum L(const Vec3f &, const Vec3f &, Vec2f, const Vec3f &, const SampledWavelengths &) const {
        return SampledSpectrum(0.f);
    }
    // Infinite lights: radiance arriving along a ray that escapes the scene.
    virtual SampledSpectrum Le(const Ray &, const SampledWavelengths &) const { return SampledSpectrum(0.f); }

    virtual std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const = 0;
    virtual void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const = 0;

    const Medium *medium = nullptr;  // medium surrounding point/spot lights

  private:
    LightType type_;
};

// Emission profile: L(w) = Lemit * scale * cos^cosPower(theta); cosPower = 0 is Lambertian.
class DiffuseAreaLight : public Light {
  public:
    DiffuseAreaLight(const Shape *shape, SpectrumPtr Lemit, float scale, bool twoSided, float cosPower,
                     const MediumInterface *mi);
    float PowerY() const override;
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &ctx, const Vec3f &wi) const override;
    SampledSpectrum L(const Vec3f &p, const Vec3f &n, Vec2f uv, const Vec3f &w,
                      const SampledWavelengths &lambda) const override;
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;
    const Shape *GetShape() const { return shape_; }

  private:
    float Profile(float cosTheta) const { return cosPower_ > 0 ? std::pow(cosTheta, cosPower_) : 1.f; }
    const Shape *shape_;
    SpectrumPtr Lemit_;
    float scale_;
    bool twoSided_;
    float cosPower_;
    const MediumInterface *mi_;
};

class PointLight : public Light {
  public:
    PointLight(const Vec3f &p, SpectrumPtr I, float scale) : Light(LightType::DeltaPosition), p_(p), I_(std::move(I)), scale_(scale) {}
    float PowerY() const override { return 4 * Pi * scale_ * SpectrumToY(*I_); }
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &, const Vec3f &) const override { return 0; }
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;

  protected:
    Vec3f p_;
    SpectrumPtr I_;
    float scale_;
};

class SpotLight : public Light {
  public:
    SpotLight(const Vec3f &p, const Vec3f &dir, SpectrumPtr I, float scale, float totalWidthDeg, float falloffStartDeg);
    float PowerY() const override;
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &, const Vec3f &) const override { return 0; }
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;

  private:
    float Falloff(const Vec3f &wWorld) const;
    Vec3f p_;
    Frame frame_;
    SpectrumPtr I_;
    float scale_, cosFalloffStart_, cosFalloffEnd_;
};

// Environment light: constant spectrum or equirectangular HDR map (y-up, u = phi / 2pi, v = theta / pi).
class EnvironmentLight : public Light {
  public:
    EnvironmentLight(SpectrumPtr constant, float scale);
    EnvironmentLight(std::shared_ptr<Image> map, float scale, float rotationDeg);
    float PowerY() const override;
    void Preprocess(const Bounds3f &b) override { b.BoundingSphere(&center_, &radius_); radius_ = std::max(radius_, 1e-3f); }
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &ctx, const Vec3f &wi) const override;
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const override;
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;
    float SceneRadius() const { return radius_; }

  private:
    SampledSpectrum Lookup(const Vec3f &wWorld, const SampledWavelengths &lambda) const;
    Vec3f ToLight(const Vec3f &w) const { return rot_.Vector(w); }
    Vec3f ToWorld(const Vec3f &w) const { return rotInv_.Vector(w); }
    SpectrumPtr constant_;
    std::shared_ptr<Image> map_;
    Distribution2D distrib_;
    float scale_;
    Transform rot_, rotInv_;
    Vec3f center_;
    float radius_ = 1;
    float avgY_ = 0;
};

// Picks lights proportionally to their power.
class LightSampler {
  public:
    LightSampler() = default;
    explicit LightSampler(const std::vector<const Light *> &lights);
    const Light *Sample(float u, float *pmf) const;
    float PMF(const Light *light) const;
    bool Empty() const { return lights_.empty(); }

  private:
    std::vector<const Light *> lights_;
    Distribution1D distrib_;
    std::unordered_map<const Light *, int> index_;
};

// Direction <-> equirect uv (y-up).
inline Vec3f EquirectToDir(Vec2f uv) {
    float theta = uv.y * Pi, phi = uv.x * 2 * Pi;
    float s = std::sin(theta);
    return {s * std::cos(phi), std::cos(theta), s * std::sin(phi)};
}
inline Vec2f DirToEquirect(const Vec3f &d) {
    float theta = SafeACos(d.y);
    float phi = std::atan2(d.z, d.x);
    if (phi < 0) phi += 2 * Pi;
    return {phi * Inv2Pi, theta * InvPi};
}

} // namespace pr

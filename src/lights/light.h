#pragma once

#include "core/image.h"
#include "core/sampling.h"
#include "core/spectrum.h"
#include "geometry/shape.h"
#include "lights/ies.h"
#include "scene/interaction.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace pr {

enum class LightType { DeltaPosition, DeltaDirection, Area, Infinite };

// Cone of directions (cosTheta = +inf: empty).
struct DirectionCone {
    Vec3f w{0, 0, 1};
    float cosTheta = Infinity;
    DirectionCone() = default;
    DirectionCone(const Vec3f &w, float c) : w(Normalize(w)), cosTheta(c) {}
    bool IsEmpty() const { return cosTheta == Infinity; }
    static DirectionCone EntireSphere() { return DirectionCone(Vec3f(0, 0, 1), -1); }
};
DirectionCone Union(const DirectionCone &a, const DirectionCone &b);

// Spatial and directional emission bounds of a light (for the light BVH).
struct LightBounds {
    Bounds3f bounds;
    Vec3f w{0, 0, 1};
    float phi = 0;
    float cosTheta_o = 1, cosTheta_e = 0;
    bool twoSided = false;
    float Importance(const Vec3f &p, const Vec3f &n) const;
};
LightBounds Union(const LightBounds &a, const LightBounds &b);
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
    // Emission bounds; nullopt for infinite and distant lights.
    virtual std::optional<LightBounds> Bounds() const { return std::nullopt; }

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
    const SpectrumPtr &Emission() const { return Lemit_; }
    float Scale() const { return scale_; }
    bool TwoSided() const { return twoSided_; }
    float CosPower() const { return cosPower_; }
    std::optional<LightBounds> Bounds() const override;

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
    std::optional<LightBounds> Bounds() const override;
    const Vec3f &Position() const { return p_; }
    const SpectrumPtr &Intensity() const { return I_; }
    float Scale() const { return scale_; }
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
    std::optional<LightBounds> Bounds() const override;
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
    const SpectrumPtr &ConstantRadiance() const { return constant_; }
    float Scale() const { return scale_; }

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

// Point light with an IES photometric distribution; frame.z is the luminaire's nadir.
class GoniometricLight : public Light {
  public:
    GoniometricLight(const Vec3f &p, const Frame &frame, std::shared_ptr<IESProfile> profile, SpectrumPtr I, float scale)
        : Light(LightType::DeltaPosition), p_(p), frame_(frame), profile_(std::move(profile)), I_(std::move(I)), scale_(scale) {}
    float PowerY() const override { return scale_ * SpectrumToY(*I_) * profile_->Integral(); }
    std::optional<LightBounds> Bounds() const override {
        LightBounds lb;
        lb.bounds = Bounds3f(p_);
        lb.phi = 4 * Pi * scale_ * I_->MaxValue();
        lb.cosTheta_o = -1;
        lb.cosTheta_e = 0;
        return lb;
    }
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &, const Vec3f &) const override { return 0; }
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;

  private:
    Vec3f p_;
    Frame frame_;
    std::shared_ptr<IESProfile> profile_;
    SpectrumPtr I_;
    float scale_;
};

// Directional light (delta direction): irradiance E arriving from direction wLight.
class DistantLight : public Light {
  public:
    DistantLight(const Vec3f &wLight, SpectrumPtr E, float scale)
        : Light(LightType::DeltaDirection), w_(Normalize(wLight)), E_(std::move(E)), scale_(scale) {}
    void Preprocess(const Bounds3f &b) override { b.BoundingSphere(&center_, &radius_); radius_ = std::max(radius_, 1e-3f); }
    float PowerY() const override { return scale_ * SpectrumToY(*E_) * Pi * radius_ * radius_; }
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &, const Vec3f &) const override { return 0; }
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;

  private:
    Vec3f w_;
    SpectrumPtr E_;
    float scale_;
    Vec3f center_;
    float radius_ = 1;
};

// The sun as a small infinitely distant disk of uniform radiance (finite solid angle, so it can
// be hit by rays and sampled with MIS, unlike a delta light).
class SunLight : public Light {
  public:
    SunLight(const Vec3f &wSun, float angularRadiusDeg, SpectrumPtr L, float scale);
    void Preprocess(const Bounds3f &b) override { b.BoundingSphere(&center_, &radius_); radius_ = std::max(radius_, 1e-3f); }
    float PowerY() const override;
    std::optional<LightLiSample> SampleLi(const LightSampleContext &ctx, Vec2f u,
                                          const SampledWavelengths &lambda) const override;
    float PDF_Li(const LightSampleContext &ctx, const Vec3f &wi) const override;
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const override;
    std::optional<LightLeSample> SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const override;
    void PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const override;

  private:
    Frame frame_;
    float cosThetaMax_;
    SpectrumPtr L_;
    float scale_;
    Vec3f center_;
    float radius_ = 1;
};

// Preetham et al. 1999 analytic daylight. Produces an equirectangular linear-sRGB radiance map
// of the sky (without the sun disk) in kcd/m^2, and the sun's radiance spectrum (kcd/m^2 luminance
// scale) including an air-mass dependent reddening and attenuation.
std::shared_ptr<Image> MakePreethamSky(const Vec3f &sunDir, float turbidity, float groundAlbedo, int width, int height);
SpectrumPtr MakeSunRadiance(const Vec3f &sunDir, float turbidity, float angularRadiusDeg);

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

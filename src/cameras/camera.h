#pragma once

#include "core/math.h"
#include "core/spectrum.h"
#include "scene/interaction.h"

#include <optional>
#include <string>

namespace pr {

struct CameraSample {
    Vec2f pFilm;  // raster position in the extended film domain
    Vec2f pLens;  // [0,1)^2
};

struct CameraWiSample {
    SampledSpectrum Wi;
    Vec3f wi;       // from the reference point towards the lens
    float pdf = 0;  // solid angle at the reference point
    Vec2f pRaster;
    Interaction pLens;
};

struct CameraFilmInfo {
    int width, height;  // image resolution
    int margin;         // extra raster margin sampled on each side
};

class Camera {
  public:
    virtual ~Camera() = default;
    virtual std::optional<Ray> GenerateRay(const CameraSample &s) const = 0;
    // Wavelength-aware variant (lens dispersion may terminate secondary wavelengths) returning a
    // ray weight (vignetting). Defaults to the plain version with weight 1.
    virtual std::optional<Ray> GenerateRay(const CameraSample &s, SampledWavelengths &, float *weight) const {
        *weight = 1;
        return GenerateRay(s);
    }
    // Importance / pdfs for light tracing and BDPT (perspective cameras only).
    virtual bool SupportsLightTracing() const { return false; }
    virtual SampledSpectrum We(const Ray &, Vec2f *) const { return SampledSpectrum(0.f); }
    virtual void PDF_We(const Ray &, float *pdfPos, float *pdfDir) const { *pdfPos = *pdfDir = 0; }
    virtual std::optional<CameraWiSample> SampleWi(const Interaction &, Vec2f) const { return std::nullopt; }
    virtual Vec3f Position() const = 0;

    const Medium *medium = nullptr;
};

struct CameraParams {
    Transform cameraToWorld;
    float fovY = 40;             // degrees, vertical
    float lensRadius = 0;        // scene units
    float focalDistance = 1;     // scene units
    float orthoHeight = 2;       // orthographic: visible height in scene units
};

// Perspective camera with a thin lens (pinhole when lensRadius == 0). Camera space: +x right,
// +y up, +z forward.
class PerspectiveCamera : public Camera {
  public:
    PerspectiveCamera(const CameraParams &p, const CameraFilmInfo &film);
    std::optional<Ray> GenerateRay(const CameraSample &s) const override;
    bool SupportsLightTracing() const override { return true; }
    SampledSpectrum We(const Ray &ray, Vec2f *pRaster) const override;
    void PDF_We(const Ray &ray, float *pdfPos, float *pdfDir) const override;
    std::optional<CameraWiSample> SampleWi(const Interaction &ref, Vec2f u) const override;
    Vec3f Position() const override { return cameraToWorld_.Point(Vec3f(0, 0, 0)); }
    const Transform &CameraToWorld() const { return cameraToWorld_; }
    float TanHalfFov() const { return tanHalf_; }
    float LensRadius() const { return lensRadius_; }
    float FocalDistance() const { return focalDistance_; }

  private:
    // Point on the z=1 plane (camera space) for a raster position, and back.
    Vec3f RasterToCamera(Vec2f p) const;
    Vec2f CameraToRaster(const Vec3f &pz1) const;
    bool InsideSampleBounds(Vec2f p) const;

    Transform cameraToWorld_, worldToCamera_;
    CameraFilmInfo film_;
    float tanHalf_, aspect_;
    float lensRadius_, focalDistance_;
    float A_;  // area of the (extended) sampling window on the z=1 plane
    Vec3f forward_;
};

class OrthographicCamera : public Camera {
  public:
    OrthographicCamera(const CameraParams &p, const CameraFilmInfo &film);
    std::optional<Ray> GenerateRay(const CameraSample &s) const override;
    Vec3f Position() const override { return cameraToWorld_.Point(Vec3f(0, 0, 0)); }

  private:
    Transform cameraToWorld_;
    CameraFilmInfo film_;
    float halfH_, halfW_;
};

} // namespace pr

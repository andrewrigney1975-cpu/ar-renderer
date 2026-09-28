#pragma once

#include "cameras/camera.h"

#include <string>
#include <vector>

namespace pr {

struct LensElement {
    float curvatureRadius;  // scene units; 0 = aperture stop
    float thickness;        // distance to the next element towards the film
    float eta;              // refractive index at the d line (0 or 1 = air)
    float abbe;             // Abbe number (dispersion); 0 = none
    float apertureRadius;
};

// Physically based lens system (after pbrt's RealisticCamera): rays are traced from the film
// through a lens prescription. With dispersion enabled each element's IOR follows a Cauchy law
// fitted to its (n_d, Abbe) pair, so chromatic aberration emerges naturally (the hero
// wavelength is kept and secondary wavelengths are terminated).
// Only the path tracer supports this camera (it has no closed-form importance).
class RealisticCamera : public Camera {
  public:
    // elements: front (scene side) to rear (film side), lengths in scene units.
    RealisticCamera(const Transform &cameraToWorld, std::vector<LensElement> elements, float filmDiagonal,
                    const CameraFilmInfo &film, float focusDistance, bool dispersion);

    std::optional<Ray> GenerateRay(const CameraSample &s) const override;
    std::optional<Ray> GenerateRay(const CameraSample &s, SampledWavelengths &lambda, float *weight) const override;
    Vec3f Position() const override { return cameraToWorld_.Point(Vec3f(0, 0, 0)); }

    // Parses a pbrt lens description (radius, thickness, eta, aperture diameter [, abbe]) in mm.
    static bool LoadLensFile(const std::string &path, float unitsPerMm, std::vector<LensElement> *out, std::string *err);
    static std::vector<LensElement> DoubleGauss50(float unitsPerMm);

  private:
    bool TraceFromFilm(Ray r, float lambda, bool useLambda, Ray *out) const;
    float LensRearZ() const { return elements_.back().thickness; }
    float Eta(const LensElement &e, float lambda, bool useLambda) const;
    float FocusRearThickness(float focusDistance);
    void ComputeExitPupils();

    struct Pupil {
        float x0, y0, x1, y1;  // 2D bounds on the rear element plane (film point on +x axis)
        bool Empty() const { return x0 > x1 || y0 > y1; }
        float Area() const { return Empty() ? 0.f : (x1 - x0) * (y1 - y0); }
    };
    std::vector<Pupil> pupils_;
    float filmRadius_ = 1, exposureNorm_ = 1;

    Transform cameraToWorld_;
    std::vector<LensElement> elements_;
    CameraFilmInfo film_;
    float filmW_, filmH_;
    bool dispersion_;
};

} // namespace pr

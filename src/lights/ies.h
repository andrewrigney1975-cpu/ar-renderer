#pragma once

#include "core/math.h"
#include "core/sampling.h"

#include <memory>
#include <string>
#include <vector>

namespace pr {

// IESNA LM-63 photometric profile (type C). Directions are in the luminaire frame: +z is the
// nadir (the IES 0-degree vertical angle, the light's aiming direction), phi is measured from +x.
class IESProfile {
  public:
    static std::shared_ptr<IESProfile> Load(const std::string &path, std::string *err);
    static std::shared_ptr<IESProfile> Parse(const std::string &text, std::string *err);

    // Relative intensity in [0,1] (normalized by the peak candela).
    float Evaluate(const Vec3f &wLocal) const;
    float PeakCandela() const { return peak_; }
    // Integral of the relative intensity over the sphere (steradians).
    float Integral() const { return integral_; }

    // Importance sampling proportional to the profile; pdf w.r.t. solid angle.
    Vec3f Sample(Vec2f u, float *pdf) const;
    float PDF(const Vec3f &wLocal) const;

  private:
    float EvaluateAngles(float thetaDeg, float phiDeg) const;
    void Build();
    std::vector<float> vAngles_, hAngles_;
    std::vector<float> candela_;  // [h][v], normalized
    float peak_ = 0, integral_ = 0;
    Distribution2D distrib_;
    static constexpr int NU = 128, NV = 64;  // phi x cos(theta) tabulation
};

} // namespace pr

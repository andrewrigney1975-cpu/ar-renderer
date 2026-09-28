#pragma once

#include "core/spectrum.h"

#include <string>

namespace pr {

// Homogeneous participating medium with spectrally varying coefficients (units: 1/scene unit)
// and a Henyey-Greenstein phase function.
class Medium {
  public:
    Medium(SpectrumPtr sigma_a, SpectrumPtr sigma_s, float scale, float g)
        : sigma_a_(std::move(sigma_a)), sigma_s_(std::move(sigma_s)), scale_(scale), g_(Clamp(g, -0.99f, 0.99f)) {}

    SampledSpectrum SigmaA(const SampledWavelengths &lambda) const { return sigma_a_->Sample(lambda) * scale_; }
    SampledSpectrum SigmaS(const SampledWavelengths &lambda) const { return sigma_s_->Sample(lambda) * scale_; }
    float G() const { return g_; }
    bool IsChromatic() const { return !(sigma_a_->IsConstant() && sigma_s_->IsConstant()); }
    std::string name;

    SampledSpectrum Transmittance(float dist, const SampledWavelengths &lambda) const {
        if (std::isinf(dist)) {
            SampledSpectrum st = SigmaA(lambda) + SigmaS(lambda);
            SampledSpectrum r;
            for (int i = 0; i < NSpectrumSamples; ++i) r[i] = st[i] > 0 ? 0.f : 1.f;
            return r;
        }
        return Exp(-(SigmaA(lambda) + SigmaS(lambda)) * dist);
    }

    struct DistanceSample {
        bool scattered = false;
        float t = 0;
        SampledSpectrum weight;  // throughput multiplier (already divided by the sampling pdf)
    };
    // Sample a free-flight distance in [0, tMax) using one-sample spectral MIS over the active
    // wavelengths (balance heuristic). Unbiased for every wavelength.
    DistanceSample SampleDistance(float tMax, float u, const SampledWavelengths &lambda) const;

  private:
    SpectrumPtr sigma_a_, sigma_s_;
    float scale_, g_;
};

} // namespace pr

#pragma once

#include "core/memory.h"
#include "integrators/integrator.h"

namespace pr {

// Unidirectional volumetric path tracer with next-event estimation, MIS and Russian roulette.
// Serves as the reference integrator for validating BDPT and MLT.
class PathIntegrator : public Integrator {
  public:
    PathIntegrator(const IntegratorSettings &s, uint64_t seed) : settings_(s), seed_(seed) {}
    bool Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) override;

    // Radiance along a camera ray (exposed for tests).
    SampledSpectrum Li(const Scene &scene, Ray ray, SampledWavelengths &lambda, Sampler &sampler,
                       ScratchBuffer &buf) const;

  private:
    SampledSpectrum SampleLd(const Scene &scene, const Interaction &intr, const BSDF *bsdf, float g,
                             const Vec3f &ns, SampledWavelengths &lambda, Sampler &sampler) const;
    IntegratorSettings settings_;
    uint64_t seed_;
};

// Runs progressive passes of per-pixel sampling over the extended film domain. The callback
// renders one pixel sample. Returns completed samples per pixel.
using PixelSampleFn = std::function<void(int px, int py, int sampleIndex, int threadIndex)>;
int RunProgressivePasses(const Scene &scene, Film &film, RenderControl &control, const IntegratorSettings &s,
                         const char *stage, const PixelSampleFn &fn);

} // namespace pr

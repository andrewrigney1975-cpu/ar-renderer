#pragma once

#include "integrators/integrator.h"

#include <string>
#include <vector>

namespace pr {

struct RenderDeviceInfo {
    int index = -1;  // -1 = CPU
    std::string name;
    std::string backend;
    uint64_t memory = 0;
    int computeUnits = 0;
    bool isGpu = false;
};

// Devices usable for rendering: the CPU first, then SYCL GPU devices exposed by prender_gpu.dll
// (loaded on demand; absent module or runtime simply means "CPU only").
std::vector<RenderDeviceInfo> ListRenderDevices(std::string *gpuStatus = nullptr);

// GPU path tracer (oneAPI/SYCL trial). Unbiased spectral path tracing with NEE + MIS on the
// device; materials/lights outside the supported subset are approximated and reported.
class GpuPathIntegrator : public Integrator {
  public:
    GpuPathIntegrator(const IntegratorSettings &s, uint64_t seed, int device) : settings_(s), seed_(seed), device_(device) {}
    bool Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) override;

  private:
    IntegratorSettings settings_;
    uint64_t seed_;
    int device_;
};

} // namespace pr

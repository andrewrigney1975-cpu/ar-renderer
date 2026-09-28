#include "integrators/integrator.h"

#include "integrators/bdpt.h"
#include "integrators/mlt.h"
#include "integrators/path.h"

namespace pr {

std::unique_ptr<Integrator> CreateIntegrator(const IntegratorSettings &settings, uint64_t seed) {
    IntegratorSettings s = settings;
    switch (s.type) {
    case IntegratorType::Path:
        // Unlimited depth: Russian roulette alone terminates paths (unbiased).
        if (s.maxDepth <= 0) s.maxDepth = 0;
        return std::make_unique<PathIntegrator>(s, seed);
    case IntegratorType::BDPT:
        if (s.maxDepth <= 0) s.maxDepth = 256;
        return std::make_unique<BDPTIntegrator>(s, seed);
    case IntegratorType::MMLT:
        if (s.maxDepth <= 0) s.maxDepth = 32;
        return std::make_unique<MLTIntegrator>(s, seed, true);
    case IntegratorType::PSSMLT:
        if (s.maxDepth <= 0) s.maxDepth = 32;
        return std::make_unique<MLTIntegrator>(s, seed, false);
    }
    return nullptr;
}

} // namespace pr

#pragma once

#include "core/film.h"
#include "core/rng.h"
#include "core/serialize.h"
#include "scene/scene.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace pr {

// Source of sample values for path construction.
class Sampler {
  public:
    virtual ~Sampler() = default;
    virtual float Get1D() = 0;
    virtual Vec2f Get2D() {
        float a = Get1D();
        float b = Get1D();
        return {a, b};
    }
    virtual void StartStream(int) {}
};

class IndependentSampler : public Sampler {
  public:
    void StartPixelSample(uint64_t pixelIndex, uint64_t sampleIndex, uint64_t seed) {
        rng_.SetSequence(Hash(pixelIndex, seed), RNG::MixBits(sampleIndex * 0x9E3779B97F4A7C15ull + 1));
    }
    float Get1D() override { return rng_.UniformFloat(); }

  private:
    RNG rng_;
};

struct ProgressInfo {
    std::string stage;
    double fraction = 0;       // 0..1 (-1 if unknown)
    double elapsed = 0;        // seconds
    double samplesPerPixel = 0;  // spp (path/bdpt) or mutations per pixel (mlt)
    double acceptRate = -1;    // mlt only
};

// Shared between the integrator and its caller (CLI) for progress, previews and cancellation.
class RenderControl {
  public:
    std::atomic<bool> cancel{false};
    std::function<void(const ProgressInfo &)> onProgress;
    std::function<void(const Film &, double scale)> onPreview;
    double previewInterval = 0;  // seconds; 0 disables previews
    double progressInterval = 0.5;
    // Checkpointing: state is written to checkpointPath periodically, on cancel and at the end;
    // a render resumes from resumePath when set (it must match the scene and settings).
    std::string checkpointPath, resumePath;
    double checkpointInterval = 60;
    uint64_t fingerprint = 0, seed = 0;

    bool CheckpointDue() {
        if (checkpointPath.empty()) return false;
        auto now = Clock::now();
        if (std::chrono::duration<double>(now - lastCheckpoint_).count() < checkpointInterval) return false;
        lastCheckpoint_ = now;
        return true;
    }

    void Start() {
        start_ = Clock::now();
        lastPreview_ = lastProgress_ = lastCheckpoint_ = start_;
    }
    double Elapsed() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }
    void Progress(const ProgressInfo &p, bool force = false) {
        auto now = Clock::now();
        if (!onProgress) return;
        if (force || std::chrono::duration<double>(now - lastProgress_).count() >= progressInterval) {
            lastProgress_ = now;
            onProgress(p);
        }
    }
    void MaybePreview(const Film &film, double scale, bool force = false) {
        if (!onPreview || previewInterval <= 0) return;
        auto now = Clock::now();
        if (force || std::chrono::duration<double>(now - lastPreview_).count() >= previewInterval) {
            lastPreview_ = now;
            onPreview(film, scale);
        }
    }

  private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_, lastPreview_, lastProgress_, lastCheckpoint_;
};

class Integrator {
  public:
    virtual ~Integrator() = default;
    // Renders into film. On success returns true and sets *scale, the factor to apply when
    // resolving the film (the film accumulates unnormalized sums).
    virtual bool Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) = 0;
};

// Checkpoint files: a validated header (integrator, settings, film size, scene fingerprint),
// the film accumulators and an integrator-specific body.
bool WriteCheckpoint(const RenderControl &control, const IntegratorSettings &s, const Film &film,
                     const std::function<void(BinaryWriter &)> &body, std::string *err);
bool ReadCheckpoint(const RenderControl &control, const IntegratorSettings &s, Film &film,
                    const std::function<bool(BinaryReader &)> &body, std::string *err);

std::unique_ptr<Integrator> CreateIntegrator(const IntegratorSettings &settings, uint64_t seed);

} // namespace pr

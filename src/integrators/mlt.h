#pragma once

#include "integrators/bdpt.h"

namespace pr {

// Primary-sample-space sampler with Kelemen mutations (lazy per-coordinate updates).
class MLTSampler : public Sampler {
  public:
    MLTSampler(uint64_t rngSequenceIndex, float sigma, float largeStepProbability, int streamCount)
        : rng_(rngSequenceIndex), sigma_(sigma), largeStepProbability_(largeStepProbability), streamCount_(streamCount) {}

    void StartIteration() {
        currentIteration_++;
        largeStep_ = rng_.UniformFloat() < largeStepProbability_;
    }
    void Accept() {
        if (largeStep_) lastLargeStepIteration_ = currentIteration_;
    }
    void Reject() {
        for (auto &x : X_)
            if (x.lastModificationIteration == currentIteration_) x.Restore();
        --currentIteration_;
    }
    void StartStream(int index) override {
        streamIndex_ = index;
        sampleIndex_ = 0;
    }
    float Get1D() override {
        int64_t index = GetNextIndex();
        EnsureReady(index);
        return X_[index].value;
    }
    bool LargeStep() const { return largeStep_; }
    RNG &Rng() { return rng_; }

  private:
    struct PrimarySample {
        float value = 0;
        int64_t lastModificationIteration = 0;
        float valueBackup = 0;
        int64_t modifyBackup = 0;
        void Backup() {
            valueBackup = value;
            modifyBackup = lastModificationIteration;
        }
        void Restore() {
            value = valueBackup;
            lastModificationIteration = modifyBackup;
        }
    };
    int64_t GetNextIndex() { return streamIndex_ + int64_t(streamCount_) * sampleIndex_++; }
    void EnsureReady(int64_t index);

    RNG rng_;
    float sigma_, largeStepProbability_;
    int streamCount_;
    std::vector<PrimarySample> X_;
    int64_t currentIteration_ = 0;
    bool largeStep_ = true;
    int64_t lastLargeStepIteration_ = 0;
    int streamIndex_ = 0, sampleIndex_ = 0;
};

class MLTIntegrator : public Integrator {
  public:
    MLTIntegrator(const IntegratorSettings &s, uint64_t seed, bool multiplexed)
        : settings_(s), seed_(seed), multiplexed_(multiplexed) {}
    bool Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) override;

  private:
    IntegratorSettings settings_;
    uint64_t seed_;
    bool multiplexed_;
};

} // namespace pr

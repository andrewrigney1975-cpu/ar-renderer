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

    void Serialize(BinaryWriter &w) const {
        uint64_t st, inc;
        rng_.GetState(&st, &inc);
        w.Put(st);
        w.Put(inc);
        w.Put(currentIteration_);
        w.Put(lastLargeStepIteration_);
        w.Put(uint8_t(largeStep_));
        w.Put(uint64_t(X_.size()));
        for (const PrimarySample &x : X_) {
            w.Put(x.value);
            w.Put(x.lastModificationIteration);
        }
    }
    bool Deserialize(BinaryReader &r) {
        uint64_t st, inc, n;
        uint8_t ls;
        if (!r.Get(&st) || !r.Get(&inc) || !r.Get(&currentIteration_) || !r.Get(&lastLargeStepIteration_) ||
            !r.Get(&ls) || !r.Get(&n) || n > (uint64_t(1) << 24))
            return false;
        rng_.SetState(st, inc);
        largeStep_ = ls != 0;
        X_.resize(size_t(n));
        for (PrimarySample &x : X_)
            if (!r.Get(&x.value) || !r.Get(&x.lastModificationIteration)) return false;
        return true;
    }
    RNG &Rng() { return rng_; }

  private:
    struct PrimarySample {
        float value = 0;
        // -1 = never drawn. A coordinate that a path uses for the first time must start from a
        // uniform value (conceptually drawn by the initial large step), not from 0: otherwise,
        // until the chain's first accepted large step, it is small-stepped away from 0 and stays
        // near 0/1 for ~1/sigma^2 iterations. Paths of varying length (random walks in media)
        // keep reaching new coordinates, which biased PSSMLT (e.g. 16% too red on chromatic
        // subsurface scattering).
        int64_t lastModificationIteration = -1;
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

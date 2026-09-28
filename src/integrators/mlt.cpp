#include "integrators/mlt.h"

#include "core/log.h"
#include "core/parallel.h"

namespace pr {

void MLTSampler::EnsureReady(int64_t index) {
    if (index >= int64_t(X_.size())) X_.resize(size_t(index) + 1);
    PrimarySample &Xi = X_[size_t(index)];
    // Bring the sample up to date with the last large step, if it was not touched since.
    if (Xi.lastModificationIteration < lastLargeStepIteration_) {
        Xi.value = rng_.UniformFloat();
        Xi.lastModificationIteration = lastLargeStepIteration_;
    }
    Xi.Backup();
    if (largeStep_) {
        Xi.value = rng_.UniformFloat();
    } else {
        // Apply the accumulated small-step perturbations (Gaussian, variance grows with count).
        int64_t nSmall = currentIteration_ - Xi.lastModificationIteration;
        float normalSample = Sqrt2 * ErfInv(2 * rng_.UniformFloat() - 1);
        float effSigma = sigma_ * std::sqrt(float(nSmall));
        Xi.value += normalSample * effSigma;
        Xi.value -= std::floor(Xi.value);
        if (!(Xi.value < 1)) Xi.value = OneMinusEpsilon;
        if (Xi.value < 0) Xi.value = 0;
    }
    Xi.lastModificationIteration = currentIteration_;
}

namespace {

constexpr int kCameraStream = 0, kLightStream = 1, kConnectionStream = 2, kNumStreams = 3;

struct Splat {
    Vec2f p;
    XYZ xyz;
};

struct PathSample {
    std::vector<Splat> splats;
    float c = 0;
    void Clear() {
        splats.clear();
        c = 0;
    }
};

float Contribution(const XYZ &x) {
    float c = std::abs(x.x) + std::abs(x.y) + std::abs(x.z);
    return std::isfinite(c) ? c : 0.f;
}

struct ThreadData {
    ScratchBuffer buf;
    std::vector<Vertex> cam, light;
};

class PathEvaluator {
  public:
    PathEvaluator(const Scene &scene, const Film &film, int maxDepth, bool multiplexed)
        : scene_(scene), maxDepth_(maxDepth), multiplexed_(multiplexed), margin_(film.Margin()),
          sw_(float(film.SampleWidth())), sh_(float(film.SampleHeight())) {}

    int DepthSlots() const { return multiplexed_ ? maxDepth_ + 1 : 1; }

    void Eval(MLTSampler &sampler, int depth, ThreadData &td, PathSample *out) const {
        out->Clear();
        if (multiplexed_) EvalMultiplexed(sampler, depth, td, out);
        else EvalFull(sampler, td, out);
        td.buf.Reset();
        if (!std::isfinite(out->c)) out->Clear();
    }

  private:
    Vec2f FilmPoint(Vec2f u) const { return {-float(margin_) + u.x * sw_, -float(margin_) + u.y * sh_}; }

    void EvalMultiplexed(MLTSampler &sampler, int depth, ThreadData &td, PathSample *out) const {
        sampler.StartStream(kCameraStream);
        int s, t, nStrategies;
        if (depth == 0) {
            nStrategies = 1;
            s = 0;
            t = 2;
        } else {
            nStrategies = depth + 2;
            s = std::min(int(sampler.Get1D() * nStrategies), nStrategies - 1);
            t = nStrategies - s;
        }
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
        Vec2f pRaster = FilmPoint(sampler.Get2D());
        BDPTContext ctx{scene_, lambda, sampler, td.buf, false};
        if (GenerateCameraSubpath(ctx, t, pRaster, td.cam.data()) != t) return;
        sampler.StartStream(kLightStream);
        if (GenerateLightSubpath(ctx, s, td.light.data()) != s) return;
        sampler.StartStream(kConnectionStream);
        SampledSpectrum L = ConnectBDPT(ctx, td.light.data(), td.cam.data(), s, t, &pRaster) * float(nStrategies);
        if (!L) return;
        XYZ xyz = ToXYZ(L, lambda);
        out->c = Contribution(xyz);
        if (out->c > 0) out->splats.push_back({pRaster, xyz});
    }

    void EvalFull(MLTSampler &sampler, ThreadData &td, PathSample *out) const {
        sampler.StartStream(kCameraStream);
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(sampler.Get1D());
        Vec2f pFilm = FilmPoint(sampler.Get2D());
        BDPTContext ctx{scene_, lambda, sampler, td.buf, false};
        int nCamera = GenerateCameraSubpath(ctx, maxDepth_ + 2, pFilm, td.cam.data());
        sampler.StartStream(kLightStream);
        int nLight = GenerateLightSubpath(ctx, maxDepth_ + 1, td.light.data());
        sampler.StartStream(kConnectionStream);
        SampledSpectrum Lcam(0.f);
        std::vector<std::pair<Vec2f, SampledSpectrum>> lightSplats;
        for (int t = 1; t <= nCamera; ++t)
            for (int s = 0; s <= nLight; ++s) {
                int depth = t + s - 2;
                if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth_) continue;
                Vec2f pNew = pFilm;
                SampledSpectrum Lpath = ConnectBDPT(ctx, td.light.data(), td.cam.data(), s, t, &pNew);
                if (!Lpath) continue;
                if (t != 1) Lcam += Lpath;
                else lightSplats.push_back({pNew, Lpath});
            }
        // Convert after all subpaths exist, so dispersion-terminated wavelengths are respected.
        float c = 0;
        if (Lcam) {
            XYZ x = ToXYZ(Lcam, lambda);
            out->splats.push_back({pFilm, x});
            c += Contribution(x);
        }
        for (auto &[p, L] : lightSplats) {
            XYZ x = ToXYZ(L, lambda);
            out->splats.push_back({p, x});
            c += Contribution(x);
        }
        out->c = c;
    }

    const Scene &scene_;
    int maxDepth_;
    bool multiplexed_;
    int margin_;
    float sw_, sh_;
};

void SplatSample(Film &film, const PathSample &ps, float weight) {
    if (weight <= 0 || ps.c <= 0) return;
    float w = weight / ps.c;
    for (const Splat &s : ps.splats) film.AddXYZ(s.p, s.xyz * w);
}

uint64_t BootstrapSequence(uint64_t seed, uint64_t index) { return Hash(seed, index) ^ index; }

} // namespace

bool MLTIntegrator::Render(const Scene &scene, Film &film, RenderControl &control, double *scale, std::string *err) {
    if (!scene.camera || !scene.camera->SupportsLightTracing()) {
        *err = "MLT requires a perspective or thin-lens camera";
        return false;
    }
    const int maxDepth = settings_.maxDepth;
    PathEvaluator eval(scene, film, maxDepth, multiplexed_);
    const int slots = eval.DepthSlots();
    const int nThreads = ThreadCount();
    std::vector<std::unique_ptr<ThreadData>> td(nThreads);
    for (auto &t : td) {
        t = std::make_unique<ThreadData>();
        t->cam.resize(maxDepth + 2);
        t->light.resize(maxDepth + 1);
    }
    const float sigma = settings_.sigma, largeStep = settings_.largeStepProbability;
    *scale = 0;

    struct Chain {
        std::unique_ptr<MLTSampler> sampler;
        PathSample current, proposed;
        int depth = 0;
        RNG rng;
        int64_t accepted = 0, total = 0;
    };
    const int nChains = std::max(1, settings_.chains);
    std::vector<Chain> chains(nChains);
    double b = 0;
    int64_t totalMutations = 0;

    auto writeState = [&](BinaryWriter &w) {
        w.Put(b);
        w.Put(totalMutations);
        w.Put(int32_t(nChains));
        w.Put(int32_t(slots));
        for (const Chain &c : chains) {
            w.Put(int32_t(c.depth));
            uint64_t st, inc;
            c.rng.GetState(&st, &inc);
            w.Put(st);
            w.Put(inc);
            w.Put(c.accepted);
            w.Put(c.total);
            w.Put(c.current.c);
            w.Put(uint32_t(c.current.splats.size()));
            for (const Splat &sp : c.current.splats) {
                w.Put(sp.p);
                w.Put(sp.xyz);
            }
            c.sampler->Serialize(w);
        }
    };
    auto readState = [&](BinaryReader &r) {
        int32_t nc, sl;
        if (!r.Get(&b) || !r.Get(&totalMutations) || !r.Get(&nc) || !r.Get(&sl) || nc != nChains || sl != slots) return false;
        for (Chain &c : chains) {
            int32_t depth;
            uint64_t st, inc;
            uint32_t nSplats;
            if (!r.Get(&depth) || !r.Get(&st) || !r.Get(&inc) || !r.Get(&c.accepted) || !r.Get(&c.total) ||
                !r.Get(&c.current.c) || !r.Get(&nSplats) || nSplats > 100000)
                return false;
            c.depth = depth;
            c.rng.SetState(st, inc);
            c.current.splats.resize(nSplats);
            for (Splat &sp : c.current.splats)
                if (!r.Get(&sp.p) || !r.Get(&sp.xyz)) return false;
            c.sampler = std::make_unique<MLTSampler>(0, sigma, largeStep, kNumStreams);
            if (!c.sampler->Deserialize(r)) return false;
        }
        return true;
    };
    auto checkpoint = [&] {
        std::string e;
        if (b > 0 && !WriteCheckpoint(control, settings_, film, writeState, &e)) LogWarning("{}", e);
    };

    if (!control.resumePath.empty()) {
        // Resume: restore the film, the normalization and every chain's exact state.
        if (!ReadCheckpoint(control, settings_, film, readState, err)) return false;
        LogInfo("resumed at {:.1f} mutations per pixel", double(totalMutations) / film.SamplePixelCount());
    } else {
        // ---- Bootstrap: estimate the normalization b and seed the chains (removes start-up bias).
        const int64_t nBootstrap = std::max<int64_t>(1, settings_.bootstrapSamples);
        std::vector<float> weights(size_t(nBootstrap * slots), 0.f);
        std::atomic<int64_t> bootstrapDone{0};
        ParallelFor(nBootstrap, [&](int64_t i, int ti) {
            if (control.cancel) return;
            PathSample ps;
            for (int depth = 0; depth < slots; ++depth) {
                int64_t idx = i * slots + depth;
                MLTSampler sampler(BootstrapSequence(seed_, uint64_t(idx)), sigma, largeStep, kNumStreams);
                eval.Eval(sampler, depth, *td[ti], &ps);
                weights[size_t(idx)] = ps.c;
            }
            int64_t done = ++bootstrapDone;
            if (ti == 0) {
                ProgressInfo pi;
                pi.stage = "bootstrap";
                pi.elapsed = control.Elapsed();
                pi.fraction = double(done) / nBootstrap;
                control.Progress(pi);
            }
        }, 64);
        if (control.cancel) return true;
        Distribution1D bootstrap(weights);
        b = double(bootstrap.Integral()) * slots;
        {
            ProgressInfo pi;
            pi.stage = "bootstrap";
            pi.elapsed = control.Elapsed();
            pi.fraction = 1;
            control.Progress(pi, true);
        }
        if (b <= 0) {
            LogWarning("MLT bootstrap found no light-carrying paths; the image is black");
            return true;
        }
        LogVerbose("MLT bootstrap: b = {}", b);

        // ---- Initialize chains by resampling the bootstrap paths.
        ParallelFor(nChains, [&](int64_t i, int ti) {
            Chain &c = chains[size_t(i)];
            c.rng.SetSequence(Hash(seed_, uint64_t(i), 0x5eedull));
            int idx = bootstrap.SampleDiscrete(c.rng.UniformFloat());
            c.depth = idx % slots;
            c.sampler = std::make_unique<MLTSampler>(BootstrapSequence(seed_, uint64_t(idx)), sigma, largeStep, kNumStreams);
            eval.Eval(*c.sampler, c.depth, *td[ti], &c.current);
        });
    }

    // ---- Run the chains in rounds (each round ~1 mutation per film pixel).
    const int64_t pixels = film.SamplePixelCount();
    const int64_t perChainPerRound = std::max<int64_t>(1, pixels / nChains);
    const double targetMpp = settings_.timeLimit > 0 ? std::numeric_limits<double>::infinity()
                                                     : std::max(0.01, settings_.mutationsPerPixel);
    // Integer per-chain budget: every chain runs exactly the same number of mutations, and the
    // total does not depend on how the render was split by checkpoints.
    const int64_t targetPerChain = std::isfinite(targetMpp)
                                       ? int64_t(std::ceil(targetMpp * double(pixels) / nChains - 1e-9))
                                       : std::numeric_limits<int64_t>::max();
    while (!control.cancel) {
        int64_t donePerChain = totalMutations / nChains;
        if (donePerChain >= targetPerChain) break;
        int64_t k = std::min(perChainPerRound, targetPerChain - donePerChain);
        ParallelFor(nChains, [&](int64_t i, int ti) {
            Chain &c = chains[size_t(i)];
            for (int64_t j = 0; j < k; ++j) {
                c.sampler->StartIteration();
                eval.Eval(*c.sampler, c.depth, *td[ti], &c.proposed);
                float accept = c.current.c > 0 ? std::min(1.f, c.proposed.c / c.current.c) : 1.f;
                // Expected-value splatting (Veach): both states contribute every iteration.
                if (accept > 0) SplatSample(film, c.proposed, accept);
                if (accept < 1) SplatSample(film, c.current, 1 - accept);
                if (c.rng.UniformFloat() < accept) {
                    std::swap(c.current, c.proposed);
                    c.sampler->Accept();
                    ++c.accepted;
                } else {
                    c.sampler->Reject();
                }
                ++c.total;
            }
        });
        totalMutations += k * nChains;
        double mpp = double(totalMutations) / pixels;
        double curScale = b / mpp;
        int64_t acc = 0, tot = 0;
        for (const Chain &c : chains) {
            acc += c.accepted;
            tot += c.total;
        }
        double elapsed = control.Elapsed();
        ProgressInfo pi;
        pi.stage = "render";
        pi.elapsed = elapsed;
        pi.samplesPerPixel = mpp;
        pi.acceptRate = tot > 0 ? double(acc) / tot : 0;
        pi.fraction = settings_.timeLimit > 0 ? std::min(1.0, elapsed / settings_.timeLimit) : std::min(1.0, mpp / targetMpp);
        control.Progress(pi, mpp >= targetMpp);
        control.MaybePreview(film, curScale);
        if (control.CheckpointDue()) checkpoint();
        if (settings_.timeLimit > 0 && elapsed >= settings_.timeLimit) break;
    }
    checkpoint();
    *scale = totalMutations > 0 ? b / (double(totalMutations) / pixels) : 0.0;
    return true;
}

// ---------------------------------------------------------------------------------------------

} // namespace pr

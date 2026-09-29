#pragma once

#include "core/math.h"
#include "core/spectrum.h"

#include <memory>
#include <string>
#include <vector>

namespace pr {

// Participating medium with spectrally varying coefficients (units: 1/scene unit) and a
// Henyey-Greenstein phase function. Rays passed in must have normalized directions.
// Component-wise a / b, with 0 where b is 0.
inline SampledSpectrum SafeDiv(const SampledSpectrum &a, const SampledSpectrum &b) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i) r[i] = b[i] != 0 ? a[i] / b[i] : 0.f;
    return r;
}

class Medium {
  public:
    explicit Medium(float g) : g_(Clamp(g, -0.99f, 0.99f)) {}
    virtual ~Medium() = default;

    float G() const { return g_; }
    std::string name;

    struct DistanceSample {
        bool scattered = false;
        float t = 0;
        SampledSpectrum weight;  // throughput multiplier (already divided by the sampling pdf)
        // Hero sampling only: pdf of this event had wavelength i driven the sampling, divided by
        // the pdf of the hero-driven sampling that was used (1 when sampling is wavelength-neutral).
        SampledSpectrum pdfRatio{1.f};
    };

    // Transmittance along ray over [0, tMax).
    virtual SampledSpectrum Transmittance(const Ray &ray, float tMax, const SampledWavelengths &lambda) const = 0;
    // Sample a real scattering event in [0, tMax) (or pass through). u drives all randomness,
    // so the result is a deterministic function of u (required for Metropolis sampling). The
    // sampling pdf does not depend on which wavelength is the hero, as BDPT's MIS requires.
    virtual DistanceSample SampleDistance(const Ray &ray, float tMax, float u, const SampledWavelengths &lambda) const = 0;
    // Distance sampling driven by the hero wavelength alone, for path-level spectral MIS (Miller
    // et al. 2019): the caller accumulates pdfRatio over the path and divides the path's estimate
    // by its average. Per-event mixing (SampleDistance) lets per-wavelength weights grow
    // exponentially with the number of scattering events in chromatic media; this keeps them
    // bounded by the number of wavelengths. Default: wavelength-neutral sampling (ratio 1).
    virtual DistanceSample SampleDistanceHero(const Ray &ray, float tMax, float u, const SampledWavelengths &lambda) const {
        return SampleDistance(ray, tMax, u, lambda);
    }
    // sigma_s / sigma_t: the colour of a scattering interior (used for denoiser guide buffers).
    virtual SampledSpectrum ScatteringAlbedo(const SampledWavelengths &lambda) const = 0;

  private:
    float g_;
};

// Homogeneous medium: closed-form transmittance and one-sample spectral-MIS distance sampling.
class HomogeneousMedium : public Medium {
  public:
    HomogeneousMedium(SpectrumPtr sigma_a, SpectrumPtr sigma_s, float scale, float g)
        : Medium(g), sigma_a_(std::move(sigma_a)), sigma_s_(std::move(sigma_s)), scale_(scale) {}

    SampledSpectrum SigmaA(const SampledWavelengths &lambda) const { return sigma_a_->Sample(lambda) * scale_; }
    SampledSpectrum SigmaS(const SampledWavelengths &lambda) const { return sigma_s_->Sample(lambda) * scale_; }
    const SpectrumPtr &SigmaAPtr() const { return sigma_a_; }
    const SpectrumPtr &SigmaSPtr() const { return sigma_s_; }
    float Scale() const { return scale_; }

    SampledSpectrum Transmittance(const Ray &ray, float tMax, const SampledWavelengths &lambda) const override;
    DistanceSample SampleDistance(const Ray &ray, float tMax, float u, const SampledWavelengths &lambda) const override;
    DistanceSample SampleDistanceHero(const Ray &ray, float tMax, float u, const SampledWavelengths &lambda) const override;
    SampledSpectrum ScatteringAlbedo(const SampledWavelengths &lambda) const override {
        return SafeDiv(sigma_s_->Sample(lambda), sigma_a_->Sample(lambda) + sigma_s_->Sample(lambda));
    }

  private:
    SpectrumPtr sigma_a_, sigma_s_;
    float scale_;
};

// Scalar density field in medium space.
class DensityField {
  public:
    virtual ~DensityField() = default;
    virtual float Density(const Vec3f &p) const = 0;
    virtual float MaxDensity() const = 0;
    // Region outside of which the density is zero (may be set by the loader).
    Bounds3f bounds;
};

// Voxel grid with trilinear interpolation over `bounds` (cell-centred samples).
class VoxelGridField : public DensityField {
  public:
    VoxelGridField(int nx, int ny, int nz, std::vector<float> data, const Bounds3f &b);
    float Density(const Vec3f &p) const override;
    float MaxDensity() const override { return maxDensity_; }
    // Mitsuba 0.5 binary .vol (float32, 1 channel or averaged).
    static std::shared_ptr<VoxelGridField> LoadVol(const std::string &path, std::string *err);

  private:
    float At(int x, int y, int z) const {
        x = Clamp(x, 0, nx_ - 1);
        y = Clamp(y, 0, ny_ - 1);
        z = Clamp(z, 0, nz_ - 1);
        return data_[(size_t(z) * ny_ + y) * nx_ + x];
    }
    int nx_, ny_, nz_;
    std::vector<float> data_;
    float maxDensity_ = 0;
};

// Float grid from an uncompressed NanoVDB (.nvdb) file; gridName empty = first float grid.
std::shared_ptr<DensityField> LoadNanoVDB(const std::string &path, const std::string &gridName, std::string *err);

// Procedural cloud-like density: fBm noise thresholded, optionally faded towards the edges of
// its bounds by an ellipsoidal falloff.
class NoiseField : public DensityField {
  public:
    NoiseField(float frequency, int octaves, float threshold, float sharpness, bool ellipsoidFalloff, Vec3f offset)
        : freq_(frequency), octaves_(octaves), threshold_(threshold), sharpness_(sharpness),
          falloff_(ellipsoidFalloff), offset_(offset) {}
    float Density(const Vec3f &p) const override;
    float MaxDensity() const override { return 1.f; }

  private:
    float freq_;
    int octaves_;
    float threshold_, sharpness_;
    bool falloff_;
    Vec3f offset_;
};

// Heterogeneous medium sigma(p, lambda) = density(p) * sigma_base(lambda) * scale, with
// unbiased spectral tracking (Kutz et al. 2017) for sampling and ratio tracking for transmittance.
class GridMedium : public Medium {
  public:
    GridMedium(SpectrumPtr sigma_a, SpectrumPtr sigma_s, float scale, float g,
               std::shared_ptr<DensityField> field, const Transform &mediumToWorld)
        : Medium(g), sigma_a_(std::move(sigma_a)), sigma_s_(std::move(sigma_s)), scale_(scale),
          field_(std::move(field)), worldToMedium_(mediumToWorld.Inverse()) {}

    SampledSpectrum Transmittance(const Ray &ray, float tMax, const SampledWavelengths &lambda) const override;
    DistanceSample SampleDistance(const Ray &ray, float tMax, float u, const SampledWavelengths &lambda) const override;
    SampledSpectrum ScatteringAlbedo(const SampledWavelengths &lambda) const override {
        return SafeDiv(sigma_s_->Sample(lambda), sigma_a_->Sample(lambda) + sigma_s_->Sample(lambda));
    }
    DensityField &Field() { return *field_; }
    // If the field has no bounds yet, bound it by a world-space box (e.g. the enclosing object).
    void FitBoundsToWorld(const Bounds3f &world) {
        if (!field_->bounds.IsEmpty() || world.IsEmpty()) return;
        Bounds3f b;
        for (int i = 0; i < 8; ++i)
            b = Union(b, worldToMedium_.Point(Vec3f(world[i & 1].x, world[(i >> 1) & 1].y, world[(i >> 2) & 1].z)));
        field_->bounds = b;
    }
    bool HasBounds() const { return !field_->bounds.IsEmpty(); }

  private:
    // Clip the (world) ray segment to the field bounds, returning the medium-space ray.
    bool Clip(const Ray &ray, float tMax, Ray *mray, float *t0, float *t1) const;
    SpectrumPtr sigma_a_, sigma_s_;
    float scale_;
    std::shared_ptr<DensityField> field_;
    Transform worldToMedium_;
};

} // namespace pr

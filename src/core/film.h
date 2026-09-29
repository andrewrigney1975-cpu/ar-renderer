#pragma once

#include "core/color.h"
#include "core/image.h"
#include "core/serialize.h"
#include "core/spectrum.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>

namespace pr {

enum class FilterType { Box, Gaussian, BlackmanHarris };

struct FilterSettings {
    FilterType type = FilterType::BlackmanHarris;
    float radius = 1.5f;  // in pixels
};

// Separable reconstruction filter normalized so that its integral over the plane is 1.
class Filter {
  public:
    explicit Filter(const FilterSettings &s);
    float Radius() const { return radius_; }
    float Evaluate(float dx, float dy) const { return Eval1D(dx) * Eval1D(dy) * norm_; }
    float Eval1D(float x) const;
    FilterType Type() const { return type_; }
    float Sigma() const { return sigma_; }
    float Norm() const { return norm_; }

  private:
    FilterType type_;
    float radius_;
    float sigma_ = 0.5f;
    float norm_ = 1;
};

struct FilmSettings {
    int width = 640, height = 360;
    FilterSettings filter;
    ColorSpaceId colorSpace = ColorSpaceId::sRGB;
    float exposure = 0;       // EV
    float whiteBalance = 0;   // Kelvin; 0 disables
    // Firefly clamp: per camera sample, luminance (CIE Y) above this is scaled down, keeping the
    // colour. 0 disables. Biased (energy is removed), so off by default; applies to samples added
    // with AddSample (path tracer, BDPT and the GPU path tracer), not to MLT splats.
    float clampLuminance = 0;
};

// Spectral film. Samples are splatted (with the filter) as CIE XYZ into an accumulation buffer
// with atomic adds. The film sampling domain extends beyond the image by the filter radius so
// that every pixel receives the same sample density (no normalization by filter weight sums,
// which would be a biased ratio estimator).
class Film {
  public:
    explicit Film(const FilmSettings &s);

    const FilmSettings &Settings() const { return settings_; }
    int Width() const { return settings_.width; }
    int Height() const { return settings_.height; }
    // Extended sampling domain in raster space: [-margin, width+margin) x [-margin, height+margin).
    int Margin() const { return margin_; }
    const Filter &GetFilter() const { return filter_; }
    int SampleWidth() const { return settings_.width + 2 * margin_; }
    int SampleHeight() const { return settings_.height + 2 * margin_; }
    int64_t SamplePixelCount() const { return int64_t(SampleWidth()) * SampleHeight(); }

    // Add radiance L carried by wavelengths lambda at continuous raster position p.
    void AddSample(Vec2f pRaster, const SampledSpectrum &L, const SampledWavelengths &lambda,
                   float weight = 1.f);
    void AddXYZ(Vec2f pRaster, const XYZ &xyz);
    // Accumulated XYZ of an image pixel (unscaled).
    XYZ PixelXYZ(int x, int y) const {
        size_t idx = 3 * (size_t(y) * settings_.width + x);
        return {float(xyz_[idx].load()), float(xyz_[idx + 1].load()), float(xyz_[idx + 2].load())};
    }
    // Adds directly to an image pixel (box-filtered estimators, e.g. the GPU path tracer).
    void AddPixelXYZ(int x, int y, const XYZ &xyz) {
        size_t idx = 3 * (size_t(y) * settings_.width + x);
        xyz_[idx + 0].fetch_add(double(xyz.x), std::memory_order_relaxed);
        xyz_[idx + 1].fetch_add(double(xyz.y), std::memory_order_relaxed);
        xyz_[idx + 2].fetch_add(double(xyz.z), std::memory_order_relaxed);
    }
    void Clear();
    // Checkpointing of the raw accumulators, streamed to/from a file in chunks (no full copy).
    bool WriteAccumulators(FILE *f) const;
    bool ReadAccumulators(FILE *f);
    // Bytes held by the accumulators for a given resolution (for memory estimates).
    static uint64_t MemoryBytes(int width, int height) { return uint64_t(width) * height * 3 * sizeof(double); }

    // Per-row resolves (thread-safe), used by the streamed writers.
    void ResolveRow(int y, double scale, float *rgb) const;              // linear, film colour space
    void ResolveDisplayRow(int y, double scale, ToneMap tm, uint8_t *rgb) const;  // tone-mapped sRGB bytes
    // Display-referred image downscaled (box filter) to at most maxDim pixels on the long side.
    Image ResolvePreview(double scale, ToneMap tm, int maxDim) const;

    // Resolve accumulated values times 'scale' to linear RGB in the film colour space.
    Image Resolve(double scale) const;
    // Resolve to display-referred sRGB (tone mapped and encoded).
    Image ResolveDisplay(double scale, ToneMap tm) const;
    // Resolve to XYZ (for tests).
    Image ResolveXYZ(double scale) const;

  private:
    RGB ToOutput(const double xyz[3], double scale, ColorSpaceId cs) const;
    FilmSettings settings_;
    Filter filter_;
    int margin_ = 0;
    Mat3 adapt_;
    std::unique_ptr<std::atomic<double>[]> xyz_;
};

} // namespace pr

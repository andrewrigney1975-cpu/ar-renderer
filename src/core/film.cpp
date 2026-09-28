#include "core/film.h"

namespace pr {

Filter::Filter(const FilterSettings &s) : type_(s.type), radius_(s.radius) {
    if (type_ == FilterType::Box) radius_ = std::max(0.5f, std::min(radius_, 0.5f));
    if (type_ == FilterType::Gaussian) sigma_ = radius_ / 3.f;
    // Numerically normalize the separable 1D profile to unit integral.
    const int n = 4096;
    double s1 = 0;
    for (int i = 0; i < n; ++i) {
        float x = -radius_ + (i + 0.5f) * (2 * radius_ / n);
        s1 += Eval1D(x);
    }
    s1 *= 2 * radius_ / n;
    norm_ = float(1.0 / (s1 * s1));
}

float Filter::Eval1D(float x) const {
    x = std::abs(x);
    if (x > radius_) return 0;
    switch (type_) {
    case FilterType::Box: return 1;
    case FilterType::Gaussian:
        return std::max(0.f, std::exp(-x * x / (2 * sigma_ * sigma_)) -
                                 std::exp(-radius_ * radius_ / (2 * sigma_ * sigma_)));
    case FilterType::BlackmanHarris: {
        // Window over [-r, r] mapped to [0, 1].
        float t = 0.5f + x / (2 * radius_);
        const float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
        return a0 - a1 * std::cos(2 * Pi * t) + a2 * std::cos(4 * Pi * t) - a3 * std::cos(6 * Pi * t);
    }
    }
    return 0;
}

Film::Film(const FilmSettings &s) : settings_(s), filter_(s.filter) {
    margin_ = int(std::ceil(filter_.Radius() - 0.5f));
    size_t n = size_t(s.width) * s.height * 3;
    xyz_.reset(new std::atomic<double>[n]);
    for (size_t i = 0; i < n; ++i) xyz_[i].store(0.0, std::memory_order_relaxed);
    adapt_ = SpectralWhiteAdaptation();
    if (s.whiteBalance > 0) {
        // Adapt from the white of a blackbody at the given temperature to D65.
        BlackbodySpectrum bb(s.whiteBalance);
        XYZ src = SpectrumToXYZ(bb);
        src = src * (1.f / src.y);
        adapt_ = BradfordAdaptation(src, {0.95047f, 1.f, 1.08883f});
    }
}

void Film::Clear() {
    size_t n = size_t(settings_.width) * settings_.height * 3;
    for (size_t i = 0; i < n; ++i) xyz_[i].store(0.0, std::memory_order_relaxed);
}

void Film::AddSample(Vec2f p, const SampledSpectrum &L, const SampledWavelengths &lambda, float weight) {
    if (L.IsZero()) return;
    AddXYZ(p, ToXYZ(L, lambda) * weight);
}

void Film::AddXYZ(Vec2f p, const XYZ &xyz) {
    if (!std::isfinite(xyz.x) || !std::isfinite(xyz.y) || !std::isfinite(xyz.z)) return;
    float r = filter_.Radius();
    int x0 = std::max(0, int(std::ceil(p.x - r - 0.5f)));
    int x1 = std::min(settings_.width - 1, int(std::floor(p.x + r - 0.5f)));
    int y0 = std::max(0, int(std::ceil(p.y - r - 0.5f)));
    int y1 = std::min(settings_.height - 1, int(std::floor(p.y + r - 0.5f)));
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            float w = filter_.Evaluate(p.x - (x + 0.5f), p.y - (y + 0.5f));
            if (w == 0) continue;
            size_t idx = 3 * (size_t(y) * settings_.width + x);
            xyz_[idx + 0].fetch_add(double(xyz.x) * w, std::memory_order_relaxed);
            xyz_[idx + 1].fetch_add(double(xyz.y) * w, std::memory_order_relaxed);
            xyz_[idx + 2].fetch_add(double(xyz.z) * w, std::memory_order_relaxed);
        }
}

RGB Film::ToOutput(const double xyz[3], double scale, ColorSpaceId cs) const {
    double ev = std::exp2(double(settings_.exposure));
    float X = float(xyz[0] * scale * ev), Y = float(xyz[1] * scale * ev), Z = float(xyz[2] * scale * ev);
    RGB a = adapt_.Apply(X, Y, Z);
    return XYZToRGBMatrix(cs).Apply(a.r, a.g, a.b);
}

Image Film::Resolve(double scale) const {
    Image img(settings_.width, settings_.height);
    for (int y = 0; y < settings_.height; ++y)
        for (int x = 0; x < settings_.width; ++x) {
            size_t idx = 3 * (size_t(y) * settings_.width + x);
            double v[3] = {xyz_[idx].load(), xyz_[idx + 1].load(), xyz_[idx + 2].load()};
            img.Set(x, y, ToOutput(v, scale, settings_.colorSpace));
        }
    return img;
}

Image Film::ResolveDisplay(double scale, ToneMap tm) const {
    Image img(settings_.width, settings_.height);
    for (int y = 0; y < settings_.height; ++y)
        for (int x = 0; x < settings_.width; ++x) {
            size_t idx = 3 * (size_t(y) * settings_.width + x);
            double v[3] = {xyz_[idx].load(), xyz_[idx + 1].load(), xyz_[idx + 2].load()};
            RGB c = ApplyToneMap(tm, ToOutput(v, scale, ColorSpaceId::sRGB));
            img.Set(x, y, {SRGBEncode(c.r), SRGBEncode(c.g), SRGBEncode(c.b)});
        }
    return img;
}

Image Film::ResolveXYZ(double scale) const {
    Image img(settings_.width, settings_.height);
    for (int y = 0; y < settings_.height; ++y)
        for (int x = 0; x < settings_.width; ++x) {
            size_t idx = 3 * (size_t(y) * settings_.width + x);
            img.Set(x, y, RGB(float(xyz_[idx].load() * scale), float(xyz_[idx + 1].load() * scale),
                              float(xyz_[idx + 2].load() * scale)));
        }
    return img;
}

} // namespace pr

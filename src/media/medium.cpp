#include "media/medium.h"

#include "core/fsutil.h"
#include "core/noise.h"
#include "core/rng.h"

#include <cstdio>
#include <cstring>

namespace pr {

// ---------------------------------------------------------------------------------------------
// Homogeneous

SampledSpectrum HomogeneousMedium::Transmittance(const Ray &, float dist, const SampledWavelengths &lambda) const {
    SampledSpectrum st = SigmaA(lambda) + SigmaS(lambda);
    if (std::isinf(dist)) {
        SampledSpectrum r;
        for (int i = 0; i < NSpectrumSamples; ++i) r[i] = st[i] > 0 ? 0.f : 1.f;
        return r;
    }
    return Exp(-st * dist);
}

Medium::DistanceSample HomogeneousMedium::SampleDistance(const Ray &ray, float tMax, float u,
                                                         const SampledWavelengths &lambda) const {
    SampledSpectrum sa = SigmaA(lambda), ss = SigmaS(lambda);
    SampledSpectrum st = sa + ss;
    SampledSpectrum lpdf = lambda.PDF();
    int active[NSpectrumSamples];
    int nActive = 0;
    for (int i = 0; i < NSpectrumSamples; ++i)
        if (lpdf[i] != 0) active[nActive++] = i;
    if (nActive == 0) {
        active[0] = 0;
        nActive = 1;
    }
    // Choose the channel that drives distance sampling, reusing the random number.
    int ci = std::min(int(u * nActive), nActive - 1);
    float uu = std::min(u * nActive - ci, OneMinusEpsilon);
    int c = active[ci];

    DistanceSample ds;
    float t = st[c] > 0 ? -std::log(1 - uu) / st[c] : Infinity;
    if (t < tMax) {
        float pdf = 0;
        for (int k = 0; k < nActive; ++k) {
            int i = active[k];
            pdf += st[i] * std::exp(-st[i] * t);
        }
        pdf /= nActive;
        SampledSpectrum Tr = Exp(-st * t);
        ds.scattered = true;
        ds.t = t;
        ds.weight = pdf > 0 ? Tr * ss / pdf : SampledSpectrum(0.f);
        return ds;
    }
    float pSurf = 0;
    for (int k = 0; k < nActive; ++k) {
        int i = active[k];
        pSurf += std::isinf(tMax) ? (st[i] > 0 ? 0.f : 1.f) : std::exp(-st[i] * tMax);
    }
    pSurf /= nActive;
    SampledSpectrum Tr = Transmittance(ray, tMax, lambda);
    ds.scattered = false;
    ds.t = tMax;
    ds.weight = pSurf > 0 ? Tr / pSurf : SampledSpectrum(0.f);
    return ds;
}

// ---------------------------------------------------------------------------------------------
// Density fields

VoxelGridField::VoxelGridField(int nx, int ny, int nz, std::vector<float> data, const Bounds3f &b)
    : nx_(nx), ny_(ny), nz_(nz), data_(std::move(data)) {
    bounds = b;
    for (float &v : data_) {
        if (!std::isfinite(v) || v < 0) v = 0;
        maxDensity_ = std::max(maxDensity_, v);
    }
}

float VoxelGridField::Density(const Vec3f &p) const {
    if (!bounds.Inside(p)) return 0;
    Vec3f o = bounds.Offset(p);
    float x = o.x * nx_ - 0.5f, y = o.y * ny_ - 0.5f, z = o.z * nz_ - 0.5f;
    int x0 = int(std::floor(x)), y0 = int(std::floor(y)), z0 = int(std::floor(z));
    float dx = x - x0, dy = y - y0, dz = z - z0;
    float c00 = Lerp(dx, At(x0, y0, z0), At(x0 + 1, y0, z0));
    float c10 = Lerp(dx, At(x0, y0 + 1, z0), At(x0 + 1, y0 + 1, z0));
    float c01 = Lerp(dx, At(x0, y0, z0 + 1), At(x0 + 1, y0, z0 + 1));
    float c11 = Lerp(dx, At(x0, y0 + 1, z0 + 1), At(x0 + 1, y0 + 1, z0 + 1));
    return Lerp(dz, Lerp(dy, c00, c10), Lerp(dy, c01, c11));
}

std::shared_ptr<VoxelGridField> VoxelGridField::LoadVol(const std::string &path, std::string *err) {
    FILE *f = _wfopen(Utf8Path(path).wstring().c_str(), L"rb");
    if (!f) {
        *err = "cannot open " + path;
        return nullptr;
    }
    char magic[3];
    uint8_t version;
    int32_t encoding, nx, ny, nz, channels;
    float bb[6];
    bool ok = std::fread(magic, 1, 3, f) == 3 && std::memcmp(magic, "VOL", 3) == 0 &&
              std::fread(&version, 1, 1, f) == 1 && version == 3 && std::fread(&encoding, 4, 1, f) == 1 &&
              std::fread(&nx, 4, 1, f) == 1 && std::fread(&ny, 4, 1, f) == 1 && std::fread(&nz, 4, 1, f) == 1 &&
              std::fread(&channels, 4, 1, f) == 1 && std::fread(bb, 4, 6, f) == 6;
    if (!ok || encoding != 1 || nx <= 0 || ny <= 0 || nz <= 0 || channels <= 0 ||
        int64_t(nx) * ny * nz > (int64_t(1) << 31)) {
        std::fclose(f);
        *err = path + ": not a float32 Mitsuba .vol (version 3) file";
        return nullptr;
    }
    size_t n = size_t(nx) * ny * nz;
    std::vector<float> raw(n * channels);
    ok = std::fread(raw.data(), sizeof(float), raw.size(), f) == raw.size();
    std::fclose(f);
    if (!ok) {
        *err = path + ": truncated voxel data";
        return nullptr;
    }
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        float s = 0;
        for (int c = 0; c < channels; ++c) s += raw[i * channels + c];
        data[i] = s / channels;
    }
    return std::make_shared<VoxelGridField>(nx, ny, nz, std::move(data),
                                            Bounds3f(Vec3f(bb[0], bb[1], bb[2]), Vec3f(bb[3], bb[4], bb[5])));
}

float NoiseField::Density(const Vec3f &p) const {
    if (!bounds.IsEmpty() && !bounds.Inside(p)) return 0;
    float n = 0.5f + 0.5f * FBm((p + offset_) * freq_, octaves_) * 1.4f;
    if (falloff_ && !bounds.IsEmpty()) {
        Vec3f o = bounds.Offset(p) * 2.f - Vec3f(1, 1, 1);
        float r2 = LengthSquared(o);
        n *= Clamp(1 - r2, 0.f, 1.f);
    }
    return Clamp((n - threshold_) * sharpness_, 0.f, 1.f);
}

// ---------------------------------------------------------------------------------------------
// Heterogeneous medium

bool GridMedium::Clip(const Ray &ray, float tMax, Ray *mray, float *t0, float *t1) const {
    // Medium-space ray; its parameter t matches the world ray's (the transform is affine).
    *mray = Ray(worldToMedium_.Point(ray.o), worldToMedium_.Vector(ray.d));
    *t0 = 0;
    *t1 = tMax;
    const Bounds3f &b = field_->bounds;
    if (b.IsEmpty()) return std::isfinite(tMax);
    for (int a = 0; a < 3; ++a) {
        float inv = 1 / (*mray).d[a];
        float tNear = (b.pMin[a] - (*mray).o[a]) * inv, tFar = (b.pMax[a] - (*mray).o[a]) * inv;
        if (tNear > tFar) std::swap(tNear, tFar);
        if (std::isnan(tNear) || std::isnan(tFar)) {
            // Parallel to the slab: inside or outside for all t.
            if ((*mray).o[a] < b.pMin[a] || (*mray).o[a] > b.pMax[a]) return false;
            continue;
        }
        *t0 = std::max(*t0, tNear);
        *t1 = std::min(*t1, tFar);
        if (*t0 > *t1) return false;
    }
    return true;
}

SampledSpectrum GridMedium::Transmittance(const Ray &ray, float tMax, const SampledWavelengths &lambda) const {
    Ray mray;
    float t0, t1;
    if (!Clip(ray, tMax, &mray, &t0, &t1)) return SampledSpectrum(1.f);
    SampledSpectrum st = (sigma_a_->Sample(lambda) + sigma_s_->Sample(lambda)) * scale_;
    float mu = field_->MaxDensity() * st.MaxComponentValue();
    if (mu <= 0) return SampledSpectrum(1.f);
    // Ratio tracking with a hashed RNG: a deterministic function of the segment.
    RNG rng(Hash(ray.o, ray.d), Hash(tMax));
    SampledSpectrum T(1.f);
    float t = t0;
    for (int iter = 0; iter < 100000; ++iter) {
        t -= std::log(1 - rng.UniformFloat()) / mu;
        if (t >= t1) break;
        float d = field_->Density(mray(t));
        T *= ClampZero(SampledSpectrum(1.f) - st * (d / mu));
        // Russian roulette on low transmittance keeps the estimator unbiased and cheap.
        float m = T.MaxComponentValue();
        if (m <= 0) return SampledSpectrum(0.f);
        if (m < 0.1f) {
            if (rng.UniformFloat() < 0.5f) return SampledSpectrum(0.f);
            T *= 2.f;
        }
    }
    return T;
}

Medium::DistanceSample GridMedium::SampleDistance(const Ray &ray, float tMax, float u,
                                                  const SampledWavelengths &lambda) const {
    DistanceSample ds;
    ds.weight = SampledSpectrum(1.f);
    ds.t = tMax;
    Ray mray;
    float t0, t1;
    if (!Clip(ray, tMax, &mray, &t0, &t1)) return ds;
    SampledSpectrum sa = sigma_a_->Sample(lambda) * scale_, ss = sigma_s_->Sample(lambda) * scale_;
    SampledSpectrum st = sa + ss;
    float mu = field_->MaxDensity() * st.MaxComponentValue();
    if (mu <= 0) return ds;
    SampledSpectrum lpdf = lambda.PDF();
    auto avg = [&](const SampledSpectrum &s) {
        float sum = 0;
        int n = 0;
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (lpdf[i] != 0) {
                sum += std::abs(s[i]);
                ++n;
            }
        return n ? sum / n : 0.f;
    };
    RNG rng(Hash(u, ray.o), Hash(ray.d, tMax));
    float t = t0;
    for (int iter = 0; iter < 100000; ++iter) {
        t -= std::log(1 - rng.UniformFloat()) / mu;
        if (t >= t1) break;
        float d = field_->Density(mray(t));
        SampledSpectrum sA = sa * d, sS = ss * d;
        SampledSpectrum sN = SampledSpectrum(mu) - sA - sS;
        // Spectral tracking: event probabilities from the (wavelength-averaged) coefficients,
        // with per-wavelength weights sigma_e / (mu * P_e) keeping every wavelength unbiased.
        float pa = avg(sA), ps = avg(sS), pn = avg(sN);
        float norm = pa + ps + pn;
        if (norm <= 0) continue;
        float xi = rng.UniformFloat() * norm;
        if (xi < ps) {
            ds.scattered = true;
            ds.t = t;
            ds.weight *= sS / (mu * ps / norm);
            return ds;
        }
        if (xi < ps + pa) {
            ds.weight = SampledSpectrum(0.f);  // absorbed (no volumetric emission)
            ds.t = t;
            return ds;
        }
        ds.weight *= sN / (mu * pn / norm);
        if (!ds.weight) return ds;
    }
    ds.t = tMax;
    return ds;
}

} // namespace pr

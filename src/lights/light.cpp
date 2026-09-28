#include "lights/light.h"

#include "core/rgb2spec.h"

namespace pr {

// ---------------------------------------------------------------------------------------------
// Area light

DiffuseAreaLight::DiffuseAreaLight(const Shape *shape, SpectrumPtr Lemit, float scale, bool twoSided,
                                   float cosPower, const MediumInterface *mi)
    : Light(LightType::Area), shape_(shape), Lemit_(std::move(Lemit)), scale_(scale), twoSided_(twoSided),
      cosPower_(std::max(0.f, cosPower)), mi_(mi) {}

float DiffuseAreaLight::PowerY() const {
    float lobe = 2 * Pi / (cosPower_ + 2);  // integral of cos^n * cos over the hemisphere
    return scale_ * SpectrumToY(*Lemit_) * shape_->Area() * lobe * (twoSided_ ? 2 : 1);
}

SampledSpectrum DiffuseAreaLight::L(const Vec3f &, const Vec3f &n, Vec2f, const Vec3f &w,
                                    const SampledWavelengths &lambda) const {
    float c = Dot(n, w);
    if (!twoSided_ && c <= 0) return SampledSpectrum(0.f);
    return Lemit_->Sample(lambda) * (scale_ * Profile(std::abs(c)));
}

std::optional<LightLiSample> DiffuseAreaLight::SampleLi(const LightSampleContext &ctx, Vec2f u,
                                                        const SampledWavelengths &lambda) const {
    ShapeSampleContext sctx{ctx.p, ctx.n, ctx.ns};
    auto ss = shape_->Sample(sctx, u);
    if (!ss || ss->pdf == 0 || LengthSquared(ss->intr.p - ctx.p) == 0) return std::nullopt;
    Vec3f wi = Normalize(ss->intr.p - ctx.p);
    SampledSpectrum Le = L(ss->intr.p, ss->intr.n, ss->intr.uv, -wi, lambda);
    if (!Le) return std::nullopt;
    LightLiSample ls;
    ls.L = Le;
    ls.wi = wi;
    ls.pdf = ss->pdf;
    ls.pLight = ss->intr;
    ls.pLight.mediumInterface = mi_;
    return ls;
}

float DiffuseAreaLight::PDF_Li(const LightSampleContext &ctx, const Vec3f &wi) const {
    return shape_->PDF(ShapeSampleContext{ctx.p, ctx.n, ctx.ns}, wi);
}

std::optional<LightLeSample> DiffuseAreaLight::SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const {
    auto ss = shape_->Sample(u1);
    if (!ss || ss->pdf == 0) return std::nullopt;
    Vec3f n = ss->intr.n;
    float dirPdfScale = 1;
    if (twoSided_) {
        dirPdfScale = 0.5f;
        if (u2.x < 0.5f) {
            u2.x = std::min(u2.x * 2, OneMinusEpsilon);
        } else {
            u2.x = std::min((u2.x - 0.5f) * 2, OneMinusEpsilon);
            n = -n;
        }
    }
    Vec3f wLocal = cosPower_ > 0 ? SampleCosPowerLobe(u2, cosPower_) : SampleCosineHemisphere(u2);
    float pdfDir = (cosPower_ > 0 ? CosPowerLobePDF(wLocal.z, cosPower_) : CosineHemispherePDF(wLocal.z)) * dirPdfScale;
    if (pdfDir == 0) return std::nullopt;
    Frame f = Frame::FromZ(n);
    Vec3f w = f.FromLocal(wLocal);
    SampledSpectrum Le = L(ss->intr.p, ss->intr.n, ss->intr.uv, w, lambda);
    if (!Le) return std::nullopt;
    LightLeSample ls;
    ls.L = Le;
    ls.intr = ss->intr;
    ls.intr.mediumInterface = mi_;
    ls.ray = ls.intr.SpawnRay(w);
    ls.pdfPos = ss->pdf;
    ls.pdfDir = pdfDir;
    return ls;
}

void DiffuseAreaLight::PDF_Le(const Ray &ray, const Vec3f &n, float *pdfPos, float *pdfDir) const {
    *pdfPos = 1 / shape_->Area();
    float c = Dot(n, ray.d);
    if (twoSided_) {
        c = std::abs(c);
        *pdfDir = 0.5f * (cosPower_ > 0 ? CosPowerLobePDF(c, cosPower_) : CosineHemispherePDF(c));
    } else {
        *pdfDir = c <= 0 ? 0.f : (cosPower_ > 0 ? CosPowerLobePDF(c, cosPower_) : CosineHemispherePDF(c));
    }
}

std::optional<LightBounds> DiffuseAreaLight::Bounds() const {
    LightBounds lb;
    lb.bounds = shape_->Bounds();
    Vec3f w;
    float cosTheta;
    shape_->NormalBounds(&w, &cosTheta);
    lb.w = w;
    lb.cosTheta_o = cosTheta;
    // Emission spreads over the hemisphere (or a narrower cos^n lobe, conservatively bounded).
    lb.cosTheta_e = 0;
    lb.twoSided = twoSided_;
    lb.phi = scale_ * Lemit_->MaxValue() * shape_->Area() * Pi * (twoSided_ ? 2 : 1);
    return lb;
}

// ---------------------------------------------------------------------------------------------
// Point light

std::optional<LightBounds> PointLight::Bounds() const {
    LightBounds lb;
    lb.bounds = Bounds3f(p_);
    lb.phi = 4 * Pi * scale_ * I_->MaxValue();
    lb.cosTheta_o = -1;
    lb.cosTheta_e = 0;
    return lb;
}

std::optional<LightBounds> SpotLight::Bounds() const {
    LightBounds lb;
    lb.bounds = Bounds3f(p_);
    lb.w = frame_.z;
    lb.phi = 4 * Pi * scale_ * I_->MaxValue();
    lb.cosTheta_o = cosFalloffStart_;
    lb.cosTheta_e = std::cos(SafeACos(cosFalloffEnd_) - SafeACos(cosFalloffStart_));
    return lb;
}

std::optional<LightLiSample> PointLight::SampleLi(const LightSampleContext &ctx, Vec2f,
                                                  const SampledWavelengths &lambda) const {
    Vec3f d = p_ - ctx.p;
    float d2 = LengthSquared(d);
    if (d2 == 0) return std::nullopt;
    LightLiSample ls;
    ls.wi = d / std::sqrt(d2);
    ls.L = I_->Sample(lambda) * (scale_ / d2);
    ls.pdf = 1;
    ls.pLight.p = p_;
    ls.pLight.medium = medium;
    return ls;
}

std::optional<LightLeSample> PointLight::SampleLe(Vec2f u1, Vec2f, const SampledWavelengths &lambda) const {
    LightLeSample ls;
    Vec3f w = SampleUniformSphere(u1);
    ls.intr.p = p_;
    ls.intr.medium = medium;
    ls.ray = Ray(p_, w);
    ls.L = I_->Sample(lambda) * scale_;
    ls.pdfPos = 1;
    ls.pdfDir = UniformSpherePDF();
    return ls;
}

void PointLight::PDF_Le(const Ray &, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

// ---------------------------------------------------------------------------------------------
// Spot light

SpotLight::SpotLight(const Vec3f &p, const Vec3f &dir, SpectrumPtr I, float scale, float totalWidthDeg,
                     float falloffStartDeg)
    : Light(LightType::DeltaPosition), p_(p), frame_(Frame::FromZ(Normalize(dir))), I_(std::move(I)), scale_(scale) {
    cosFalloffEnd_ = std::cos(Radians(totalWidthDeg));
    cosFalloffStart_ = std::cos(Radians(std::min(falloffStartDeg, totalWidthDeg)));
}

float SpotLight::Falloff(const Vec3f &w) const {
    float c = Dot(w, frame_.z);
    return SmoothStep(c, cosFalloffEnd_, cosFalloffStart_);
}

float SpotLight::PowerY() const {
    return scale_ * SpectrumToY(*I_) * 2 * Pi * ((1 - cosFalloffStart_) + (cosFalloffStart_ - cosFalloffEnd_) / 2);
}

std::optional<LightLiSample> SpotLight::SampleLi(const LightSampleContext &ctx, Vec2f,
                                                 const SampledWavelengths &lambda) const {
    Vec3f d = p_ - ctx.p;
    float d2 = LengthSquared(d);
    if (d2 == 0) return std::nullopt;
    Vec3f wi = d / std::sqrt(d2);
    float f = Falloff(-wi);
    if (f == 0) return std::nullopt;
    LightLiSample ls;
    ls.wi = wi;
    ls.L = I_->Sample(lambda) * (scale_ * f / d2);
    ls.pdf = 1;
    ls.pLight.p = p_;
    ls.pLight.medium = medium;
    return ls;
}

std::optional<LightLeSample> SpotLight::SampleLe(Vec2f u1, Vec2f, const SampledWavelengths &lambda) const {
    Vec3f wl = SampleUniformCone(u1, cosFalloffEnd_);
    Vec3f w = frame_.FromLocal(wl);
    LightLeSample ls;
    ls.intr.p = p_;
    ls.intr.medium = medium;
    ls.ray = Ray(p_, w);
    ls.L = I_->Sample(lambda) * (scale_ * Falloff(w));
    ls.pdfPos = 1;
    ls.pdfDir = UniformConePDF(cosFalloffEnd_);
    return ls;
}

void SpotLight::PDF_Le(const Ray &ray, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = Dot(ray.d, frame_.z) >= cosFalloffEnd_ ? UniformConePDF(cosFalloffEnd_) : 0.f;
}

// ---------------------------------------------------------------------------------------------
// Environment light

EnvironmentLight::EnvironmentLight(SpectrumPtr constant, float scale)
    : Light(LightType::Infinite), constant_(std::move(constant)), scale_(scale) {
    avgY_ = SpectrumToY(*constant_);
}

EnvironmentLight::EnvironmentLight(std::shared_ptr<Image> map, float scale, float rotationDeg)
    : Light(LightType::Infinite), map_(std::move(map)), scale_(scale) {
    rotInv_ = Transform::Rotate(rotationDeg, Vec3f(0, 1, 0));
    rot_ = rotInv_.Inverse();
    int w = map_->Width(), h = map_->Height();
    std::vector<float> f(size_t(w) * h);
    double sumY = 0, sumW = 0;
    for (int y = 0; y < h; ++y) {
        float sinTheta = std::sin(Pi * (y + 0.5f) / h);
        for (int x = 0; x < w; ++x) {
            RGB c = map_->Get(x, y);
            float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
            f[size_t(y) * w + x] = std::max(lum, 0.f) * sinTheta + 1e-6f;
            sumY += lum * sinTheta;
            sumW += sinTheta;
        }
    }
    avgY_ = float(sumY / std::max(1e-9, sumW));
    distrib_ = Distribution2D(f.data(), w, h);
}

float EnvironmentLight::PowerY() const { return 4 * Pi * Pi * radius_ * radius_ * scale_ * avgY_; }

SampledSpectrum EnvironmentLight::Lookup(const Vec3f &wWorld, const SampledWavelengths &lambda) const {
    if (constant_) return constant_->Sample(lambda) * scale_;
    Vec2f uv = DirToEquirect(Normalize(ToLight(wWorld)));
    int w = map_->Width(), h = map_->Height();
    // Bilinear lookup with horizontal wrap.
    float x = uv.x * w - 0.5f, y = uv.y * h - 0.5f;
    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    float dx = x - x0, dy = y - y0;
    auto tex = [&](int xi, int yi) {
        xi = ((xi % w) + w) % w;
        yi = Clamp(yi, 0, h - 1);
        return map_->Get(xi, yi);
    };
    RGB a = tex(x0, y0), b = tex(x0 + 1, y0), c = tex(x0, y0 + 1), d = tex(x0 + 1, y0 + 1);
    RGB rgb;
    for (int i = 0; i < 3; ++i)
        rgb[i] = (1 - dx) * (1 - dy) * a[i] + dx * (1 - dy) * b[i] + (1 - dx) * dy * c[i] + dx * dy * d[i];
    return RGBIlluminantSample(rgb * scale_, lambda);
}

SampledSpectrum EnvironmentLight::Le(const Ray &ray, const SampledWavelengths &lambda) const {
    return Lookup(ray.d, lambda);
}

std::optional<LightLiSample> EnvironmentLight::SampleLi(const LightSampleContext &ctx, Vec2f u,
                                                        const SampledWavelengths &lambda) const {
    Vec3f wi;
    float pdf;
    if (constant_) {
        wi = SampleUniformSphere(u);
        pdf = UniformSpherePDF();
    } else {
        float mapPdf;
        Vec2f uv = distrib_.SampleContinuous(u, &mapPdf);
        if (mapPdf == 0) return std::nullopt;
        float sinTheta = std::sin(uv.y * Pi);
        if (sinTheta == 0) return std::nullopt;
        wi = ToWorld(EquirectToDir(uv));
        pdf = mapPdf / (2 * Pi * Pi * sinTheta);
    }
    LightLiSample ls;
    ls.wi = wi;
    ls.pdf = pdf;
    ls.L = Lookup(wi, lambda);
    ls.pLight.p = ctx.p + wi * (2 * radius_);
    return ls;
}

float EnvironmentLight::PDF_Li(const LightSampleContext &, const Vec3f &wi) const {
    if (constant_) return UniformSpherePDF();
    Vec2f uv = DirToEquirect(Normalize(ToLight(wi)));
    float sinTheta = std::sin(uv.y * Pi);
    if (sinTheta == 0) return 0;
    return distrib_.PDF(uv) / (2 * Pi * Pi * sinTheta);
}

std::optional<LightLeSample> EnvironmentLight::SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const {
    // Direction towards the light, then a point on a disk perpendicular to it.
    LightSampleContext dummy;
    auto li = SampleLi(dummy, u1, lambda);
    if (!li) return std::nullopt;
    Vec3f d = -li->wi;  // propagation direction
    Vec3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Vec2f cd = SampleUniformDiskConcentric(u2);
    Vec3f pDisk = center_ + (v1 * cd.x + v2 * cd.y) * radius_;
    LightLeSample ls;
    ls.intr.p = pDisk + (-d) * radius_;
    ls.intr.n = Vec3f();  // not a surface
    ls.ray = Ray(ls.intr.p, d);
    ls.L = li->L;
    ls.pdfPos = 1 / (Pi * radius_ * radius_);
    ls.pdfDir = li->pdf;
    return ls;
}

void EnvironmentLight::PDF_Le(const Ray &ray, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfDir = PDF_Li(LightSampleContext(), -ray.d);
    *pdfPos = 1 / (Pi * radius_ * radius_);
}

// ---------------------------------------------------------------------------------------------

LightSampler::LightSampler(const std::vector<const Light *> &lights) : lights_(lights) {
    std::vector<float> power;
    for (size_t i = 0; i < lights.size(); ++i) {
        float p = lights[i]->PowerY();
        power.push_back(std::isfinite(p) && p > 0 ? p : 0.f);
        index_[lights[i]] = int(i);
    }
    // Keep every light samplable: give zero-power lights a tiny share.
    float maxP = 0;
    for (float p : power) maxP = std::max(maxP, p);
    for (float &p : power) p = std::max(p, maxP > 0 ? maxP * 1e-4f : 1.f);
    if (!power.empty()) distrib_ = Distribution1D(power);
}

const Light *LightSampler::Sample(float u, float *pmf) const {
    if (lights_.empty()) return nullptr;
    int i = distrib_.SampleDiscrete(u, pmf);
    return lights_[i];
}

float LightSampler::PMF(const Light *light) const {
    auto it = index_.find(light);
    if (it == index_.end()) return 0;
    return distrib_.DiscretePMF(it->second);
}

} // namespace pr

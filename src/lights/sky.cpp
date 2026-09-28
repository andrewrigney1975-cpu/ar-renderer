// Goniometric (IES), distant and sun lights, and the Preetham daylight model.

#include "lights/light.h"

#include "core/color.h"

namespace pr {

// ---------------------------------------------------------------------------------------------
// Goniometric (IES) light

std::optional<LightLiSample> GoniometricLight::SampleLi(const LightSampleContext &ctx, Vec2f,
                                                        const SampledWavelengths &lambda) const {
    Vec3f d = p_ - ctx.p;
    float d2 = LengthSquared(d);
    if (d2 == 0) return std::nullopt;
    Vec3f wi = d / std::sqrt(d2);
    float prof = profile_->Evaluate(frame_.ToLocal(-wi));
    if (prof <= 0) return std::nullopt;
    LightLiSample ls;
    ls.wi = wi;
    ls.L = I_->Sample(lambda) * (scale_ * prof / d2);
    ls.pdf = 1;
    ls.pLight.p = p_;
    ls.pLight.medium = medium;
    return ls;
}

std::optional<LightLeSample> GoniometricLight::SampleLe(Vec2f u1, Vec2f, const SampledWavelengths &lambda) const {
    float pdf;
    Vec3f wl = profile_->Sample(u1, &pdf);
    if (pdf == 0) return std::nullopt;
    Vec3f w = frame_.FromLocal(wl);
    LightLeSample ls;
    ls.intr.p = p_;
    ls.intr.medium = medium;
    ls.ray = Ray(p_, w);
    ls.L = I_->Sample(lambda) * (scale_ * profile_->Evaluate(wl));
    ls.pdfPos = 1;
    ls.pdfDir = pdf;
    return ls;
}

void GoniometricLight::PDF_Le(const Ray &ray, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = profile_->PDF(frame_.ToLocal(ray.d));
}

// ---------------------------------------------------------------------------------------------
// Distant light

std::optional<LightLiSample> DistantLight::SampleLi(const LightSampleContext &ctx, Vec2f,
                                                    const SampledWavelengths &lambda) const {
    LightLiSample ls;
    ls.wi = w_;
    ls.L = E_->Sample(lambda) * scale_;
    ls.pdf = 1;
    ls.pLight.p = ctx.p + w_ * (2 * radius_);
    return ls;
}

std::optional<LightLeSample> DistantLight::SampleLe(Vec2f u1, Vec2f, const SampledWavelengths &lambda) const {
    Vec3f v1, v2;
    CoordinateSystem(w_, &v1, &v2);
    Vec2f cd = SampleUniformDiskConcentric(u1);
    Vec3f pDisk = center_ + (v1 * cd.x + v2 * cd.y) * radius_;
    LightLeSample ls;
    ls.intr.p = pDisk + w_ * radius_;
    ls.ray = Ray(ls.intr.p, -w_);
    ls.L = E_->Sample(lambda) * scale_;
    ls.pdfPos = 1 / (Pi * radius_ * radius_);
    ls.pdfDir = 1;
    return ls;
}

void DistantLight::PDF_Le(const Ray &, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfPos = 1 / (Pi * radius_ * radius_);
    *pdfDir = 0;
}

// ---------------------------------------------------------------------------------------------
// Sun

SunLight::SunLight(const Vec3f &wSun, float angularRadiusDeg, SpectrumPtr L, float scale)
    : Light(LightType::Infinite), frame_(Frame::FromZ(Normalize(wSun))),
      cosThetaMax_(std::cos(Radians(Clamp(angularRadiusDeg, 0.01f, 45.f)))), L_(std::move(L)), scale_(scale) {}

float SunLight::PowerY() const {
    return scale_ * SpectrumToY(*L_) * 2 * Pi * (1 - cosThetaMax_) * Pi * radius_ * radius_;
}

SampledSpectrum SunLight::Le(const Ray &ray, const SampledWavelengths &lambda) const {
    if (Dot(Normalize(ray.d), frame_.z) < cosThetaMax_) return SampledSpectrum(0.f);
    return L_->Sample(lambda) * scale_;
}

std::optional<LightLiSample> SunLight::SampleLi(const LightSampleContext &ctx, Vec2f u,
                                                const SampledWavelengths &lambda) const {
    LightLiSample ls;
    ls.wi = frame_.FromLocal(SampleUniformCone(u, cosThetaMax_));
    ls.pdf = UniformConePDF(cosThetaMax_);
    ls.L = L_->Sample(lambda) * scale_;
    ls.pLight.p = ctx.p + ls.wi * (2 * radius_);
    return ls;
}

float SunLight::PDF_Li(const LightSampleContext &, const Vec3f &wi) const {
    return Dot(Normalize(wi), frame_.z) >= cosThetaMax_ ? UniformConePDF(cosThetaMax_) : 0.f;
}

std::optional<LightLeSample> SunLight::SampleLe(Vec2f u1, Vec2f u2, const SampledWavelengths &lambda) const {
    Vec3f wi = frame_.FromLocal(SampleUniformCone(u1, cosThetaMax_));
    Vec3f v1, v2;
    CoordinateSystem(wi, &v1, &v2);
    Vec2f cd = SampleUniformDiskConcentric(u2);
    Vec3f pDisk = center_ + (v1 * cd.x + v2 * cd.y) * radius_;
    LightLeSample ls;
    ls.intr.p = pDisk + wi * radius_;
    ls.ray = Ray(ls.intr.p, -wi);
    ls.L = L_->Sample(lambda) * scale_;
    ls.pdfPos = 1 / (Pi * radius_ * radius_);
    ls.pdfDir = UniformConePDF(cosThetaMax_);
    return ls;
}

void SunLight::PDF_Le(const Ray &ray, const Vec3f &, float *pdfPos, float *pdfDir) const {
    *pdfPos = 1 / (Pi * radius_ * radius_);
    *pdfDir = PDF_Li(LightSampleContext(), -ray.d);
}

// ---------------------------------------------------------------------------------------------
// Preetham sky

namespace {

struct Perez {
    float A, B, C, D, E;
    float F(float cosTheta, float gamma, float cosGamma) const {
        cosTheta = std::max(cosTheta, 0.01f);
        return (1 + A * std::exp(B / cosTheta)) * (1 + C * std::exp(D * gamma) + E * cosGamma * cosGamma);
    }
};

} // namespace

std::shared_ptr<Image> MakePreethamSky(const Vec3f &sunDirIn, float T, float groundAlbedo, int width, int height) {
    Vec3f sunDir = Normalize(sunDirIn);
    T = Clamp(T, 1.7f, 10.f);
    float thetaS = SafeACos(std::max(sunDir.y, 0.f));  // below the horizon: use the horizon sun
    Perez pY{0.1787f * T - 1.4630f, -0.3554f * T + 0.4275f, -0.0227f * T + 5.3251f, 0.1206f * T - 2.5771f,
             -0.0670f * T + 0.3703f};
    Perez px{-0.0193f * T - 0.2592f, -0.0665f * T + 0.0008f, -0.0004f * T + 0.2125f, -0.0641f * T - 0.8989f,
             -0.0033f * T + 0.0452f};
    Perez py{-0.0167f * T - 0.2608f, -0.0950f * T + 0.0092f, -0.0079f * T + 0.2102f, -0.0441f * T - 1.6537f,
             -0.0109f * T + 0.0529f};
    float chi = (4.f / 9.f - T / 120.f) * (Pi - 2 * thetaS);
    float Yz = std::max(0.f, (4.0453f * T - 4.9710f) * std::tan(chi) - 0.2155f * T + 2.4192f);  // kcd/m^2
    float t2 = thetaS * thetaS, t3 = t2 * thetaS;
    float xz = T * T * (0.00166f * t3 - 0.00375f * t2 + 0.00209f * thetaS) +
               T * (-0.02903f * t3 + 0.06377f * t2 - 0.03202f * thetaS + 0.00394f) +
               (0.11693f * t3 - 0.21196f * t2 + 0.06052f * thetaS + 0.25886f);
    float yz = T * T * (0.00275f * t3 - 0.00610f * t2 + 0.00317f * thetaS) +
               T * (-0.04214f * t3 + 0.08970f * t2 - 0.04153f * thetaS + 0.00516f) +
               (0.15346f * t3 - 0.26756f * t2 + 0.06670f * thetaS + 0.26688f);
    float cosS = std::cos(thetaS);
    float denY = pY.F(1, thetaS, cosS), denx = px.F(1, thetaS, cosS), deny = py.F(1, thetaS, cosS);
    // Daylight fades out as the sun sets below the horizon.
    float night = SmoothStep(sunDir.y, -0.1f, 0.02f);
    const Mat3 &toRGB = XYZToRGBMatrix(ColorSpaceId::sRGB);
    Vec3f sunUp = Normalize(Vec3f(sunDir.x, std::max(sunDir.y, 0.f), sunDir.z));
    auto img = std::make_shared<Image>(width, height);
    double skyE = 0;  // horizontal illuminance from the sky (kcd/m^2 * sr)
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i) {
            Vec3f w = EquirectToDir(Vec2f((i + 0.5f) / width, (j + 0.5f) / height));
            RGB c;
            if (w.y > 0) {
                float cosG = Clamp(Dot(w, sunUp), -1.f, 1.f);
                float gamma = std::acos(cosG);
                float Y = Yz * pY.F(w.y, gamma, cosG) / denY * night;
                float x = xz * px.F(w.y, gamma, cosG) / denx;
                float y = yz * py.F(w.y, gamma, cosG) / deny;
                if (y > 0 && Y > 0) {
                    float X = x / y * Y, Z = (1 - x - y) / y * Y;
                    c = toRGB.Apply(X, Y, Z);
                    c = RGB(std::max(0.f, c.r), std::max(0.f, c.g), std::max(0.f, c.b));
                    float dOmega = (2 * Pi / width) * (Pi / height) * SafeSqrt(1 - w.y * w.y);
                    skyE += Y * w.y * dOmega;
                }
            }
            img->Set(i, j, c);
        }
    // Lambertian ground lit by sun and sky: L = albedo / pi * E.
    float sunE = 0;
    if (sunDir.y > 0) {
        const float r = Radians(0.2665f);
        SpectrumPtr sun = MakeSunRadiance(sunDir, T, 0.2665f);
        sunE = SpectrumToY(*sun) * Pi * r * r * sunDir.y;
    }
    float groundY = groundAlbedo * InvPi * (float(skyE) + sunE);
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            if (EquirectToDir(Vec2f((i + 0.5f) / width, (j + 0.5f) / height)).y <= 0) img->Set(i, j, RGB(groundY, groundY, groundY));
    return img;
}

SpectrumPtr MakeSunRadiance(const Vec3f &sunDir, float T, float angularRadiusDeg) {
    Vec3f d = Normalize(sunDir);
    float elevDeg = Degrees(std::asin(Clamp(d.y, -1.f, 1.f)));
    float zenithDeg = 90 - elevDeg;
    // Kasten & Young (1989) relative optical air mass.
    float m = zenithDeg < 95
                  ? 1 / (std::cos(Radians(std::min(zenithDeg, 90.f))) + 0.50572f * std::pow(96.07995f - zenithDeg, -1.6364f))
                  : 1e3f;
    float beta = std::max(0.f, 0.04608f * T - 0.04586f);  // Angstrom turbidity coefficient
    BlackbodySpectrum bb(5778);
    float yBB = SpectrumToY(bb);
    // Nominal solar disk luminance ~2e6 kcd/m^2 for a 0.2665 degree radius; an enlarged disk
    // keeps the same irradiance (softer shadows, same exposure).
    float sizeScale = Sqr(0.2665f / std::max(angularRadiusDeg, 0.01f));
    std::vector<float> l, v;
    for (int lambda = 360; lambda <= 830; lambda += 5) {
        float um = lambda * 1e-3f;
        float tauR = 0.008735f * std::pow(um, -4.08f);  // Rayleigh
        float tauA = beta * std::pow(um, -1.3f);         // aerosols
        float trans = elevDeg < -1 ? 0.f : std::exp(-m * (tauR + tauA));
        l.push_back(float(lambda));
        v.push_back(2.0e6f * sizeScale * bb(float(lambda)) / yBB * trans);
    }
    return std::make_shared<PiecewiseLinearSpectrum>(l, v);
}

} // namespace pr

#include "core/spectrum.h"

#include <map>
#include <mutex>

namespace pr {

// ---------------------------------------------------------------------------------------------
// Wavelength sampling

float SampleVisibleWavelength(float u) {
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

float VisibleWavelengthPDF(float lambda) {
    if (lambda < LambdaMin || lambda > LambdaMax) return 0;
    return 0.0039398042f / Sqr(std::cosh(0.0072f * (lambda - 538)));
}

SampledWavelengths SampledWavelengths::SampleUniform(float u) {
    SampledWavelengths swl;
    swl.lambda_[0] = Lerp(u, LambdaMin, LambdaMax);
    float delta = (LambdaMax - LambdaMin) / NSpectrumSamples;
    for (int i = 1; i < NSpectrumSamples; ++i) {
        swl.lambda_[i] = swl.lambda_[i - 1] + delta;
        if (swl.lambda_[i] > LambdaMax) swl.lambda_[i] = LambdaMin + (swl.lambda_[i] - LambdaMax);
    }
    for (int i = 0; i < NSpectrumSamples; ++i) swl.pdf_[i] = 1 / (LambdaMax - LambdaMin);
    return swl;
}

SampledWavelengths SampledWavelengths::SampleVisible(float u) {
    SampledWavelengths swl;
    for (int i = 0; i < NSpectrumSamples; ++i) {
        float up = u + float(i) / NSpectrumSamples;
        if (up > 1) up -= 1;
        swl.lambda_[i] = Clamp(SampleVisibleWavelength(up), LambdaMin, LambdaMax);
        swl.pdf_[i] = VisibleWavelengthPDF(swl.lambda_[i]);
    }
    return swl;
}

// ---------------------------------------------------------------------------------------------
// CIE colour matching functions: Wyman, Sloan & Shirley, "Simple Analytic Approximations to the
// CIE XYZ Color Matching Functions", JCGT 2013 (multi-lobe fit).

static inline float G(float x, float mu, float s1, float s2) {
    float t = (x - mu) / (x < mu ? s1 : s2);
    return std::exp(-0.5f * t * t);
}

float CIE_X(float l) {
    return 1.056f * G(l, 599.8f, 37.9f, 31.0f) + 0.362f * G(l, 442.0f, 16.0f, 26.7f) -
           0.065f * G(l, 501.1f, 20.4f, 26.2f);
}
float CIE_Y(float l) { return 0.821f * G(l, 568.8f, 46.9f, 40.5f) + 0.286f * G(l, 530.9f, 16.3f, 31.1f); }
float CIE_Z(float l) { return 1.217f * G(l, 437.0f, 11.8f, 36.0f) + 0.681f * G(l, 459.0f, 26.0f, 13.8f); }

float CIE_Y_Integral() {
    static const float integral = [] {
        double s = 0;
        for (int l = int(LambdaMin); l <= int(LambdaMax); ++l) s += CIE_Y(float(l));
        return float(s);
    }();
    return integral;
}

XYZ ToXYZ(const SampledSpectrum &s, const SampledWavelengths &lambda) {
    SampledSpectrum pdf = lambda.PDF();
    double X = 0, Y = 0, Z = 0;
    for (int i = 0; i < NSpectrumSamples; ++i) {
        if (pdf[i] == 0) continue;
        float v = s[i] / pdf[i];
        X += CIE_X(lambda[i]) * v;
        Y += CIE_Y(lambda[i]) * v;
        Z += CIE_Z(lambda[i]) * v;
    }
    float norm = 1.f / (NSpectrumSamples * CIE_Y_Integral());
    return {float(X) * norm, float(Y) * norm, float(Z) * norm};
}

XYZ SpectrumToXYZ(const Spectrum &s) {
    double X = 0, Y = 0, Z = 0;
    for (int l = int(LambdaMin); l <= int(LambdaMax); ++l) {
        float v = s(float(l));
        X += CIE_X(float(l)) * v;
        Y += CIE_Y(float(l)) * v;
        Z += CIE_Z(float(l)) * v;
    }
    float inv = 1.f / CIE_Y_Integral();
    return {float(X) * inv, float(Y) * inv, float(Z) * inv};
}

float SpectrumToY(const Spectrum &s) { return SpectrumToXYZ(s).y; }

// ---------------------------------------------------------------------------------------------

PiecewiseLinearSpectrum::PiecewiseLinearSpectrum(std::vector<float> lambdas, std::vector<float> values)
    : lambdas_(std::move(lambdas)), values_(std::move(values)) {}

float PiecewiseLinearSpectrum::operator()(float lambda) const {
    if (lambdas_.empty()) return 0;
    if (lambda <= lambdas_.front()) return values_.front();
    if (lambda >= lambdas_.back()) return values_.back();
    int o = FindInterval(int(lambdas_.size()), [&](int i) { return lambdas_[i] <= lambda; });
    float t = (lambda - lambdas_[o]) / (lambdas_[o + 1] - lambdas_[o]);
    return Lerp(t, values_[o], values_[o + 1]);
}

float PiecewiseLinearSpectrum::MaxValue() const {
    float m = 0;
    for (float v : values_) m = std::max(m, v);
    return m;
}

float BlackbodySpectrum::Planck(float lambdaNm, float T) {
    if (T <= 0) return 0;
    const double c = 299792458., h = 6.62606957e-34, kb = 1.3806488e-23;
    double l = lambdaNm * 1e-9;
    double Le = (2 * h * c * c) / (std::pow(l, 5) * (std::exp((h * c) / (l * kb * T)) - 1));
    return float(Le);
}

BlackbodySpectrum::BlackbodySpectrum(float T) : T_(T) {
    float lambdaMax = 2.8977721e-3f / T * 1e9f;
    normalizationFactor_ = 1 / Planck(lambdaMax, T);
}

float BlackbodySpectrum::operator()(float lambda) const { return Planck(lambda, T_) * normalizationFactor_; }

float SellmeierIOR::operator()(float lambda) const {
    double l2 = Sqr(double(lambda) * 1e-3);
    double n2 = 1;
    for (size_t i = 0; i < B_.size(); ++i) n2 += B_[i] * l2 / (l2 - C_[i]);
    return float(std::sqrt(std::max(n2, 1.0)));
}

// ---------------------------------------------------------------------------------------------
// CIE D65 relative spectral power distribution, 300-830nm in 10nm steps.
static const float D65_values[] = {
    0.0341f,  3.2945f,  20.236f,  37.0535f, 39.9488f, 44.9117f, 46.6383f, 52.0891f, 49.9755f,
    54.6482f, 82.7549f, 91.486f,  93.4318f, 86.6823f, 104.865f, 117.008f, 117.812f, 114.861f,
    115.923f, 108.811f, 109.354f, 107.802f, 104.79f,  107.689f, 104.405f, 104.046f, 100.0f,
    96.3342f, 95.788f,  88.6856f, 90.0062f, 89.5991f, 87.6987f, 83.2886f, 83.6992f, 80.0268f,
    80.2146f, 82.2778f, 78.2842f, 69.7213f, 71.6091f, 74.349f,  61.604f,  69.8856f, 75.087f,
    63.5927f, 46.4182f, 66.8054f, 63.3828f, 64.304f,  59.4519f, 51.959f,  57.4406f, 60.3125f};

const Spectrum &StdIlluminantD65() {
    static const std::shared_ptr<Spectrum> d65 = [] {
        std::vector<float> l, v;
        for (int i = 0; i < int(std::size(D65_values)); ++i) {
            l.push_back(300.f + 10.f * i);
            v.push_back(D65_values[i]);
        }
        PiecewiseLinearSpectrum raw(l, v);
        float Y = SpectrumToY(raw);
        for (float &x : v) x /= Y;
        return std::make_shared<PiecewiseLinearSpectrum>(l, v);
    }();
    return *d65;
}

// ---------------------------------------------------------------------------------------------
// Metal optical constants (approximate, after Johnson & Christy 1972 and Rakic 1995), sampled
// coarsely across the visible range. Values outside the tabulated range are clamped.
namespace {
struct MetalData {
    const char *name;
    std::vector<float> lambda, n, k;
};
const MetalData kMetals[] = {
    {"Au",
     {360, 400, 450, 500, 550, 600, 650, 700, 750, 830},
     {1.72f, 1.66f, 1.40f, 0.97f, 0.43f, 0.25f, 0.17f, 0.16f, 0.16f, 0.17f},
     {1.92f, 1.96f, 1.88f, 1.87f, 2.46f, 2.98f, 3.50f, 3.95f, 4.40f, 5.10f}},
    {"Cu",
     {360, 400, 450, 500, 550, 600, 650, 700, 750, 830},
     {1.26f, 1.18f, 1.17f, 1.13f, 1.03f, 0.27f, 0.21f, 0.21f, 0.23f, 0.27f},
     {2.30f, 2.40f, 2.53f, 2.56f, 2.58f, 3.24f, 3.67f, 4.18f, 4.60f, 5.20f}},
    {"Ag",
     {360, 400, 450, 500, 550, 600, 650, 700, 750, 830},
     {0.07f, 0.05f, 0.04f, 0.05f, 0.06f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f},
     {1.60f, 2.10f, 2.66f, 3.13f, 3.59f, 4.00f, 4.44f, 4.80f, 5.20f, 5.80f}},
    {"Al",
     {360, 400, 450, 500, 550, 600, 650, 700, 750, 830},
     {0.40f, 0.49f, 0.62f, 0.77f, 0.96f, 1.20f, 1.47f, 1.83f, 2.40f, 2.70f},
     {4.35f, 4.86f, 5.47f, 6.08f, 6.69f, 7.26f, 7.79f, 8.31f, 8.62f, 8.30f}},
};

struct GlassData {
    const char *name;
    std::vector<float> B, C;
};
const GlassData kGlasses[] = {
    {"BK7", {1.03961212f, 0.231792344f, 1.01046945f}, {0.00600069867f, 0.0200179144f, 103.560653f}},
    {"SF11", {1.73759695f, 0.313747346f, 1.89878101f}, {0.013188707f, 0.0623068142f, 155.23629f}},
    {"fused_silica", {0.6961663f, 0.4079426f, 0.8974794f}, {0.00467914826f, 0.0135120631f, 97.9340025f}},
    {"diamond", {0.3306f, 4.3356f}, {0.030625f, 0.011236f}},
    {"sapphire", {1.4313493f, 0.65054713f, 5.3414021f}, {0.00527993f, 0.0142383f, 325.017834f}},
};
} // namespace

bool GetNamedMetal(const std::string &name, SpectrumPtr *eta, SpectrumPtr *k) {
    for (const MetalData &m : kMetals) {
        if (name == m.name) {
            *eta = std::make_shared<PiecewiseLinearSpectrum>(m.lambda, m.n);
            *k = std::make_shared<PiecewiseLinearSpectrum>(m.lambda, m.k);
            return true;
        }
    }
    return false;
}

SpectrumPtr GetNamedIOR(const std::string &name) {
    for (const GlassData &g : kGlasses)
        if (name == g.name) return std::make_shared<SellmeierIOR>(g.B, g.C);
    if (name == "water") return std::make_shared<CauchyIOR>(1.3240f, 3089.f);
    if (name == "air") return std::make_shared<ConstantSpectrum>(1.000293f);
    return nullptr;
}

SpectrumPtr GetNamedSpectrum(const std::string &name) {
    if (name == "D65" || name == "stdillum-D65") {
        struct Ref : Spectrum {
            float operator()(float l) const override { return StdIlluminantD65()(l); }
            float MaxValue() const override { return StdIlluminantD65().MaxValue(); }
        };
        return std::make_shared<Ref>();
    }
    if (auto ior = GetNamedIOR(name)) return ior;
    return nullptr;
}

} // namespace pr

#include "doctest/doctest.h"

#include "core/memory.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "materials/bxdf.h"

#include <functional>
#include <string>

using namespace pr;

namespace {

struct Case {
    std::string name;
    std::function<const BxDF *(ScratchBuffer &)> make;
    bool exactPdf = true;
};

// Compares sampling-based and uniform-sphere-based estimates of the directional albedo, and
// checks that Sample_f's pdf agrees with PDF() and that PDF integrates to at most one.
void CheckBxDF(const Case &c, const Vec3f &wo, TransportMode mode) {
    ScratchBuffer buf;
    const BxDF *bxdf = c.make(buf);
    RNG rng(12345);
    const int N = 200000;
    double albedoSampled = 0;
    int pdfMismatch = 0, nonSpecular = 0;
    for (int i = 0; i < N; ++i) {
        float uc = rng.UniformFloat();
        Vec2f u = rng.Uniform2D();
        auto bs = bxdf->Sample_f(wo, uc, u, mode, BxDFReflTransFlags::All);
        if (!bs || bs->pdf == 0) continue;
        albedoSampled += bs->f.Average() * AbsCosTheta(bs->wi) / bs->pdf;
        if (!bs->IsSpecular() && c.exactPdf) {
            ++nonSpecular;
            float p = bxdf->PDF(wo, bs->wi, mode, BxDFReflTransFlags::All);
            if (std::abs(p - bs->pdf) > 1e-3f * std::max(1.f, bs->pdf)) ++pdfMismatch;
            SampledSpectrum f = bxdf->f(wo, bs->wi, mode);
            if (std::abs(f.Average() - bs->f.Average()) > 1e-3f * std::max(1.f, f.Average())) ++pdfMismatch;
        }
    }
    albedoSampled /= N;
    INFO(c.name << " cos(wo)=" << wo.z);
    CHECK(pdfMismatch <= nonSpecular / 1000);
    CHECK(albedoSampled <= 1.02);

    if (IsSpecular(bxdf->Flags())) return;
    double albedoUniform = 0, pdfIntegral = 0;
    for (int i = 0; i < N; ++i) {
        Vec3f wi = SampleUniformSphere(rng.Uniform2D());
        albedoUniform += bxdf->f(wo, wi, mode).Average() * AbsCosTheta(wi) / UniformSpherePDF();
        pdfIntegral += bxdf->PDF(wo, wi, mode, BxDFReflTransFlags::All) / UniformSpherePDF();
    }
    albedoUniform /= N;
    pdfIntegral /= N;
    CHECK(pdfIntegral <= 1.03);
    CHECK(albedoSampled == doctest::Approx(albedoUniform).epsilon(0.04).scale(1));
}

} // namespace

TEST_CASE("BxDF sampling is consistent with evaluation") {
    std::vector<Case> cases = {
        {"diffuse", [](ScratchBuffer &b) { return b.New<DiffuseBxDF>(SampledSpectrum(0.8f)); }},
        {"rough conductor",
         [](ScratchBuffer &b) {
             return b.New<ConductorBxDF>(TrowbridgeReitzDistribution(0.3f, 0.3f), SampledSpectrum(0.2f), SampledSpectrum(3.f));
         }},
        {"anisotropic conductor",
         [](ScratchBuffer &b) {
             return b.New<ConductorBxDF>(TrowbridgeReitzDistribution(0.1f, 0.5f), SampledSpectrum(1.2f), SampledSpectrum(7.f));
         }},
        {"rough dielectric", [](ScratchBuffer &b) { return b.New<DielectricBxDF>(1.5f, TrowbridgeReitzDistribution(0.3f, 0.3f)); }},
        {"rough dielectric (inside)",
         [](ScratchBuffer &b) { return b.New<DielectricBxDF>(1.0f / 1.5f, TrowbridgeReitzDistribution(0.2f, 0.2f)); }},
        {"sheen over diffuse",
         [](ScratchBuffer &b) {
             auto *base = b.New<DiffuseBxDF>(SampledSpectrum(0.5f));
             return b.New<SheenBxDF>(base, SampledSpectrum(1.f), 0.3f, 1.f);
         }},
        {"coated diffuse",
         [](ScratchBuffer &b) {
             auto *top = b.New<DielectricBxDF>(1.5f, TrowbridgeReitzDistribution(0, 0));
             auto *bottom = b.New<DiffuseBxDF>(SampledSpectrum(0.7f));
             return b.New<LayeredBxDF>(top, bottom, 0.01f, SampledSpectrum(0.f), 0.f, true);
         },
         false},
    };
    for (const Case &c : cases)
        for (float cosTheta : {0.9f, 0.5f, 0.15f}) {
            Vec3f wo(SafeSqrt(1 - cosTheta * cosTheta), 0, cosTheta);
            // Importance mode: no 1/eta^2 radiance scaling, so the albedo must be <= 1.
            CheckBxDF(c, wo, TransportMode::Importance);
        }
}

TEST_CASE("smooth dielectric conserves energy") {
    ScratchBuffer buf;
    DielectricBxDF d(1.5f, TrowbridgeReitzDistribution(0, 0));
    RNG rng(7);
    // Importance transport (no 1/eta^2 radiance scaling): R + T == 1.
    double sum = 0;
    const int N = 100000;
    Vec3f wo = Normalize(Vec3f(0.3f, 0.1f, 0.8f));
    for (int i = 0; i < N; ++i) {
        auto bs = d.Sample_f(wo, rng.UniformFloat(), rng.Uniform2D(), TransportMode::Importance, BxDFReflTransFlags::All);
        if (bs) sum += bs->f.Average() * AbsCosTheta(bs->wi) / bs->pdf;
    }
    CHECK(sum / N == doctest::Approx(1.0).epsilon(0.005));
}

TEST_CASE("Fresnel limits") {
    CHECK(FrDielectric(1.f, 1.5f) == doctest::Approx(0.04).epsilon(0.01));
    CHECK(FrDielectric(0.1f, 1.f / 1.5f) == doctest::Approx(1.0));  // total internal reflection
    SampledSpectrum F = FrComplex(1.f, SampledSpectrum(1.f), SampledSpectrum(0.f));
    CHECK(F[0] == doctest::Approx(0.0).epsilon(1e-4));
}

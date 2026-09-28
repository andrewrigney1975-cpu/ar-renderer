#include "doctest/doctest.h"

#include "core/color.h"
#include "core/rgb2spec.h"
#include "core/sampling.h"
#include "core/spectrum.h"

using namespace pr;

static RGB ReflectanceToRGB(const Spectrum &s) {
    // Reflectance under the renderer's D65, through the same adaptation as the film.
    struct Lit : Spectrum {
        const Spectrum &r;
        explicit Lit(const Spectrum &r) : r(r) {}
        float operator()(float l) const override { return r(l) * StdIlluminantD65()(l); }
        float MaxValue() const override { return 1; }
    } lit(s);
    XYZ xyz = SpectrumToXYZ(lit);
    RGB a = SpectralWhiteAdaptation().Apply(xyz.x, xyz.y, xyz.z);
    return XYZToRGBMatrix(ColorSpaceId::sRGB).Apply(a.r, a.g, a.b);
}

TEST_CASE("visible wavelength sampling pdf integrates to one") {
    double sum = 0;
    for (int l = 360; l < 830; ++l) sum += VisibleWavelengthPDF(l + 0.5f);
    CHECK(sum == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("D65 is normalized to Y = 1 and renders white") {
    CHECK(SpectrumToY(StdIlluminantD65()) == doctest::Approx(1.0).epsilon(1e-3));
    ConstantSpectrum one(1.f);
    RGB c = ReflectanceToRGB(one);
    CHECK(c.r == doctest::Approx(1.0).epsilon(0.01));
    CHECK(c.g == doctest::Approx(1.0).epsilon(0.01));
    CHECK(c.b == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("RGB albedo uplift round-trips") {
    const RGB colors[] = {{0.5f, 0.5f, 0.5f}, {0.8f, 0.2f, 0.1f}, {0.1f, 0.6f, 0.2f}, {0.2f, 0.3f, 0.9f},
                          {0.95f, 0.9f, 0.1f}, {0.05f, 0.05f, 0.05f}};
    for (const RGB &c : colors) {
        RGBAlbedoSpectrum s(c);
        RGB r = ReflectanceToRGB(s);
        CHECK(r.r == doctest::Approx(c.r).epsilon(0.03).scale(1));
        CHECK(r.g == doctest::Approx(c.g).epsilon(0.03).scale(1));
        CHECK(r.b == doctest::Approx(c.b).epsilon(0.03).scale(1));
        CHECK(s.MaxValue() <= 1.0001f);
    }
}

TEST_CASE("Monte Carlo spectral XYZ matches the integral") {
    BlackbodySpectrum bb(3000);
    XYZ ref = SpectrumToXYZ(bb);
    double X = 0, Y = 0, Z = 0;
    const int n = 20000;
    for (int i = 0; i < n; ++i) {
        SampledWavelengths lambda = SampledWavelengths::SampleVisible((i + 0.5f) / n);
        XYZ x = ToXYZ(bb.Sample(lambda), lambda);
        X += x.x; Y += x.y; Z += x.z;
    }
    CHECK(X / n == doctest::Approx(ref.x).epsilon(0.01));
    CHECK(Y / n == doctest::Approx(ref.y).epsilon(0.01));
    CHECK(Z / n == doctest::Approx(ref.z).epsilon(0.02));
}

TEST_CASE("dispersion data is plausible") {
    auto bk7 = GetNamedIOR("BK7");
    REQUIRE(bool(bk7));
    CHECK((*bk7)(587.6f) == doctest::Approx(1.5168).epsilon(1e-3));
    auto sf11 = GetNamedIOR("SF11");
    CHECK((*sf11)(587.6f) == doctest::Approx(1.7847).epsilon(1e-3));
    auto diamond = GetNamedIOR("diamond");
    CHECK((*diamond)(589.f) == doctest::Approx(2.417).epsilon(2e-3));
    CHECK((*bk7)(400.f) > (*bk7)(700.f));
}

TEST_CASE("distribution sampling") {
    std::vector<float> f = {1, 3, 0, 4};
    Distribution1D d(f);
    CHECK(d.DiscretePMF(1) == doctest::Approx(3.0 / 8));
    CHECK(d.DiscretePMF(2) == 0);
    float pmf;
    CHECK(d.SampleDiscrete(0.05f, &pmf) == 0);
    CHECK(d.SampleDiscrete(0.99f, &pmf) == 3);
}

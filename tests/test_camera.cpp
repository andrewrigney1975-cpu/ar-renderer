#include "doctest/doctest.h"

#include "cameras/realistic.h"

using namespace pr;

TEST_CASE("realistic lens: focus, image orientation and dispersion") {
    CameraFilmInfo film{64, 48, 0};
    auto lens = RealisticCamera::DoubleGauss50(0.001f);  // metres
    RealisticCamera cam(Transform(), lens, 0.035f, film, 2.0f, true);
    // Rays through the film centre converge near the focus distance.
    int hits = 0;
    double sumX = 0;
    for (int i = 0; i < 64; ++i) {
        CameraSample cs{Vec2f(32, 24), Vec2f((i % 8 + 0.5f) / 8, (i / 8 + 0.5f) / 8)};
        SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.3f);
        float w;
        auto r = cam.GenerateRay(cs, lambda, &w);
        if (!r) continue;
        ++hits;
        CHECK(r->d.z > 0.99f);
        CHECK(lambda.SecondaryTerminated());
        float t = (2.0f - r->o.z) / r->d.z;
        sumX += std::abs(r->o.x + r->d.x * t) + std::abs(r->o.y + r->d.y * t);
    }
    REQUIRE(hits > 16);
    CHECK(sumX / hits < 0.01);  // within 1 cm of the axis at the focal plane
    // The image is upright: the top-left of the film looks up and to the left.
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.3f);
    float w;
    std::optional<Ray> r;
    for (int i = 0; i < 64 && !r; ++i) r = cam.GenerateRay(CameraSample{Vec2f(4, 4), Vec2f((i % 8 + 0.5f) / 8, (i / 8 + 0.5f) / 8)}, lambda, &w);
    REQUIRE(r);
    CHECK(r->d.x < 0);
    CHECK(r->d.y > 0);
}

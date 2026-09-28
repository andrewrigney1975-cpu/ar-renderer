#include "doctest/doctest.h"

#include "core/film.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "integrators/integrator.h"
#include "lights/ies.h"
#include "scene/scene_loader.h"

#include <filesystem>
#include <fstream>
#include <map>

using namespace pr;
namespace fs = std::filesystem;

namespace {

// A narrow-ish downlight, rotationally symmetric.
const char *kIES = R"(IESNA:LM-63-2002
[TEST] test
TILT=NONE
1 1000 1 7 1 1 2 0.1 0.1 0
1 1 50
0 15 30 45 60 75 90
0
1000 950 700 300 80 10 0
)";

fs::path TestDir() {
    fs::path d = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(d);
    return d;
}

double MeanY(const std::string &json, const char *tag, IntegratorType type, int spp, double mpp = 64) {
    fs::path file = TestDir() / (std::string(tag) + ".prscene.json");
    {
        std::ofstream out(file);
        out << json;
    }
    std::string err;
    auto scene = LoadScene(file.string(), LoadOptions(), &err);
    REQUIRE_MESSAGE(scene, err);
    IntegratorSettings is;
    is.type = type;
    is.spp = spp;
    is.mutationsPerPixel = mpp;
    is.bootstrapSamples = 20000;
    is.chains = 256;
    auto integ = CreateIntegrator(is, 3);
    Film film(scene->settings.film);
    RenderControl rc;
    rc.Start();
    double scale;
    REQUIRE(integ->Render(*scene, film, rc, &scale, &err));
    Image xyz = film.ResolveXYZ(scale);
    double s = 0;
    for (int y = 0; y < xyz.Height(); ++y)
        for (int x = 0; x < xyz.Width(); ++x) s += xyz.Get(x, y).g;
    return s / (double(xyz.Width()) * xyz.Height());
}

std::string LitScene(const std::string &lightJson) {
    return R"({
  "materials": { "floor": { "type": "diffuse", "reflectance": 0.7 },
                 "ball": { "type": "conductor", "metal": "Cu", "roughness": 0.3 } },
  "geometry": { "ground": { "type": "rect", "size": [10, 10] }, "s": { "type": "sphere", "radius": 0.5 } },
  "objects": [ { "geometry": "ground", "material": "floor" },
               { "geometry": "s", "material": "ball", "transform": { "translate": [0, 0.5, 0] } } ],
  "lights": { "l": )" + lightJson + R"( },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 2.5, 4], "look_at": [0, 0.3, 0], "fov_y": 45 } },
  "render": { "camera": "cam", "film": { "width": 32, "height": 24, "filter": "box" } }
})";
}

} // namespace

TEST_CASE("IES profile parsing, evaluation and sampling") {
    std::string err;
    auto p = IESProfile::Parse(kIES, &err);
    REQUIRE_MESSAGE(p, err);
    CHECK(p->PeakCandela() == doctest::Approx(1000));
    CHECK(p->Evaluate(Vec3f(0, 0, 1)) == doctest::Approx(1.0));
    CHECK(p->Evaluate(Vec3f(0, 0, -1)) == 0.f);  // above 90 degrees: dark
    Vec3f w30 = SphericalDirection(std::sin(Radians(30)), std::cos(Radians(30)), 1.0f);
    CHECK(p->Evaluate(w30) == doctest::Approx(0.7).epsilon(0.01));
    // The sampling pdf integrates to one and matches PDF().
    RNG rng(5);
    double integral = 0, maxErr = 0;
    const int n = 50000;
    for (int i = 0; i < n; ++i) {
        integral += p->PDF(SampleUniformSphere(rng.Uniform2D())) / UniformSpherePDF();
        float pdf;
        Vec3f w = p->Sample(rng.Uniform2D(), &pdf);
        maxErr = std::max(maxErr, double(std::abs(pdf - p->PDF(w)) / std::max(pdf, 1e-3f)));
    }
    CHECK(integral / n == doctest::Approx(1.0).epsilon(0.02));
    CHECK(maxErr < 1e-2);
}

TEST_CASE("IES light: path tracer and BDPT agree") {
    fs::path ies = TestDir() / "test.ies";
    {
        std::ofstream out(ies);
        out << kIES;
    }
    std::string light = R"({ "type": "point", "position": [0, 2.5, 0.5], "ies": ")" + ies.generic_string() +
                        R"(", "intensity": 8, "temperature": 3000 })";
    double pt = MeanY(LitScene(light), "ies_pt", IntegratorType::Path, 256);
    double bd = MeanY(LitScene(light), "ies_bdpt", IntegratorType::BDPT, 64);
    MESSAGE("ies: pt=" << pt << " bdpt=" << bd);
    CHECK(pt > 0);
    CHECK(bd == doctest::Approx(pt).epsilon(0.03));
}

TEST_CASE("sun and sky: integrators agree") {
    std::string light = R"({ "type": "sun_sky", "elevation": 35, "azimuth": 40, "turbidity": 3, "sun_size": 4 })";
    double pt = MeanY(LitScene(light), "sun_pt", IntegratorType::Path, 256);
    double bd = MeanY(LitScene(light), "sun_bdpt", IntegratorType::BDPT, 64);
    double ml = MeanY(LitScene(light), "sun_mmlt", IntegratorType::MMLT, 0, 256);
    MESSAGE("sun_sky: pt=" << pt << " bdpt=" << bd << " mmlt=" << ml);
    CHECK(pt > 0.05);
    CHECK(bd == doctest::Approx(pt).epsilon(0.03));
    CHECK(ml == doctest::Approx(pt).epsilon(0.06));
}

TEST_CASE("distant light: path tracer and BDPT agree") {
    std::string light = R"({ "type": "distant", "direction": [-0.3, -1, -0.4], "irradiance": 2, "temperature": 5500 })";
    double pt = MeanY(LitScene(light), "dist_pt", IntegratorType::Path, 128);
    double bd = MeanY(LitScene(light), "dist_bdpt", IntegratorType::BDPT, 32);
    MESSAGE("distant: pt=" << pt << " bdpt=" << bd);
    CHECK(pt > 0);
    CHECK(bd == doctest::Approx(pt).epsilon(0.03));
}

TEST_CASE("light BVH: PMF matches sampling frequencies and renders agree with power sampling") {
    // A grid of small emitters of different sizes and orientations above a floor.
    std::string lights;
    for (int i = 0; i < 48; ++i) {
        float x = float(i % 8) - 3.5f, z = float(i / 8) - 2.5f;
        lights += (i ? "," : "") + std::string("\"l") + std::to_string(i) + "\": { \"type\": \"" + (i % 3 == 0 ? "sphere" : "rect") +
                  "\", \"position\": [" + std::to_string(x) + ", " + std::to_string(1.0f + 0.3f * (i % 5)) + ", " + std::to_string(z) +
                  "], \"radius\": 0.08, \"size\": [0.2, 0.2], \"direction\": [" + std::to_string(0.2f * (i % 4) - 0.3f) +
                  ", -1, 0.1], \"luminance\": " + std::to_string(5 + 7 * (i % 7)) + ", \"temperature\": " + std::to_string(2500 + 150 * i) + " }";
    }
    std::string scene = R"({
  "materials": { "floor": { "type": "diffuse", "reflectance": 0.6 } },
  "geometry": { "ground": { "type": "rect", "size": [12, 12] } },
  "objects": [ { "geometry": "ground", "material": "floor" } ],
  "lights": { )" + lights + R"( },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 5, 7], "look_at": [0, 0, 0], "fov_y": 50 } },
  "render": { "camera": "cam", "film": { "width": 32, "height": 24, "filter": "box" } }
})";
    fs::path file = TestDir() / "manylights.prscene.json";
    std::ofstream(file) << scene;
    std::string err;
    auto sc = LoadScene(file.string(), LoadOptions(), &err);
    REQUIRE_MESSAGE(sc, err);
    // Empirical frequencies at a shading point vs. PMF.
    Vec3f p(0.3f, 0, -0.7f), n(0, 1, 0);
    std::map<const Light *, int> counts;
    const int N = 400000;
    RNG rng(17);
    double pmfSum = 0;
    for (const auto &l : sc->lights) pmfSum += sc->lightBVH.PMF(p, n, l.get());
    CHECK(pmfSum == doctest::Approx(1.0).epsilon(1e-3));
    for (int i = 0; i < N; ++i) {
        float pmf;
        const Light *l = sc->lightBVH.Sample(p, n, rng.UniformFloat(), &pmf);
        REQUIRE(l);
        CHECK(pmf == doctest::Approx(sc->lightBVH.PMF(p, n, l)).epsilon(1e-4));
        counts[l]++;
    }
    double maxDev = 0;
    for (auto &[l, c] : counts) {
        double expect = sc->lightBVH.PMF(p, n, l) * N;
        if (expect > 500) maxDev = std::max(maxDev, std::abs(c - expect) / expect);
    }
    CHECK(maxDev < 0.1);

    auto meanY = [&](bool bvh, IntegratorType type, int spp) {
        IntegratorSettings is;
        is.type = type;
        is.spp = spp;
        is.lightBVH = bvh;
        auto integ = CreateIntegrator(is, 4);
        Film film(sc->settings.film);
        RenderControl rc;
        rc.Start();
        double scale;
        REQUIRE(integ->Render(*sc, film, rc, &scale, &err));
        Image xyz = film.ResolveXYZ(scale);
        double s = 0;
        for (float v : xyz.Data()) s += v;
        return s / xyz.Data().size();
    };
    double withBVH = meanY(true, IntegratorType::Path, 128);
    double withPower = meanY(false, IntegratorType::Path, 128);
    double bdpt = meanY(false, IntegratorType::BDPT, 32);
    MESSAGE("many lights: bvh=" << withBVH << " power=" << withPower << " bdpt=" << bdpt);
    CHECK(withBVH == doctest::Approx(withPower).epsilon(0.02));
    CHECK(withBVH == doctest::Approx(bdpt).epsilon(0.03));
}

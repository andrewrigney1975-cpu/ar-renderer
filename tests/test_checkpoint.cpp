#include "doctest/doctest.h"

#include "core/film.h"
#include "integrators/integrator.h"
#include "scene/scene_loader.h"

#include <filesystem>
#include <fstream>

using namespace pr;
namespace fs = std::filesystem;

namespace {

const char *kScene = R"({
  "materials": { "white": { "type": "diffuse", "reflectance": 0.7 },
                 "glass": { "type": "dielectric", "ior": 1.5 } },
  "geometry": { "floor": { "type": "rect", "size": [4, 4] }, "ball": { "type": "sphere", "radius": 0.5 } },
  "objects": [ { "geometry": "floor", "material": "white" },
               { "geometry": "ball", "material": "glass", "transform": { "translate": [0, 0.5, 0] } } ],
  "lights": { "key": { "type": "rect", "position": [0, 2.5, 0.5], "look_at": [0, 0, 0], "size": [0.6, 0.6], "luminance": 20 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 1.4, 3.5], "look_at": [0, 0.4, 0], "fov_y": 45 } },
  "render": { "camera": "cam", "film": { "width": 24, "height": 16 } }
})";

struct Result {
    Image xyz;
    bool ok;
    std::string err;
};

Result Render(IntegratorType type, int spp, double mpp, const std::string &checkpoint, const std::string &resume) {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path file = dir / "ckpt.prscene.json";
    std::ofstream(file) << kScene;
    std::string err;
    auto scene = LoadScene(file.string(), LoadOptions(), &err);
    REQUIRE_MESSAGE(scene, err);
    IntegratorSettings is;
    is.type = type;
    is.spp = spp;
    is.mutationsPerPixel = mpp;
    is.bootstrapSamples = 5000;
    is.chains = 64;
    is.maxDepth = type == IntegratorType::Path ? -1 : 16;
    auto integ = CreateIntegrator(is, 9);
    Film film(scene->settings.film);
    RenderControl rc;
    rc.checkpointPath = checkpoint;
    rc.resumePath = resume;
    rc.fingerprint = scene->Fingerprint();
    rc.seed = 9;
    rc.Start();
    double scale = 0;
    Result r;
    r.ok = integ->Render(*scene, film, rc, &scale, &r.err);
    r.xyz = film.ResolveXYZ(scale);
    return r;
}

double MaxRelDiff(const Image &a, const Image &b) {
    double m = 0;
    for (size_t i = 0; i < a.Data().size(); ++i) {
        double x = a.Data()[i], y = b.Data()[i];
        m = std::max(m, std::abs(x - y) / std::max(1e-3, std::abs(y)));
    }
    return m;
}

} // namespace

TEST_CASE("checkpoint + resume equals an uninterrupted render (path tracer)") {
    fs::path ck = fs::temp_directory_path() / "prender_tests" / "pt.prck";
    fs::remove(ck);
    Result half = Render(IntegratorType::Path, 8, 0, ck.string(), "");
    REQUIRE(half.ok);
    REQUIRE(fs::exists(ck));
    Result resumed = Render(IntegratorType::Path, 16, 0, "", ck.string());
    REQUIRE_MESSAGE(resumed.ok, resumed.err);
    Result direct = Render(IntegratorType::Path, 16, 0, "", "");
    CHECK(MaxRelDiff(resumed.xyz, direct.xyz) < 1e-4);
    CHECK(MaxRelDiff(half.xyz, direct.xyz) > 1e-3);
}

TEST_CASE("checkpoint + resume equals an uninterrupted render (MMLT)") {
    fs::path ck = fs::temp_directory_path() / "prender_tests" / "mmlt.prck";
    fs::remove(ck);
    Result half = Render(IntegratorType::MMLT, 0, 16, ck.string(), "");
    REQUIRE(half.ok);
    Result resumed = Render(IntegratorType::MMLT, 0, 32, "", ck.string());
    REQUIRE_MESSAGE(resumed.ok, resumed.err);
    Result direct = Render(IntegratorType::MMLT, 0, 32, "", "");
    double d = MaxRelDiff(resumed.xyz, direct.xyz);
    MESSAGE("MMLT resume max relative difference: " << d);
    CHECK(d < 1e-3);
}

TEST_CASE("resume rejects a checkpoint from different settings") {
    fs::path ck = fs::temp_directory_path() / "prender_tests" / "pt2.prck";
    fs::remove(ck);
    REQUIRE(Render(IntegratorType::Path, 2, 0, ck.string(), "").ok);
    Result wrong = Render(IntegratorType::BDPT, 4, 0, "", ck.string());
    CHECK_FALSE(wrong.ok);
    CHECK(wrong.err.find("different") != std::string::npos);
}

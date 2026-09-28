#include "doctest/doctest.h"

#include "core/film.h"
#include "core/image.h"
#include "core/parallel.h"
#include "integrators/integrator.h"
#include "scene/scene_loader.h"

#include <filesystem>
#include <fstream>

using namespace pr;
namespace fs = std::filesystem;

namespace {

// Renders a scene given as JSON text and returns the mean luminance (Y) over the image.
double RenderMeanY(const std::string &sceneJson, IntegratorType type, int spp, double mpp, const char *tag,
                   int maxDepth = -1) {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path file = dir / (std::string(tag) + ".prscene.json");
    {
        std::ofstream out(file);
        out << sceneJson;
    }
    std::string err;
    LoadOptions opts;
    auto scene = LoadScene(file.string(), opts, &err);
    REQUIRE_MESSAGE(scene, err);
    IntegratorSettings is = scene->settings.integrator;
    is.type = type;
    is.spp = spp;
    is.mutationsPerPixel = mpp;
    is.bootstrapSamples = 20000;
    is.chains = 256;
    is.maxDepth = maxDepth;
    auto integrator = CreateIntegrator(is, 1);
    Film film(scene->settings.film);
    RenderControl control;
    control.Start();
    double scale = 0;
    REQUIRE(integrator->Render(*scene, film, control, &scale, &err));
    Image xyz = film.ResolveXYZ(scale);
    double sum = 0;
    for (int y = 0; y < xyz.Height(); ++y)
        for (int x = 0; x < xyz.Width(); ++x) sum += xyz.Get(x, y).g;
    return sum / (double(xyz.Width()) * xyz.Height());
}

const char *kFurnace = R"({
  "materials": { "white": { "type": "diffuse", "reflectance": 1.0 } },
  "geometry": { "ball": { "type": "sphere", "radius": 1 } },
  "objects": [ { "geometry": "ball", "material": "white" } ],
  "lights": { "sky": { "type": "environment", "color": [1, 1, 1], "intensity": 1 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 0, 4], "look_at": [0, 0, 0], "fov_y": 40 } },
  "render": { "camera": "cam", "film": { "width": 24, "height": 24, "filter": "box" } }
})";

const char *kCornellish = R"({
  "materials": {
    "white": { "type": "diffuse", "reflectance": [0.7, 0.7, 0.7] },
    "red":   { "type": "diffuse", "reflectance": [0.7, 0.1, 0.1] },
    "glass": { "type": "dielectric", "ior": 1.5, "roughness": 0.2 },
    "gold":  { "type": "conductor", "metal": "Au", "roughness": 0.3 }
  },
  "geometry": {
    "floor": { "type": "rect", "size": [4, 4] },
    "wall":  { "type": "rect", "size": [4, 4] },
    "ball":  { "type": "sphere", "radius": 0.5 }
  },
  "objects": [
    { "geometry": "floor", "material": "white" },
    { "geometry": "wall", "material": "red", "transform": { "translate": [0, 2, -2], "rotate": [90, 1, 0, 0] } },
    { "geometry": "ball", "material": "glass", "transform": { "translate": [-0.6, 0.5, 0] } },
    { "geometry": "ball", "material": "gold", "transform": { "translate": [0.7, 0.5, 0.3] } }
  ],
  "lights": { "key": { "type": "rect", "position": [0, 3, 0.5], "look_at": [0, 0, 0], "size": [1, 1], "luminance": 8, "temperature": 5000 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 1.5, 4.5], "look_at": [0, 0.6, 0], "fov_y": 45 } },
  "render": { "camera": "cam", "film": { "width": 32, "height": 24, "filter": "box" } }
})";

} // namespace

TEST_CASE("white furnace: path tracer") {
    double y = RenderMeanY(kFurnace, IntegratorType::Path, 64, 0, "furnace_pt");
    CHECK(y == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("white furnace: BDPT") {
    double y = RenderMeanY(kFurnace, IntegratorType::BDPT, 32, 0, "furnace_bdpt", 64);
    CHECK(y == doctest::Approx(1.0).epsilon(0.02));
}

TEST_CASE("white furnace: MMLT") {
    double y = RenderMeanY(kFurnace, IntegratorType::MMLT, 0, 64, "furnace_mmlt", 24);
    // MMLT truncates at max depth; albedo 1 means ~(1 - tiny) of the energy is captured.
    CHECK(y == doctest::Approx(1.0).epsilon(0.03));
}

TEST_CASE("integrators agree on a glossy/refractive scene") {
    double pt = RenderMeanY(kCornellish, IntegratorType::Path, 512, 0, "agree_pt");
    double bdpt = RenderMeanY(kCornellish, IntegratorType::BDPT, 128, 0, "agree_bdpt", 64);
    double mmlt = RenderMeanY(kCornellish, IntegratorType::MMLT, 0, 256, "agree_mmlt", 32);
    double pss = RenderMeanY(kCornellish, IntegratorType::PSSMLT, 0, 128, "agree_pss", 32);
    MESSAGE("pt=" << pt << " bdpt=" << bdpt << " mmlt=" << mmlt << " pssmlt=" << pss);
    CHECK(pt > 0);
    CHECK(bdpt == doctest::Approx(pt).epsilon(0.03));
    CHECK(mmlt == doctest::Approx(pt).epsilon(0.05));
    CHECK(pss == doctest::Approx(pt).epsilon(0.05));
}

TEST_CASE("normal and bump mapping") {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    // Flat tangent-space normal map (0.5, 0.5, 1): must leave shading unchanged.
    Image flat(4, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) flat.Set(x, y, RGB(0.5f, 0.5f, 1.f));
    std::string err;
    REQUIRE(WritePFM((dir / "flatnormal.pfm").string(), flat, &err));
    auto scene = [&](const std::string &extra) {
        return std::string(R"({
  "textures": { "nm": { "type": "image", "file": ")") + (dir / "flatnormal.pfm").generic_string() + R"(", "colorspace": "linear" },
                "bumps": { "type": "noise", "frequency": 6, "octaves": 3 } },
  "materials": { "m": { "type": "conductor", "metal": "Cu", "roughness": 0.2 )" + extra + R"( } },
  "geometry": { "ball": { "type": "sphere", "radius": 1 } },
  "objects": [ { "geometry": "ball", "material": "m" } ],
  "lights": { "sky": { "type": "environment", "color": [1, 1, 1] },
              "key": { "type": "distant", "direction": [-0.5, -1, -0.3], "irradiance": 3 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 0, 4], "look_at": [0, 0, 0], "fov_y": 40 } },
  "render": { "camera": "cam", "film": { "width": 24, "height": 24, "filter": "box" } }
})";
    };
    double plain = RenderMeanY(scene(""), IntegratorType::Path, 32, 0, "nm_plain");
    double flatNm = RenderMeanY(scene(R"(, "normal_map": { "texture": "nm" })"), IntegratorType::Path, 32, 0, "nm_flat");
    double bumped = RenderMeanY(scene(R"(, "bump_map": { "texture": "bumps", "scale": 0.05 })"), IntegratorType::Path, 32, 0, "nm_bump");
    MESSAGE("plain=" << plain << " flat normal map=" << flatNm << " bumped=" << bumped);
    CHECK(flatNm == doctest::Approx(plain).epsilon(1e-3));
    CHECK(std::abs(bumped - plain) > 1e-3 * plain);
}

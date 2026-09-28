#include "doctest/doctest.h"

#include "core/film.h"
#include "integrators/integrator.h"
#include "media/medium.h"
#include "scene/scene_loader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace pr;
namespace fs = std::filesystem;

namespace {

void WriteVol(const fs::path &p, int n, float value, float bmin, float bmax) {
    FILE *f = std::fopen(p.string().c_str(), "wb");
    REQUIRE(f);
    std::fwrite("VOL", 1, 3, f);
    uint8_t version = 3;
    std::fwrite(&version, 1, 1, f);
    int32_t hdr[5] = {1, n, n, n, 1};
    std::fwrite(hdr, 4, 5, f);
    float bb[6] = {bmin, bmin, bmin, bmax, bmax, bmax};
    std::fwrite(bb, 4, 6, f);
    std::vector<float> data(size_t(n) * n * n, value);
    std::fwrite(data.data(), 4, data.size(), f);
    std::fclose(f);
}

double MeanY(const std::string &json, const char *tag, IntegratorType type, int spp) {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path file = dir / (std::string(tag) + ".prscene.json");
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
    auto integ = CreateIntegrator(is, 7);
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

std::string FogScene(const std::string &mediumJson) {
    return R"({
  "media": { "fog": )" + mediumJson + R"( },
  "materials": { "boundary": { "type": "interface", "medium": "fog" },
                 "floor": { "type": "diffuse", "reflectance": 0.6 } },
  "geometry": { "ball": { "type": "sphere", "radius": 1 }, "ground": { "type": "rect", "size": [8, 8] } },
  "objects": [ { "geometry": "ball", "material": "boundary", "transform": { "translate": [0, 1, 0] } },
               { "geometry": "ground", "material": "floor" } ],
  "lights": { "key": { "type": "rect", "position": [0, 4, 1], "look_at": [0, 0, 0], "size": [1, 1], "luminance": 10 },
              "sky": { "type": "environment", "color": [0.3, 0.4, 0.6] } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 1.2, 4.5], "look_at": [0, 1, 0], "fov_y": 40 } },
  "render": { "camera": "cam", "film": { "width": 24, "height": 24, "filter": "box" } }
})";
}

} // namespace

TEST_CASE("grid medium transmittance matches Beer-Lambert for constant density") {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path vol = dir / "const.vol";
    WriteVol(vol, 4, 1.f, -1, 1);
    std::string err;
    auto field = VoxelGridField::LoadVol(vol.string(), &err);
    REQUIRE_MESSAGE(field, err);
    GridMedium m(std::make_shared<ConstantSpectrum>(0.3f), std::make_shared<ConstantSpectrum>(0.9f), 1.f, 0.f, field,
                 Transform());
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.3f);
    double sum = 0;
    const int n = 20000;
    for (int i = 0; i < n; ++i) {
        Ray r(Vec3f(-2, 0.1f * (i % 7) - 0.3f, 0.01f * (i % 13)), Vec3f(1, 0, 0));
        sum += m.Transmittance(r, 10, lambda)[0];
    }
    CHECK(sum / n == doctest::Approx(std::exp(-1.2 * 2)).epsilon(0.03));
}

TEST_CASE("constant grid medium renders like the equivalent homogeneous medium") {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path vol = dir / "const2.vol";
    WriteVol(vol, 8, 1.f, -1.5f, 1.5f);
    std::string volPath = vol.generic_string();
    double homo = MeanY(FogScene(R"({ "type": "homogeneous", "sigma_s": [1.5, 1.0, 0.6], "sigma_a": 0.2, "g": 0.3 })"),
                        "fog_homo", IntegratorType::Path, 256);
    double grid = MeanY(FogScene(R"({ "type": "grid", "file": ")" + volPath +
                                 R"(", "sigma_s": [1.5, 1.0, 0.6], "sigma_a": 0.2, "g": 0.3,
                                        "transform": { "translate": [0, 1, 0] } })"),
                        "fog_grid", IntegratorType::Path, 256);
    double gridBdpt = MeanY(FogScene(R"({ "type": "grid", "file": ")" + volPath +
                                     R"(", "sigma_s": [1.5, 1.0, 0.6], "sigma_a": 0.2, "g": 0.3,
                                            "transform": { "translate": [0, 1, 0] } })"),
                            "fog_grid_bdpt", IntegratorType::BDPT, 64);
    MESSAGE("homogeneous=" << homo << " grid=" << grid << " grid(bdpt)=" << gridBdpt);
    CHECK(grid == doctest::Approx(homo).epsilon(0.03));
    CHECK(gridBdpt == doctest::Approx(homo).epsilon(0.03));
}

TEST_CASE("noise medium loads and renders") {
    double y = MeanY(FogScene(R"({ "type": "noise", "sigma_s": 6, "sigma_a": 0.1, "frequency": 2, "threshold": 0.4 })"),
                     "fog_noise", IntegratorType::Path, 16);
    CHECK(y > 0);
}

// ---------------------------------------------------------------------------------------------
// NanoVDB: build a fog sphere with NanoVDB's own tools, write it, load it through the renderer.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244 4267 4146 4456 4457 4458 4100 4127 4702)
#endif
#include <nanovdb/tools/CreatePrimitives.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

TEST_CASE("NanoVDB fog volume loads and attenuates") {
    fs::path dir = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(dir);
    fs::path file = dir / "fogsphere.nvdb";
    {
        auto handle = nanovdb::tools::createFogVolumeSphere<float>(1.0, nanovdb::Vec3d(0.0), 0.02, 3.0,
                                                                   nanovdb::Vec3d(0.0), "density");
        std::ofstream os(file, std::ios::binary);
        nanovdb::io::writeUncompressedGrid(os, handle.gridData());
    }
    std::string err;
    auto field = LoadNanoVDB(file.string(), "density", &err);
    REQUIRE_MESSAGE(field, err);
    CHECK(field->Density(Vec3f(0, 0, 0)) == doctest::Approx(1.0).epsilon(0.02));
    CHECK(field->Density(Vec3f(1.5f, 0, 0)) == 0.f);
    GridMedium m(std::make_shared<ConstantSpectrum>(0.f), std::make_shared<ConstantSpectrum>(0.5f), 1.f, 0.f, field,
                 Transform());
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
    double sum = 0;
    const int n = 20000;
    for (int i = 0; i < n; ++i) sum += m.Transmittance(Ray(Vec3f(-3, 1e-6f * i, 0), Vec3f(1, 0, 0)), 6, lambda)[0];
    // Density ~1 over a diameter of ~2 (slightly less at the fog ramp).
    CHECK(sum / n == doctest::Approx(std::exp(-0.5 * 2.0)).epsilon(0.08));
}

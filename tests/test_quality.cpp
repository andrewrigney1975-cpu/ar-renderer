// Speckle and denoising work: the MLT sampler fix, path-level spectral MIS, the firefly clamp,
// denoiser guide buffers and OIDN denoising.

#include "doctest/doctest.h"

#include "core/denoise.h"
#include "core/film.h"
#include "integrators/aov.h"
#include "integrators/integrator.h"
#include "integrators/mlt.h"
#include "scene/scene_loader.h"

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace pr;
namespace fs = std::filesystem;

namespace {

std::unique_ptr<Scene> Load(const std::string &json, const char *tag) {
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
    return scene;
}

// Renders and returns the image in linear sRGB.
Image Render(const Scene &scene, IntegratorType type, int spp, double mpp, int maxDepth, uint64_t seed = 1,
             int chains = 256, int64_t bootstrap = 40000) {
    IntegratorSettings is = scene.settings.integrator;
    is.type = type;
    is.spp = spp;
    is.mutationsPerPixel = mpp;
    is.bootstrapSamples = bootstrap;
    is.chains = chains;
    is.maxDepth = maxDepth;
    auto integ = CreateIntegrator(is, seed);
    Film film(scene.settings.film);
    RenderControl rc;
    rc.Start();
    double scale = 0;
    std::string err;
    REQUIRE_MESSAGE(integ->Render(scene, film, rc, &scale, &err), err);
    Image img(film.Width(), film.Height());
    std::vector<float> row(size_t(film.Width()) * 3);
    for (int y = 0; y < film.Height(); ++y) {
        film.ResolveRow(y, scale, row.data());
        for (int x = 0; x < film.Width(); ++x) img.Set(x, y, RGB(row[3 * x], row[3 * x + 1], row[3 * x + 2]));
    }
    return img;
}

RGB Mean(const Image &img) {
    double s[3] = {0, 0, 0};
    for (int y = 0; y < img.Height(); ++y)
        for (int x = 0; x < img.Width(); ++x) {
            RGB c = img.Get(x, y);
            s[0] += c.r;
            s[1] += c.g;
            s[2] += c.b;
        }
    double n = double(img.Width()) * img.Height();
    return RGB(float(s[0] / n), float(s[1] / n), float(s[2] / n));
}

// A wax-like sphere: rough dielectric boundary around a strongly chromatic scattering interior.
// Its random walks are long and wavelength-dependent, which exposed both the MLT sampler bug and
// the exponential variance of per-event spectral MIS.
const char *kWax = R"({
  "materials": { "wax": { "type": "subsurface", "ior": 1.45, "roughness": 0.3,
                          "albedo": [0.96, 0.86, 0.70], "mfp": [0.05, 0.03, 0.015] } },
  "geometry": { "ball": { "type": "sphere", "radius": 0.5 } },
  "objects": [ { "geometry": "ball", "material": "wax" } ],
  "lights": { "key": { "type": "rect", "position": [-1.5, 2.5, 2], "look_at": [0, 0, 0], "size": [1, 1],
                       "temperature": 5200, "luminance": 40 },
              "sky": { "type": "environment", "color": [0.55, 0.68, 1.0], "intensity": 0.12 } },
  "cameras": { "cam": { "type": "perspective", "position": [0, 0, 3], "look_at": [0, 0, 0], "fov_y": 22 } },
  "render": { "camera": "cam", "film": { "width": 16, "height": 16, "filter": "box" } }
})";

} // namespace

TEST_CASE("MLT sampler: coordinates used for the first time are uniform") {
    // Small steps only, so a first-used coordinate relies entirely on its initial value. Before
    // the fix it started from 0 and stayed within a few sigma of 0/1.
    const int n = 4000;
    double distToEdge = 0;
    for (int i = 0; i < n; ++i) {
        MLTSampler s(uint64_t(i) + 1, 0.01f, 0.f, 1);
        s.StartIteration();
        s.StartStream(0);
        (void)s.Get1D();
        s.Accept();
        for (int k = 0; k < 5; ++k) {  // a few small steps that only use the first coordinate
            s.StartIteration();
            s.StartStream(0);
            (void)s.Get1D();
            s.Accept();
        }
        s.StartIteration();
        s.StartStream(0);
        (void)s.Get1D();
        float v = s.Get1D();  // second coordinate: first use
        distToEdge += std::min(v, 1 - v);
    }
    distToEdge /= n;
    MESSAGE("mean distance to 0/1 of first-used coordinates: " << distToEdge << " (uniform: 0.25)");
    CHECK(distToEdge == doctest::Approx(0.25).epsilon(0.08));
}

TEST_CASE("integrators agree per channel on chromatic subsurface scattering") {
    auto scene = Load(kWax, "wax");
    const int depth = 96;
    RGB pt = Mean(Render(*scene, IntegratorType::Path, 4096, 0, depth));
    RGB bdpt = Mean(Render(*scene, IntegratorType::BDPT, 1024, 0, depth));
    // Many chains: on a tiny image a few chains stuck on bright subsurface paths would otherwise
    // dominate (MLT variance, not bias: seed-to-seed spread drops from ~17% to ~3%).
    RGB mmlt = Mean(Render(*scene, IntegratorType::MMLT, 0, 8192, depth, 1, 4096, 200000));
    RGB pss = Mean(Render(*scene, IntegratorType::PSSMLT, 0, 8192, depth, 1, 4096, 200000));
    auto show = [](const RGB &c) { return std::to_string(c.r) + " " + std::to_string(c.g) + " " + std::to_string(c.b); };
    MESSAGE("path " << show(pt) << " | bdpt " << show(bdpt) << " | mmlt " << show(mmlt) << " | pssmlt " << show(pss));
    // Colour (channel ratios) is compared separately from overall brightness, since MLT's
    // normalisation noise scales all channels alike. With per-event spectral MIS and the MLT
    // sampler bug, the red/green ratio was off by ~30%; with bounded (path-level) spectral MIS
    // the sampler bug alone shifts it by only ~3%, which the unit test above catches instead.
    auto rg = [](const RGB &c) { return c.r / c.g; };
    auto bg = [](const RGB &c) { return c.b / c.g; };
    for (const RGB *c : {&bdpt, &mmlt, &pss}) {
        CHECK(rg(*c) == doctest::Approx(rg(pt)).epsilon(0.10));
        CHECK(bg(*c) == doctest::Approx(bg(pt)).epsilon(0.10));
    }
    CHECK(bdpt.g == doctest::Approx(pt.g).epsilon(0.05));
    CHECK(mmlt.g == doctest::Approx(pt.g).epsilon(0.12));
    CHECK(pss.g == doctest::Approx(pt.g).epsilon(0.12));
}

TEST_CASE("path tracer: spectral MIS keeps chromatic subsurface noise low") {
    // Per-pixel spread across independent renders. With per-event spectral MIS the weights of
    // long random walks grew exponentially (variance ~24x higher on this sphere).
    auto scene = Load(kWax, "wax_var");
    const int runs = 4;
    std::vector<Image> imgs;
    for (int i = 0; i < runs; ++i) imgs.push_back(Render(*scene, IntegratorType::Path, 256, 0, 0, 100 + i));
    double var = 0, mean = 0;
    const int w = imgs[0].Width(), h = imgs[0].Height();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double m = 0, m2 = 0;
            for (const Image &im : imgs) {
                double v = im.Get(x, y).g;
                m += v;
                m2 += v * v;
            }
            m /= runs;
            var += (m2 / runs - m * m) * runs / (runs - 1);
            mean += m;
        }
    var /= double(w) * h;
    mean /= double(w) * h;
    double relStd = std::sqrt(var) / mean;
    MESSAGE("per-pixel relative std at 256 spp: " << relStd);
    // Measured ~0.7 with path-level spectral MIS; per-event MIS had ~24x the variance (~3.5).
    CHECK(relStd < 1.5);
}

TEST_CASE("firefly clamp limits per-sample luminance and is off by default") {
    FilmSettings fs;
    fs.width = fs.height = 1;
    fs.filter.type = FilterType::Box;
    fs.filter.radius = 0.5f;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.3f);
    SampledSpectrum huge(1000.f);
    Film raw(fs);
    raw.AddSample(Vec2f(0.5f, 0.5f), huge, lambda);
    fs.clampLuminance = 2.f;
    Film clamped(fs);
    clamped.AddSample(Vec2f(0.5f, 0.5f), huge, lambda);
    XYZ a = raw.PixelXYZ(0, 0), b = clamped.PixelXYZ(0, 0);
    MESSAGE("raw Y " << a.y << ", clamped Y " << b.y);
    CHECK(a.y > 100.f);
    CHECK(b.y == doctest::Approx(2.f).epsilon(1e-4));
    CHECK(b.x / b.y == doctest::Approx(a.x / a.y).epsilon(1e-4));  // colour is kept
}

TEST_CASE("denoiser guides: albedo and normal follow specular chains through glass") {
    // A smooth glass sphere in front of a red wall; the centre pixel looks through the glass.
    auto scene = Load(R"({
  "materials": { "glass": { "type": "dielectric", "ior": 1.5 },
                 "red": { "type": "diffuse", "reflectance": [0.8, 0.05, 0.05] } },
  "geometry": { "ball": { "type": "sphere", "radius": 0.6 }, "wall": { "type": "rect", "size": [20, 20] } },
  "objects": [ { "geometry": "ball", "material": "glass" },
               { "geometry": "wall", "material": "red", "transform": { "translate": [0, 0, -2], "rotate": [90, 1, 0, 0] } } ],
  "lights": { "sky": { "type": "environment", "color": [1, 1, 1] } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 0, 4], "look_at": [0, 0, 0], "fov_y": 8 } },
  "render": { "camera": "cam", "film": { "width": 8, "height": 8, "filter": "box" } }
})", "guides");
    Image plain = RenderAOV(*scene, AOVType::Albedo, 64, 1, false);
    Image through = RenderAOV(*scene, AOVType::Albedo, 64, 1, true);
    RGB p = plain.Get(4, 4), t = through.Get(4, 4);
    MESSAGE("first-hit albedo " << p.r << " " << p.g << " " << p.b << "; through glass " << t.r << " " << t.g << " " << t.b);
    CHECK(std::abs(p.r - p.g) < 0.2f);  // glass: roughly neutral
    CHECK(t.r > 3 * t.g);                // the red wall seen through it
    for (float v : through.Data()) CHECK((v >= 0.f && v <= 1.f));
    Image n = RenderAOV(*scene, AOVType::Normal, 16, 1, true);
    RGB c = n.Get(4, 4);
    CHECK(c.b > 0.9f);  // the wall's normal faces the camera (+z)
}

TEST_CASE("denoising reduces error (OIDN)") {
    std::string status;
    if (!DenoiserAvailable(&status)) {
        MESSAGE("skipped: " << status);
        return;
    }
    auto scene = Load(R"({
  "materials": { "white": { "type": "diffuse", "reflectance": [0.7, 0.7, 0.7] },
                 "red": { "type": "diffuse", "reflectance": [0.7, 0.1, 0.1] } },
  "geometry": { "floor": { "type": "rect", "size": [4, 4] }, "ball": { "type": "sphere", "radius": 0.5 } },
  "objects": [ { "geometry": "floor", "material": "white" },
               { "geometry": "ball", "material": "red", "transform": { "translate": [0, 0.5, 0] } } ],
  "lights": { "key": { "type": "rect", "position": [0, 3, 1], "look_at": [0, 0, 0], "size": [0.5, 0.5], "luminance": 30 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 1.2, 3.5], "look_at": [0, 0.4, 0], "fov_y": 45 } },
  "render": { "camera": "cam", "film": { "width": 64, "height": 48, "filter": "box" } }
})", "denoise");
    Image ref = Render(*scene, IntegratorType::Path, 2048, 0, 0, 3);
    Image noisy = Render(*scene, IntegratorType::Path, 8, 0, 0, 4);
    Image albedo = RenderAOV(*scene, AOVType::Albedo, 16, 1, true);
    Image normal = RenderAOV(*scene, AOVType::Normal, 16, 1, true);
    Image den = noisy;
    std::string err;
    REQUIRE_MESSAGE(Denoise(den, &albedo, &normal, DenoiseOptions{}, &err), err);
    auto relMse = [&](const Image &a) {
        double e = 0;
        for (size_t i = 0; i < a.Data().size(); ++i) {
            double r = ref.Data()[i], d = a.Data()[i] - r;
            e += d * d / (r * r + 1e-2);
        }
        return e / double(a.Data().size());
    };
    double eNoisy = relMse(noisy), eDen = relMse(den);
    MESSAGE("relMSE noisy " << eNoisy << ", denoised " << eDen);
    CHECK(eDen < 0.5 * eNoisy);
    RGB mr = Mean(ref), md = Mean(den);
    CHECK(md.g == doctest::Approx(mr.g).epsilon(0.05));  // denoising must not change brightness much
}

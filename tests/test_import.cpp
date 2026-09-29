#include "doctest/doctest.h"

#include "core/film.h"
#include "core/image.h"
#include "core/rng.h"
#include "integrators/integrator.h"
#include "scene/importers.h"
#include "scene/scene_loader.h"

#include <cstring>
#include <filesystem>
#include <fstream>

using namespace pr;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path TestDir() {
    fs::path d = fs::temp_directory_path() / "prender_tests";
    fs::create_directories(d);
    return d;
}

std::string Base64(const std::vector<uint8_t> &data) {
    static const char *tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
    }
    if (i < data.size()) {
        uint32_t v = data[i] << 16;
        if (i + 1 < data.size()) v |= data[i + 1] << 8;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < data.size() ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// Renders a scene file and returns (left-half mean Y, right-half mean Y).
std::pair<double, double> RenderHalves(const fs::path &file, IntegratorType type, int spp) {
    std::string err;
    auto scene = LoadScene(file.string(), LoadOptions(), &err);
    REQUIRE_MESSAGE(scene, err);
    IntegratorSettings is;
    is.type = type;
    is.spp = spp;
    auto integ = CreateIntegrator(is, 11);
    Film film(scene->settings.film);
    RenderControl rc;
    rc.Start();
    double scale;
    REQUIRE(integ->Render(*scene, film, rc, &scale, &err));
    Image xyz = film.ResolveXYZ(scale);
    double l = 0, r = 0;
    for (int y = 0; y < xyz.Height(); ++y)
        for (int x = 0; x < xyz.Width(); ++x) (x < xyz.Width() / 2 ? l : r) += xyz.Get(x, y).g;
    double n = double(xyz.Width()) * xyz.Height() / 2;
    return {l / n, r / n};
}

fs::path WriteTestGLTF() {
    // Three quads: floor (y = 0, up), lamp (y = 2, down) and a vertical "prism" panel.
    std::vector<float> pos = {-2, 0, -2, 2, 0, -2, 2, 0, 2, -2, 0, 2,                  // floor
                              -0.5f, 2, -0.5f, 0.5f, 2, -0.5f, 0.5f, 2, 0.5f, -0.5f, 2, 0.5f,  // lamp
                              -0.5f, 0.2f, 0, 0.5f, 0.2f, 0, 0.5f, 1.2f, 0, -0.5f, 1.2f, 0};  // panel
    std::vector<uint16_t> idx = {0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3, 0, 1, 2, 0, 2, 3};
    std::vector<uint8_t> buf(pos.size() * 4 + idx.size() * 2);
    std::memcpy(buf.data(), pos.data(), pos.size() * 4);
    std::memcpy(buf.data() + pos.size() * 4, idx.data(), idx.size() * 2);
    json g = json::parse(R"({
      "asset": { "version": "2.0" },
      "extensionsUsed": ["KHR_materials_emissive_strength", "KHR_materials_transmission", "KHR_materials_ior",
                         "KHR_materials_dispersion", "KHR_materials_volume", "KHR_lights_punctual"],
      "materials": [
        { "name": "floor", "pbrMetallicRoughness": { "baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 0.6 } },
        { "name": "lamp", "emissiveFactor": [1, 0.9, 0.8], "extensions": { "KHR_materials_emissive_strength": { "emissiveStrength": 20 } } },
        { "name": "prism", "pbrMetallicRoughness": { "metallicFactor": 0, "roughnessFactor": 0.0 },
          "extensions": { "KHR_materials_transmission": { "transmissionFactor": 1 }, "KHR_materials_ior": { "ior": 1.7 },
                          "KHR_materials_dispersion": { "dispersion": 0.5 }, "KHR_materials_volume": { "thicknessFactor": 0.1 } } }
      ],
      "meshes": [
        { "primitives": [ { "attributes": { "POSITION": 0 }, "indices": 3, "material": 0 } ] },
        { "primitives": [ { "attributes": { "POSITION": 1 }, "indices": 4, "material": 1 } ] },
        { "primitives": [ { "attributes": { "POSITION": 2 }, "indices": 5, "material": 2 } ] }
      ],
      "cameras": [ { "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1 } } ],
      "extensions": { "KHR_lights_punctual": { "lights": [ { "type": "point", "color": [1, 1, 1], "intensity": 3 } ] } },
      "nodes": [ { "mesh": 0 }, { "mesh": 1 }, { "mesh": 2 }, { "camera": 0, "translation": [0, 1.2, 5] },
                 { "translation": [1, 1, 1], "extensions": { "KHR_lights_punctual": { "light": 0 } } } ],
      "scenes": [ { "nodes": [0, 1, 2, 3, 4] } ],
      "scene": 0
    })");
    g["buffers"] = json::array({{{"byteLength", buf.size()}, {"uri", "data:application/octet-stream;base64," + Base64(buf)}}});
    json views = json::array(), acc = json::array();
    for (int q = 0; q < 3; ++q) {
        views.push_back({{"buffer", 0}, {"byteOffset", q * 48}, {"byteLength", 48}});
        float mn[3] = {1e9f, 1e9f, 1e9f}, mx[3] = {-1e9f, -1e9f, -1e9f};
        for (int v = 0; v < 4; ++v)
            for (int c = 0; c < 3; ++c) {
                mn[c] = std::min(mn[c], pos[q * 12 + v * 3 + c]);
                mx[c] = std::max(mx[c], pos[q * 12 + v * 3 + c]);
            }
        acc.push_back({{"bufferView", q}, {"componentType", 5126}, {"count", 4}, {"type", "VEC3"},
                       {"min", {mn[0], mn[1], mn[2]}}, {"max", {mx[0], mx[1], mx[2]}}});
    }
    for (int q = 0; q < 3; ++q) {
        views.push_back({{"buffer", 0}, {"byteOffset", pos.size() * 4 + q * 12}, {"byteLength", 12}});
        acc.push_back({{"bufferView", 3 + q}, {"componentType", 5123}, {"count", 6}, {"type", "SCALAR"}});
    }
    g["bufferViews"] = views;
    g["accessors"] = acc;
    fs::path file = TestDir() / "test_scene.gltf";
    std::ofstream(file) << g.dump(1);
    return file;
}

} // namespace

TEST_CASE("glTF import: materials, extensions, camera and lights") {
    fs::path file = WriteTestGLTF();
    std::string err;
    json doc = ImportGLTF(file.string(), ImportOptions(), &err);
    REQUIRE_MESSAGE(!doc.is_null(), err);
    CHECK(doc["objects"].size() == 3);
    CHECK(doc["cameras"].size() == 1);
    CHECK(doc["lights"].size() == 1);
    bool foundDispersive = false, foundEmission = false;
    for (auto &[id, m] : doc["materials"].items()) {
        if (m.value("type", "") == "dielectric" && m["ior"].is_object() && m["ior"].contains("cauchy")) foundDispersive = true;
        if (m.contains("emission")) foundEmission = true;
    }
    CHECK(foundDispersive);
    CHECK(foundEmission);
    auto [l, r] = RenderHalves(file, IntegratorType::Path, 32);
    CHECK(l + r > 0);
}

TEST_CASE("pbrt import matches the equivalent native scene (camera handedness, lights, materials)") {
    fs::path pbrt = TestDir() / "test_scene.pbrt";
    std::ofstream(pbrt) << R"(# test scene
Scale -1 1 1
LookAt 0 1.5 5  0 0.8 0  0 1 0
Camera "perspective" "float fov" [ 40 ]
Film "rgb" "integer xresolution" [ 32 ] "integer yresolution" [ 24 ] "string filename" "out.exr"
Sampler "zsobol" "integer pixelsamples" 16
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "blackbody L" [ 5000 ] "float scale" [ 8 ]
  Shape "trianglemesh" "point3 P" [ -0.5 2 -0.5  0.5 2 -0.5  0.5 2 0.5  -0.5 2 0.5 ] "integer indices" [ 0 1 2 0 2 3 ]
AttributeEnd
Material "diffuse" "rgb reflectance" [ 0.7 0.7 0.7 ]
Shape "trianglemesh" "point3 P" [ -3 0 -3  3 0 -3  3 0 3  -3 0 3 ] "integer indices" [ 0 2 1 0 3 2 ]
AttributeBegin
  Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.05
  Translate 1 0.5 0
  Shape "sphere" "float radius" 0.5
AttributeEnd
)";
    fs::path native = TestDir() / "test_native.prscene.json";
    std::ofstream(native) << R"({
  "materials": { "floor": { "type": "diffuse", "reflectance": [0.7, 0.7, 0.7] },
                 "gold": { "type": "conductor", "metal": "Au", "roughness": 0.47287 } },
  "geometry": { "floor": { "type": "rect", "size": [6, 6] }, "ball": { "type": "sphere", "radius": 0.5 } },
  "objects": [ { "geometry": "floor", "material": "floor" },
               { "geometry": "ball", "material": "gold", "transform": { "translate": [1, 0.5, 0] } } ],
  "lights": { "key": { "type": "rect", "position": [0, 2, 0], "direction": [0, -1, 0], "size": [1, 1],
                       "temperature": 5000, "luminance": 8 } },
  "cameras": { "cam": { "type": "pinhole", "position": [0, 1.5, 5], "look_at": [0, 0.8, 0], "fov_y": 40 } },
  "render": { "camera": "cam", "film": { "width": 32, "height": 24 } }
})";
    auto [pl, pr_] = RenderHalves(pbrt, IntegratorType::Path, 256);
    auto [nl, nr] = RenderHalves(native, IntegratorType::Path, 256);
    MESSAGE("pbrt L/R = " << pl << " / " << pr_ << "   native L/R = " << nl << " / " << nr);
    CHECK(pl == doctest::Approx(nl).epsilon(0.04));
    CHECK(pr_ == doctest::Approx(nr).epsilon(0.04));
    // The gold sphere sits at +x: brighter right half in both.
    CHECK(pr_ > pl);

    std::string err;
    fs::path out = TestDir() / "converted.prscene.json";
    REQUIRE_MESSAGE(ConvertSceneToJson(pbrt.string(), out.string(), &err), err);
    CHECK(fs::file_size(out) > 100);
}

TEST_CASE("EXR round trip") {
    Image img(3, 2);
    img.Set(0, 0, RGB(1, 2, 3));
    img.Set(2, 1, RGB(0.25f, 1e3f, 0));
    fs::path p = TestDir() / "rt.exr";
    std::string err;
    REQUIRE(WriteEXR(p.string(), img, false, &err));
    Image back;
    REQUIRE_MESSAGE(ReadImage(p.string(), TextureEncoding::Linear, &back, &err), err);
    REQUIRE(back.Width() == 3);
    CHECK(back.Get(0, 0).b == doctest::Approx(3));
    CHECK(back.Get(2, 1).g == doctest::Approx(1e3));
}

TEST_CASE("EXR ZIP writer: multi-block, half and float, partial last block") {
    const int w = 53, h = 37;  // 37 rows = 2 full 16-line blocks + a partial one
    Image img(w, h);
    RNG rng(3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img.Set(x, y, RGB(rng.UniformFloat() * 4, float(x) / w, float(y) / h + (x == y ? 100.f : 0.f)));
    std::string err;
    for (bool half : {false, true}) {
        fs::path p = TestDir() / (half ? "zip_half.exr" : "zip_float.exr");
        REQUIRE(WriteEXRRows(p.string(), w, h, [&](int y, float *rgb) {
            for (int x = 0; x < w; ++x) {
                RGB c = img.Get(x, y);
                rgb[3 * x] = c.r; rgb[3 * x + 1] = c.g; rgb[3 * x + 2] = c.b;
            }
        }, half, true, &err));
        Image back;
        REQUIRE_MESSAGE(ReadImage(p.string(), TextureEncoding::Linear, &back, &err), err);
        REQUIRE(back.Width() == w);
        REQUIRE(back.Height() == h);
        double maxErr = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    maxErr = std::max(maxErr, double(std::abs(back.Get(x, y)[c] - img.Get(x, y)[c]) / std::max(1.f, img.Get(x, y)[c])));
        CHECK(maxErr < (half ? 1e-3 : 1e-7));
    }
}

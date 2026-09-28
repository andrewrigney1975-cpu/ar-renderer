#include "scene/scene_loader.h"

#include "core/fsutil.h"
#include "core/log.h"
#include "core/rgb2spec.h"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace pr {

bool ParseDuration(const std::string &s, double *seconds) {
    if (s.empty()) return false;
    char *end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return false;
    std::string unit(end);
    if (unit.empty() || unit == "s") *seconds = v;
    else if (unit == "m" || unit == "min") *seconds = v * 60;
    else if (unit == "h") *seconds = v * 3600;
    else if (unit == "ms") *seconds = v / 1000;
    else return false;
    return true;
}

namespace {

struct LoadError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void Fail(const std::string &msg) { throw LoadError(msg); }

// ---------------------------------------------------------------------------------------------
// JSON file loading, path resolution and composition

void ResolvePaths(json &j, const fs::path &dir) {
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            if ((it.key() == "file") && it.value().is_string()) {
                fs::path p = Utf8Path(it.value().get<std::string>());
                if (p.is_relative()) it.value() = PathUtf8((dir / p).lexically_normal());
            } else if (it.key() == "rigs" && it.value().is_object()) {
                // Rig entries may be bare file paths.
                for (auto r = it.value().begin(); r != it.value().end(); ++r) {
                    if (r.value().is_string()) {
                        fs::path p = Utf8Path(r.value().get<std::string>());
                        if (p.is_relative()) r.value() = PathUtf8((dir / p).lexically_normal());
                    } else {
                        ResolvePaths(r.value(), dir);
                    }
                }
            } else {
                ResolvePaths(it.value(), dir);
            }
        }
    } else if (j.is_array()) {
        for (auto &e : j) ResolvePaths(e, dir);
    }
}

json ReadJsonFile(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) Fail("cannot open " + PathUtf8(path));
    json j;
    try {
        j = json::parse(in, nullptr, true, /*ignore_comments=*/true);
    } catch (const json::parse_error &e) {
        Fail(PathUtf8(path) + ": " + e.what());
    }
    ResolvePaths(j, path.parent_path());
    return j;
}

const std::set<std::string> kDictSections = {"textures", "materials", "media", "geometry", "lights", "cameras", "rigs"};

void DeepMerge(json &dst, const json &src) {
    for (auto it = src.begin(); it != src.end(); ++it) {
        if (dst.contains(it.key()) && dst[it.key()].is_object() && it.value().is_object())
            DeepMerge(dst[it.key()], it.value());
        else dst[it.key()] = it.value();
    }
}

void MergeDocument(json &dst, const json &src) {
    for (auto it = src.begin(); it != src.end(); ++it) {
        const std::string &k = it.key();
        if (k == "include") continue;
        if (kDictSections.count(k)) {
            if (!it.value().is_object()) Fail("section '" + k + "' must be an object");
            for (auto e = it.value().begin(); e != it.value().end(); ++e) dst[k][e.key()] = e.value();
        } else if (k == "objects") {
            if (!it.value().is_array()) Fail("'objects' must be an array");
            for (const auto &o : it.value()) dst["objects"].push_back(o);
        } else if (k == "render" && dst.contains("render") && it.value().is_object()) {
            DeepMerge(dst["render"], it.value());
        } else {
            dst[k] = it.value();
        }
    }
}

json LoadComposed(const fs::path &path, int depth = 0) {
    if (depth > 16) Fail("include depth exceeded at " + PathUtf8(path));
    json doc = ReadJsonFile(path);
    json result = json::object();
    if (doc.contains("include")) {
        for (const auto &inc : doc["include"]) {
            fs::path p = Utf8Path(inc.get<std::string>());
            if (p.is_relative()) p = path.parent_path() / p;
            MergeDocument(result, LoadComposed(p, depth + 1));
        }
    }
    MergeDocument(result, doc);
    return result;
}

// ---------------------------------------------------------------------------------------------
// Value parsing

Vec3f ParseVec3(const json &j, const char *what) {
    if (j.is_number()) {
        float v = j.get<float>();
        return {v, v, v};
    }
    if (!j.is_array() || j.size() != 3) Fail(std::string(what) + ": expected [x, y, z]");
    return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
}

RGB ParseRGB(const json &j, const char *what) {
    Vec3f v = ParseVec3(j, what);
    return {v.x, v.y, v.z};
}

float GetFloat(const json &j, const char *key, float def) {
    if (!j.contains(key)) return def;
    if (!j[key].is_number()) Fail(std::string("'") + key + "' must be a number");
    return j[key].get<float>();
}

std::string GetString(const json &j, const char *key, const std::string &def) {
    if (!j.contains(key)) return def;
    if (!j[key].is_string()) Fail(std::string("'") + key + "' must be a string");
    return j[key].get<std::string>();
}

bool GetBool(const json &j, const char *key, bool def) {
    if (!j.contains(key)) return def;
    if (!j[key].is_boolean()) Fail(std::string("'") + key + "' must be true/false");
    return j[key].get<bool>();
}

SpectrumPtr NormalizedToY(SpectrumPtr s) {
    float y = SpectrumToY(*s);
    if (y <= 0) return s;
    return std::make_shared<ScaledSpectrum>(1 / y, s);
}

SpectrumPtr LoadSpectrumCSV(const std::string &file) {
    std::ifstream in(Utf8Path(file));
    if (!in) Fail("cannot open spectrum file " + file);
    std::vector<float> l, v;
    std::string line;
    while (std::getline(in, line)) {
        for (char &c : line)
            if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line);
        float a, b;
        if (ss >> a >> b) {
            l.push_back(a);
            v.push_back(b);
        }
    }
    if (l.size() < 2) Fail("spectrum file " + file + " has fewer than 2 samples");
    return std::make_shared<PiecewiseLinearSpectrum>(l, v);
}

// Parses a spectrum value. 'type' decides how RGB triples are uplifted.
SpectrumPtr ParseSpectrum(const json &j, SpectrumType type, const char *what) {
    if (j.is_number()) {
        float v = j.get<float>();
        if (type == SpectrumType::Illuminant) return std::make_shared<ScaledSpectrum>(v, NormalizedToY(GetNamedSpectrum("D65")));
        return std::make_shared<ConstantSpectrum>(v);
    }
    if (j.is_array()) {
        RGB c = ParseRGB(j, what);
        switch (type) {
        case SpectrumType::Albedo:
            return std::make_shared<RGBAlbedoSpectrum>(RGB(Clamp(c.r, 0.f, 1.f), Clamp(c.g, 0.f, 1.f), Clamp(c.b, 0.f, 1.f)));
        case SpectrumType::Unbounded: return std::make_shared<RGBUnboundedSpectrum>(c);
        case SpectrumType::Illuminant: return std::make_shared<RGBIlluminantSpectrum>(c);
        }
    }
    if (j.is_string()) {
        std::string name = j.get<std::string>();
        if (auto s = GetNamedSpectrum(name)) return type == SpectrumType::Illuminant ? NormalizedToY(s) : s;
        Fail(std::string(what) + ": unknown named spectrum '" + name + "'");
    }
    if (j.is_object()) {
        if (j.contains("blackbody") || j.contains("temperature")) {
            float T = j.contains("blackbody") ? j["blackbody"].get<float>() : j["temperature"].get<float>();
            SpectrumPtr bb = std::make_shared<BlackbodySpectrum>(T);
            return NormalizedToY(bb);
        }
        if (j.contains("sellmeier")) {
            const json &s = j["sellmeier"];
            std::vector<float> B, C;
            if (s.is_array() && s.size() % 2 == 0) {
                size_t n = s.size() / 2;
                for (size_t i = 0; i < n; ++i) B.push_back(s[i].get<float>());
                for (size_t i = 0; i < n; ++i) C.push_back(s[n + i].get<float>());
            } else if (s.is_object()) {
                for (auto &b : s["B"]) B.push_back(b.get<float>());
                for (auto &c : s["C"]) C.push_back(c.get<float>());
            }
            if (B.empty() || B.size() != C.size()) Fail(std::string(what) + ": bad sellmeier coefficients");
            return std::make_shared<SellmeierIOR>(B, C);
        }
        if (j.contains("cauchy")) {
            const json &c = j["cauchy"];
            if (!c.is_array() || c.size() != 2) Fail(std::string(what) + ": cauchy expects [A, B_nm2]");
            return std::make_shared<CauchyIOR>(c[0].get<float>(), c[1].get<float>());
        }
        if (j.contains("samples")) {
            std::vector<float> l, v;
            for (const auto &p : j["samples"]) {
                l.push_back(p[0].get<float>());
                v.push_back(p[1].get<float>());
            }
            SpectrumPtr s = std::make_shared<PiecewiseLinearSpectrum>(l, v);
            return type == SpectrumType::Illuminant ? NormalizedToY(s) : s;
        }
        if (j.contains("file")) {
            SpectrumPtr s = LoadSpectrumCSV(j["file"].get<std::string>());
            return type == SpectrumType::Illuminant ? NormalizedToY(s) : s;
        }
    }
    Fail(std::string(what) + ": cannot parse spectrum");
}

RGB ApproxRGB(const Spectrum &s, SpectrumType type) {
    // Rough preview colour for power estimates.
    if (type == SpectrumType::Illuminant) {
        XYZ xyz = SpectrumToXYZ(s);
        return XYZToRGBMatrix(ColorSpaceId::sRGB).Apply(xyz.x, xyz.y, xyz.z);
    }
    XYZ xyz = {0, 0, 0};
    for (int l = 400; l <= 700; l += 5) {
        float v = s(float(l));
        xyz.x += CIE_X(float(l)) * v;
        xyz.y += CIE_Y(float(l)) * v;
        xyz.z += CIE_Z(float(l)) * v;
    }
    float n = 0;
    for (int l = 400; l <= 700; l += 5) n += CIE_Y(float(l));
    return XYZToRGBMatrix(ColorSpaceId::sRGB).Apply(xyz.x / n, xyz.y / n, xyz.z / n);
}

Transform ParseTransform(const json &j) {
    if (j.is_null()) return Transform();
    if (j.contains("matrix")) {
        const json &m = j["matrix"];
        if (!m.is_array() || m.size() != 16) Fail("transform.matrix must have 16 numbers (row-major)");
        Mat4 mat;
        for (int i = 0; i < 16; ++i) mat.m[i / 4][i % 4] = m[i].get<float>();
        return Transform(mat);
    }
    Transform T, R, S;
    if (j.contains("translate")) T = Transform::Translate(ParseVec3(j["translate"], "translate"));
    if (j.contains("rotate")) {
        const json &r = j["rotate"];
        // [angle, ax, ay, az] or list of those, or {"axis":[...], "angle":deg}
        auto one = [](const json &e) -> Transform {
            if (e.is_array() && e.size() == 4)
                return Transform::Rotate(e[0].get<float>(), Vec3f(e[1].get<float>(), e[2].get<float>(), e[3].get<float>()));
            if (e.is_object()) return Transform::Rotate(e["angle"].get<float>(), ParseVec3(e["axis"], "rotate.axis"));
            Fail("rotate: expected [angle, ax, ay, az]");
        };
        if (r.is_array() && !r.empty() && r[0].is_array()) {
            for (const auto &e : r) R = R * one(e);
        } else {
            R = one(r);
        }
    }
    if (j.contains("euler")) {
        Vec3f e = ParseVec3(j["euler"], "euler");
        R = R * Transform::Rotate(e.z, {0, 0, 1}) * Transform::Rotate(e.y, {0, 1, 0}) * Transform::Rotate(e.x, {1, 0, 0});
    }
    if (j.contains("scale")) S = Transform::Scale(ParseVec3(j["scale"], "scale"));
    return T * R * S;
}

// ---------------------------------------------------------------------------------------------

struct ObjectRecord {
    std::string name;
    std::vector<int> prims;
    Bounds3f bounds;
    Vec3f probe;                      // a point inside the object (for containment tests)
    const Medium *interior = nullptr;
    bool autoExterior = true;
    const Medium *exterior = nullptr;
    bool closed = false;
};

struct EmissionSpec {
    SpectrumPtr spectrum;
    float scale = 1;
    bool powerMode = false;
    float power = 0;
    bool twoSided = false;
    float cosPower = 0;
};

class Loader {
  public:
    Loader(json doc, const LoadOptions &opts, fs::path sceneDir)
        : doc_(std::move(doc)), opts_(opts), dir_(std::move(sceneDir)) {}

    std::unique_ptr<Scene> Load();

  private:
    SpectrumTexturePtr GetSpectrumTexture(const json &v, SpectrumType type, const char *what);
    FloatTexturePtr GetFloatTexture(const json &v, const char *what);
    std::shared_ptr<const ImageData> GetImage(const json &tex, const std::string &id);
    const Medium *GetMedium(const std::string &id);
    std::shared_ptr<Medium> ParseMedium(const json &j, const std::string &name);
    Material *GetMaterial(const std::string &id);
    std::shared_ptr<Material> ParseMaterial(const json &j, const std::string &id);
    std::optional<EmissionSpec> ParseEmission(const json &j);
    SpectrumPtr ParseIOR(const json &j, float def);
    void ParseConductor(const json &j, ConductorParams *p);
    void AddObject(const json &o, int index);
    void AddLight(const std::string &id, const json &l);
    int AddPrimitive(std::unique_ptr<Shape> shape, const Material *mat, ObjectRecord &rec);
    void AttachEmission(ObjectRecord &rec, const EmissionSpec &e);
    void ResolveMedia();
    void ParseRender();
    void ParseCamera();
    const Medium *EnclosingMedium(const Vec3f &p, int excludeObject) const;

    json doc_;
    LoadOptions opts_;
    fs::path dir_;
    std::unique_ptr<Scene> scene_;
    std::map<std::string, SpectrumTexturePtr> specTexCache_;
    std::map<std::string, FloatTexturePtr> floatTexCache_;
    std::map<std::string, std::shared_ptr<const ImageData>> imageCache_;
    std::map<std::string, const Medium *> mediumCache_;
    std::map<std::string, Material *> materialCache_;
    std::map<std::string, const Medium *> materialInterior_;
    std::map<std::string, EmissionSpec> materialEmission_;
    std::shared_ptr<Material> blackMaterial_;
    std::vector<ObjectRecord> objects_;
    std::vector<std::pair<int, EmissionSpec>> pendingEmission_;
};

std::shared_ptr<const ImageData> Loader::GetImage(const json &tex, const std::string &id) {
    std::string file = GetString(tex, "file", "");
    if (file.empty()) Fail("texture '" + id + "': missing 'file'");
    std::string cs = GetString(tex, "colorspace", "srgb");
    TextureEncoding enc = (cs == "srgb" || cs == "sRGB") ? TextureEncoding::sRGB : TextureEncoding::Linear;
    std::string wrapS = GetString(tex, "wrap", "repeat");
    WrapMode wrap = wrapS == "clamp" ? WrapMode::Clamp : (wrapS == "mirror" ? WrapMode::Mirror : WrapMode::Repeat);
    bool flip = GetBool(tex, "flip_v", true);
    std::string key = file + "|" + cs + "|" + wrapS + (flip ? "|f" : "|n");
    auto it = imageCache_.find(key);
    if (it != imageCache_.end()) return it->second;
    Image img;
    std::string err;
    if (!ReadImage(file, enc, &img, &err)) Fail("texture '" + id + "': " + err);
    auto data = std::make_shared<const ImageData>(std::move(img), wrap, flip);
    imageCache_[key] = data;
    return data;
}

static UVMapping ParseUVMapping(const json &t) {
    UVMapping m;
    if (t.contains("uv_scale")) {
        const json &s = t["uv_scale"];
        if (s.is_number()) m.scale = {s.get<float>(), s.get<float>()};
        else m.scale = {s[0].get<float>(), s[1].get<float>()};
    }
    if (t.contains("uv_offset")) m.offset = {t["uv_offset"][0].get<float>(), t["uv_offset"][1].get<float>()};
    return m;
}

SpectrumTexturePtr Loader::GetSpectrumTexture(const json &v, SpectrumType type, const char *what) {
    if (v.is_object() && v.contains("texture")) {
        std::string id = v["texture"].get<std::string>();
        std::string key = id + "#" + std::to_string(int(type));
        auto it = specTexCache_.find(key);
        if (it != specTexCache_.end()) return it->second;
        if (!doc_["textures"].contains(id)) Fail(std::string(what) + ": unknown texture '" + id + "'");
        const json &t = doc_["textures"][id];
        std::string ttype = GetString(t, "type", "image");
        SpectrumTexturePtr tex;
        if (ttype == "image") {
            tex = std::make_shared<ImageSpectrumTexture>(GetImage(t, id), ParseUVMapping(t), type, GetFloat(t, "multiplier", 1.f));
        } else if (ttype == "checker") {
            json a = t.contains("a") ? t["a"] : json(0.8), b = t.contains("b") ? t["b"] : json(0.2);
            tex = std::make_shared<CheckerSpectrumTexture>(GetSpectrumTexture(a, type, what), GetSpectrumTexture(b, type, what),
                                                           ParseUVMapping(t));
        } else if (ttype == "constant") {
            tex = GetSpectrumTexture(t["value"], type, what);
        } else {
            Fail("texture '" + id + "': unknown type '" + ttype + "'");
        }
        specTexCache_[key] = tex;
        return tex;
    }
    SpectrumPtr s = ParseSpectrum(v, type, what);
    RGB avg = v.is_array() ? ParseRGB(v, what) : ApproxRGB(*s, type);
    return std::make_shared<ConstantSpectrumTexture>(s, avg);
}

FloatTexturePtr Loader::GetFloatTexture(const json &v, const char *what) {
    if (v.is_number()) return std::make_shared<ConstantFloatTexture>(v.get<float>());
    if (v.is_object() && v.contains("texture")) {
        std::string id = v["texture"].get<std::string>();
        std::string ch = GetString(v, "channel", "avg");
        int channel = ch == "r" ? 0 : ch == "g" ? 1 : ch == "b" ? 2 : -1;
        std::string key = id + "#" + ch;
        auto it = floatTexCache_.find(key);
        if (it != floatTexCache_.end()) return it->second;
        if (!doc_["textures"].contains(id)) Fail(std::string(what) + ": unknown texture '" + id + "'");
        const json &t = doc_["textures"][id];
        std::string ttype = GetString(t, "type", "image");
        FloatTexturePtr tex;
        if (ttype == "image") {
            json tt = t;
            if (!tt.contains("colorspace")) tt["colorspace"] = "linear";
            tex = std::make_shared<ImageFloatTexture>(GetImage(tt, id), ParseUVMapping(t), channel, GetFloat(t, "multiplier", 1.f));
        } else if (ttype == "checker") {
            json a = t.contains("a") ? t["a"] : json(0.8), b = t.contains("b") ? t["b"] : json(0.2);
            auto fa = a.is_number() ? GetFloatTexture(a, what) : GetFloatTexture(json(ParseRGB(a, what).Max()), what);
            auto fb = b.is_number() ? GetFloatTexture(b, what) : GetFloatTexture(json(ParseRGB(b, what).Max()), what);
            tex = std::make_shared<CheckerFloatTexture>(fa, fb, ParseUVMapping(t));
        } else if (ttype == "constant") {
            tex = GetFloatTexture(t["value"], what);
        } else {
            Fail("texture '" + id + "': unknown type '" + ttype + "'");
        }
        floatTexCache_[key] = tex;
        return tex;
    }
    Fail(std::string(what) + ": expected a number or {\"texture\": id}");
}

std::shared_ptr<Medium> Loader::ParseMedium(const json &j, const std::string &name) {
    std::string type = GetString(j, "type", "homogeneous");
    if (type != "homogeneous") Fail("medium '" + name + "': only 'homogeneous' media are supported");
    float g = GetFloat(j, "g", 0.f);
    float scale = GetFloat(j, "scale", 1.f);
    SpectrumPtr sa, ss;
    if (j.contains("albedo") || j.contains("mfp")) {
        // Artist parameterization: multiple-scattering albedo and mean free path per channel,
        // inverted to single-scattering coefficients (van de Hulst / Christensen-Burley fit).
        RGB A = j.contains("albedo") ? ParseRGB(j["albedo"], "albedo") : RGB(0.8f, 0.8f, 0.8f);
        RGB mfp = j.contains("mfp") ? ParseRGB(j["mfp"], "mfp") : RGB(1, 1, 1);
        RGB sigS, sigA;
        for (int c = 0; c < 3; ++c) {
            float a = Clamp(A[c], 0.f, 0.999f);
            float alpha = 1 - Sqr(4.09712f + 4.20863f * a - std::sqrt(9.59217f + 41.6808f * a + 17.7126f * a * a));
            alpha = Clamp(alpha, 0.f, 0.9999f);
            float sigmaT = 1 / std::max(mfp[c], 1e-6f);
            sigS[c] = alpha * sigmaT;
            sigA[c] = sigmaT - sigS[c];
        }
        ss = std::make_shared<RGBUnboundedSpectrum>(sigS);
        sa = std::make_shared<RGBUnboundedSpectrum>(sigA);
    } else {
        sa = j.contains("sigma_a") ? ParseSpectrum(j["sigma_a"], SpectrumType::Unbounded, "sigma_a")
                                   : std::make_shared<ConstantSpectrum>(0.f);
        ss = j.contains("sigma_s") ? ParseSpectrum(j["sigma_s"], SpectrumType::Unbounded, "sigma_s")
                                   : std::make_shared<ConstantSpectrum>(0.f);
    }
    auto m = std::make_shared<Medium>(sa, ss, scale, g);
    m->name = name;
    return m;
}

const Medium *Loader::GetMedium(const std::string &id) {
    auto it = mediumCache_.find(id);
    if (it != mediumCache_.end()) return it->second;
    if (!doc_["media"].contains(id)) Fail("unknown medium '" + id + "'");
    auto m = ParseMedium(doc_["media"][id], id);
    scene_->media.push_back(m);
    mediumCache_[id] = m.get();
    return m.get();
}

SpectrumPtr Loader::ParseIOR(const json &j, float def) {
    if (j.is_null()) return std::make_shared<ConstantSpectrum>(def);
    if (j.is_string()) {
        auto s = GetNamedIOR(j.get<std::string>());
        if (!s) Fail("unknown IOR '" + j.get<std::string>() + "'");
        return s;
    }
    return ParseSpectrum(j, SpectrumType::Unbounded, "ior");
}

void Loader::ParseConductor(const json &j, ConductorParams *p) {
    if (j.contains("metal")) {
        SpectrumPtr eta, k;
        std::string m = j["metal"].get<std::string>();
        if (!GetNamedMetal(m, &eta, &k)) Fail("unknown metal '" + m + "' (known: Au, Ag, Cu, Al)");
        p->eta = std::make_shared<ConstantSpectrumTexture>(eta, RGB(1, 1, 1));
        p->k = std::make_shared<ConstantSpectrumTexture>(k, RGB(1, 1, 1));
    } else if (j.contains("eta") && j.contains("k")) {
        p->eta = GetSpectrumTexture(j["eta"], SpectrumType::Unbounded, "eta");
        p->k = GetSpectrumTexture(j["k"], SpectrumType::Unbounded, "k");
    } else {
        p->reflectance = GetSpectrumTexture(j.contains("reflectance") ? j["reflectance"] : json(json::array({0.9, 0.9, 0.9})),
                                            SpectrumType::Albedo, "reflectance");
    }
    json r = j.contains("roughness") ? j["roughness"] : json(0.0);
    p->uRoughness = GetFloatTexture(j.contains("roughness_u") ? j["roughness_u"] : r, "roughness_u");
    p->vRoughness = GetFloatTexture(j.contains("roughness_v") ? j["roughness_v"] : r, "roughness_v");
}

std::optional<EmissionSpec> Loader::ParseEmission(const json &j) {
    if (j.is_null()) return std::nullopt;
    EmissionSpec e;
    if (j.contains("temperature")) e.spectrum = NormalizedToY(std::make_shared<BlackbodySpectrum>(j["temperature"].get<float>()));
    else if (j.contains("spectrum")) e.spectrum = ParseSpectrum(j["spectrum"], SpectrumType::Illuminant, "emission.spectrum");
    else if (j.contains("color")) e.spectrum = ParseSpectrum(j["color"], SpectrumType::Illuminant, "emission.color");
    else e.spectrum = NormalizedToY(GetNamedSpectrum("D65"));
    e.scale = GetFloat(j, "luminance", GetFloat(j, "intensity", 1.f));
    if (j.contains("power")) {
        e.powerMode = true;
        e.power = GetFloat(j, "power", 1.f);
    }
    e.twoSided = GetBool(j, "two_sided", false);
    e.cosPower = GetFloat(j, "cos_power", 0.f);
    return e;
}

std::shared_ptr<Material> Loader::ParseMaterial(const json &j, const std::string &id) {
    std::string type = GetString(j, "type", "diffuse");
    std::shared_ptr<Material> m;
    auto floatOr = [&](const char *key, float def) { return GetFloatTexture(j.contains(key) ? j[key] : json(def), key); };
    if (type == "diffuse") {
        json r = j.contains("reflectance") ? j["reflectance"] : (j.contains("color") ? j["color"] : json(0.5));
        m = std::make_shared<DiffuseMaterial>(GetSpectrumTexture(r, SpectrumType::Albedo, "reflectance"));
    } else if (type == "conductor" || type == "metal") {
        ConductorParams p;
        ParseConductor(j, &p);
        m = std::make_shared<ConductorMaterial>(p);
    } else if (type == "dielectric" || type == "glass" || type == "subsurface") {
        json r = j.contains("roughness") ? j["roughness"] : json(0.0);
        auto ur = GetFloatTexture(j.contains("roughness_u") ? j["roughness_u"] : r, "roughness_u");
        auto vr = GetFloatTexture(j.contains("roughness_v") ? j["roughness_v"] : r, "roughness_v");
        m = std::make_shared<DielectricMaterial>(ParseIOR(j.contains("ior") ? j["ior"] : json(), 1.5f), ur, vr,
                                                 GetBool(j, "thin", false));
        if (type == "subsurface") {
            // The scattering interior is a medium owned by this material.
            json mj = j.contains("medium_params") ? j["medium_params"] : j;
            mj["type"] = "homogeneous";
            auto med = ParseMedium(mj, id + ".interior");
            scene_->media.push_back(med);
            materialInterior_[id] = med.get();
        }
    } else if (type == "coated_diffuse") {
        CoatParams c;
        c.eta = ParseIOR(j.contains("coat_ior") ? j["coat_ior"] : json(), 1.5f);
        c.roughness = floatOr("coat_roughness", 0.f);
        c.thickness = floatOr("thickness", 0.01f);
        if (j.contains("albedo")) c.albedo = GetSpectrumTexture(j["albedo"], SpectrumType::Albedo, "albedo");
        c.g = GetFloat(j, "g", 0.f);
        json r = j.contains("reflectance") ? j["reflectance"] : (j.contains("color") ? j["color"] : json(0.5));
        m = std::make_shared<CoatedDiffuseMaterial>(c, GetSpectrumTexture(r, SpectrumType::Albedo, "reflectance"));
    } else if (type == "coated_conductor") {
        CoatParams c;
        c.eta = ParseIOR(j.contains("coat_ior") ? j["coat_ior"] : json(), 1.5f);
        c.roughness = floatOr("coat_roughness", 0.f);
        c.thickness = floatOr("thickness", 0.01f);
        if (j.contains("albedo")) c.albedo = GetSpectrumTexture(j["albedo"], SpectrumType::Albedo, "albedo");
        c.g = GetFloat(j, "g", 0.f);
        ConductorParams p;
        ParseConductor(j, &p);
        m = std::make_shared<CoatedConductorMaterial>(c, p);
    } else if (type == "interface") {
        m = nullptr;
    } else {
        Fail("material '" + id + "': unknown type '" + type + "'");
    }
    if (m) {
        m->name = id;
        if (j.contains("sheen")) {
            const json &s = j["sheen"];
            SheenParams sp;
            sp.color = GetSpectrumTexture(s.contains("color") ? s["color"] : json(1.0), SpectrumType::Albedo, "sheen.color");
            sp.roughness = GetFloatTexture(s.contains("roughness") ? s["roughness"] : json(0.3), "sheen.roughness");
            sp.weight = GetFloat(s, "weight", 1.f);
            m->SetSheen(sp);
        }
    }
    if (j.contains("medium")) materialInterior_[id] = GetMedium(j["medium"].get<std::string>());
    if (j.contains("emission")) {
        auto e = ParseEmission(j["emission"]);
        if (e) materialEmission_[id] = *e;
    }
    return m;
}

Material *Loader::GetMaterial(const std::string &id) {
    auto it = materialCache_.find(id);
    if (it != materialCache_.end()) return it->second;
    if (!doc_["materials"].contains(id)) Fail("unknown material '" + id + "'");
    auto m = ParseMaterial(doc_["materials"][id], id);
    if (m) scene_->materials.push_back(m);
    materialCache_[id] = m.get();
    return m.get();
}

int Loader::AddPrimitive(std::unique_ptr<Shape> shape, const Material *mat, ObjectRecord &rec) {
    Primitive p;
    p.shape = shape.get();
    p.material = mat;
    p.object = int(objects_.size());
    rec.bounds = Union(rec.bounds, shape->Bounds());
    scene_->shapes.push_back(std::move(shape));
    scene_->primitives.push_back(p);
    int idx = int(scene_->primitives.size() - 1);
    rec.prims.push_back(idx);
    return idx;
}

void Loader::AttachEmission(ObjectRecord &rec, const EmissionSpec &e) {
    float scale = e.scale;
    if (e.powerMode) {
        float area = 0;
        for (int pi : rec.prims) area += scene_->primitives[pi].shape->Area();
        float lobe = 2 * Pi / (e.cosPower + 2) * (e.twoSided ? 2 : 1);
        float y = SpectrumToY(*e.spectrum);
        scale = (area > 0 && y > 0) ? e.power / (y * area * lobe) : 0.f;
    }
    for (int pi : rec.prims) {
        Primitive &p = scene_->primitives[pi];
        auto light = std::make_shared<DiffuseAreaLight>(p.shape, e.spectrum, scale, e.twoSided, e.cosPower,
                                                        &p.mediumInterface);
        p.areaLight = light.get();
        scene_->lights.push_back(light);
    }
}

void Loader::AddObject(const json &o, int index) {
    ObjectRecord rec;
    rec.name = GetString(o, "name", "object" + std::to_string(index));
    if (!o.contains("geometry")) Fail("object '" + rec.name + "': missing 'geometry'");
    std::string gid = o["geometry"].get<std::string>();
    if (!doc_["geometry"].contains(gid)) Fail("object '" + rec.name + "': unknown geometry '" + gid + "'");
    const json &g = doc_["geometry"][gid];
    Transform xf = ParseTransform(o.contains("transform") ? o["transform"] : json());

    const Material *mat = nullptr;
    std::string matId;
    if (o.contains("material") && !o["material"].is_null()) {
        matId = o["material"].get<std::string>();
        mat = GetMaterial(matId);
    } else {
        Fail("object '" + rec.name + "': missing 'material' (use a material of type 'interface' for pure medium boundaries)");
    }

    bool flip = GetBool(o, "flip_normals", false);
    std::string gtype = GetString(g, "type", "mesh");
    if (gtype == "sphere") {
        float r = GetFloat(g, "radius", 1.f);
        Vec3f c = g.contains("center") ? ParseVec3(g["center"], "center") : Vec3f(0, 0, 0);
        Vec3f wc = xf.Point(c);
        Vec3f sx = xf.Vector(Vec3f(1, 0, 0)), sy = xf.Vector(Vec3f(0, 1, 0)), sz = xf.Vector(Vec3f(0, 0, 1));
        float lx = Length(sx), ly = Length(sy), lz = Length(sz);
        if (std::abs(lx - ly) > 1e-3f * lx || std::abs(lx - lz) > 1e-3f * lx)
            LogWarning("object '{}': non-uniform scale on a sphere; using the average scale", rec.name);
        float wr = r * (lx + ly + lz) / 3;
        rec.probe = wc;
        rec.closed = true;
        AddPrimitive(std::make_unique<Sphere>(wc, wr, flip), mat, rec);
    } else {
        std::shared_ptr<TriangleMesh> mesh;
        if (gtype == "rect" || gtype == "quad") {
            json s = g.contains("size") ? g["size"] : json::array({1, 1});
            mesh = MakeRectMesh(s[0].get<float>(), s[1].get<float>());
        } else if (gtype == "box" || gtype == "cube") {
            mesh = MakeBoxMesh(g.contains("size") ? ParseVec3(g["size"], "size") : Vec3f(1, 1, 1));
        } else if (gtype == "disk") {
            mesh = MakeDiskMesh(GetFloat(g, "radius", 1.f), int(GetFloat(g, "segments", 64)));
        } else if (gtype == "mesh") {
            std::string err;
            auto loaded = LoadMesh(GetString(g, "file", ""), &err);
            if (!loaded) Fail("geometry '" + gid + "': " + err);
            mesh = std::make_shared<TriangleMesh>(*loaded);
            mesh->closed = GetBool(g, "closed", false);
        } else {
            Fail("geometry '" + gid + "': unknown type '" + gtype + "'");
        }
        mesh = std::make_shared<TriangleMesh>(*mesh);  // per-instance copy (world space)
        mesh->ApplyTransform(xf);
        if (flip) {
            for (size_t i = 0; i + 2 < mesh->indices.size(); i += 3) std::swap(mesh->indices[i + 1], mesh->indices[i + 2]);
            for (auto &n : mesh->n) n = -n;
        }
        scene_->meshes.push_back(mesh);
        rec.closed = mesh->closed;
        rec.probe = mesh->Bounds().Centroid();
        for (int t = 0; t < mesh->TriangleCount(); ++t) AddPrimitive(std::make_unique<Triangle>(mesh, t), mat, rec);
    }

    // Media: interior from the object, else from the material.
    if (o.contains("interior")) rec.interior = o["interior"].is_null() ? nullptr : GetMedium(o["interior"].get<std::string>());
    else if (!matId.empty() && materialInterior_.count(matId)) rec.interior = materialInterior_[matId];
    if (o.contains("exterior")) {
        const json &e = o["exterior"];
        if (e.is_string() && e.get<std::string>() == "auto") rec.autoExterior = true;
        else {
            rec.autoExterior = false;
            rec.exterior = e.is_null() ? nullptr : GetMedium(e.get<std::string>());
        }
    }
    if (rec.interior && !rec.closed)
        LogWarning("object '{}' has an interior medium but is not a closed shape", rec.name);

    objects_.push_back(rec);
    ObjectRecord &stored = objects_.back();

    std::optional<EmissionSpec> emission;
    if (o.contains("emission")) emission = ParseEmission(o["emission"]);
    else if (!matId.empty() && materialEmission_.count(matId)) emission = materialEmission_[matId];
    if (emission) pendingEmission_.push_back({int(objects_.size() - 1), *emission});
    scene_->objectNames.push_back(stored.name);
}

void Loader::AddLight(const std::string &id, const json &l) {
    if (!GetBool(l, "enabled", true)) return;
    std::string type = GetString(l, "type", "");
    auto emissionSpectrum = [&]() -> SpectrumPtr {
        if (l.contains("temperature")) return NormalizedToY(std::make_shared<BlackbodySpectrum>(l["temperature"].get<float>()));
        if (l.contains("spectrum")) return ParseSpectrum(l["spectrum"], SpectrumType::Illuminant, "spectrum");
        if (l.contains("color")) return ParseSpectrum(l["color"], SpectrumType::Illuminant, "color");
        return NormalizedToY(GetNamedSpectrum("D65"));
    };
    float scaleMul = GetFloat(l, "scale", 1.f);
    if (type == "point" || type == "spot") {
        SpectrumPtr I = emissionSpectrum();
        Vec3f pos = ParseVec3(l.at("position"), "position");
        float scale = GetFloat(l, "intensity", 1.f) * scaleMul;
        std::shared_ptr<Light> light;
        if (type == "point") {
            if (l.contains("power")) scale = GetFloat(l, "power", 1.f) / (4 * Pi * SpectrumToY(*I));
            light = std::make_shared<PointLight>(pos, I, scale);
        } else {
            Vec3f dir = l.contains("look_at") ? ParseVec3(l["look_at"], "look_at") - pos : ParseVec3(l.at("direction"), "direction");
            float width = GetFloat(l, "cone_angle", 30.f), start = GetFloat(l, "falloff_start", width * 0.8f);
            light = std::make_shared<SpotLight>(pos, dir, I, scale, width, start);
        }
        light->medium = EnclosingMedium(pos, -1);
        scene_->lights.push_back(light);
        return;
    }
    if (type == "environment" || type == "env" || type == "sky") {
        float scale = GetFloat(l, "intensity", 1.f) * scaleMul;
        std::shared_ptr<Light> light;
        if (l.contains("file")) {
            Image img;
            std::string err;
            if (!ReadImage(l["file"].get<std::string>(), TextureEncoding::sRGB, &img, &err)) Fail("light '" + id + "': " + err);
            light = std::make_shared<EnvironmentLight>(std::make_shared<Image>(std::move(img)), scale, GetFloat(l, "rotation", 0.f));
        } else {
            light = std::make_shared<EnvironmentLight>(emissionSpectrum(), scale);
        }
        scene_->lights.push_back(light);
        return;
    }
    if (type == "rect" || type == "sphere" || type == "disk") {
        Vec3f pos = ParseVec3(l.at("position"), "position");
        json o;
        o["name"] = "light:" + id;
        json geom;
        Transform xf = Transform::Translate(pos);
        if (type == "sphere") {
            geom["type"] = "sphere";
            geom["radius"] = GetFloat(l, "radius", 0.1f);
        } else {
            Vec3f dir = l.contains("look_at") ? ParseVec3(l["look_at"], "look_at") - pos
                                              : (l.contains("direction") ? ParseVec3(l["direction"], "direction") : Vec3f(0, -1, 0));
            dir = Normalize(dir);
            Frame f = Frame::FromZ(dir);
            // Local +Y (the emitting side) maps to dir; the basis (f.y, dir, f.x) is a proper rotation.
            Mat4 m;
            m.m[0][0] = f.y.x; m.m[1][0] = f.y.y; m.m[2][0] = f.y.z;
            m.m[0][1] = dir.x; m.m[1][1] = dir.y; m.m[2][1] = dir.z;
            m.m[0][2] = f.x.x; m.m[1][2] = f.x.y; m.m[2][2] = f.x.z;
            xf = xf * Transform(m);
            if (type == "rect") {
                geom["type"] = "rect";
                geom["size"] = l.contains("size") ? l["size"] : json::array({1, 1});
            } else {
                geom["type"] = "disk";
                geom["radius"] = GetFloat(l, "radius", 0.5f);
            }
        }
        std::string gid = "__light_geom_" + id;
        doc_["geometry"][gid] = geom;
        if (!blackMaterial_) {
            blackMaterial_ = std::make_shared<DiffuseMaterial>(
                std::make_shared<ConstantSpectrumTexture>(std::make_shared<ConstantSpectrum>(0.f), RGB()));
            blackMaterial_->name = "__light_black";
            scene_->materials.push_back(blackMaterial_);
            materialCache_["__light_black"] = blackMaterial_.get();
        }
        o["geometry"] = gid;
        o["material"] = "__light_black";
        json e;
        if (l.contains("temperature")) e["temperature"] = l["temperature"];
        if (l.contains("spectrum")) e["spectrum"] = l["spectrum"];
        if (l.contains("color")) e["color"] = l["color"];
        if (l.contains("power")) e["power"] = GetFloat(l, "power", 1.f) * scaleMul;
        e["luminance"] = GetFloat(l, "luminance", GetFloat(l, "intensity", 1.f)) * scaleMul;
        e["two_sided"] = GetBool(l, "two_sided", false);
        e["cos_power"] = GetFloat(l, "cos_power", 0.f);
        o["emission"] = e;
        // Build the object with the light's transform encoded as a matrix.
        json withXf = o;
        json mat = json::array();
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k) mat.push_back(xf.Matrix().m[i][k]);
        withXf["transform"] = {{"matrix", mat}};
        AddObject(withXf, int(objects_.size()));
        return;
    }
    Fail("light '" + id + "': unknown type '" + type + "'");
}

const Medium *Loader::EnclosingMedium(const Vec3f &p, int excludeObject) const {
    const Medium *best = nullptr;
    float bestVol = Infinity;
    for (size_t i = 0; i < objects_.size(); ++i) {
        const ObjectRecord &o = objects_[i];
        if (int(i) == excludeObject || !o.interior || !o.closed) continue;
        if (!o.bounds.Inside(p)) continue;
        if (!scene_->primitives[o.prims[0]].shape->Contains(p)) continue;
        Vec3f d = o.bounds.Diagonal();
        float vol = d.x * d.y * d.z;
        if (vol < bestVol) {
            bestVol = vol;
            best = o.interior;
        }
    }
    return best;
}

void Loader::ResolveMedia() {
    for (size_t i = 0; i < objects_.size(); ++i) {
        ObjectRecord &o = objects_[i];
        if (o.autoExterior) {
            // The containing medium must also enclose this object's probe and be larger than it.
            const Medium *best = nullptr;
            float bestVol = Infinity;
            Vec3f d0 = o.bounds.Diagonal();
            float myVol = d0.x * d0.y * d0.z;
            for (size_t k = 0; k < objects_.size(); ++k) {
                const ObjectRecord &c = objects_[k];
                if (k == i || !c.interior || !c.closed) continue;
                Vec3f dc = c.bounds.Diagonal();
                float vol = dc.x * dc.y * dc.z;
                if (vol <= myVol) continue;
                if (!c.bounds.Inside(o.probe) || !scene_->primitives[c.prims[0]].shape->Contains(o.probe)) continue;
                if (vol < bestVol) {
                    bestVol = vol;
                    best = c.interior;
                }
            }
            o.exterior = best;
        }
        for (int pi : o.prims) {
            Primitive &p = scene_->primitives[pi];
            p.mediumInterface.inside = o.interior ? o.interior : o.exterior;
            p.mediumInterface.outside = o.exterior;
            // Non-transmissive surfaces never need an inside medium; keep it consistent anyway.
        }
    }
}

void Loader::ParseCamera() {
    const json &cams = doc_["cameras"];
    std::string name = !opts_.camera.empty() ? opts_.camera : scene_->settings.cameraName;
    if (name.empty() && !cams.empty()) name = cams.begin().key();
    if (name.empty() || !cams.contains(name)) Fail("camera '" + name + "' not found");
    scene_->settings.cameraName = name;
    const json &c = cams[name];
    std::string type = GetString(c, "type", "perspective");
    Vec3f pos = ParseVec3(c.at("position"), "camera.position");
    Vec3f target = c.contains("look_at") ? ParseVec3(c["look_at"], "camera.look_at")
                                         : pos + ParseVec3(c.at("direction"), "camera.direction");
    Vec3f up = c.contains("up") ? ParseVec3(c["up"], "camera.up") : Vec3f(0, 1, 0);
    CameraParams p;
    p.cameraToWorld = Transform::LookAt(pos, target, up);
    const FilmSettings &fs = scene_->settings.film;
    Filter filter(fs.filter);
    CameraFilmInfo info{fs.width, fs.height, int(std::ceil(filter.Radius() - 0.5f))};
    float sensorH = GetFloat(c, "sensor_height_mm", 24.f);
    if (c.contains("fov_y")) p.fovY = GetFloat(c, "fov_y", 40.f);
    else if (c.contains("focal_length_mm")) p.fovY = Degrees(2 * std::atan(sensorH / (2 * GetFloat(c, "focal_length_mm", 50.f))));
    p.focalDistance = c.contains("focus_on") ? Distance(pos, ParseVec3(c["focus_on"], "focus_on"))
                                             : GetFloat(c, "focus_distance", Distance(pos, target));
    if (c.contains("aperture_radius")) p.lensRadius = GetFloat(c, "aperture_radius", 0.f);
    else if (c.contains("f_stop") && type != "pinhole") {
        float focalMm = sensorH / (2 * std::tan(Radians(p.fovY) / 2));
        p.lensRadius = (focalMm / 1000.f) / (2 * GetFloat(c, "f_stop", 8.f)) * GetFloat(c, "units_per_meter", 1.f);
    }
    if (type == "pinhole") p.lensRadius = 0;
    p.orthoHeight = GetFloat(c, "ortho_height", 2.f);
    if (type == "orthographic" || type == "ortho") scene_->camera = std::make_unique<OrthographicCamera>(p, info);
    else if (type == "perspective" || type == "thin_lens" || type == "pinhole")
        scene_->camera = std::make_unique<PerspectiveCamera>(p, info);
    else Fail("camera '" + name + "': unknown type '" + type + "'");
    scene_->camera->medium = EnclosingMedium(pos, -1);
}

void Loader::ParseRender() {
    RenderSettings &rs = scene_->settings;
    const json r = doc_.contains("render") ? doc_["render"] : json::object();
    rs.cameraName = GetString(r, "camera", "");
    rs.seed = r.contains("seed") ? r["seed"].get<uint64_t>() : 0;
    if (r.contains("film")) {
        const json &f = r["film"];
        rs.film.width = int(GetFloat(f, "width", 640));
        rs.film.height = int(GetFloat(f, "height", 360));
        rs.film.exposure = GetFloat(f, "exposure", 0.f);
        rs.film.whiteBalance = GetFloat(f, "white_balance", 0.f);
        if (f.contains("colorspace") && !ParseColorSpace(f["colorspace"].get<std::string>(), &rs.film.colorSpace))
            Fail("unknown colorspace '" + f["colorspace"].get<std::string>() + "'");
        if (f.contains("filter")) {
            const json &fl = f["filter"];
            std::string t = fl.is_string() ? fl.get<std::string>() : GetString(fl, "type", "blackman_harris");
            if (t == "box") rs.film.filter.type = FilterType::Box, rs.film.filter.radius = 0.5f;
            else if (t == "gaussian") rs.film.filter.type = FilterType::Gaussian;
            else if (t == "blackman_harris") rs.film.filter.type = FilterType::BlackmanHarris;
            else Fail("unknown filter '" + t + "'");
            if (fl.is_object() && fl.contains("radius")) rs.film.filter.radius = fl["radius"].get<float>();
        }
    }
    if (opts_.width > 0) rs.film.width = opts_.width;
    if (opts_.height > 0) rs.film.height = opts_.height;
    if (r.contains("integrator")) {
        const json &i = r["integrator"];
        IntegratorSettings &is = rs.integrator;
        std::string t = GetString(i, "type", "mmlt");
        if (t == "path" || t == "pt") is.type = IntegratorType::Path;
        else if (t == "bdpt") is.type = IntegratorType::BDPT;
        else if (t == "mmlt" || t == "mlt") is.type = IntegratorType::MMLT;
        else if (t == "pssmlt") is.type = IntegratorType::PSSMLT;
        else Fail("unknown integrator '" + t + "'");
        is.maxDepth = int(GetFloat(i, "max_depth", float(is.maxDepth)));
        is.spp = int(GetFloat(i, "spp", float(is.spp)));
        is.mutationsPerPixel = GetFloat(i, "mutations_per_pixel", float(is.mutationsPerPixel));
        is.bootstrapSamples = int64_t(GetFloat(i, "bootstrap_samples", float(is.bootstrapSamples)));
        is.chains = int(GetFloat(i, "chains", float(is.chains)));
        is.largeStepProbability = GetFloat(i, "large_step_probability", is.largeStepProbability);
        is.sigma = GetFloat(i, "sigma", is.sigma);
        is.russianRoulette = GetBool(i, "russian_roulette", is.russianRoulette);
        if (i.contains("time_limit")) {
            const json &tl = i["time_limit"];
            if (tl.is_number()) is.timeLimit = tl.get<double>();
            else if (!ParseDuration(tl.get<std::string>(), &is.timeLimit)) Fail("bad time_limit");
        }
    }
    json outs = r.contains("output") ? r["output"] : (r.contains("outputs") ? r["outputs"] : json::array());
    if (outs.is_object() || outs.is_string()) outs = json::array({outs});
    for (const auto &o : outs) {
        OutputSpec spec;
        if (o.is_string()) spec.file = o.get<std::string>();
        else {
            spec.file = GetString(o, "file", "");
            if (o.contains("tonemap") && !ParseToneMap(o["tonemap"].get<std::string>(), &spec.toneMap))
                Fail("unknown tonemap '" + o["tonemap"].get<std::string>() + "'");
            spec.half = GetBool(o, "half", false);
        }
        if (!spec.file.empty()) rs.outputs.push_back(spec);
    }
}

std::unique_ptr<Scene> Loader::Load() {
    scene_ = std::make_unique<Scene>();
    for (const char *sec : {"textures", "materials", "media", "geometry", "lights", "cameras", "rigs"})
        if (!doc_.contains(sec)) doc_[sec] = json::object();
    if (!doc_.contains("objects")) doc_["objects"] = json::array();

    // Light rigs: merge the active rigs' documents on top of the scene.
    std::vector<std::string> active;
    if (opts_.rigs) active = *opts_.rigs;
    else if (doc_.contains("active_rigs"))
        for (const auto &r : doc_["active_rigs"]) active.push_back(r.get<std::string>());
    for (const auto &r : opts_.addRigs)
        if (std::find(active.begin(), active.end(), r) == active.end()) active.push_back(r);
    for (const auto &rig : active) {
        if (!doc_["rigs"].contains(rig)) Fail("unknown light rig '" + rig + "'");
        const json &rj = doc_["rigs"][rig];
        json rigDoc;
        if (rj.is_string()) rigDoc = LoadComposed(Utf8Path(rj.get<std::string>()));
        else if (rj.is_object() && rj.contains("file")) rigDoc = LoadComposed(Utf8Path(rj["file"].get<std::string>()));
        else rigDoc = rj;
        MergeDocument(doc_, rigDoc);
    }
    scene_->settings.outputs.clear();
    ParseRender();

    int idx = 0;
    for (const auto &o : doc_["objects"]) {
        if (GetBool(o, "visible", true)) AddObject(o, idx);
        ++idx;
    }
    // Explicit lights (area lights become objects too, so add them before resolving media).
    json lights = doc_["lights"];
    for (auto it = lights.begin(); it != lights.end(); ++it) AddLight(it.key(), it.value());
    // The primitive array is final now, so area lights can safely point at medium interfaces.
    for (auto &[objIndex, spec] : pendingEmission_) AttachEmission(objects_[objIndex], spec);
    ResolveMedia();
    ParseCamera();
    scene_->Build();
    if (scene_->lights.empty()) LogWarning("scene has no lights; the image will be black");
    return std::move(scene_);
}

} // namespace

std::unique_ptr<Scene> LoadScene(const std::string &path, const LoadOptions &opts, std::string *err) {
    try {
        fs::path p = Utf8Path(path);
        json doc = LoadComposed(p);
        Loader loader(std::move(doc), opts, p.parent_path());
        return loader.Load();
    } catch (const LoadError &e) {
        *err = e.what();
    } catch (const json::exception &e) {
        *err = std::string("scene format error: ") + e.what();
    } catch (const std::exception &e) {
        *err = e.what();
    }
    return nullptr;
}

} // namespace pr

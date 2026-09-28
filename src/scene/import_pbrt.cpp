// pbrt-v4 scene importer (subset), converting to the native scene JSON.
//
// Supported: Camera (perspective), Film, Sampler (pixelsamples), Integrator (maxdepth), transforms
// (LookAt, Translate, Scale, Rotate, Transform, ConcatTransform, CoordinateSystem/CoordSysTransform),
// Attribute/Transform blocks, WorldBegin, Material / MakeNamedMaterial / NamedMaterial (diffuse,
// coateddiffuse, conductor, coatedconductor, dielectric, thindielectric, diffusetransmission,
// subsurface, mix [first], interface), Texture (imagemap, checkerboard, constant, scale [partial]),
// Shape (sphere, trianglemesh, plymesh, bilinearmesh, loopsubdiv [no subdivision], disk),
// LightSource (point, spot, distant, infinite [constant or equal-area image], goniometric [IES only]),
// AreaLightSource diffuse, MakeNamedMedium (homogeneous, uniformgrid, nanovdb), MediumInterface,
// ReverseOrientation, ObjectBegin/End/ObjectInstance, Include/Import.

#include "scene/importers.h"

#include "core/fsutil.h"
#include "core/log.h"
#include "core/rng.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <variant>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace pr {

namespace {

// ---------------------------------------------------------------------------------------------
// Tokenizer

struct Token {
    enum Kind { Identifier, String, Number, LBracket, RBracket, End } kind = End;
    std::string text;
    double number = 0;
};

class Tokenizer {
  public:
    Tokenizer(std::string text, std::string file) : text_(std::move(text)), file_(std::move(file)) {}

    Token Next() {
        if (peeked_) {
            Token t = std::move(*peeked_);
            peeked_.reset();
            return t;
        }
        return Read();
    }
    const Token &Peek() {
        if (!peeked_) peeked_ = Read();
        return *peeked_;
    }
    const std::string &File() const { return file_; }
    int Line() const { return line_; }

  private:
    Token Read() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == '\n') ++line_;
            if (std::isspace((unsigned char)c)) {
                ++pos_;
                continue;
            }
            if (c == '#') {
                while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
                continue;
            }
            break;
        }
        Token t;
        if (pos_ >= text_.size()) return t;
        char c = text_[pos_];
        if (c == '[') {
            ++pos_;
            t.kind = Token::LBracket;
            return t;
        }
        if (c == ']') {
            ++pos_;
            t.kind = Token::RBracket;
            return t;
        }
        if (c == '"') {
            size_t end = text_.find('"', pos_ + 1);
            if (end == std::string::npos) throw std::runtime_error(file_ + ": unterminated string");
            t.kind = Token::String;
            t.text = text_.substr(pos_ + 1, end - pos_ - 1);
            pos_ = end + 1;
            return t;
        }
        size_t start = pos_;
        while (pos_ < text_.size() && !std::isspace((unsigned char)text_[pos_]) && text_[pos_] != '[' &&
               text_[pos_] != ']' && text_[pos_] != '"' && text_[pos_] != '#')
            ++pos_;
        t.text = text_.substr(start, pos_ - start);
        char *endp = nullptr;
        t.number = std::strtod(t.text.c_str(), &endp);
        t.kind = (endp && *endp == '\0' && !t.text.empty() &&
                  (std::isdigit((unsigned char)t.text[0]) || t.text[0] == '-' || t.text[0] == '+' || t.text[0] == '.'))
                     ? Token::Number
                     : Token::Identifier;
        return t;
    }
    std::string text_, file_;
    size_t pos_ = 0;
    int line_ = 1;
    std::optional<Token> peeked_;
};

// ---------------------------------------------------------------------------------------------
// Parameter lists

struct Param {
    std::string type, name;
    std::vector<double> nums;
    std::vector<std::string> strs;
    bool used = false;
};

struct ParamSet {
    std::vector<Param> params;
    Param *Find(const std::string &name) {
        for (auto &p : params)
            if (p.name == name) {
                p.used = true;
                return &p;
            }
        return nullptr;
    }
    double Float(const std::string &name, double def) {
        Param *p = Find(name);
        return p && !p->nums.empty() ? p->nums[0] : def;
    }
    int Int(const std::string &name, int def) { return int(Float(name, def)); }
    bool Bool(const std::string &name, bool def) {
        Param *p = Find(name);
        if (!p) return def;
        if (!p->strs.empty()) return p->strs[0] == "true";
        return !p->nums.empty() ? p->nums[0] != 0 : def;
    }
    std::string String(const std::string &name, const std::string &def) {
        Param *p = Find(name);
        return p && !p->strs.empty() ? p->strs[0] : def;
    }
    std::vector<double> Floats(const std::string &name) {
        Param *p = Find(name);
        return p ? p->nums : std::vector<double>();
    }
};

ParamSet ParseParams(Tokenizer &tk) {
    ParamSet ps;
    while (tk.Peek().kind == Token::String) {
        Token decl = tk.Next();
        std::istringstream ds(decl.text);
        Param p;
        ds >> p.type >> p.name;
        if (p.name.empty()) throw std::runtime_error(tk.File() + ":" + std::to_string(tk.Line()) + ": bad parameter '" + decl.text + "'");
        auto addValue = [&](const Token &v) {
            if (v.kind == Token::Number) p.nums.push_back(v.number);
            else if (v.kind == Token::String) p.strs.push_back(v.text);
            else if (v.kind == Token::Identifier && (v.text == "true" || v.text == "false")) p.strs.push_back(v.text);
            else throw std::runtime_error(tk.File() + ":" + std::to_string(tk.Line()) + ": bad value for '" + p.name + "'");
        };
        if (tk.Peek().kind == Token::LBracket) {
            tk.Next();
            while (tk.Peek().kind != Token::RBracket && tk.Peek().kind != Token::End) addValue(tk.Next());
            tk.Next();
        } else {
            addValue(tk.Next());
        }
        if (p.type == "bool")
            for (auto &s : p.strs) s = (s == "true") ? "true" : "false";
        ps.params.push_back(std::move(p));
    }
    return ps;
}

// ---------------------------------------------------------------------------------------------
// Matrices (row-major, column vectors, like pr::Mat4)

Mat4 MatMul(const Mat4 &a, const Mat4 &b) { return a * b; }
Mat4 MatInverse(const Mat4 &m) {
    Mat4 r;
    if (!m.Inverse(&r)) return Mat4();
    return r;
}
Mat4 MatTranslate(double x, double y, double z) { return Transform::Translate(Vec3f(float(x), float(y), float(z))).Matrix(); }
Mat4 MatScale(double x, double y, double z) { return Transform::Scale(Vec3f(float(x), float(y), float(z))).Matrix(); }
Mat4 MatRotate(double deg, double x, double y, double z) {
    return Transform::Rotate(float(deg), Vec3f(float(x), float(y), float(z))).Matrix();
}
// pbrt's LookAt: returns cameraFromWorld with right = up x dir.
Mat4 MatLookAt(Vec3f pos, Vec3f look, Vec3f up) {
    Vec3f dir = Normalize(look - pos);
    Vec3f right = Cross(Normalize(up), dir);
    if (LengthSquared(right) == 0) right = Vec3f(1, 0, 0);
    right = Normalize(right);
    Vec3f newUp = Cross(dir, right);
    Mat4 worldFromCamera;
    worldFromCamera.m[0][0] = right.x; worldFromCamera.m[1][0] = right.y; worldFromCamera.m[2][0] = right.z;
    worldFromCamera.m[0][1] = newUp.x; worldFromCamera.m[1][1] = newUp.y; worldFromCamera.m[2][1] = newUp.z;
    worldFromCamera.m[0][2] = dir.x;   worldFromCamera.m[1][2] = dir.y;   worldFromCamera.m[2][2] = dir.z;
    worldFromCamera.m[0][3] = pos.x;   worldFromCamera.m[1][3] = pos.y;   worldFromCamera.m[2][3] = pos.z;
    return MatInverse(worldFromCamera);
}
// pbrt lists matrices column by column.
Mat4 MatFromPbrt(const std::vector<double> &v) {
    Mat4 m;
    for (int i = 0; i < 16; ++i) m.m[i % 4][i / 4] = float(v[i]);
    return m;
}
json MatJson(const Mat4 &m) {
    json a = json::array();
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) a.push_back(m.m[i][k]);
    return a;
}
Vec3f MatPoint(const Mat4 &m, Vec3f p) { return Transform(m, Mat4()).Point(p); }
Vec3f MatVector(const Mat4 &m, Vec3f v) { return Transform(m, Mat4()).Vector(v); }
json VecJ(const Vec3f &v) { return json::array({v.x, v.y, v.z}); }

// pbrt-v4 equal-area octahedral mapping (Clarberg 2008).
Vec3f EqualAreaSquareToSphere(Vec2f p) {
    float u = 2 * p.x - 1, v = 2 * p.y - 1;
    float up = std::abs(u), vp = std::abs(v);
    float signedDistance = 1 - (up + vp);
    float d = std::abs(signedDistance);
    float r = 1 - d;
    float phi = (r == 0 ? 1 : (vp - up) / r + 1) * Pi / 4;
    float z = std::copysign(1 - r * r, signedDistance);
    float cosPhi = std::copysign(std::cos(phi), u), sinPhi = std::copysign(std::sin(phi), v);
    return {cosPhi * r * SafeSqrt(2 - r * r), sinPhi * r * SafeSqrt(2 - r * r), z};
}
Vec2f EqualAreaSphereToSquare(Vec3f d) {
    float x = std::abs(d.x), y = std::abs(d.y), z = std::abs(d.z);
    float r = SafeSqrt(1 - z);
    float a = std::max(x, y), b = std::min(x, y);
    b = a == 0 ? 0 : b / a;
    float phi = std::atan(b) * 2 / Pi;
    if (x < y) phi = 1 - phi;
    float v = phi * r, u = r - v;
    if (d.z < 0) {
        std::swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }
    u = std::copysign(u, d.x);
    v = std::copysign(v, d.y);
    return {(u + 1) * 0.5f, (v + 1) * 0.5f};
}

// ---------------------------------------------------------------------------------------------

struct GraphicsState {
    Mat4 ctm;
    bool reverseOrientation = false;
    std::string material;       // material id ("" = default)
    json areaLight;             // emission block or null
    std::string inside, outside;
};

class PbrtImporter {
  public:
    PbrtImporter(const ImportOptions &opts) : opts_(opts) {
        doc_ = {{"textures", json::object()}, {"materials", json::object()}, {"media", json::object()},
                {"geometry", json::object()}, {"objects", json::array()},  {"lights", json::object()},
                {"cameras", json::object()}, {"render", json::object()}};
        cacheDir_ = fs::temp_directory_path() / "prender_import";
        std::error_code ec;
        fs::create_directories(cacheDir_, ec);
    }

    void ParseFile(const fs::path &path);
    json Finish();

  private:
    void Parse(Tokenizer &tk, const fs::path &dir);
    [[noreturn]] void Error(Tokenizer &tk, const std::string &msg) {
        throw std::runtime_error(tk.File() + ":" + std::to_string(tk.Line()) + ": " + msg);
    }
    void Warn(const std::string &msg) {
        if (warned_.insert(msg).second) LogWarning("pbrt import: {}", msg);
    }
    std::vector<double> Numbers(Tokenizer &tk, int n) {
        std::vector<double> v;
        bool bracket = tk.Peek().kind == Token::LBracket;
        if (bracket) tk.Next();
        while (int(v.size()) < n) {
            Token t = tk.Next();
            if (t.kind != Token::Number) Error(tk, "expected a number");
            v.push_back(t.number);
        }
        if (bracket && tk.Next().kind != Token::RBracket) Error(tk, "expected ']'");
        return v;
    }
    std::string ResolvePath(const fs::path &dir, const std::string &f) {
        fs::path p = Utf8Path(f);
        if (p.is_relative()) p = dir / p;
        return PathUtf8(p.lexically_normal());
    }

    json SpectrumParam(ParamSet &ps, const std::string &name, const json &def, bool illuminant, const fs::path &dir);
    json SpectrumOrTexture(ParamSet &ps, const std::string &name, const json &def, const fs::path &dir);
    json FloatOrTexture(ParamSet &ps, const std::string &name, const json &def);
    json Roughness(ParamSet &ps, const std::string &prefix, bool remap);
    json IOR(ParamSet &ps, const std::string &name, double def);
    json MakeMaterial(const std::string &type, ParamSet &ps, const fs::path &dir);
    void AddShape(const std::string &type, ParamSet &ps, const fs::path &dir, Tokenizer &tk);
    void AddLight(const std::string &type, ParamSet &ps, const fs::path &dir);
    void MakeMedium(const std::string &name, ParamSet &ps, const fs::path &dir);
    void MakeTexture(const std::string &name, const std::string &kind, const std::string &cls, ParamSet &ps, const fs::path &dir);
    void EmitObject(json obj);

    ImportOptions opts_;
    json doc_;
    std::vector<GraphicsState> stack_;
    GraphicsState gs_;
    std::map<std::string, Mat4> namedCS_;
    std::map<std::string, std::string> namedMaterials_;
    std::map<std::string, json> objectDefs_;  // name -> array of objects
    std::string currentObject_;
    bool inWorld_ = false;
    int counter_ = 0;
    std::set<std::string> warned_;
    fs::path cacheDir_;
    // Camera / film
    bool haveCamera_ = false;
    Mat4 cameraFromWorld_;
    double fov_ = 90, lensRadius_ = 0, focalDistance_ = 1e6;
    int xres_ = 1280, yres_ = 720;
    std::string filmName_;
    Bounds3f bounds_;
};

void PbrtImporter::ParseFile(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + PathUtf8(path));
    std::stringstream ss;
    ss << in.rdbuf();
    Tokenizer tk(ss.str(), PathUtf8(path));
    Parse(tk, path.parent_path());
}

json PbrtImporter::SpectrumParam(ParamSet &ps, const std::string &name, const json &def, bool illuminant,
                                 const fs::path &dir) {
    Param *p = ps.Find(name);
    if (!p) return def;
    if (p->type == "rgb" && p->nums.size() >= 3) return json::array({p->nums[0], p->nums[1], p->nums[2]});
    if (p->type == "float" && !p->nums.empty()) return p->nums[0];
    if (p->type == "blackbody" && !p->nums.empty()) return json{{"blackbody", p->nums[0]}};
    if (p->type == "spectrum") {
        if (!p->strs.empty()) {
            std::string s = p->strs[0];
            if (s.rfind("stdillum-", 0) == 0) return "D65";
            if (s.rfind("glass-", 0) == 0) {
                std::string g = s.substr(6);
                if (g == "BK7" || g == "SF11" || g == "fused-silica" || g == "F11") return g == "fused-silica" ? "fused_silica" : (g == "F11" ? "SF11" : g);
                Warn("glass spectrum '" + s + "' approximated by BK7");
                return "BK7";
            }
            // Spectrum file: pairs of lambda value
            std::ifstream in(Utf8Path(ResolvePath(dir, s)));
            if (in) {
                json samples = json::array();
                double l, v;
                while (in >> l >> v) samples.push_back(json::array({l, v}));
                if (!samples.empty()) return json{{"samples", samples}};
            }
            Warn("named spectrum '" + s + "' is not supported; using 1");
            return 1.0;
        }
        if (p->nums.size() >= 4) {
            json samples = json::array();
            for (size_t i = 0; i + 1 < p->nums.size(); i += 2) samples.push_back(json::array({p->nums[i], p->nums[i + 1]}));
            return json{{"samples", samples}};
        }
    }
    (void)illuminant;
    return def;
}

json PbrtImporter::SpectrumOrTexture(ParamSet &ps, const std::string &name, const json &def, const fs::path &dir) {
    Param *p = ps.Find(name);
    if (p && p->type == "texture" && !p->strs.empty()) {
        std::string id = "pbrt_tex:" + p->strs[0];
        if (!doc_["textures"].contains(id)) {
            Warn("undefined texture '" + p->strs[0] + "'");
            return def;
        }
        return json{{"texture", id}};
    }
    if (p) p->used = false;
    return SpectrumParam(ps, name, def, false, dir);
}

json PbrtImporter::FloatOrTexture(ParamSet &ps, const std::string &name, const json &def) {
    Param *p = ps.Find(name);
    if (!p) return def;
    if (p->type == "texture" && !p->strs.empty()) {
        std::string id = "pbrt_tex:" + p->strs[0];
        if (!doc_["textures"].contains(id)) return def;
        return json{{"texture", id}};
    }
    return p->nums.empty() ? def : json(p->nums[0]);
}

// pbrt alpha = remap ? sqrt(r) : r; our roughness r' has alpha = r'^2.
json PbrtImporter::Roughness(ParamSet &ps, const std::string &prefix, bool remap) {
    auto conv = [&](json v) -> json {
        if (v.is_number()) {
            double r = v.get<double>();
            double alpha = remap ? std::sqrt(std::max(0.0, r)) : r;
            return std::sqrt(std::max(0.0, alpha));
        }
        if (v.is_object()) Warn("roughness textures are used without pbrt's remapping");
        return v;
    };
    json r = FloatOrTexture(ps, prefix + "roughness", 0.0);
    json u = FloatOrTexture(ps, prefix + "uroughness", r), v = FloatOrTexture(ps, prefix + "vroughness", r);
    return json{{"u", conv(u)}, {"v", conv(v)}};
}

json PbrtImporter::IOR(ParamSet &ps, const std::string &name, double def) {
    Param *p = ps.Find(name);
    if (!p) return def;
    if (p->type == "float" && !p->nums.empty()) return p->nums[0];
    p->used = false;
    return SpectrumParam(ps, name, def, false, fs::path());
}

json PbrtImporter::MakeMaterial(const std::string &type, ParamSet &ps, const fs::path &dir) {
    bool remap = ps.Bool("remaproughness", true);
    json m;
    auto conductor = [&](json &mat, const std::string &prefix) {
        Param *eta = ps.Find(prefix + "eta");
        Param *k = ps.Find(prefix + "k");
        if (Param *refl = ps.Find("reflectance")) {
            refl->used = false;
            mat["reflectance"] = SpectrumOrTexture(ps, "reflectance", 0.9, dir);
            return;
        }
        auto metalOf = [](const Param *p) -> std::string {
            if (!p || p->strs.empty()) return "";
            const std::string &s = p->strs[0];
            if (s.rfind("metal-", 0) != 0) return "";
            std::string metal = s.substr(6, s.find('-', 6) - 6);
            return metal;
        };
        std::string metal = metalOf(eta);
        if (metal.empty() && !eta && !k) metal = "Cu";  // pbrt's default conductor is copper
        if (!metal.empty()) {
            static const std::set<std::string> known = {"Au", "Ag", "Cu", "Al"};
            if (!known.count(metal)) {
                Warn("metal '" + metal + "' approximated by " + (metal == "CuZn" ? "Au" : "Al"));
                metal = metal == "CuZn" ? "Au" : "Al";
            }
            mat["metal"] = metal;
        } else {
            if (eta) eta->used = false;
            if (k) k->used = false;
            mat["eta"] = SpectrumParam(ps, prefix + "eta", 0.2, false, dir);
            mat["k"] = SpectrumParam(ps, prefix + "k", 3.9, false, dir);
        }
    };
    if (type == "diffuse" || type == "diffusetransmission") {
        if (type == "diffusetransmission") Warn("diffusetransmission is approximated by diffuse reflection");
        m = {{"type", "diffuse"}, {"reflectance", SpectrumOrTexture(ps, "reflectance", 0.5, dir)}};
    } else if (type == "coateddiffuse") {
        json r = Roughness(ps, "", remap);
        m = {{"type", "coated_diffuse"}, {"reflectance", SpectrumOrTexture(ps, "reflectance", 0.5, dir)},
             {"coat_roughness", r["u"]}, {"thickness", FloatOrTexture(ps, "thickness", 0.01)}, {"coat_ior", IOR(ps, "eta", 1.5)},
             {"g", ps.Float("g", 0)}};
        json albedo = SpectrumOrTexture(ps, "albedo", 0.0, dir);
        if (!(albedo.is_number() && albedo.get<double>() == 0)) m["albedo"] = albedo;
    } else if (type == "conductor") {
        json r = Roughness(ps, "", remap);
        m = {{"type", "conductor"}, {"roughness_u", r["u"]}, {"roughness_v", r["v"]}};
        conductor(m, "");
    } else if (type == "coatedconductor") {
        json ri = Roughness(ps, "interface.", remap), rc = Roughness(ps, "conductor.", remap);
        m = {{"type", "coated_conductor"}, {"coat_roughness", ri["u"]}, {"coat_ior", IOR(ps, "interface.eta", 1.5)},
             {"thickness", FloatOrTexture(ps, "thickness", 0.01)}, {"roughness_u", rc["u"]}, {"roughness_v", rc["v"]}};
        conductor(m, "conductor.");
    } else if (type == "dielectric" || type == "thindielectric") {
        json r = Roughness(ps, "", remap);
        m = {{"type", "dielectric"}, {"ior", IOR(ps, "eta", 1.5)}, {"roughness_u", r["u"]}, {"roughness_v", r["v"]}};
        if (type == "thindielectric") m["thin"] = true;
    } else if (type == "subsurface") {
        json r = Roughness(ps, "", remap);
        m = {{"type", "subsurface"}, {"ior", IOR(ps, "eta", 1.33)}, {"roughness_u", r["u"]}, {"roughness_v", r["v"]}};
        double scale = ps.Float("scale", 1);
        if (ps.Find("reflectance")) {
            ps.Find("reflectance")->used = false;
            m["albedo"] = SpectrumParam(ps, "reflectance", 0.8, false, dir);
            m["mfp"] = SpectrumParam(ps, "mfp", 1.0, false, dir);
        } else {
            m["sigma_a"] = SpectrumParam(ps, "sigma_a", json::array({0.0011, 0.0024, 0.014}), false, dir);
            m["sigma_s"] = SpectrumParam(ps, "sigma_s", json::array({2.55, 3.21, 3.77}), false, dir);
            m["scale"] = scale;
        }
        m["g"] = ps.Float("g", 0);
    } else if (type == "mix") {
        Param *mats = ps.Find("materials");
        Warn("mix materials use their first component");
        if (mats && !mats->strs.empty() && namedMaterials_.count(mats->strs[0])) return json{{"__ref", namedMaterials_[mats->strs[0]]}};
        m = {{"type", "diffuse"}, {"reflectance", 0.5}};
    } else if (type == "interface" || type == "" || type == "none") {
        m = {{"type", "interface"}};
    } else {
        Warn("material '" + type + "' is not supported; using diffuse");
        m = {{"type", "diffuse"}, {"reflectance", SpectrumOrTexture(ps, "reflectance", 0.5, dir)}};
    }
    if (Param *d = ps.Find("displacement")) {
        (void)d;
        Warn("displacement is not supported");
    }
    if (Param *nm = ps.Find("normalmap"); nm && !nm->strs.empty()) {
        std::string id = "pbrt_normalmap:" + std::to_string(counter_++);
        doc_["textures"][id] = {{"type", "image"}, {"file", ResolvePath(dir, nm->strs[0])}, {"colorspace", "linear"}};
        m["normal_map"] = {{"texture", id}};
    }
    return m;
}

void PbrtImporter::MakeTexture(const std::string &name, const std::string &kind, const std::string &cls, ParamSet &ps,
                               const fs::path &dir) {
    std::string id = "pbrt_tex:" + name;
    json t;
    json uvScale = json::array({ps.Float("uscale", 1), ps.Float("vscale", 1)});
    json uvOffset = json::array({ps.Float("udelta", 0), ps.Float("vdelta", 0)});
    if (cls == "imagemap") {
        std::string enc = ps.String("encoding", kind == "float" ? "linear" : "sRGB");
        t = {{"type", "image"}, {"file", ResolvePath(dir, ps.String("filename", ""))},
             {"colorspace", enc == "sRGB" ? "srgb" : "linear"}, {"uv_scale", uvScale}, {"uv_offset", uvOffset}};
        if (ps.Float("scale", 1) != 1) t["multiplier"] = ps.Float("scale", 1);
        if (ps.Bool("invert", false)) Warn("inverted image textures are not supported");
    } else if (cls == "checkerboard") {
        t = {{"type", "checker"}, {"a", SpectrumOrTexture(ps, "tex1", 1.0, dir)}, {"b", SpectrumOrTexture(ps, "tex2", 0.0, dir)},
             {"uv_scale", uvScale}, {"uv_offset", uvOffset}};
    } else if (cls == "constant") {
        t = {{"type", "constant"}, {"value", SpectrumOrTexture(ps, "value", 1.0, dir)}};
    } else if (cls == "scale") {
        json tex = SpectrumOrTexture(ps, "tex", 1.0, dir);
        json sc = FloatOrTexture(ps, "scale", 1.0);
        if (tex.is_object() && tex.contains("texture") && sc.is_number()) {
            t = doc_["textures"][tex["texture"].get<std::string>()];
            if (t.value("type", "") == "image") t["multiplier"] = sc;
            else Warn("scale texture approximated");
        } else if (!tex.is_object() && sc.is_number()) {
            json v = tex;
            if (v.is_number()) v = v.get<double>() * sc.get<double>();
            else if (v.is_array()) for (auto &c : v) c = c.get<double>() * sc.get<double>();
            t = {{"type", "constant"}, {"value", v}};
        } else {
            Warn("scale texture approximated by its input");
            t = {{"type", "constant"}, {"value", tex}};
        }
    } else if (cls == "mix") {
        Warn("mix textures use tex1");
        t = {{"type", "constant"}, {"value", SpectrumOrTexture(ps, "tex1", 0.0, dir)}};
    } else {
        Warn("texture class '" + cls + "' is not supported; using constant 0.5");
        t = {{"type", "constant"}, {"value", 0.5}};
    }
    doc_["textures"][id] = t;
}

void PbrtImporter::MakeMedium(const std::string &name, ParamSet &ps, const fs::path &dir) {
    std::string type = ps.String("type", "homogeneous");
    std::string id = "pbrt_medium:" + name;
    json sa = SpectrumParam(ps, "sigma_a", 1.0, false, dir), ss = SpectrumParam(ps, "sigma_s", 1.0, false, dir);
    if (ps.Find("preset")) Warn("measured medium presets are not supported; using sigma_a/sigma_s");
    json m = {{"sigma_a", sa}, {"sigma_s", ss}, {"scale", ps.Float("scale", 1)}, {"g", ps.Float("g", 0)}};
    Mat4 xf = gs_.ctm;
    if (type == "homogeneous") {
        m["type"] = "homogeneous";
    } else if (type == "nanovdb") {
        m["type"] = "grid";
        m["file"] = ResolvePath(dir, ps.String("filename", ""));
        m["transform"] = {{"matrix", MatJson(xf)}};
    } else if (type == "uniformgrid") {
        int nx = ps.Int("nx", 1), ny = ps.Int("ny", 1), nz = ps.Int("nz", 1);
        std::vector<double> dens = ps.Floats("density");
        if (dens.size() != size_t(nx) * ny * nz) {
            Warn("uniformgrid medium '" + name + "' has no density; skipped");
            return;
        }
        std::vector<double> p0 = ps.Floats("p0"), p1 = ps.Floats("p1");
        if (p0.size() < 3) p0 = {0, 0, 0};
        if (p1.size() < 3) p1 = {1, 1, 1};
        uint64_t h = Hash(uint64_t(dens.size()), dens[0], dens[dens.size() / 2], dens.back());
        fs::path vol = cacheDir_ / ("grid_" + std::to_string(h) + ".vol");
        FILE *f = _wfopen(vol.wstring().c_str(), L"wb");
        if (!f) throw std::runtime_error("cannot write " + PathUtf8(vol));
        std::fwrite("VOL", 1, 3, f);
        uint8_t version = 3;
        std::fwrite(&version, 1, 1, f);
        int32_t hdr[5] = {1, nx, ny, nz, 1};
        std::fwrite(hdr, 4, 5, f);
        float bb[6] = {float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])};
        std::fwrite(bb, 4, 6, f);
        for (double d : dens) {
            float v = float(d);
            std::fwrite(&v, 4, 1, f);
        }
        std::fclose(f);
        m["type"] = "grid";
        m["file"] = PathUtf8(vol);
        m["transform"] = {{"matrix", MatJson(xf)}};
    } else {
        Warn("medium type '" + type + "' is not supported; using homogeneous");
        m["type"] = "homogeneous";
    }
    doc_["media"][id] = m;
}

void PbrtImporter::EmitObject(json obj) {
    if (!currentObject_.empty()) {
        objectDefs_[currentObject_].push_back(std::move(obj));
        return;
    }
    doc_["objects"].push_back(std::move(obj));
}

void PbrtImporter::AddShape(const std::string &type, ParamSet &ps, const fs::path &dir, Tokenizer &tk) {
    std::string gid = "pbrt_shape:" + std::to_string(counter_++);
    json g;
    Mat4 xf = gs_.ctm;
    auto flatten = [](const std::vector<double> &v) {
        json a = json::array();
        for (double x : v) a.push_back(x);
        return a;
    };
    if (type == "sphere") {
        g = {{"type", "sphere"}, {"radius", ps.Float("radius", 1)}};
        if (ps.Find("zmin") || ps.Find("zmax") || ps.Find("phimax")) Warn("partial spheres are rendered as full spheres");
        float r = float(ps.Float("radius", 1));
        Vec3f c = MatPoint(xf, Vec3f());
        float s = Length(MatVector(xf, Vec3f(1, 0, 0)));
        bounds_ = Union(bounds_, Bounds3f(c - Vec3f(r * s, r * s, r * s), c + Vec3f(r * s, r * s, r * s)));
    } else if (type == "trianglemesh" || type == "bilinearmesh" || type == "loopsubdiv") {
        std::vector<double> P = ps.Floats("P");
        std::vector<double> idx = ps.Floats("indices");
        if (type == "loopsubdiv") Warn("loopsubdiv meshes are rendered without subdivision");
        if (type == "bilinearmesh") {
            // Bilinear patches (p00 p10 p01 p11) -> two triangles each.
            std::vector<double> tri;
            if (idx.empty())
                for (size_t i = 0; i < P.size() / 3; ++i) idx.push_back(double(i));
            for (size_t q = 0; q + 3 < idx.size(); q += 4) {
                double a = idx[q], b = idx[q + 1], c = idx[q + 2], d = idx[q + 3];
                tri.insert(tri.end(), {a, b, d, a, d, c});
            }
            idx = tri;
        }
        if (idx.empty() && P.size() == 9) idx = {0, 1, 2};
        g = {{"type", "mesh"}, {"positions", flatten(P)}, {"indices", flatten(idx)}};
        std::vector<double> N = ps.Floats("N"), uv = ps.Floats("uv");
        if (uv.empty()) uv = ps.Floats("st");
        if (!N.empty()) g["normals"] = flatten(N);
        if (!uv.empty()) g["uvs"] = flatten(uv);
        for (size_t i = 0; i + 2 < P.size(); i += 3)
            bounds_ = Union(bounds_, MatPoint(xf, Vec3f(float(P[i]), float(P[i + 1]), float(P[i + 2]))));
    } else if (type == "plymesh") {
        g = {{"type", "mesh"}, {"file", ResolvePath(dir, ps.String("filename", ""))}};
        if (ps.Find("displacement")) Warn("plymesh displacement is not supported");
        bounds_ = Union(bounds_, MatPoint(xf, Vec3f()));
    } else if (type == "disk") {
        g = {{"type", "disk"}, {"radius", ps.Float("radius", 1)}};
        // pbrt disks lie in z = height with normal +z; ours lie in y = 0 with normal +y.
        xf = MatMul(xf, MatMul(MatTranslate(0, 0, ps.Float("height", 0)), MatRotate(90, 1, 0, 0)));
        if (ps.Find("innerradius")) Warn("disk inner radius is not supported");
    } else {
        Warn("shape '" + type + "' is not supported (skipped)");
        (void)tk;
        return;
    }
    doc_["geometry"][gid] = g;
    json obj = {{"name", gid}, {"geometry", gid}, {"transform", {{"matrix", MatJson(xf)}}}};
    std::string mat = gs_.material;
    if (mat.empty()) {
        mat = "pbrt_default";
        if (!doc_["materials"].contains(mat)) doc_["materials"][mat] = {{"type", "diffuse"}, {"reflectance", 0.5}};
    }
    obj["material"] = mat;
    if (!gs_.areaLight.is_null()) obj["emission"] = gs_.areaLight;
    if (gs_.reverseOrientation) obj["flip_normals"] = true;
    if (!gs_.inside.empty() && doc_["media"].contains("pbrt_medium:" + gs_.inside)) {
        obj["interior"] = "pbrt_medium:" + gs_.inside;
        if (type == "trianglemesh" || type == "plymesh") g["closed"] = true, doc_["geometry"][gid]["closed"] = true;
    }
    if (!gs_.outside.empty() && doc_["media"].contains("pbrt_medium:" + gs_.outside)) obj["exterior"] = "pbrt_medium:" + gs_.outside;
    EmitObject(std::move(obj));
}

void PbrtImporter::AddLight(const std::string &type, ParamSet &ps, const fs::path &dir) {
    std::string id = "pbrt_light:" + std::to_string(counter_++);
    float scale = float(ps.Float("scale", 1)) * opts_.lightScale;
    json l;
    auto colorOf = [&](const char *name) {
        json c = SpectrumParam(ps, name, "D65", true, dir);
        if (c.is_object() && c.contains("blackbody")) return json{{"temperature", c["blackbody"]}};
        return json{{"color", c}};
    };
    const Mat4 &M = gs_.ctm;
    if (type == "point") {
        std::vector<double> from = ps.Floats("from");
        Vec3f p = from.size() >= 3 ? Vec3f(float(from[0]), float(from[1]), float(from[2])) : Vec3f();
        l = {{"type", "point"}, {"position", VecJ(MatPoint(M, p))}, {"intensity", scale}};
        l.update(colorOf("I"));
        if (ps.Find("power")) l["power"] = ps.Float("power", 1) * opts_.lightScale;
    } else if (type == "spot") {
        std::vector<double> from = ps.Floats("from"), to = ps.Floats("to");
        Vec3f f = from.size() >= 3 ? Vec3f(float(from[0]), float(from[1]), float(from[2])) : Vec3f();
        Vec3f t = to.size() >= 3 ? Vec3f(float(to[0]), float(to[1]), float(to[2])) : Vec3f(0, 0, 1);
        double cone = ps.Float("coneangle", 30), delta = ps.Float("conedeltaangle", 5);
        l = {{"type", "spot"}, {"position", VecJ(MatPoint(M, f))}, {"direction", VecJ(MatVector(M, t - f))},
             {"cone_angle", cone}, {"falloff_start", std::max(0.0, cone - delta)}, {"intensity", scale}};
        l.update(colorOf("I"));
    } else if (type == "distant") {
        std::vector<double> from = ps.Floats("from"), to = ps.Floats("to");
        Vec3f f = from.size() >= 3 ? Vec3f(float(from[0]), float(from[1]), float(from[2])) : Vec3f();
        Vec3f t = to.size() >= 3 ? Vec3f(float(to[0]), float(to[1]), float(to[2])) : Vec3f(0, 0, 1);
        // pbrt's light arrives from 'from' towards 'to'.
        l = {{"type", "distant"}, {"direction", VecJ(MatVector(M, t - f))}, {"irradiance", scale}};
        l.update(colorOf("L"));
    } else if (type == "infinite") {
        std::string file = ps.String("filename", "");
        if (file.empty()) {
            l = {{"type", "environment"}, {"intensity", scale}};
            l.update(colorOf("L"));
        } else {
            // pbrt-v4 environment maps use the equal-area octahedral parameterization in the
            // light's frame; resample into an equirectangular (y-up) map in world space.
            std::string src = ResolvePath(dir, file);
            Image img;
            std::string err;
            if (!ReadImage(src, TextureEncoding::Linear, &img, &err)) throw std::runtime_error(err);
            Mat4 lightFromWorld = MatInverse(M);
            int W = std::max(256, img.Width() * 2), H = W / 2;
            Image eq(W, H);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float th = Pi * (y + 0.5f) / H, ph = 2 * Pi * (x + 0.5f) / W;
                    Vec3f w(std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph));
                    Vec3f wl = Normalize(MatVector(lightFromWorld, w));
                    Vec2f uv = EqualAreaSphereToSquare(wl);
                    int px = Clamp(int(uv.x * img.Width()), 0, img.Width() - 1);
                    int py = Clamp(int(uv.y * img.Height()), 0, img.Height() - 1);
                    eq.Set(x, y, img.Get(px, py));
                }
            fs::path out = cacheDir_ / ("env_" + std::to_string(Hash(uint64_t(W), uint64_t(src.size()), float(M.m[0][0]),
                                                                     float(M.m[1][2]))) + ".pfm");
            if (!WritePFM(PathUtf8(out), eq, &err)) throw std::runtime_error(err);
            l = {{"type", "environment"}, {"file", PathUtf8(out)}, {"intensity", scale}};
        }
    } else if (type == "goniometric") {
        Warn("goniometric lights need an IES file in this renderer; skipped");
        return;
    } else {
        Warn("light '" + type + "' is not supported (skipped)");
        return;
    }
    doc_["lights"][id] = l;
}

void PbrtImporter::Parse(Tokenizer &tk, const fs::path &dir) {
    while (true) {
        Token t = tk.Next();
        if (t.kind == Token::End) break;
        if (t.kind != Token::Identifier) Error(tk, "expected a directive, got '" + t.text + "'");
        const std::string &d = t.text;
        auto nextString = [&]() {
            Token s = tk.Next();
            if (s.kind != Token::String) Error(tk, d + ": expected a quoted string");
            return s.text;
        };
        if (d == "AttributeBegin" || d == "TransformBegin") {
            stack_.push_back(gs_);
        } else if (d == "AttributeEnd" || d == "TransformEnd") {
            if (stack_.empty()) Error(tk, "unmatched " + d);
            if (d == "TransformEnd") gs_.ctm = stack_.back().ctm;  // only the transform is scoped
            else gs_ = stack_.back();
            stack_.pop_back();
        } else if (d == "Identity") {
            gs_.ctm = Mat4();
        } else if (d == "Translate") {
            auto v = Numbers(tk, 3);
            gs_.ctm = MatMul(gs_.ctm, MatTranslate(v[0], v[1], v[2]));
        } else if (d == "Scale") {
            auto v = Numbers(tk, 3);
            gs_.ctm = MatMul(gs_.ctm, MatScale(v[0], v[1], v[2]));
        } else if (d == "Rotate") {
            auto v = Numbers(tk, 4);
            gs_.ctm = MatMul(gs_.ctm, MatRotate(v[0], v[1], v[2], v[3]));
        } else if (d == "LookAt") {
            auto v = Numbers(tk, 9);
            gs_.ctm = MatMul(gs_.ctm, MatLookAt(Vec3f(float(v[0]), float(v[1]), float(v[2])), Vec3f(float(v[3]), float(v[4]), float(v[5])),
                                                Vec3f(float(v[6]), float(v[7]), float(v[8]))));
        } else if (d == "Transform") {
            gs_.ctm = MatFromPbrt(Numbers(tk, 16));
        } else if (d == "ConcatTransform") {
            gs_.ctm = MatMul(gs_.ctm, MatFromPbrt(Numbers(tk, 16)));
        } else if (d == "CoordinateSystem") {
            namedCS_[nextString()] = gs_.ctm;
        } else if (d == "CoordSysTransform") {
            std::string n = nextString();
            if (namedCS_.count(n)) gs_.ctm = namedCS_[n];
            else Warn("unknown coordinate system '" + n + "'");
        } else if (d == "ReverseOrientation") {
            gs_.reverseOrientation = !gs_.reverseOrientation;
        } else if (d == "Camera") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            if (type != "perspective") Warn("camera '" + type + "' imported as perspective");
            haveCamera_ = true;
            cameraFromWorld_ = gs_.ctm;
            namedCS_["camera"] = MatInverse(gs_.ctm);
            fov_ = ps.Float("fov", 90);
            lensRadius_ = ps.Float("lensradius", 0);
            focalDistance_ = ps.Float("focaldistance", 1e6);
        } else if (d == "Film") {
            nextString();
            ParamSet ps = ParseParams(tk);
            xres_ = ps.Int("xresolution", 1280);
            yres_ = ps.Int("yresolution", 720);
            filmName_ = ps.String("filename", "");
        } else if (d == "Sampler") {
            nextString();
            ParamSet ps = ParseParams(tk);
            doc_["render"]["integrator"]["spp"] = ps.Int("pixelsamples", 16);
        } else if (d == "Integrator") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            doc_["render"]["integrator"]["max_depth"] = ps.Int("maxdepth", 5);
            doc_["render"]["integrator"]["type"] = type == "bdpt" ? "bdpt" : (type == "mlt" ? "mmlt" : "path");
        } else if (d == "PixelFilter") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            doc_["render"]["film"]["filter"] = type == "box" ? "box" : (type == "gaussian" ? "gaussian" : "blackman_harris");
        } else if (d == "ColorSpace") {
            nextString();
        } else if (d == "Option") {
            ParseParams(tk);
        } else if (d == "Accelerator" || d == "Attribute") {
            nextString();
            ParseParams(tk);
            if (d == "Attribute") Warn("Attribute defaults are ignored");
        } else if (d == "TransformTimes") {
            Numbers(tk, 2);
        } else if (d == "ActiveTransform") {
            Token which = tk.Next();
            if (which.text != "All") Warn("motion blur (ActiveTransform) is not supported");
        } else if (d == "WorldBegin") {
            inWorld_ = true;
            gs_.ctm = Mat4();
            namedCS_["world"] = Mat4();
        } else if (d == "WorldEnd") {
            // pbrt-v3 compatibility.
        } else if (d == "Texture") {
            std::string name = nextString(), kind = nextString(), cls = nextString();
            ParamSet ps = ParseParams(tk);
            MakeTexture(name, kind, cls, ps, dir);
        } else if (d == "Material" || d == "MakeNamedMaterial") {
            std::string name = d == "MakeNamedMaterial" ? nextString() : "";
            std::string type;
            ParamSet ps;
            if (d == "Material") {
                type = nextString();
                ps = ParseParams(tk);
            } else {
                ps = ParseParams(tk);
                type = ps.String("type", "diffuse");
            }
            json m = MakeMaterial(type, ps, dir);
            std::string id;
            if (m.contains("__ref")) id = m["__ref"].get<std::string>();
            else {
                id = name.empty() ? "pbrt_mat:" + std::to_string(counter_++) : "pbrt_mat:" + name;
                doc_["materials"][id] = m;
            }
            if (d == "Material") gs_.material = id;
            else namedMaterials_[name] = id;
        } else if (d == "NamedMaterial") {
            std::string name = nextString();
            if (namedMaterials_.count(name)) gs_.material = namedMaterials_[name];
            else Warn("unknown named material '" + name + "'");
        } else if (d == "LightSource") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            AddLight(type, ps, dir);
        } else if (d == "AreaLightSource") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            json e = {{"luminance", ps.Float("scale", 1) * opts_.emissionScale}, {"two_sided", ps.Bool("twosided", false)}};
            json L = SpectrumParam(ps, "L", "D65", true, dir);
            if (L.is_object() && L.contains("blackbody")) e["temperature"] = L["blackbody"];
            else e["color"] = L;
            if (ps.Find("power")) e["power"] = ps.Float("power", 1) * opts_.emissionScale;
            if (ps.Find("filename")) Warn("image area lights are not supported");
            gs_.areaLight = e;
        } else if (d == "Shape") {
            std::string type = nextString();
            ParamSet ps = ParseParams(tk);
            AddShape(type, ps, dir, tk);
        } else if (d == "MakeNamedMedium") {
            std::string name = nextString();
            ParamSet ps = ParseParams(tk);
            MakeMedium(name, ps, dir);
        } else if (d == "MediumInterface") {
            std::string inside = nextString();
            std::string outside = inside;
            if (tk.Peek().kind == Token::String) outside = nextString();
            gs_.inside = inside;
            gs_.outside = outside;
        } else if (d == "ObjectBegin") {
            stack_.push_back(gs_);
            currentObject_ = nextString();
            objectDefs_[currentObject_] = json::array();
        } else if (d == "ObjectEnd") {
            currentObject_.clear();
            if (!stack_.empty()) {
                gs_ = stack_.back();
                stack_.pop_back();
            }
        } else if (d == "ObjectInstance") {
            std::string name = nextString();
            auto it = objectDefs_.find(name);
            if (it == objectDefs_.end()) {
                Warn("unknown object '" + name + "'");
                continue;
            }
            for (const json &o : it->second) {
                json inst = o;
                Mat4 m;
                const json &a = o["transform"]["matrix"];
                for (int i = 0; i < 16; ++i) m.m[i / 4][i % 4] = a[i].get<float>();
                inst["transform"] = {{"matrix", MatJson(MatMul(gs_.ctm, m))}};
                inst["name"] = o["name"].get<std::string>() + "@" + std::to_string(counter_++);
                doc_["objects"].push_back(inst);
            }
        } else if (d == "Include" || d == "Import") {
            fs::path p = Utf8Path(ResolvePath(dir, nextString()));
            std::string ext = p.extension().string();
            if (ext == ".gz") throw std::runtime_error("gzipped pbrt files are not supported: " + PathUtf8(p));
            ParseFile(p);
        } else {
            Error(tk, "unknown directive '" + d + "'");
        }
    }
}

json PbrtImporter::Finish() {
    if (haveCamera_) {
        Mat4 worldFromCamera = MatInverse(cameraFromWorld_);
        // pbrt's fov applies to the shorter image axis.
        double fovY = fov_;
        if (xres_ < yres_) fovY = Degrees(float(2 * std::atan(std::tan(Radians(float(fov_)) / 2) * double(yres_) / xres_)));
        Vec3f p = MatPoint(worldFromCamera, Vec3f()), f = Normalize(MatVector(worldFromCamera, Vec3f(0, 0, 1)));
        Vec3f up = Normalize(MatVector(worldFromCamera, Vec3f(0, 1, 0)));
        json cam = {{"type", lensRadius_ > 0 ? "thin_lens" : "perspective"}, {"matrix", MatJson(worldFromCamera)},
                    {"position", VecJ(p)}, {"look_at", VecJ(p + f)}, {"up", VecJ(up)}, {"fov_y", fovY}};
        if (lensRadius_ > 0) {
            cam["aperture_radius"] = lensRadius_;
            cam["focus_distance"] = focalDistance_;
        }
        doc_["cameras"]["pbrt"] = cam;
        doc_["render"]["camera"] = "pbrt";
    }
    doc_["render"]["film"]["width"] = xres_;
    doc_["render"]["film"]["height"] = yres_;
    if (!filmName_.empty()) {
        std::string stem = PathUtf8(Utf8Path(filmName_).stem());
        doc_["render"]["output"] = json::array({{{"file", stem + ".exr"}}, {{"file", stem + ".png"}}});
    }
    doc_["import_bounds"] = {VecJ(bounds_.pMin), VecJ(bounds_.pMax)};
    return doc_;
}

} // namespace

json ImportPBRT(const std::string &path, const ImportOptions &opts, std::string *err) {
    try {
        PbrtImporter imp(opts);
        imp.ParseFile(Utf8Path(path));
        return imp.Finish();
    } catch (const std::exception &e) {
        *err = e.what();
        return json();
    }
}

} // namespace pr

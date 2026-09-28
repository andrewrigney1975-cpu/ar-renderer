#include "geometry/shape.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/fsutil.h"
#include <fstream>
#include <map>
#include <sstream>
#include <tuple>

namespace pr {

namespace {

std::string Lower(std::string s) {
    for (char &c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

std::shared_ptr<TriangleMesh> LoadOBJ(const std::string &path, std::string *err) {
    std::ifstream in(Utf8Path(path), std::ios::binary);
    if (!in) {
        *err = "cannot open " + path;
        return nullptr;
    }
    std::vector<Vec3f> P, N;
    std::vector<Vec2f> T;
    auto mesh = std::make_shared<TriangleMesh>();
    std::map<std::tuple<int, int, int>, int> remap;
    bool anyNormals = false, anyUVs = false;
    std::string line;
    auto resolve = [](int idx, size_t count) { return idx < 0 ? int(count) + idx : idx - 1; };
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            Vec3f v;
            ss >> v.x >> v.y >> v.z;
            P.push_back(v);
        } else if (tag == "vn") {
            Vec3f v;
            ss >> v.x >> v.y >> v.z;
            N.push_back(v);
        } else if (tag == "vt") {
            Vec2f v;
            ss >> v.x >> v.y;
            T.push_back(v);
        } else if (tag == "f") {
            std::vector<int> face;
            std::string tok;
            while (ss >> tok) {
                int vi = 0, ti = 0, ni = 0;
                const char *s = tok.c_str();
                vi = std::atoi(s);
                const char *slash = std::strchr(s, '/');
                if (slash) {
                    if (slash[1] != '/') ti = std::atoi(slash + 1);
                    const char *slash2 = std::strchr(slash + 1, '/');
                    if (slash2) ni = std::atoi(slash2 + 1);
                }
                int pv = resolve(vi, P.size());
                int pt = ti ? resolve(ti, T.size()) : -1;
                int pn = ni ? resolve(ni, N.size()) : -1;
                if (pv < 0 || pv >= int(P.size())) {
                    *err = "OBJ: bad vertex index in " + path;
                    return nullptr;
                }
                auto key = std::make_tuple(pv, pt, pn);
                auto it = remap.find(key);
                int idx;
                if (it == remap.end()) {
                    idx = int(mesh->p.size());
                    remap[key] = idx;
                    mesh->p.push_back(P[pv]);
                    mesh->uv.push_back(pt >= 0 && pt < int(T.size()) ? T[pt] : Vec2f(0, 0));
                    mesh->n.push_back(pn >= 0 && pn < int(N.size()) ? Normalize(N[pn]) : Vec3f());
                    anyUVs |= pt >= 0;
                    anyNormals |= pn >= 0;
                } else {
                    idx = it->second;
                }
                face.push_back(idx);
            }
            for (size_t i = 1; i + 1 < face.size(); ++i)
                mesh->indices.insert(mesh->indices.end(), {face[0], face[i], face[i + 1]});
        }
    }
    if (!anyUVs) mesh->uv.clear();
    if (!anyNormals) mesh->n.clear();
    else {
        // Vertices without normals: fall back to no shading normals at all.
        for (const auto &n : mesh->n)
            if (n.IsZero()) {
                mesh->n.clear();
                break;
            }
    }
    if (mesh->indices.empty()) {
        *err = "OBJ: no faces in " + path;
        return nullptr;
    }
    return mesh;
}

// PLY: ascii or binary_little_endian; vertex x,y,z [nx,ny,nz] [u,v|s,t]; face vertex_indices list.
std::shared_ptr<TriangleMesh> LoadPLY(const std::string &path, std::string *err) {
    std::ifstream in(Utf8Path(path), std::ios::binary);
    if (!in) {
        *err = "cannot open " + path;
        return nullptr;
    }
    struct Prop {
        std::string name, type, countType;
        bool isList = false;
    };
    struct Element {
        std::string name;
        size_t count = 0;
        std::vector<Prop> props;
    };
    std::vector<Element> elements;
    std::string line, format;
    std::getline(in, line);
    if (line.rfind("ply", 0) != 0) {
        *err = "not a PLY file: " + path;
        return nullptr;
    }
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "format") ss >> format;
        else if (tag == "element") {
            Element e;
            ss >> e.name >> e.count;
            elements.push_back(e);
        } else if (tag == "property" && !elements.empty()) {
            Prop p;
            std::string t;
            ss >> t;
            if (t == "list") {
                p.isList = true;
                ss >> p.countType >> p.type >> p.name;
            } else {
                p.type = t;
                ss >> p.name;
            }
            elements.back().props.push_back(p);
        } else if (tag == "end_header") break;
    }
    bool ascii = format == "ascii";
    if (!ascii && format != "binary_little_endian") {
        *err = "unsupported PLY format '" + format + "'";
        return nullptr;
    }
    auto typeSize = [](const std::string &t) -> int {
        if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
        if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
        if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
        if (t == "double" || t == "float64") return 8;
        return 0;
    };
    auto readValue = [&](const std::string &t) -> double {
        if (ascii) {
            double v;
            in >> v;
            return v;
        }
        unsigned char buf[8];
        int sz = typeSize(t);
        in.read(reinterpret_cast<char *>(buf), sz);
        if (t == "char" || t == "int8") return double(int8_t(buf[0]));
        if (t == "uchar" || t == "uint8") return double(buf[0]);
        if (t == "short" || t == "int16") { int16_t v; std::memcpy(&v, buf, 2); return v; }
        if (t == "ushort" || t == "uint16") { uint16_t v; std::memcpy(&v, buf, 2); return v; }
        if (t == "int" || t == "int32") { int32_t v; std::memcpy(&v, buf, 4); return v; }
        if (t == "uint" || t == "uint32") { uint32_t v; std::memcpy(&v, buf, 4); return v; }
        if (t == "float" || t == "float32") { float v; std::memcpy(&v, buf, 4); return v; }
        double v;
        std::memcpy(&v, buf, 8);
        return v;
    };
    auto mesh = std::make_shared<TriangleMesh>();
    bool hasN = false, hasUV = false;
    for (const Element &e : elements) {
        if (e.name == "vertex") {
            for (const Prop &p : e.props) {
                hasN |= p.name == "nx";
                hasUV |= p.name == "u" || p.name == "s" || p.name == "texture_u";
            }
            mesh->p.resize(e.count);
            if (hasN) mesh->n.resize(e.count);
            if (hasUV) mesh->uv.resize(e.count);
            for (size_t i = 0; i < e.count; ++i) {
                for (const Prop &p : e.props) {
                    if (p.isList) {
                        int n = int(readValue(p.countType));
                        for (int k = 0; k < n; ++k) readValue(p.type);
                        continue;
                    }
                    float v = float(readValue(p.type));
                    if (p.name == "x") mesh->p[i].x = v;
                    else if (p.name == "y") mesh->p[i].y = v;
                    else if (p.name == "z") mesh->p[i].z = v;
                    else if (p.name == "nx") mesh->n[i].x = v;
                    else if (p.name == "ny") mesh->n[i].y = v;
                    else if (p.name == "nz") mesh->n[i].z = v;
                    else if (p.name == "u" || p.name == "s" || p.name == "texture_u") mesh->uv[i].x = v;
                    else if (p.name == "v" || p.name == "t" || p.name == "texture_v") mesh->uv[i].y = v;
                }
            }
        } else if (e.name == "face") {
            for (size_t i = 0; i < e.count; ++i) {
                for (const Prop &p : e.props) {
                    if (!p.isList) {
                        readValue(p.type);
                        continue;
                    }
                    int n = int(readValue(p.countType));
                    std::vector<int> face(n);
                    for (int k = 0; k < n; ++k) face[k] = int(readValue(p.type));
                    if (p.name == "vertex_indices" || p.name == "vertex_index")
                        for (int k = 1; k + 1 < n; ++k)
                            mesh->indices.insert(mesh->indices.end(), {face[0], face[k], face[k + 1]});
                }
            }
        } else {
            for (size_t i = 0; i < e.count; ++i)
                for (const Prop &p : e.props) {
                    if (p.isList) {
                        int n = int(readValue(p.countType));
                        for (int k = 0; k < n; ++k) readValue(p.type);
                    } else readValue(p.type);
                }
        }
        if (!in) {
            *err = "PLY: truncated file " + path;
            return nullptr;
        }
    }
    for (auto &n : mesh->n) {
        float l = Length(n);
        if (l > 0) n = n / l;
    }
    for (int idx : mesh->indices)
        if (idx < 0 || idx >= int(mesh->p.size())) {
            *err = "PLY: bad vertex index in " + path;
            return nullptr;
        }
    return mesh;
}

} // namespace

std::shared_ptr<TriangleMesh> LoadMesh(const std::string &path, std::string *err) {
    std::string ext = Lower(std::filesystem::path(path).extension().string());
    if (ext == ".obj") return LoadOBJ(path, err);
    if (ext == ".ply") return LoadPLY(path, err);
    *err = "unsupported mesh format: " + ext;
    return nullptr;
}

} // namespace pr

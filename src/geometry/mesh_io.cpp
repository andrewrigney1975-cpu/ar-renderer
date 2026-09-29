#include "geometry/shape.h"

#include "core/log.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/fsutil.h"
#include <fstream>
#include <map>
#include <sstream>
#include <charconv>
#include <tuple>
#include <unordered_map>

#include "core/rng.h"

namespace pr {

namespace {

std::string Lower(std::string s) {
    for (char &c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

// Fast OBJ parser: the whole file is read at once and scanned with std::from_chars (no streams,
// no per-line allocations). Vertex/uv/normal triples are deduplicated with a hash map.
std::shared_ptr<TriangleMesh> LoadOBJ(const std::string &path, std::string *err) {
    std::string text;
    {
        std::ifstream in(Utf8Path(path), std::ios::binary);
        if (!in) {
            *err = "cannot open " + path;
            return nullptr;
        }
        in.seekg(0, std::ios::end);
        text.resize(size_t(in.tellg()));
        in.seekg(0);
        in.read(text.data(), std::streamsize(text.size()));
    }
    std::vector<Vec3f> P, N;
    std::vector<Vec2f> T;
    auto mesh = std::make_shared<TriangleMesh>();
    struct KeyHash {
        size_t operator()(const std::tuple<int, int, int> &k) const {
            return size_t(RNG::MixBits(uint64_t(uint32_t(std::get<0>(k))) * 0x9E3779B97F4A7C15ull ^
                                       (uint64_t(uint32_t(std::get<1>(k))) << 21) ^ uint64_t(uint32_t(std::get<2>(k)))));
        }
    };
    std::unordered_map<std::tuple<int, int, int>, int, KeyHash> remap;
    bool anyNormals = false, anyUVs = false;
    auto resolve = [](int idx, size_t count) { return idx < 0 ? int(count) + idx : idx - 1; };
    const char *c = text.data(), *end = c + text.size();
    auto skipSpace = [&] {
        while (c < end && (*c == ' ' || *c == '\t' || *c == '\r')) ++c;
    };
    auto readFloat = [&](float *v) {
        skipSpace();
        auto r = std::from_chars(c, end, *v);
        if (r.ec != std::errc()) *v = 0;
        c = r.ptr;
    };
    auto readInt = [&](int *v) {
        auto r = std::from_chars(c, end, *v);
        if (r.ec != std::errc()) {
            *v = 0;
            return false;
        }
        c = r.ptr;
        return true;
    };
    std::vector<int> face;
    while (c < end) {
        skipSpace();
        const char *lineStart = c;
        if (c + 1 < end && c[0] == 'v' && (c[1] == ' ' || c[1] == '\t')) {
            c += 1;
            Vec3f v;
            readFloat(&v.x);
            readFloat(&v.y);
            readFloat(&v.z);
            P.push_back(v);
        } else if (c + 2 < end && c[0] == 'v' && c[1] == 'n' && (c[2] == ' ' || c[2] == '\t')) {
            c += 2;
            Vec3f v;
            readFloat(&v.x);
            readFloat(&v.y);
            readFloat(&v.z);
            N.push_back(v);
        } else if (c + 2 < end && c[0] == 'v' && c[1] == 't' && (c[2] == ' ' || c[2] == '\t')) {
            c += 2;
            Vec2f v;
            readFloat(&v.x);
            readFloat(&v.y);
            T.push_back(v);
        } else if (c + 1 < end && c[0] == 'f' && (c[1] == ' ' || c[1] == '\t')) {
            c += 1;
            face.clear();
            while (true) {
                skipSpace();
                if (c >= end || *c == '\n' || *c == '#') break;
                int vi = 0, ti = 0, ni = 0;
                if (!readInt(&vi)) {
                    *err = "OBJ: malformed face in " + path;
                    return nullptr;
                }
                if (c < end && *c == '/') {
                    ++c;
                    if (c < end && *c != '/') readInt(&ti);
                    if (c < end && *c == '/') {
                        ++c;
                        readInt(&ni);
                    }
                }
                int pv = resolve(vi, P.size());
                int pt = ti ? resolve(ti, T.size()) : -1;
                int pn = ni ? resolve(ni, N.size()) : -1;
                if (pv < 0 || pv >= int(P.size())) {
                    *err = "OBJ: bad vertex index in " + path;
                    return nullptr;
                }
                auto [it, inserted] = remap.try_emplace(std::make_tuple(pv, pt, pn), int(mesh->p.size()));
                if (inserted) {
                    mesh->p.push_back(P[pv]);
                    mesh->uv.push_back(pt >= 0 && pt < int(T.size()) ? T[pt] : Vec2f(0, 0));
                    mesh->n.push_back(pn >= 0 && pn < int(N.size()) ? Normalize(N[pn]) : Vec3f());
                    anyUVs |= pt >= 0;
                    anyNormals |= pn >= 0;
                }
                face.push_back(it->second);
            }
            for (size_t i = 1; i + 1 < face.size(); ++i)
                mesh->indices.insert(mesh->indices.end(), {face[0], face[i], face[i + 1]});
        }
        (void)lineStart;
        while (c < end && *c != '\n') ++c;  // skip the rest of the line (comments, unknown tags)
        if (c < end) ++c;
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
    auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        std::chrono::steady_clock::time_point t;
        std::string p;
        ~Timer() { LogVerbose("load: mesh {} parsed in {:.3f}s", p, std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count()); }
    } timer{t0, path};
    std::string ext = Lower(std::filesystem::path(path).extension().string());
    if (ext == ".obj") return LoadOBJ(path, err);
    if (ext == ".ply") return LoadPLY(path, err);
    *err = "unsupported mesh format: " + ext;
    return nullptr;
}

} // namespace pr

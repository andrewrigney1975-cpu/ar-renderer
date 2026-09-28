// glTF 2.0 importer (cgltf).

#include "scene/importers.h"

#include "core/fsutil.h"
#include "core/log.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <cstring>
#include <functional>
#include <map>
#include <mutex>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace pr {

namespace {

struct GltfFile {
    cgltf_data *data = nullptr;
    ~GltfFile() {
        if (data) cgltf_free(data);
    }
};

std::mutex gGltfMutex;
std::map<std::string, std::shared_ptr<GltfFile>> gGltfCache;

std::shared_ptr<GltfFile> OpenGLTF(const std::string &path, std::string *err) {
    std::lock_guard<std::mutex> lock(gGltfMutex);
    auto it = gGltfCache.find(path);
    if (it != gGltfCache.end()) return it->second;
    cgltf_options options{};
    auto f = std::make_shared<GltfFile>();
    cgltf_result r = cgltf_parse_file(&options, path.c_str(), &f->data);
    if (r != cgltf_result_success) {
        *err = "cannot parse glTF " + path + " (cgltf error " + std::to_string(int(r)) + ")";
        return nullptr;
    }
    r = cgltf_load_buffers(&options, f->data, path.c_str());
    if (r != cgltf_result_success) {
        *err = "cannot load buffers of " + path + " (cgltf error " + std::to_string(int(r)) + ")";
        return nullptr;
    }
    gGltfCache[path] = f;
    return f;
}

std::vector<unsigned char> Base64Decode(const char *s, size_t n) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    std::vector<unsigned char> out;
    out.reserve(n * 3 / 4);
    unsigned buf = 0;
    int bits = 0;
    for (size_t i = 0; i < n; ++i) {
        int v = val(s[i]);
        if (v < 0) continue;
        buf = (buf << 6) | unsigned(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((unsigned char)((buf >> bits) & 0xff));
        }
    }
    return out;
}

json Rgb(const float *c) { return json::array({c[0], c[1], c[2]}); }
json Vec(const Vec3f &v) { return json::array({v.x, v.y, v.z}); }

Vec3f XformPoint(const float *m, const Vec3f &p) {  // column-major 4x4
    return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12], m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
}
Vec3f XformVector(const float *m, const Vec3f &v) {
    return {m[0] * v.x + m[4] * v.y + m[8] * v.z, m[1] * v.x + m[5] * v.y + m[9] * v.z, m[2] * v.x + m[6] * v.y + m[10] * v.z};
}

std::string Sanitize(const char *name) {
    std::string s = name ? name : "";
    for (char &c : s)
        if (c == '"' || c == '\\' || std::iscntrl((unsigned char)c)) c = '_';
    return s;
}

} // namespace

std::shared_ptr<TriangleMesh> LoadGLTFPrimitive(const std::string &path, int meshIndex, int primIndex, std::string *err) {
    auto f = OpenGLTF(path, err);
    if (!f) return nullptr;
    cgltf_data *d = f->data;
    if (meshIndex < 0 || size_t(meshIndex) >= d->meshes_count ||
        primIndex < 0 || size_t(primIndex) >= d->meshes[meshIndex].primitives_count) {
        *err = path + ": bad mesh/primitive index";
        return nullptr;
    }
    const cgltf_primitive &prim = d->meshes[meshIndex].primitives[primIndex];
    if (prim.has_draco_mesh_compression) {
        *err = path + ": Draco-compressed meshes are not supported";
        return nullptr;
    }
    const cgltf_accessor *pos = nullptr, *nrm = nullptr, *uv = nullptr;
    for (size_t a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute &at = prim.attributes[a];
        if (at.type == cgltf_attribute_type_position) pos = at.data;
        else if (at.type == cgltf_attribute_type_normal) nrm = at.data;
        else if (at.type == cgltf_attribute_type_texcoord && at.index == 0) uv = at.data;
    }
    if (!pos) {
        *err = path + ": primitive has no POSITION";
        return nullptr;
    }
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->p.resize(pos->count);
    for (size_t i = 0; i < pos->count; ++i) cgltf_accessor_read_float(pos, i, &mesh->p[i].x, 3);
    if (nrm && nrm->count == pos->count) {
        mesh->n.resize(nrm->count);
        for (size_t i = 0; i < nrm->count; ++i) {
            cgltf_accessor_read_float(nrm, i, &mesh->n[i].x, 3);
            float l = Length(mesh->n[i]);
            mesh->n[i] = l > 0 ? mesh->n[i] / l : Vec3f(0, 1, 0);
        }
    }
    if (uv && uv->count == pos->count) {
        mesh->uv.resize(uv->count);
        for (size_t i = 0; i < uv->count; ++i) cgltf_accessor_read_float(uv, i, &mesh->uv[i].x, 2);
    }
    if (prim.indices) {
        mesh->indices.resize(prim.indices->count);
        for (size_t i = 0; i < prim.indices->count; ++i) mesh->indices[i] = int(cgltf_accessor_read_index(prim.indices, i));
    } else {
        mesh->indices.resize(pos->count);
        for (size_t i = 0; i < pos->count; ++i) mesh->indices[i] = int(i);
    }
    mesh->indices.resize(mesh->indices.size() / 3 * 3);
    for (int idx : mesh->indices)
        if (idx < 0 || size_t(idx) >= mesh->p.size()) {
            *err = path + ": index out of range";
            return nullptr;
        }
    return mesh;
}

bool LoadGLTFImage(const std::string &path, int imageIndex, TextureEncoding enc, Image *out, std::string *err) {
    auto f = OpenGLTF(path, err);
    if (!f) return false;
    cgltf_data *d = f->data;
    if (imageIndex < 0 || size_t(imageIndex) >= d->images_count) {
        *err = path + ": bad image index";
        return false;
    }
    const cgltf_image &im = d->images[imageIndex];
    if (im.buffer_view && im.buffer_view->buffer && im.buffer_view->buffer->data) {
        const unsigned char *p = static_cast<const unsigned char *>(im.buffer_view->buffer->data) + im.buffer_view->offset;
        return ReadImageFromMemory(p, im.buffer_view->size, enc, out, err);
    }
    if (im.uri && std::strncmp(im.uri, "data:", 5) == 0) {
        const char *comma = std::strchr(im.uri, ',');
        if (!comma) {
            *err = path + ": malformed data URI";
            return false;
        }
        auto bytes = Base64Decode(comma + 1, std::strlen(comma + 1));
        return ReadImageFromMemory(bytes.data(), bytes.size(), enc, out, err);
    }
    *err = path + ": image " + std::to_string(imageIndex) + " has no embedded data";
    return false;
}

json ImportGLTF(const std::string &path, const ImportOptions &opts, std::string *err) {
    auto f = OpenGLTF(path, err);
    if (!f) return json();
    cgltf_data *d = f->data;
    const fs::path dir = Utf8Path(path).parent_path();
    const std::string prefix = PathUtf8(Utf8Path(path).stem()) + "/";
    json doc = {{"textures", json::object()}, {"materials", json::object()}, {"media", json::object()},
                {"geometry", json::object()}, {"objects", json::array()},  {"lights", json::object()},
                {"cameras", json::object()}};

    // Textures are keyed by image, colour space and multiplier (glTF factors multiply textures).
    auto texture = [&](const cgltf_texture_view &tv, bool srgb, const json &multiplier) -> std::string {
        if (!tv.texture || !tv.texture->image) return "";
        if (tv.texcoord != 0) LogWarning("{}: only TEXCOORD_0 is supported", path);
        int img = int(tv.texture->image - d->images);
        std::string id = prefix + "img" + std::to_string(img) + (srgb ? "" : "_lin") + "*" + multiplier.dump();
        if (doc["textures"].contains(id)) return id;
        json t = {{"type", "image"}, {"colorspace", srgb ? "srgb" : "linear"}, {"flip_v", false}, {"multiplier", multiplier}};
        const cgltf_image *im = tv.texture->image;
        if (im->uri && std::strncmp(im->uri, "data:", 5) != 0) {
            std::string uri = im->uri;
            cgltf_decode_uri(uri.data());
            uri.resize(std::strlen(uri.c_str()));
            t["file"] = PathUtf8((dir / Utf8Path(uri)).lexically_normal());
        } else {
            t["gltf"] = path;
            t["image"] = img;
        }
        if (const cgltf_sampler *s = tv.texture->sampler) {
            t["wrap"] = s->wrap_s == 33071 ? "clamp" : (s->wrap_s == 33648 ? "mirror" : "repeat");
        }
        doc["textures"][id] = t;
        return id;
    };
    auto param = [&](const cgltf_texture_view &tv, bool srgb, const json &constant, const json &multiplier) -> json {
        std::string id = texture(tv, srgb, multiplier);
        if (id.empty()) return constant;
        return json{{"texture", id}};
    };

    auto materialId = [&](const cgltf_material *m) -> std::string {
        if (!m) {
            std::string id = prefix + "default";
            if (!doc["materials"].contains(id))
                doc["materials"][id] = {{"type", "coated_diffuse"}, {"reflectance", 0.8}, {"coat_roughness", 0.5}, {"thickness", 0.001}};
            return id;
        }
        int mi = int(m - d->materials);
        std::string id = prefix + "mat" + std::to_string(mi) + (m->name ? ":" + Sanitize(m->name) : "");
        if (doc["materials"].contains(id)) return id;
        const cgltf_pbr_metallic_roughness &pbr = m->pbr_metallic_roughness;
        const float *bc = pbr.base_color_factor;
        float metallic = m->has_pbr_metallic_roughness ? pbr.metallic_factor : 0.f;
        float rough = m->has_pbr_metallic_roughness ? pbr.roughness_factor : 0.5f;
        float ior = m->has_ior ? m->ior.ior : 1.5f;
        json baseColor = param(pbr.base_color_texture, true, Rgb(bc), Rgb(bc));
        json roughness = param(pbr.metallic_roughness_texture, false, rough, rough);
        if (roughness.is_object()) roughness["channel"] = "g";
        json mat;
        if (m->has_transmission && m->transmission.transmission_factor >= 0.5f) {
            mat = {{"type", "dielectric"}, {"roughness", roughness}};
            if (m->has_dispersion && m->dispersion.dispersion > 0) {
                // KHR_materials_dispersion: dispersion = 20 / Abbe number; fit a Cauchy law.
                float V = 20.f / m->dispersion.dispersion;
                float B = (ior - 1) / (V * (1 / (486.1f * 486.1f) - 1 / (656.3f * 656.3f)));
                mat["ior"] = {{"cauchy", json::array({ior - B / (587.6f * 587.6f), B})}};
            } else {
                mat["ior"] = ior;
            }
            bool thick = m->has_volume && m->volume.thickness_factor > 0;
            if (!thick) mat["thin"] = true;
            const float *ac = m->volume.attenuation_color;
            float dist = m->volume.attenuation_distance;
            if (thick && std::isfinite(dist) && dist > 0 && (ac[0] < 1 || ac[1] < 1 || ac[2] < 1)) {
                std::string med = id + ":volume";
                json sa = json::array();
                for (int c = 0; c < 3; ++c) sa.push_back(-std::log(std::max(ac[c], 1e-4f)) / dist);
                doc["media"][med] = {{"type", "homogeneous"}, {"sigma_a", sa}, {"sigma_s", 0}};
                mat["medium"] = med;
            }
        } else if (metallic >= 0.5f) {
            bool coat = m->has_clearcoat && m->clearcoat.clearcoat_factor >= 0.5f;
            mat = {{"type", coat ? "coated_conductor" : "conductor"}, {"reflectance", baseColor}, {"roughness", roughness}};
            if (coat) {
                mat["coat_roughness"] = m->clearcoat.clearcoat_roughness_factor;
                mat["thickness"] = 0.001;
            }
        } else {
            mat = {{"type", "coated_diffuse"}, {"reflectance", baseColor}, {"coat_roughness", roughness}, {"coat_ior", ior},
                   {"thickness", 0.001}};
        }
        if (m->has_sheen) {
            const float *sc = m->sheen.sheen_color_factor;
            if (sc[0] > 0 || sc[1] > 0 || sc[2] > 0)
                mat["sheen"] = {{"color", Rgb(sc)}, {"roughness", std::max(0.05f, m->sheen.sheen_roughness_factor)}, {"weight", 1.0}};
        }
        const float *ef = m->emissive_factor;
        float strength = (m->has_emissive_strength ? m->emissive_strength.emissive_strength : 1.f) * opts.emissionScale;
        if ((ef[0] > 0 || ef[1] > 0 || ef[2] > 0) && strength > 0) {
            mat["emission"] = {{"color", json::array({ef[0] * strength, ef[1] * strength, ef[2] * strength})}, {"luminance", 1.0}};
            if (m->emissive_texture.texture) LogWarning("{}: emissive textures are approximated by their factor", path);
        }
        if (m->normal_texture.texture) {
            std::string nt = texture(m->normal_texture, false, 1.0);
            // glTF normal maps: +Y is image-up, i.e. towards decreasing v.
            mat["normal_map"] = {{"texture", nt}, {"strength", m->normal_texture.scale}, {"flip_green", true}};
        }
        doc["materials"][id] = mat;
        return id;
    };

    Bounds3f worldBounds;
    int unnamed = 0;
    std::function<void(const cgltf_node *)> visit = [&](const cgltf_node *node) {
        float M[16];
        cgltf_node_transform_world(node, M);
        std::string nodeName = node->name ? Sanitize(node->name) : "node" + std::to_string(unnamed++);
        if (node->mesh) {
            int mi = int(node->mesh - d->meshes);
            for (size_t pi = 0; pi < node->mesh->primitives_count; ++pi) {
                const cgltf_primitive &prim = node->mesh->primitives[pi];
                if (prim.type != cgltf_primitive_type_triangles) continue;
                std::string gid = prefix + "mesh" + std::to_string(mi) + "_" + std::to_string(pi);
                doc["geometry"][gid] = {{"type", "gltf"}, {"file", path}, {"mesh", mi}, {"primitive", int(pi)}};
                json matrix = json::array({M[0], M[4], M[8], M[12], M[1], M[5], M[9], M[13], M[2], M[6], M[10], M[14],
                                           M[3], M[7], M[11], M[15]});
                doc["objects"].push_back({{"name", prefix + nodeName + (node->mesh->primitives_count > 1 ? "#" + std::to_string(pi) : "")},
                                          {"geometry", gid},
                                          {"material", materialId(prim.material)},
                                          {"transform", {{"matrix", matrix}}}});
                for (size_t a = 0; a < prim.attributes_count; ++a)
                    if (prim.attributes[a].type == cgltf_attribute_type_position && prim.attributes[a].data->has_min &&
                        prim.attributes[a].data->has_max) {
                        const float *mn = prim.attributes[a].data->min, *mx = prim.attributes[a].data->max;
                        for (int c = 0; c < 8; ++c)
                            worldBounds = Union(worldBounds, XformPoint(M, Vec3f(c & 1 ? mx[0] : mn[0], c & 2 ? mx[1] : mn[1],
                                                                                 c & 4 ? mx[2] : mn[2])));
                    }
            }
        }
        if (node->camera && node->camera->type == cgltf_camera_type_perspective) {
            Vec3f p = XformPoint(M, Vec3f(0, 0, 0));
            Vec3f fwd = Normalize(XformVector(M, Vec3f(0, 0, -1))), up = Normalize(XformVector(M, Vec3f(0, 1, 0)));
            std::string cid = node->camera->name ? Sanitize(node->camera->name) : "camera" + std::to_string(doc["cameras"].size());
            doc["cameras"][cid] = {{"type", "perspective"}, {"position", Vec(p)}, {"look_at", Vec(p + fwd)}, {"up", Vec(up)},
                                   {"fov_y", Degrees(node->camera->data.perspective.yfov)}};
        }
        if (node->light) {
            const cgltf_light *l = node->light;
            Vec3f p = XformPoint(M, Vec3f(0, 0, 0));
            Vec3f dirv = Normalize(XformVector(M, Vec3f(0, 0, -1)));
            float I = l->intensity * opts.lightScale;
            std::string lid = prefix + (l->name ? Sanitize(l->name) : "light" + std::to_string(doc["lights"].size()));
            if (l->type == cgltf_light_type_point)
                doc["lights"][lid] = {{"type", "point"}, {"position", Vec(p)}, {"color", Rgb(l->color)}, {"intensity", I}};
            else if (l->type == cgltf_light_type_spot)
                doc["lights"][lid] = {{"type", "spot"}, {"position", Vec(p)}, {"direction", Vec(dirv)}, {"color", Rgb(l->color)},
                                      {"intensity", I}, {"cone_angle", Degrees(l->spot_outer_cone_angle)},
                                      {"falloff_start", Degrees(l->spot_inner_cone_angle)}};
            else if (l->type == cgltf_light_type_directional)
                doc["lights"][lid] = {{"type", "distant"}, {"direction", Vec(dirv)}, {"color", Rgb(l->color)}, {"irradiance", I}};
        }
        for (size_t c = 0; c < node->children_count; ++c) visit(node->children[c]);
    };
    const cgltf_scene *scene = d->scene ? d->scene : (d->scenes_count ? &d->scenes[0] : nullptr);
    if (scene) {
        for (size_t i = 0; i < scene->nodes_count; ++i) visit(scene->nodes[i]);
    } else {
        for (size_t i = 0; i < d->nodes_count; ++i)
            if (!d->nodes[i].parent) visit(&d->nodes[i]);
    }
    if (!doc["cameras"].empty()) doc["render"] = {{"camera", doc["cameras"].begin().key()}};
    doc["import_bounds"] = {Vec(worldBounds.pMin), Vec(worldBounds.pMax)};
    return doc;
}

void AddImportDefaults(json &doc, const Bounds3f &b) {
    Vec3f c(0, 0, 0);
    float r = 1;
    if (!b.IsEmpty()) b.BoundingSphere(&c, &r);
    r = std::max(r, 1e-3f);
    if (!doc.contains("cameras") || doc["cameras"].empty()) {
        Vec3f dir = Normalize(Vec3f(0.55f, 0.4f, 1.f));
        Vec3f p = c + dir * (r * 2.9f);
        doc["cameras"]["auto"] = {{"type", "perspective"}, {"position", Vec(p)}, {"look_at", Vec(c)}, {"fov_y", 40.0}};
        doc["render"]["camera"] = "auto";
    }
    if (!doc.contains("lights") || doc["lights"].empty()) {
        bool emissive = false;
        for (auto &m : doc["materials"])
            if (m.contains("emission")) emissive = true;
        if (doc.contains("objects"))
            for (auto &o : doc["objects"])
                if (o.contains("emission")) emissive = true;
        if (!emissive) {
            doc["lights"]["default_sky"] = {{"type", "environment"}, {"color", json::array({1.0, 1.0, 1.0})}, {"intensity", 1.0}};
            doc["lights"]["default_key"] = {{"type", "distant"}, {"direction", json::array({-0.4, -1.0, -0.3})}, {"irradiance", 2.0},
                                            {"temperature", 5500}};
        }
    }
}

} // namespace pr

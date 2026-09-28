#pragma once

#include "core/image.h"
#include "geometry/shape.h"

#include "nlohmann/json.hpp"

#include <string>

namespace pr {

// Importers convert foreign formats into the native scene JSON (same schema as .prscene.json),
// so they work as top-level scenes, as 'include' entries and with `prender --convert`.
struct ImportOptions {
    float lightScale = 1.f;     // multiplies imported light intensities
    float emissionScale = 1.f;  // multiplies imported emission
};

// glTF 2.0 (.gltf / .glb). Geometry stays in the glTF file (geometry type "gltf").
nlohmann::json ImportGLTF(const std::string &path, const ImportOptions &opts, std::string *err);
// Loads one primitive of a glTF mesh (object space).
std::shared_ptr<TriangleMesh> LoadGLTFPrimitive(const std::string &path, int mesh, int primitive, std::string *err);
// Decodes an image stored inside a glTF (buffer view or data: URI).
bool LoadGLTFImage(const std::string &path, int image, TextureEncoding enc, Image *out, std::string *err);

// pbrt-v4 scene description (subset). Meshes are emitted inline (geometry type "mesh" with
// "positions"/"indices"/...), PLY meshes by reference.
nlohmann::json ImportPBRT(const std::string &path, const ImportOptions &opts, std::string *err);

// Adds an automatic camera and default lighting when an imported scene has none.
void AddImportDefaults(nlohmann::json &doc, const Bounds3f &bounds);

} // namespace pr

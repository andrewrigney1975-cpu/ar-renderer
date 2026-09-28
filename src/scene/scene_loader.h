#pragma once

#include "scene/scene.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pr {

struct LoadOptions {
    std::string camera;                    // override render.camera
    std::optional<std::vector<std::string>> rigs;  // override active light rigs
    std::vector<std::string> addRigs;      // rigs added on top of the active set
    int width = 0, height = 0;             // override film resolution
};

std::unique_ptr<Scene> LoadScene(const std::string &path, const LoadOptions &opts, std::string *err);

// Writes the fully composed scene (includes, imports, active rigs not applied) as native JSON,
// e.g. to convert glTF / pbrt scenes into .prscene.json.
bool ConvertSceneToJson(const std::string &path, const std::string &outPath, std::string *err);

// Parse a duration such as "90", "90s", "10m", "1.5h" into seconds. Returns false on error.
bool ParseDuration(const std::string &s, double *seconds);

} // namespace pr

#pragma once

#include "core/image.h"
#include "scene/scene.h"

#include <string>

namespace pr {

enum class AOVType { Albedo, Normal, Depth, Position };
bool ParseAOV(const std::string &name, AOVType *out);

// Renders an auxiliary buffer at the film resolution (box-filtered, `spp` jittered samples per
// pixel). Albedo is the directional albedo of the first visible surface in linear sRGB (through
// perfectly specular interfaces it is 1, as recommended for denoisers); normals are world-space
// shading normals; depth is the distance along the camera ray.
Image RenderAOV(const Scene &scene, AOVType type, int spp, uint64_t seed);

} // namespace pr

#pragma once

#include "core/image.h"
#include "scene/scene.h"

#include <string>

namespace pr {

enum class AOVType { Albedo, Normal, Depth, Position };
bool ParseAOV(const std::string &name, AOVType *out);

// Renders an auxiliary buffer at the film resolution (box-filtered, `spp` jittered samples per
// pixel). Albedo is the directional albedo of the first visible surface in linear sRGB; normals
// are world-space shading normals; depth is the distance along the camera ray.
// throughSpecular (denoiser guides): albedo and normal follow perfectly specular reflection and
// refraction (glass, mirrors) to the first non-specular surface, tinted by the chain's
// throughput, so detail seen through glass is preserved; a chain that enters a scattering
// medium (subsurface) takes the medium's single-scattering albedo.
Image RenderAOV(const Scene &scene, AOVType type, int spp, uint64_t seed, bool throughSpecular = false);

} // namespace pr

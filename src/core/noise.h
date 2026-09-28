#pragma once

#include "core/math.h"

namespace pr {

// Improved Perlin noise (Perlin 2002) in [-1, 1] and fractional Brownian motion built on it.
float PerlinNoise(const Vec3f &p);
// Sum of octaves of |noise| (turbulence = true) or signed noise; result roughly in [-1, 1] / [0, 1].
float FBm(const Vec3f &p, int octaves, float lacunarity = 2.f, float gain = 0.5f, bool turbulence = false);

} // namespace pr

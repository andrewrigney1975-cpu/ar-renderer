#pragma once

#include "core/image.h"

#include <string>

namespace pr {

// Optional denoising with Intel Open Image Denoise (OIDN), loaded at runtime from
// OpenImageDenoise.dll next to the executable (`build.cmd oidn` fetches it). Denoising trades
// the renderer's unbiasedness for less noise, so it only ever produces an *additional* output.

// True when OIDN could be loaded; otherwise *status says why.
bool DenoiserAvailable(std::string *status = nullptr);

struct DenoiseOptions {
    bool highQuality = true;  // OIDN "high" (final frames) vs "balanced" (previews)
    bool cleanAux = false;    // albedo/normal are noise-free (they usually are not)
};

// Denoises a linear HDR RGB image in place. albedo (reflectance, [0,1]) and normal ([-1,1]) are
// optional guides of the same size; either may be null. Negative inputs are clamped to zero.
bool Denoise(Image &color, const Image *albedo, const Image *normal, const DenoiseOptions &opts, std::string *err);

} // namespace pr

// OpenEXR input via tinyexr (textures and environment maps).

#include "core/image.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_USE_THREAD 0
#include "tinyexr.h"
#undef RGB  // windows.h (pulled in by tinyexr) defines an RGB macro
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace pr {

bool ReadEXR(const std::string &path, Image *out, std::string *err) {
    float *rgba = nullptr;
    int w = 0, h = 0;
    const char *msg = nullptr;
    int rc = LoadEXR(&rgba, &w, &h, path.c_str(), &msg);
    if (rc != TINYEXR_SUCCESS) {
        *err = "failed to read " + path + (msg ? std::string(": ") + msg : std::string());
        if (msg) FreeEXRErrorMessage(msg);
        return false;
    }
    *out = Image(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float *p = rgba + 4 * (size_t(y) * w + x);
            out->Set(x, y, RGB(p[0], p[1], p[2]));
        }
    free(rgba);
    return true;
}

} // namespace pr

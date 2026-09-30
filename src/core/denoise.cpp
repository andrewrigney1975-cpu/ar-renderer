#include "core/denoise.h"

#include "core/log.h"

#include <algorithm>
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef RGB
#endif

namespace pr {

namespace {

// The subset of the OIDN 2.x C API that we use, declared here so that the renderer builds and
// runs without OIDN; the library is only needed when denoising is requested.
using OIDNDevice = void *;
using OIDNFilter = void *;
using OIDNBuffer = void *;
constexpr int kDeviceTypeCPU = 1;
constexpr int kFormatFloat3 = 3;
constexpr int kQualityBalanced = 5, kQualityHigh = 6;

struct OidnApi {
    bool loaded = false;
    std::string status;
    OIDNDevice (*newDevice)(int) = nullptr;
    void (*commitDevice)(OIDNDevice) = nullptr;
    int (*getDeviceError)(OIDNDevice, const char **) = nullptr;
    void (*releaseDevice)(OIDNDevice) = nullptr;
    OIDNFilter (*newFilter)(OIDNDevice, const char *) = nullptr;
    void (*setFilterImage)(OIDNFilter, const char *, OIDNBuffer, int, size_t, size_t, size_t, size_t, size_t) = nullptr;
    void (*setFilterBool)(OIDNFilter, const char *, bool) = nullptr;
    void (*setFilterInt)(OIDNFilter, const char *, int) = nullptr;
    void (*commitFilter)(OIDNFilter) = nullptr;
    void (*executeFilter)(OIDNFilter) = nullptr;
    void (*releaseFilter)(OIDNFilter) = nullptr;
    OIDNBuffer (*newBuffer)(OIDNDevice, size_t) = nullptr;
    void (*writeBuffer)(OIDNBuffer, size_t, size_t, const void *) = nullptr;
    void (*readBuffer)(OIDNBuffer, size_t, size_t, void *) = nullptr;
    void (*releaseBuffer)(OIDNBuffer) = nullptr;
};

OidnApi &Api() {
    static OidnApi api = [] {
        OidnApi a;
#ifdef _WIN32
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        std::filesystem::path dir = std::filesystem::path(std::wstring(buf, n)).parent_path();
        // LOAD_WITH_ALTERED_SEARCH_PATH: OIDN's own dependencies (core, CPU device, TBB) are
        // resolved next to OpenImageDenoise.dll.
        HMODULE h = LoadLibraryExW((dir / L"OpenImageDenoise.dll").wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) {
            a.status = "OpenImageDenoise.dll not found next to prender.exe (run build.cmd oidn)";
            return a;
        }
        auto get = [&](auto &fn, const char *name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(h, name));
            return fn != nullptr;
        };
        bool ok = get(a.newDevice, "oidnNewDevice") && get(a.commitDevice, "oidnCommitDevice") &&
                  get(a.getDeviceError, "oidnGetDeviceError") && get(a.releaseDevice, "oidnReleaseDevice") &&
                  get(a.newFilter, "oidnNewFilter") && get(a.setFilterImage, "oidnSetFilterImage") &&
                  get(a.setFilterBool, "oidnSetFilterBool") && get(a.setFilterInt, "oidnSetFilterInt") &&
                  get(a.commitFilter, "oidnCommitFilter") && get(a.executeFilter, "oidnExecuteFilter") &&
                  get(a.releaseFilter, "oidnReleaseFilter") && get(a.newBuffer, "oidnNewBuffer") &&
                  get(a.writeBuffer, "oidnWriteBuffer") && get(a.readBuffer, "oidnReadBuffer") &&
                  get(a.releaseBuffer, "oidnReleaseBuffer");
        if (!ok) {
            a.status = "OpenImageDenoise.dll is not an OIDN 2.x library";
            return a;
        }
        a.loaded = true;
        a.status = "ok";
#else
        a.status = "denoising is only available on Windows builds";
#endif
        return a;
    }();
    return api;
}

} // namespace

bool DenoiserAvailable(std::string *status) {
    OidnApi &a = Api();
    if (status) *status = a.status;
    return a.loaded;
}

bool Denoise(Image &color, const Image *albedo, const Image *normal, const DenoiseOptions &opts, std::string *err) {
    OidnApi &a = Api();
    if (!a.loaded) {
        *err = "denoising unavailable: " + a.status;
        return false;
    }
    const int w = color.Width(), h = color.Height();
    if (w == 0 || h == 0) return true;
    if (albedo && (albedo->Width() != w || albedo->Height() != h)) albedo = nullptr;
    if (normal && (normal->Width() != w || normal->Height() != h)) normal = nullptr;
    if (!albedo) normal = nullptr;  // OIDN accepts a normal guide only together with albedo

    // The CPU device only: OIDN's GPU device ships a SYCL runtime that would clash with the one
    // prender_gpu.dll uses.
    OIDNDevice dev = a.newDevice(kDeviceTypeCPU);
    if (!dev) {
        *err = "OIDN: cannot create the CPU device";
        return false;
    }
    a.commitDevice(dev);
    const size_t bytes = size_t(w) * h * 3 * sizeof(float);
    std::vector<float> in(color.Data());
    for (float &v : in)
        if (!(v > 0)) v = 0;  // spectral rendering yields out-of-gamut negatives; OIDN expects >= 0
    OIDNBuffer bColor = a.newBuffer(dev, bytes), bOut = a.newBuffer(dev, bytes);
    OIDNBuffer bAlb = albedo ? a.newBuffer(dev, bytes) : nullptr, bNrm = normal ? a.newBuffer(dev, bytes) : nullptr;
    OIDNFilter f = a.newFilter(dev, "RT");
    const char *msg = nullptr;
    bool ok = bColor && bOut && f && (!albedo || bAlb) && (!normal || bNrm);
    if (ok) {
        a.writeBuffer(bColor, 0, bytes, in.data());
        a.setFilterImage(f, "color", bColor, kFormatFloat3, size_t(w), size_t(h), 0, 0, 0);
        if (albedo) {
            a.writeBuffer(bAlb, 0, bytes, albedo->Data().data());
            a.setFilterImage(f, "albedo", bAlb, kFormatFloat3, size_t(w), size_t(h), 0, 0, 0);
        }
        if (normal) {
            a.writeBuffer(bNrm, 0, bytes, normal->Data().data());
            a.setFilterImage(f, "normal", bNrm, kFormatFloat3, size_t(w), size_t(h), 0, 0, 0);
        }
        a.setFilterImage(f, "output", bOut, kFormatFloat3, size_t(w), size_t(h), 0, 0, 0);
        a.setFilterBool(f, "hdr", true);
        a.setFilterBool(f, "cleanAux", opts.cleanAux && albedo);
        a.setFilterInt(f, "quality", opts.highQuality ? kQualityHigh : kQualityBalanced);
        a.commitFilter(f);
        a.executeFilter(f);
        if (a.getDeviceError(dev, &msg) != 0) ok = false;
        else a.readBuffer(bOut, 0, bytes, color.Data().data());
    }
    if (!ok) *err = std::string("OIDN: ") + (msg ? msg : "denoising failed");
    if (f) a.releaseFilter(f);
    for (OIDNBuffer b : {bColor, bOut, bAlb, bNrm})
        if (b) a.releaseBuffer(b);
    a.releaseDevice(dev);
    return ok;
}

} // namespace pr

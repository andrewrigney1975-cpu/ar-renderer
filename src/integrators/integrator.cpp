#include "integrators/integrator.h"

#include "integrators/bdpt.h"
#include "integrators/gpu_path.h"
#include "integrators/mlt.h"
#include "integrators/path.h"

#include "core/fsutil.h"
#include "core/log.h"

#include <cstdio>
#include <filesystem>

namespace pr {

static const char kCheckpointMagic[8] = {'P', 'R', 'C', 'K', 'P', 'T', '0', '2'};

static void PutHeader(BinaryWriter &w, const RenderControl &c, const IntegratorSettings &s, const Film &film) {
    w.PutBytes(kCheckpointMagic, 8);
    w.Put(uint32_t(s.type));
    w.Put(int32_t(s.maxDepth));
    w.Put(int32_t(film.Width()));
    w.Put(int32_t(film.Height()));
    w.Put(int32_t(s.chains));
    w.Put(c.fingerprint);
    w.Put(c.seed);
}

bool WriteCheckpoint(const RenderControl &control, const IntegratorSettings &s, const Film &film,
                     const std::function<void(BinaryWriter &)> &body, std::string *err) {
    if (control.checkpointPath.empty()) return true;
    BinaryWriter hdr, w;
    PutHeader(hdr, control, s, film);
    body(w);
    std::string tmp = control.checkpointPath + ".tmp";
    FILE *f = _wfopen(Utf8Path(tmp).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot write checkpoint " + tmp;
        return false;
    }
    // Header, then the film streamed in chunks (it can be gigabytes at 16K), then the body.
    bool ok = std::fwrite(hdr.Data().data(), 1, hdr.Data().size(), f) == hdr.Data().size() && film.WriteAccumulators(f);
    uint64_t bodySize = w.Data().size();
    ok = ok && std::fwrite(&bodySize, sizeof(bodySize), 1, f) == 1 &&
         std::fwrite(w.Data().data(), 1, w.Data().size(), f) == w.Data().size();
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(Utf8Path(tmp), Utf8Path(control.checkpointPath), ec);
    if (!ok || ec) {
        *err = "failed to write checkpoint " + control.checkpointPath;
        return false;
    }
    return true;
}

bool ReadCheckpoint(const RenderControl &control, const IntegratorSettings &s, Film &film,
                    const std::function<bool(BinaryReader &)> &body, std::string *err) {
    FILE *f = _wfopen(Utf8Path(control.resumePath).wstring().c_str(), L"rb");
    if (!f) {
        *err = "cannot open checkpoint " + control.resumePath;
        return false;
    }
    BinaryWriter expected;
    PutHeader(expected, control, s, film);
    const auto &hdr = expected.Data();
    std::vector<uint8_t> got(hdr.size());
    bool readHeader = std::fread(got.data(), 1, got.size(), f) == got.size();
    if (!readHeader || std::memcmp(got.data(), hdr.data(), 8) != 0) {
        std::fclose(f);
        *err = control.resumePath + " is not a prender checkpoint";
        return false;
    }
    if (std::memcmp(got.data(), hdr.data(), hdr.size()) != 0) {
        std::fclose(f);
        *err = control.resumePath + " was made for a different scene, integrator, resolution, seed or depth";
        return false;
    }
    uint64_t bodySize = 0;
    bool ok = film.ReadAccumulators(f) && std::fread(&bodySize, sizeof(bodySize), 1, f) == 1 && bodySize < (uint64_t(1) << 40);
    std::vector<uint8_t> bodyData(ok ? size_t(bodySize) : 0);
    ok = ok && std::fread(bodyData.data(), 1, bodyData.size(), f) == bodyData.size();
    std::fclose(f);
    BinaryReader r(std::move(bodyData));
    if (!ok || !body(r) || !r.Ok()) {
        *err = control.resumePath + " is truncated or corrupt";
        return false;
    }
    return true;
}

std::unique_ptr<Integrator> CreateIntegrator(const IntegratorSettings &settings, uint64_t seed) {
    IntegratorSettings s = settings;
    if (s.device >= 0) {
        if (s.type == IntegratorType::Path) {
            if (s.maxDepth <= 0) s.maxDepth = 0;
            return std::make_unique<GpuPathIntegrator>(s, seed, s.device);
        }
        LogWarning("the GPU device currently supports the path tracer only; rendering on the CPU");
    }
    switch (s.type) {
    case IntegratorType::Path:
        // Unlimited depth: Russian roulette alone terminates paths (unbiased).
        if (s.maxDepth <= 0) s.maxDepth = 0;
        return std::make_unique<PathIntegrator>(s, seed);
    case IntegratorType::BDPT:
        if (s.maxDepth <= 0) s.maxDepth = 256;
        return std::make_unique<BDPTIntegrator>(s, seed);
    case IntegratorType::MMLT:
        if (s.maxDepth <= 0) s.maxDepth = 32;
        return std::make_unique<MLTIntegrator>(s, seed, true);
    case IntegratorType::PSSMLT:
        if (s.maxDepth <= 0) s.maxDepth = 32;
        return std::make_unique<MLTIntegrator>(s, seed, false);
    }
    return nullptr;
}

} // namespace pr

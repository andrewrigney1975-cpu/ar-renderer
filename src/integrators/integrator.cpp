#include "integrators/integrator.h"

#include "integrators/bdpt.h"
#include "integrators/mlt.h"
#include "integrators/path.h"

#include "core/fsutil.h"

#include <cstdio>
#include <filesystem>

namespace pr {

static const char kCheckpointMagic[8] = {'P', 'R', 'C', 'K', 'P', 'T', '0', '1'};

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
    BinaryWriter w;
    PutHeader(w, control, s, film);
    film.Serialize(w);
    body(w);
    std::string tmp = control.checkpointPath + ".tmp";
    FILE *f = _wfopen(Utf8Path(tmp).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot write checkpoint " + tmp;
        return false;
    }
    bool ok = std::fwrite(w.Data().data(), 1, w.Data().size(), f) == w.Data().size();
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
    std::vector<uint8_t> data;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    BinaryWriter expected;
    PutHeader(expected, control, s, film);
    const auto &hdr = expected.Data();
    if (data.size() < hdr.size() || std::memcmp(data.data(), hdr.data(), 8) != 0) {
        *err = control.resumePath + " is not a prender checkpoint";
        return false;
    }
    if (std::memcmp(data.data(), hdr.data(), hdr.size()) != 0) {
        *err = control.resumePath + " was made for a different scene, integrator, resolution, seed or depth";
        return false;
    }
    BinaryReader r(std::vector<uint8_t>(data.begin() + hdr.size(), data.end()));
    if (!film.Deserialize(r) || !body(r) || !r.Ok()) {
        *err = control.resumePath + " is truncated or corrupt";
        return false;
    }
    return true;
}

std::unique_ptr<Integrator> CreateIntegrator(const IntegratorSettings &settings, uint64_t seed) {
    IntegratorSettings s = settings;
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

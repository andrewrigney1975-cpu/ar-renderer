#include "lights/ies.h"

#include "core/fsutil.h"

#include <fstream>
#include <sstream>

namespace pr {

std::shared_ptr<IESProfile> IESProfile::Load(const std::string &path, std::string *err) {
    std::ifstream in(Utf8Path(path), std::ios::binary);
    if (!in) {
        *err = "cannot open " + path;
        return nullptr;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    auto p = Parse(ss.str(), err);
    if (!p) *err = path + ": " + *err;
    return p;
}

std::shared_ptr<IESProfile> IESProfile::Parse(const std::string &text, std::string *err) {
    // Skip keyword lines up to and including TILT=.
    size_t pos = text.find("TILT=");
    if (pos == std::string::npos) {
        *err = "missing TILT= line (not an LM-63 file)";
        return nullptr;
    }
    size_t eol = text.find('\n', pos);
    std::string tilt = text.substr(pos + 5, eol == std::string::npos ? std::string::npos : eol - pos - 5);
    std::istringstream in(eol == std::string::npos ? std::string() : text.substr(eol + 1));
    if (tilt.rfind("INCLUDE", 0) == 0) {
        // Lamp-to-luminaire geometry, then n tilt angles and n multipliers: skipped.
        float geometry;
        int n;
        in >> geometry >> n;
        for (int i = 0; i < 2 * n; ++i) {
            float skip;
            in >> skip;
        }
    }
    float nLamps, lumensPerLamp, multiplier, photometricType, units, w, l, h, ballast, future, watts;
    int nV, nH;
    in >> nLamps >> lumensPerLamp >> multiplier >> nV >> nH >> photometricType >> units >> w >> l >> h >> ballast >>
        future >> watts;
    if (!in || nV <= 0 || nH <= 0 || nV > 100000 || nH > 100000) {
        *err = "bad photometric header";
        return nullptr;
    }
    if (int(photometricType) != 1) {
        *err = "only type C photometry is supported";
        return nullptr;
    }
    auto p = std::make_shared<IESProfile>();
    p->vAngles_.resize(nV);
    p->hAngles_.resize(nH);
    p->candela_.resize(size_t(nV) * nH);
    for (float &a : p->vAngles_) in >> a;
    for (float &a : p->hAngles_) in >> a;
    for (float &c : p->candela_) in >> c;
    if (!in) {
        *err = "truncated candela data";
        return nullptr;
    }
    for (float &c : p->candela_) {
        c = std::max(0.f, c * multiplier * ballast);
        p->peak_ = std::max(p->peak_, c);
    }
    if (p->peak_ <= 0) {
        *err = "all candela values are zero";
        return nullptr;
    }
    for (float &c : p->candela_) c /= p->peak_;
    p->Build();
    return p;
}

float IESProfile::EvaluateAngles(float theta, float phi) const {
    const int nV = int(vAngles_.size()), nH = int(hAngles_.size());
    if (theta < vAngles_.front() || theta > vAngles_.back()) return 0;
    // Fold phi by the symmetry implied by the last horizontal angle.
    float hLast = hAngles_.back();
    phi = std::fmod(phi, 360.f);
    if (phi < 0) phi += 360;
    if (nH == 1 || hLast == 0) phi = 0;
    else if (hLast <= 90) {
        phi = std::fmod(phi, 180.f);
        if (phi > 90) phi = 180 - phi;
    } else if (hLast <= 180) {
        if (phi > 180) phi = 360 - phi;
    }
    auto lookupV = [&](int hi) {
        if (nV == 1) return candela_[size_t(hi) * nV];
        int vi = FindInterval(nV, [&](int i) { return vAngles_[i] <= theta; });
        float t = (theta - vAngles_[vi]) / std::max(1e-6f, vAngles_[vi + 1] - vAngles_[vi]);
        return Lerp(Clamp(t, 0.f, 1.f), candela_[size_t(hi) * nV + vi], candela_[size_t(hi) * nV + vi + 1]);
    };
    if (nH == 1) return lookupV(0);
    int hi = FindInterval(nH, [&](int i) { return hAngles_[i] <= phi; });
    float t = (phi - hAngles_[hi]) / std::max(1e-6f, hAngles_[hi + 1] - hAngles_[hi]);
    return Lerp(Clamp(t, 0.f, 1.f), lookupV(hi), lookupV(hi + 1));
}

float IESProfile::Evaluate(const Vec3f &w) const {
    float theta = Degrees(SafeACos(w.z));
    float phi = Degrees(std::atan2(w.y, w.x));
    return EvaluateAngles(theta, phi);
}

void IESProfile::Build() {
    // Tabulate over (u = phi / 2pi, v = (1 - cos theta) / 2): equal-area cells, so the
    // solid-angle pdf is pdf_uv / 4pi.
    std::vector<float> f(size_t(NU) * NV);
    double sum = 0;
    for (int v = 0; v < NV; ++v)
        for (int u = 0; u < NU; ++u) {
            // Max over a few sub-samples so narrow beams never get zero probability.
            float m = 0, s = 0;
            for (int k = 0; k < 4; ++k) {
                float uu = (u + (k & 1 ? 0.75f : 0.25f)) / NU, vv = (v + (k & 2 ? 0.75f : 0.25f)) / NV;
                float cosT = 1 - 2 * vv, phi = 2 * Pi * uu;
                float sinT = SafeSqrt(1 - cosT * cosT);
                float val = Evaluate(Vec3f(sinT * std::cos(phi), sinT * std::sin(phi), cosT));
                m = std::max(m, val);
                s += val / 4;
            }
            f[size_t(v) * NU + u] = m + 1e-4f;
            sum += s;
        }
    integral_ = float(sum / (double(NU) * NV) * 4 * Pi);
    distrib_ = Distribution2D(f.data(), NU, NV);
}

Vec3f IESProfile::Sample(Vec2f u, float *pdf) const {
    float pdfUV;
    Vec2f uv = distrib_.SampleContinuous(u, &pdfUV);
    float cosT = 1 - 2 * uv.y, phi = 2 * Pi * uv.x;
    float sinT = SafeSqrt(1 - cosT * cosT);
    *pdf = pdfUV / (4 * Pi);
    return {sinT * std::cos(phi), sinT * std::sin(phi), cosT};
}

float IESProfile::PDF(const Vec3f &w) const {
    float phi = std::atan2(w.y, w.x);
    if (phi < 0) phi += 2 * Pi;
    Vec2f uv(phi * Inv2Pi, (1 - Clamp(w.z, -1.f, 1.f)) / 2);
    return distrib_.PDF(uv) / (4 * Pi);
}

} // namespace pr

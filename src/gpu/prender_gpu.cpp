// prender_gpu.dll: oneAPI / SYCL path tracer (trial).
//
// Spectral (4 hero wavelengths), unbiased path tracing with next-event estimation and MIS,
// mirroring pr::PathIntegrator for the supported subset: diffuse (with checker), GGX conductors,
// smooth/rough/thin dielectrics with dispersion, area/point lights and a constant environment.
// One work-item renders all samples of one pixel, so the film needs no atomics.

#include "gpu/gpu_api.h"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

using namespace prgpu;

namespace {

// ---------------------------------------------------------------------------------------------
// Device math

struct V3 {
    float x, y, z;
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
inline float Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 Cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float Len2(V3 a) { return Dot(a, a); }
inline float Len(V3 a) { return sycl::sqrt(Len2(a)); }
inline V3 Norm(V3 a) { return a * (1.f / Len(a)); }
inline V3 ToV3(Float3 f) { return {f.x, f.y, f.z}; }
inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float SafeSqrt(float x) { return sycl::sqrt(sycl::fmax(0.f, x)); }
inline float Sqr(float x) { return x * x; }
inline V3 FaceForward(V3 n, V3 v) { return Dot(n, v) < 0 ? -n : n; }

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInv2Pi = 0.15915494309189533577f;
constexpr float kInv4Pi = 0.07957747154594766788f;
constexpr float kOneMinusEps = 0x1.fffffep-1f;
constexpr float kInf = 3.402823466e38f;

struct Frame {
    V3 x, y, z;
    V3 ToLocal(V3 v) const { return {Dot(v, x), Dot(v, y), Dot(v, z)}; }
    V3 FromLocal(V3 v) const { return x * v.x + y * v.y + z * v.z; }
};

inline Frame FrameFromZ(V3 z) {
    float sign = sycl::copysign(1.f, z.z);
    float a = -1 / (sign + z.z);
    float b = z.x * z.y * a;
    return {{1 + sign * z.x * z.x * a, sign * b, -sign * z.x}, {b, sign + z.y * z.y * a, -z.y}, z};
}

inline Frame FrameFromXZ(V3 xIn, V3 z) {
    V3 x = xIn - z * Dot(xIn, z);
    float l2 = Len2(x);
    if (!(l2 > 1e-12f) || !sycl::isfinite(l2)) return FrameFromZ(z);
    x = x * (1.f / sycl::sqrt(l2));
    return {x, Cross(z, x), z};
}

// ---------------------------------------------------------------------------------------------
// RNG (PCG32, as on the CPU)

struct RNG {
    uint64_t state, inc;
    static uint64_t Mix(uint64_t v) {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185ULL;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44dULL;
        v ^= (v >> 33);
        return v;
    }
    void Seed(uint64_t seq, uint64_t offset) {
        state = 0;
        inc = (seq << 1u) | 1u;
        Next();
        state += offset;
        Next();
    }
    uint32_t Next() {
        uint64_t old = state;
        state = old * 0x5851f42d4c957f2dULL + inc;
        uint32_t xs = uint32_t(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = uint32_t(old >> 59u);
        return (xs >> rot) | (xs << ((~rot + 1u) & 31));
    }
    float U() { return sycl::fmin(kOneMinusEps, float(Next()) * 0x1p-32f); }
};

// ---------------------------------------------------------------------------------------------
// Spectra

struct SS {  // sampled spectrum at 4 wavelengths
    float v[4];
};
inline SS Splat(float c) { return {{c, c, c, c}}; }
inline SS operator*(SS a, SS b) { return {{a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3]}}; }
inline SS operator*(SS a, float s) { return {{a.v[0] * s, a.v[1] * s, a.v[2] * s, a.v[3] * s}}; }
inline SS operator+(SS a, SS b) { return {{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3]}}; }
inline float MaxC(SS a) { return sycl::fmax(sycl::fmax(a.v[0], a.v[1]), sycl::fmax(a.v[2], a.v[3])); }
inline bool NonZero(SS a) { return a.v[0] != 0 || a.v[1] != 0 || a.v[2] != 0 || a.v[3] != 0; }

struct Lambda {
    float l[4], pdf[4];
    void TerminateSecondary() {
        if (pdf[1] == 0 && pdf[2] == 0 && pdf[3] == 0) return;
        pdf[1] = pdf[2] = pdf[3] = 0;
        pdf[0] /= 4;
    }
};

inline Lambda SampleVisible(float u) {
    Lambda w;
    for (int i = 0; i < 4; ++i) {
        float up = u + float(i) / 4;
        if (up > 1) up -= 1;
        float l = 538 - 138.888889f * sycl::atanh(0.85691062f - 1.82750197f * up);
        l = Clampf(l, 360.f, 830.f);
        w.l[i] = l;
        w.pdf[i] = 0.0039398042f / Sqr(sycl::cosh(0.0072f * (l - 538)));
    }
    return w;
}

inline float Tab(const Spectrum *spectra, int id, float lambda) {
    if (id < 0) return 0;
    float x = (lambda - kSpectrumMin) / kSpectrumStep;
    x = Clampf(x, 0.f, float(kSpectrumSamples - 1) - 1e-3f);
    int i = int(x);
    float t = x - float(i);
    const float *v = spectra[id].v;
    return v[i] * (1 - t) + v[i + 1] * t;
}
inline SS TabSS(const Spectrum *spectra, int id, const Lambda &w) {
    return {{Tab(spectra, id, w.l[0]), Tab(spectra, id, w.l[1]), Tab(spectra, id, w.l[2]), Tab(spectra, id, w.l[3])}};
}

inline float G(float x, float mu, float s1, float s2) {
    float t = (x - mu) / (x < mu ? s1 : s2);
    return sycl::exp(-0.5f * t * t);
}
inline float CieX(float l) { return 1.056f * G(l, 599.8f, 37.9f, 31.0f) + 0.362f * G(l, 442.0f, 16.0f, 26.7f) - 0.065f * G(l, 501.1f, 20.4f, 26.2f); }
inline float CieY(float l) { return 0.821f * G(l, 568.8f, 46.9f, 40.5f) + 0.286f * G(l, 530.9f, 16.3f, 31.1f); }
inline float CieZ(float l) { return 1.217f * G(l, 437.0f, 11.8f, 36.0f) + 0.681f * G(l, 459.0f, 26.0f, 13.8f); }

// ---------------------------------------------------------------------------------------------
// Microfacets and Fresnel (pbrt-v4)

struct TR {
    float ax, ay;
    bool Smooth() const { return sycl::fmax(ax, ay) < 1e-3f; }
    static float CosT(V3 w) { return w.z; }
    static float Cos2T(V3 w) { return w.z * w.z; }
    static float Sin2T(V3 w) { return sycl::fmax(0.f, 1 - Cos2T(w)); }
    static float Tan2T(V3 w) { return Sin2T(w) / Cos2T(w); }
    static float CosP(V3 w) {
        float s = SafeSqrt(Sin2T(w));
        return s == 0 ? 1 : Clampf(w.x / s, -1.f, 1.f);
    }
    static float SinP(V3 w) {
        float s = SafeSqrt(Sin2T(w));
        return s == 0 ? 0 : Clampf(w.y / s, -1.f, 1.f);
    }
    float D(V3 wm) const {
        float t2 = Tan2T(wm);
        if (!sycl::isfinite(t2)) return 0;
        float c4 = Sqr(Cos2T(wm));
        if (c4 < 1e-16f) return 0;
        float e = t2 * (Sqr(CosP(wm) / ax) + Sqr(SinP(wm) / ay));
        return 1 / (kPi * ax * ay * c4 * Sqr(1 + e));
    }
    float Lambda(V3 w) const {
        float t2 = Tan2T(w);
        if (!sycl::isfinite(t2)) return 0;
        float a2 = Sqr(CosP(w) * ax) + Sqr(SinP(w) * ay);
        return (sycl::sqrt(1 + a2 * t2) - 1) / 2;
    }
    float G1(V3 w) const { return 1 / (1 + Lambda(w)); }
    float Gm(V3 wo, V3 wi) const { return 1 / (1 + Lambda(wo) + Lambda(wi)); }
    float PDF(V3 w, V3 wm) const { return G1(w) / sycl::fabs(w.z) * D(wm) * sycl::fabs(Dot(w, wm)); }
    V3 Sample(V3 w, float u0, float u1) const {
        V3 wh = Norm(V3{ax * w.x, ay * w.y, w.z});
        if (wh.z < 0) wh = -wh;
        V3 T1 = wh.z < 0.99999f ? Norm(Cross(V3{0, 0, 1}, wh)) : V3{1, 0, 0};
        V3 T2 = Cross(wh, T1);
        float r = sycl::sqrt(u0), th = 2 * kPi * u1;
        float px = r * sycl::cos(th), py = r * sycl::sin(th);
        float h = sycl::sqrt(1 - px * px);
        float s = (1 + wh.z) / 2;
        py = (1 - s) * h + s * py;
        float pz = SafeSqrt(1 - px * px - py * py);
        V3 nh = T1 * px + T2 * py + wh * pz;
        return Norm(V3{ax * nh.x, ay * nh.y, sycl::fmax(1e-6f, nh.z)});
    }
};

inline float FrDielectric(float cosI, float eta) {
    cosI = Clampf(cosI, -1.f, 1.f);
    if (cosI < 0) {
        eta = 1 / eta;
        cosI = -cosI;
    }
    float sin2T = (1 - cosI * cosI) / (eta * eta);
    if (sin2T >= 1) return 1;
    float cosT = SafeSqrt(1 - sin2T);
    float rpa = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rpe = (cosI - eta * cosT) / (cosI + eta * cosT);
    return (rpa * rpa + rpe * rpe) / 2;
}

inline float FrComplex(float cosI, float etaR, float etaI) {
    // Complex arithmetic done by hand (eta = etaR + i etaI).
    cosI = Clampf(cosI, 0.f, 1.f);
    float sin2I = 1 - cosI * cosI;
    // eta^2
    float e2r = etaR * etaR - etaI * etaI, e2i = 2 * etaR * etaI;
    float den = e2r * e2r + e2i * e2i;
    // sin2T = sin2I / eta^2
    float s2r = sin2I * e2r / den, s2i = -sin2I * e2i / den;
    // cosT = sqrt(1 - sin2T)
    float ar = 1 - s2r, ai = -s2i;
    float mag = sycl::sqrt(sycl::sqrt(ar * ar + ai * ai));
    float ang = 0.5f * sycl::atan2(ai, ar);
    float cr = mag * sycl::cos(ang), ci = mag * sycl::sin(ang);
    auto div = [](float ar, float ai, float br, float bi, float *rr, float *ri) {
        float d = br * br + bi * bi;
        *rr = (ar * br + ai * bi) / d;
        *ri = (ai * br - ar * bi) / d;
    };
    // r_parl = (eta cosI - cosT) / (eta cosI + cosT)
    float nr = etaR * cosI - cr, ni = etaI * cosI - ci, dr = etaR * cosI + cr, di = etaI * cosI + ci;
    float pr, pi;
    div(nr, ni, dr, di, &pr, &pi);
    // r_perp = (cosI - eta cosT) / (cosI + eta cosT)
    float ecr = etaR * cr - etaI * ci, eci = etaR * ci + etaI * cr;
    float qr, qi;
    div(cosI - ecr, -eci, cosI + ecr, eci, &qr, &qi);
    return (pr * pr + pi * pi + qr * qr + qi * qi) / 2;
}

inline bool Refract(V3 wi, V3 n, float eta, float *etap, V3 *wt) {
    float cosI = Dot(n, wi);
    if (cosI < 0) {
        eta = 1 / eta;
        cosI = -cosI;
        n = -n;
    }
    float sin2I = sycl::fmax(0.f, 1 - cosI * cosI);
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1) return false;
    float cosT = SafeSqrt(1 - sin2T);
    *wt = -wi * (1 / eta) + n * (cosI / eta - cosT);
    *etap = eta;
    return true;
}

inline float PowerHeuristic(float f, float g) {
    if (!sycl::isfinite(f * f)) return 1;
    if (f == 0 && g == 0) return 0;
    return f * f / (f * f + g * g);
}

// ---------------------------------------------------------------------------------------------
// BxDFs in the local shading frame (z = normal), mirroring pr::DiffuseBxDF, ConductorBxDF,
// DielectricBxDF, ThinDielectricBxDF, LayeredBxDF and SheenBxDF without virtual dispatch.

enum { FlagReflection = 1, FlagTransmission = 2, FlagSpecular = 16 };
enum { SampleR = 1, SampleT = 2, SampleAll = 3 };
enum { LobeDiffuse = 0, LobeConductor = 1, LobeDielectric = 2, LobeThin = 3 };

struct BSample {
    SS f;
    V3 wi;
    float pdf;
    uint32_t flags;
    float eta;
    bool proportional;  // pdf only proportional to the density (stochastic BSDFs): use EvalPDF for MIS
};

inline bool SameHemi(V3 a, V3 b) { return a.z * b.z > 0; }

inline V3 CosineHemisphere(float u0, float u1) {
    float ox = 2 * u0 - 1, oy = 2 * u1 - 1, r, th;
    if (ox == 0 && oy == 0) r = th = 0;
    else if (sycl::fabs(ox) > sycl::fabs(oy)) {
        r = ox;
        th = 0.78539816f * (oy / ox);
    } else {
        r = oy;
        th = 1.57079633f - 0.78539816f * (ox / oy);
    }
    V3 w{r * sycl::cos(th), r * sycl::sin(th), 0};
    w.z = SafeSqrt(1 - w.x * w.x - w.y * w.y);
    return w;
}

inline float HG(float cosTheta, float g) {
    float denom = 1 + g * g + 2 * g * cosTheta;
    return kInv4Pi * (1 - g * g) / (denom * SafeSqrt(denom));
}

inline V3 SampleHG(V3 wo, float g, float u0, float u1, float *pdf) {
    float cosTheta;
    if (sycl::fabs(g) < 1e-3f) cosTheta = 1 - 2 * u0;
    else cosTheta = -1 / (2 * g) * (1 + g * g - Sqr((1 - g * g) / (1 + g - 2 * g * u0)));
    cosTheta = Clampf(cosTheta, -1.f, 1.f);
    float sinTheta = SafeSqrt(1 - cosTheta * cosTheta), phi = 2 * kPi * u1;
    Frame f = FrameFromZ(wo);
    *pdf = HG(cosTheta, g);
    return f.FromLocal(V3{sinTheta * sycl::cos(phi), sinTheta * sycl::sin(phi), cosTheta});
}

struct Lobe {
    uint32_t type;
    SS R;        // diffuse
    SS eta, k;   // conductor
    float etaD;  // dielectric / thin
    TR mf;
    bool Specular() const {
        switch (type) {
        case LobeDiffuse: return false;
        case LobeThin: return true;
        default: return mf.Smooth();
        }
    }
};

inline SS ConductorF(const Lobe &l, V3 wo, V3 wi) {
    if (!SameHemi(wo, wi) || l.mf.Smooth()) return Splat(0);
    float co = sycl::fabs(wo.z), ci = sycl::fabs(wi.z);
    if (co == 0 || ci == 0) return Splat(0);
    V3 wm = wi + wo;
    if (Len2(wm) == 0) return Splat(0);
    wm = Norm(wm);
    float d = l.mf.D(wm) * l.mf.Gm(wo, wi) / (4 * ci * co);
    float c = sycl::fabs(Dot(wo, wm));
    SS F;
    for (int i = 0; i < 4; ++i) F.v[i] = FrComplex(c, l.eta.v[i], l.k.v[i]) * d;
    return F;
}

inline float DielectricF(const Lobe &l, V3 wo, V3 wi, bool radiance) {
    if (l.etaD == 1 || l.mf.Smooth()) return 0;
    float co = wo.z, ci = wi.z;
    bool refl = ci * co > 0;
    float etap = 1;
    if (!refl) etap = co > 0 ? l.etaD : 1 / l.etaD;
    V3 wm = wi * etap + wo;
    if (ci == 0 || co == 0 || Len2(wm) == 0) return 0;
    wm = FaceForward(Norm(wm), V3{0, 0, 1});
    if (Dot(wm, wi) * ci < 0 || Dot(wm, wo) * co < 0) return 0;
    float F = FrDielectric(Dot(wo, wm), l.etaD);
    if (refl) return l.mf.D(wm) * l.mf.Gm(wo, wi) * F / sycl::fabs(4 * ci * co);
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap) * ci * co;
    float ft = l.mf.D(wm) * (1 - F) * l.mf.Gm(wo, wi) * sycl::fabs(Dot(wi, wm) * Dot(wo, wm) / denom);
    return radiance ? ft / Sqr(etap) : ft;
}

inline float DielectricPDF(const Lobe &l, V3 wo, V3 wi, int flags) {
    if (l.etaD == 1 || l.mf.Smooth()) return 0;
    float co = wo.z, ci = wi.z;
    bool refl = ci * co > 0;
    float etap = 1;
    if (!refl) etap = co > 0 ? l.etaD : 1 / l.etaD;
    V3 wm = wi * etap + wo;
    if (ci == 0 || co == 0 || Len2(wm) == 0) return 0;
    wm = FaceForward(Norm(wm), V3{0, 0, 1});
    if (Dot(wm, wi) * ci < 0 || Dot(wm, wo) * co < 0) return 0;
    float R = FrDielectric(Dot(wo, wm), l.etaD), T = 1 - R;
    float pr = (flags & SampleR) ? R : 0, pt = (flags & SampleT) ? T : 0;
    if (pr == 0 && pt == 0) return 0;
    if (refl) return l.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm))) * pr / (pr + pt);
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
    return l.mf.PDF(wo, wm) * sycl::fabs(Dot(wi, wm)) / denom * pt / (pr + pt);
}

inline SS LobeF(const Lobe &l, V3 wo, V3 wi, bool radiance) {
    switch (l.type) {
    case LobeDiffuse: return SameHemi(wo, wi) ? l.R * kInvPi : Splat(0);
    case LobeConductor: return ConductorF(l, wo, wi);
    case LobeDielectric: return Splat(DielectricF(l, wo, wi, radiance));
    default: return Splat(0);
    }
}

inline float LobePDF(const Lobe &l, V3 wo, V3 wi, int flags) {
    switch (l.type) {
    case LobeDiffuse: return (flags & SampleR) && SameHemi(wo, wi) ? sycl::fabs(wi.z) * kInvPi : 0;
    case LobeConductor: {
        if (!(flags & SampleR) || !SameHemi(wo, wi) || l.mf.Smooth()) return 0;
        V3 wm = wo + wi;
        if (Len2(wm) == 0) return 0;
        wm = FaceForward(Norm(wm), V3{0, 0, 1});
        return l.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm)));
    }
    case LobeDielectric: return DielectricPDF(l, wo, wi, flags);
    default: return 0;
    }
}

inline bool LobeSample(const Lobe &l, V3 wo, float uc, float u0, float u1, bool radiance, int flags, BSample *s) {
    s->eta = 1;
    s->proportional = false;
    switch (l.type) {
    case LobeDiffuse: {
        if (!(flags & SampleR)) return false;
        V3 wi = CosineHemisphere(u0, u1);
        if (wo.z < 0) wi.z = -wi.z;
        s->pdf = sycl::fabs(wi.z) * kInvPi;
        s->f = l.R * kInvPi;
        s->wi = wi;
        s->flags = FlagReflection;
        return true;
    }
    case LobeConductor: {
        if (!(flags & SampleR)) return false;
        if (l.mf.Smooth()) {
            V3 wi{-wo.x, -wo.y, wo.z};
            float c = sycl::fabs(wi.z);
            for (int i = 0; i < 4; ++i) s->f.v[i] = FrComplex(c, l.eta.v[i], l.k.v[i]) / c;
            s->pdf = 1;
            s->wi = wi;
            s->flags = FlagReflection | FlagSpecular;
            return true;
        }
        if (wo.z == 0) return false;
        V3 wm = l.mf.Sample(wo, u0, u1);
        V3 wi = -wo + wm * (2 * Dot(wo, wm));
        if (!SameHemi(wo, wi)) return false;
        s->pdf = l.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm)));
        s->f = ConductorF(l, wo, wi);
        s->wi = wi;
        s->flags = FlagReflection;
        return true;
    }
    case LobeThin: {
        float R = FrDielectric(sycl::fabs(wo.z), l.etaD), T = 1 - R;
        if (R < 1) {
            R += T * T * R / (1 - R * R);
            T = 1 - R;
        }
        float pr = (flags & SampleR) ? R : 0, pt = (flags & SampleT) ? T : 0;
        if (pr == 0 && pt == 0) return false;
        if (uc < pr / (pr + pt)) {
            V3 wi{-wo.x, -wo.y, wo.z};
            s->f = Splat(R / sycl::fabs(wi.z));
            s->pdf = pr / (pr + pt);
            s->wi = wi;
            s->flags = FlagReflection | FlagSpecular;
        } else {
            V3 wi = -wo;
            s->f = Splat(T / sycl::fabs(wi.z));
            s->pdf = pt / (pr + pt);
            s->wi = wi;
            s->flags = FlagTransmission | FlagSpecular;
        }
        return true;
    }
    case LobeDielectric: {
        if (l.etaD == 1 || l.mf.Smooth()) {
            float R = FrDielectric(wo.z, l.etaD), T = 1 - R;
            float pr = (flags & SampleR) ? R : 0, pt = (flags & SampleT) ? T : 0;
            if (pr == 0 && pt == 0) return false;
            if (uc < pr / (pr + pt)) {
                V3 wi{-wo.x, -wo.y, wo.z};
                s->f = Splat(R / sycl::fabs(wi.z));
                s->pdf = pr / (pr + pt);
                s->wi = wi;
                s->flags = FlagReflection | FlagSpecular;
                return true;
            }
            V3 wi;
            float etap;
            if (!Refract(wo, V3{0, 0, 1}, l.etaD, &etap, &wi)) return false;
            float ft = T / sycl::fabs(wi.z);
            s->f = Splat(radiance ? ft / (etap * etap) : ft);
            s->pdf = pt / (pr + pt);
            s->wi = wi;
            s->flags = FlagTransmission | FlagSpecular;
            s->eta = etap;
            return true;
        }
        V3 wm = l.mf.Sample(wo, u0, u1);
        float R = FrDielectric(Dot(wo, wm), l.etaD), T = 1 - R;
        float pr = (flags & SampleR) ? R : 0, pt = (flags & SampleT) ? T : 0;
        if (pr == 0 && pt == 0) return false;
        if (uc < pr / (pr + pt)) {
            V3 wi = -wo + wm * (2 * Dot(wo, wm));
            if (!SameHemi(wo, wi)) return false;
            s->pdf = l.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm))) * pr / (pr + pt);
            s->f = Splat(l.mf.D(wm) * l.mf.Gm(wo, wi) * R / (4 * wi.z * wo.z));
            s->wi = wi;
            s->flags = FlagReflection;
            return true;
        }
        V3 wi;
        float etap;
        if (!Refract(wo, wm, l.etaD, &etap, &wi) || SameHemi(wo, wi) || wi.z == 0) return false;
        float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
        s->pdf = l.mf.PDF(wo, wm) * sycl::fabs(Dot(wi, wm)) / denom * pt / (pr + pt);
        float ft = T * l.mf.D(wm) * l.mf.Gm(wo, wi) * sycl::fabs(Dot(wi, wm) * Dot(wo, wm) / (wi.z * wo.z * denom));
        s->f = Splat(radiance ? ft / (etap * etap) : ft);
        s->wi = wi;
        s->flags = FlagTransmission;
        s->eta = etap;
        return true;
    }
    default: return false;
    }
}

inline bool GoodSample(bool ok, const BSample &s) { return ok && NonZero(s.f) && s.pdf > 0 && s.wi.z != 0; }

// Full BSDF: one lobe, or a two-sided layered coat (top dielectric over a bottom lobe with an
// optional scattering layer), optionally under a sheen lobe.
struct BSDF {
    Frame frame;
    bool black, layered;
    Lobe top, bottom;  // single-lobe BSDFs use top
    float thickness, g;
    SS albedo;
    bool hasAlbedo;
    int maxDepth, nSamples;
    float sheenWeight, sheenAlpha, sheenColorMax;
    SS sheenColor;
    const float *sheenTable;

    bool BaseNonSpecular() const {
        if (black) return false;
        if (!layered) return !top.Specular();
        return !top.Specular() || !bottom.Specular() || hasAlbedo;
    }
    bool NonSpecular() const { return sheenWeight > 0 || BaseNonSpecular(); }
};

inline uint64_t HashV(V3 a, uint64_t h) {
    h = RNG::Mix(h ^ sycl::bit_cast<uint32_t>(a.x));
    h = RNG::Mix(h ^ (uint64_t(sycl::bit_cast<uint32_t>(a.y)) << 21));
    return RNG::Mix(h ^ (uint64_t(sycl::bit_cast<uint32_t>(a.z)) << 42));
}

inline float LayerTr(float dz, V3 w) {
    if (sycl::fabs(dz) <= 1.17549435e-38f) return 1;
    return sycl::exp(-sycl::fabs(dz / w.z));
}

// Stochastic estimate of the layered BSDF (pbrt-v4 random walk between the interfaces). The
// layered walks are kept out of line: inlining them into every call site costs registers everywhere.
__attribute__((noinline)) SS LayeredF(const BSDF &b, V3 wo, V3 wi, bool radiance) {
    SS f = Splat(0);
    if (wo.z < 0) {  // two-sided
        wo = -wo;
        wi = -wi;
    }
    const bool enteredTop = true;
    const Lobe *enter = &b.top;
    bool same = SameHemi(wo, wi);
    const Lobe *exitI = (same ^ enteredTop) ? &b.bottom : &b.top;
    const Lobe *nonExit = (same ^ enteredTop) ? &b.top : &b.bottom;
    float exitZ = (same ^ enteredTop) ? 0 : b.thickness;
    if (same) f = LobeF(*enter, wo, wi, radiance) * float(b.nSamples);
    RNG r;
    r.Seed(HashV(wo, 0x51), HashV(wi, 0x7a));
    for (int s = 0; s < b.nSamples; ++s) {
        float a0 = r.U(), a1 = r.U(), a2 = r.U();
        BSample wos;
        if (!GoodSample(LobeSample(*enter, wo, a0, a1, a2, radiance, SampleT, &wos), wos)) continue;
        float c0 = r.U(), c1 = r.U(), c2 = r.U();
        BSample wis;
        if (!GoodSample(LobeSample(*exitI, wi, c0, c1, c2, !radiance, SampleT, &wis), wis)) continue;
        SS beta = wos.f * (sycl::fabs(wos.wi.z) / wos.pdf);
        float z = enteredTop ? b.thickness : 0;
        V3 w = wos.wi;
        for (int depth = 0; depth < b.maxDepth; ++depth) {
            if (depth > 3 && MaxC(beta) < 0.25f) {
                float q = sycl::fmax(0.f, 1 - MaxC(beta));
                if (r.U() < q) break;
                beta = beta * (1 / (1 - q));
            }
            if (!b.hasAlbedo) {
                z = (z == b.thickness) ? 0 : b.thickness;
                beta = beta * LayerTr(b.thickness, w);
            } else {
                float dz = -sycl::log(1 - r.U()) * sycl::fabs(w.z);
                float zp = w.z > 0 ? (z + dz) : (z - dz);
                if (0 < zp && zp < b.thickness) {
                    float wt = 1;
                    float phasePdfExit = HG(Dot(-w, -wis.wi), b.g);
                    if (!exitI->Specular()) wt = PowerHeuristic(wis.pdf, phasePdfExit);
                    f = f + beta * b.albedo * wis.f * (phasePdfExit * wt * LayerTr(zp - exitZ, wis.wi) / wis.pdf);
                    float p0 = r.U(), p1 = r.U(), phasePdf;
                    V3 wiPhase = SampleHG(-w, b.g, p0, p1, &phasePdf);
                    if (phasePdf == 0 || wiPhase.z == 0) continue;
                    beta = beta * b.albedo;
                    w = wiPhase;
                    z = zp;
                    if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) && !exitI->Specular()) {
                        SS fExit = LobeF(*exitI, -w, wi, radiance);
                        if (NonZero(fExit)) {
                            float exitPDF = LobePDF(*exitI, -w, wi, SampleT);
                            f = f + beta * fExit * (LayerTr(zp - exitZ, wiPhase) * PowerHeuristic(phasePdf, exitPDF));
                        }
                    }
                    continue;
                }
                z = Clampf(zp, 0.f, b.thickness);
            }
            if (z == exitZ) {
                float e0 = r.U(), e1 = r.U(), e2 = r.U();
                BSample bs;
                if (!GoodSample(LobeSample(*exitI, -w, e0, e1, e2, radiance, SampleR, &bs), bs)) break;
                beta = beta * bs.f * (sycl::fabs(bs.wi.z) / bs.pdf);
                w = bs.wi;
            } else {
                if (!nonExit->Specular()) {
                    float wt = 1;
                    if (!exitI->Specular()) wt = PowerHeuristic(wis.pdf, LobePDF(*nonExit, -w, -wis.wi, SampleAll));
                    f = f + beta * LobeF(*nonExit, -w, -wis.wi, radiance) * wis.f *
                                (sycl::fabs(wis.wi.z) * wt * LayerTr(b.thickness, wis.wi) / wis.pdf);
                }
                float e0 = r.U(), e1 = r.U(), e2 = r.U();
                BSample bs;
                if (!GoodSample(LobeSample(*nonExit, -w, e0, e1, e2, radiance, SampleR, &bs), bs)) break;
                beta = beta * bs.f * (sycl::fabs(bs.wi.z) / bs.pdf);
                w = bs.wi;
                if (!exitI->Specular()) {
                    SS fExit = LobeF(*exitI, -w, wi, radiance);
                    if (NonZero(fExit)) {
                        float wt = 1;
                        if (!nonExit->Specular()) wt = PowerHeuristic(bs.pdf, LobePDF(*exitI, -w, wi, SampleT));
                        f = f + beta * fExit * (LayerTr(b.thickness, bs.wi) * wt);
                    }
                }
            }
        }
    }
    return f * (1.f / float(b.nSamples));
}

inline bool LayeredSample(const BSDF &b, V3 wo, float uc, float u0, float u1, bool radiance, BSample *out) {
    bool flipWi = false;
    if (wo.z < 0) {
        wo = -wo;
        flipWi = true;
    }
    BSample bs;
    if (!GoodSample(LobeSample(b.top, wo, uc, u0, u1, radiance, SampleAll, &bs), bs)) return false;
    if (bs.flags & FlagReflection) {
        if (flipWi) bs.wi = -bs.wi;
        bs.proportional = true;
        *out = bs;
        return true;
    }
    V3 w = bs.wi;
    bool specularPath = (bs.flags & FlagSpecular) != 0;
    RNG r;
    r.Seed(HashV(wo, 0x33), HashV(V3{uc, u0, u1}, 0x99));
    SS f = bs.f * sycl::fabs(bs.wi.z);
    float pdf = bs.pdf;
    float z = b.thickness;
    for (int depth = 0; depth < b.maxDepth; ++depth) {
        float rrBeta = MaxC(f) / pdf;
        if (depth > 3 && rrBeta < 0.25f) {
            float q = sycl::fmax(0.f, 1 - rrBeta);
            if (r.U() < q) return false;
            pdf *= 1 - q;
        }
        if (w.z == 0) return false;
        if (b.hasAlbedo) {
            float dz = -sycl::log(1 - r.U()) * sycl::fabs(w.z);
            float zp = w.z > 0 ? (z + dz) : (z - dz);
            if (zp == z) return false;
            if (0 < zp && zp < b.thickness) {
                float p0 = r.U(), p1 = r.U(), phasePdf;
                V3 wiPhase = SampleHG(-w, b.g, p0, p1, &phasePdf);
                if (phasePdf == 0 || wiPhase.z == 0) return false;
                f = f * b.albedo * phasePdf;
                pdf *= phasePdf;
                specularPath = false;
                w = wiPhase;
                z = zp;
                continue;
            }
            z = Clampf(zp, 0.f, b.thickness);
        } else {
            z = (z == b.thickness) ? 0 : b.thickness;
            f = f * LayerTr(b.thickness, w);
        }
        const Lobe *iface = (z == 0) ? &b.bottom : &b.top;
        float e0 = r.U(), e1 = r.U(), e2 = r.U();
        BSample bs2;
        if (!GoodSample(LobeSample(*iface, -w, e0, e1, e2, radiance, SampleAll, &bs2), bs2)) return false;
        f = f * bs2.f;
        pdf *= bs2.pdf;
        specularPath = specularPath && (bs2.flags & FlagSpecular);
        w = bs2.wi;
        if (bs2.flags & FlagTransmission) {
            uint32_t flags = SameHemi(wo, w) ? FlagReflection : FlagTransmission;
            if (specularPath) flags |= FlagSpecular;
            if (flipWi) w = -w;
            *out = BSample{f, w, pdf, flags, 1.f, true};
            return true;
        }
        f = f * sycl::fabs(bs2.wi.z);
    }
    return false;
}

inline float LayeredPDF(const BSDF &b, V3 wo, V3 wi, bool radiance) {
    if (wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }
    RNG r;
    r.Seed(HashV(wi, 0x15), HashV(wo, 0x2c));
    float pdfSum = 0;
    bool same = SameHemi(wo, wi);
    if (same) pdfSum += float(b.nSamples) * LobePDF(b.top, wo, wi, SampleR);
    for (int s = 0; s < b.nSamples; ++s) {
        if (same) {
            const Lobe *rI = &b.bottom, *tI = &b.top;
            float a0 = r.U(), a1 = r.U(), a2 = r.U();
            BSample wos, wis;
            bool okO = LobeSample(*tI, wo, a0, a1, a2, radiance, SampleT, &wos);
            float c0 = r.U(), c1 = r.U(), c2 = r.U();
            bool okI = LobeSample(*tI, wi, c0, c1, c2, !radiance, SampleT, &wis);
            if (okO && NonZero(wos.f) && wos.pdf > 0 && okI && NonZero(wis.f) && wis.pdf > 0) {
                if (tI->Specular()) {
                    pdfSum += LobePDF(*rI, -wos.wi, -wis.wi, SampleAll);
                } else {
                    float d0 = r.U(), d1 = r.U(), d2 = r.U();
                    BSample rs;
                    if (LobeSample(*rI, -wos.wi, d0, d1, d2, radiance, SampleAll, &rs) && NonZero(rs.f) && rs.pdf > 0) {
                        if (rI->Specular()) {
                            pdfSum += LobePDF(*tI, -rs.wi, wi, SampleAll);
                        } else {
                            float rPDF = LobePDF(*rI, -wos.wi, -wis.wi, SampleAll);
                            pdfSum += PowerHeuristic(wis.pdf, rPDF) * rPDF;
                            float tPDF = LobePDF(*tI, -rs.wi, wi, SampleAll);
                            pdfSum += PowerHeuristic(rs.pdf, tPDF) * tPDF;
                        }
                    }
                }
            }
        } else {
            const Lobe *toI = &b.top, *tiI = &b.bottom;
            float a0 = r.U(), a1 = r.U(), a2 = r.U();
            BSample wos;
            if (!GoodSample(LobeSample(*toI, wo, a0, a1, a2, radiance, SampleAll, &wos), wos) ||
                (wos.flags & FlagReflection))
                continue;
            float c0 = r.U(), c1 = r.U(), c2 = r.U();
            BSample wis;
            if (!GoodSample(LobeSample(*tiI, wi, c0, c1, c2, !radiance, SampleAll, &wis), wis) ||
                (wis.flags & FlagReflection))
                continue;
            if (toI->Specular()) pdfSum += LobePDF(*tiI, -wos.wi, wi, SampleAll);
            else if (tiI->Specular()) pdfSum += LobePDF(*toI, wo, -wis.wi, SampleAll);
            else pdfSum += (LobePDF(*toI, wo, -wis.wi, SampleAll) + LobePDF(*tiI, -wos.wi, wi, SampleAll)) / 2;
        }
    }
    return 0.1f * kInv4Pi + 0.9f * pdfSum / float(b.nSamples);
}

inline SS BaseF(const BSDF &b, V3 wo, V3 wi, bool radiance) {
    if (b.black) return Splat(0);
    return b.layered ? LayeredF(b, wo, wi, radiance) : LobeF(b.top, wo, wi, radiance);
}
inline float BasePDF(const BSDF &b, V3 wo, V3 wi, bool radiance) {
    if (b.black) return 0;
    return b.layered ? LayeredPDF(b, wo, wi, radiance) : LobePDF(b.top, wo, wi, SampleAll);
}
inline bool BaseSample(const BSDF &b, V3 wo, float uc, float u0, float u1, bool radiance, BSample *s) {
    if (b.black) return false;
    return b.layered ? LayeredSample(b, wo, uc, u0, u1, radiance, s) : LobeSample(b.top, wo, uc, u0, u1, radiance, SampleAll, s);
}

// Sheen ("Charlie", Estevez & Kulla 2017) as in pr::SheenBxDF.
inline float SheenAlbedo(const float *t, float cosTheta, float alpha) {
    float fa = (Clampf(alpha, 0.05f, 1.f) - 0.05f) / 0.95f * float(kSheenAlpha - 1);
    float fm = Clampf(sycl::fabs(cosTheta), 0.f, 1.f) * float(kSheenMu - 1);
    int ia = sycl::min(int(fa), kSheenAlpha - 2), im = sycl::min(int(fm), kSheenMu - 2);
    float ta = fa - float(ia), tm = fm - float(im);
    auto at = [&](int a, int m) { return t[a * kSheenMu + m]; };
    float l0 = at(ia, im) + (at(ia, im + 1) - at(ia, im)) * tm;
    float l1 = at(ia + 1, im) + (at(ia + 1, im + 1) - at(ia + 1, im)) * tm;
    return l0 + (l1 - l0) * ta;
}
inline float SheenValue(V3 wo, V3 wi, float alpha) {
    if (!SameHemi(wo, wi)) return 0;
    float co = sycl::fabs(wo.z), ci = sycl::fabs(wi.z);
    if (co == 0 || ci == 0) return 0;
    V3 wh = Norm(wo + wi);
    float invAlpha = 1 / alpha;
    float sin2 = sycl::fmax(0.f, 1 - wh.z * wh.z);
    float D = (2 + invAlpha) * sycl::pow(sin2, 0.5f * invAlpha) * kInv2Pi;
    return D / (4 * (ci + co - ci * co));
}
inline SS SheenF(const BSDF &b, V3 wo, V3 wi) { return b.sheenColor * (b.sheenWeight * SheenValue(wo, wi, b.sheenAlpha)); }
inline float SheenBaseScale(const BSDF &b, V3 wo, V3 wi) {
    float e = sycl::fmax(SheenAlbedo(b.sheenTable, wo.z, b.sheenAlpha), SheenAlbedo(b.sheenTable, wi.z, b.sheenAlpha));
    return sycl::fmax(0.f, 1 - b.sheenWeight * b.sheenColorMax * e);
}
inline float SheenProb(const BSDF &b, V3 wo) {
    if (b.black) return 1;
    return Clampf(b.sheenWeight * b.sheenColorMax * SheenAlbedo(b.sheenTable, wo.z, b.sheenAlpha), 0.f, 0.9f);
}

template <bool kFull> inline SS LocalF(const BSDF &b, V3 wo, V3 wi) {
    if constexpr (!kFull) return b.black ? Splat(0) : LobeF(b.top, wo, wi, true);
    if (b.sheenWeight > 0) return SheenF(b, wo, wi) + BaseF(b, wo, wi, true) * SheenBaseScale(b, wo, wi);
    return BaseF(b, wo, wi, true);
}
template <bool kFull> inline float LocalPDF(const BSDF &b, V3 wo, V3 wi) {
    if constexpr (!kFull) return b.black ? 0.f : LobePDF(b.top, wo, wi, SampleAll);
    if (b.sheenWeight > 0) {
        float ps = SheenProb(b, wo);
        float pdfSheen = SameHemi(wo, wi) ? sycl::fabs(wi.z) * kInvPi : 0;
        return ps * pdfSheen + (1 - ps) * BasePDF(b, wo, wi, true);
    }
    return BasePDF(b, wo, wi, true);
}
template <bool kFull> inline bool LocalSample(const BSDF &b, V3 wo, float uc, float u0, float u1, BSample *s) {
    if constexpr (!kFull) return !b.black && LobeSample(b.top, wo, uc, u0, u1, true, SampleAll, s);
    if (b.sheenWeight <= 0) return BaseSample(b, wo, uc, u0, u1, true, s);
    float ps = SheenProb(b, wo);
    bool exact = !b.layered;
    if (uc < ps) {
        V3 wi = CosineHemisphere(u0, u1);
        if (wo.z < 0) wi.z = -wi.z;
        if (exact) {
            float pdf = LocalPDF<kFull>(b, wo, wi);
            if (pdf == 0) return false;
            *s = BSample{LocalF<kFull>(b, wo, wi), wi, pdf, FlagReflection, 1.f, false};
        } else {
            *s = BSample{SheenF(b, wo, wi), wi, ps * sycl::fabs(wi.z) * kInvPi, FlagReflection, 1.f, true};
        }
        return true;
    }
    float uc2 = ps < 1 ? sycl::fmin((uc - ps) / (1 - ps), kOneMinusEps) : 0.f;
    if (!BaseSample(b, wo, uc2, u0, u1, true, s) || s->pdf == 0) return false;
    bool specular = (s->flags & FlagSpecular) != 0;
    if (specular || !exact || s->proportional) {
        s->f = s->f * SheenBaseScale(b, wo, s->wi);
        s->pdf *= 1 - ps;
        if (!specular) s->proportional = true;
        return true;
    }
    float pdf = LocalPDF<kFull>(b, wo, s->wi);
    if (pdf == 0) return false;
    s->f = LocalF<kFull>(b, wo, s->wi);
    s->pdf = pdf;
    return true;
}

// World-space wrappers (as pr::BSDF).
template <bool kFull> inline SS EvalF(const BSDF &b, V3 woW, V3 wiW) {
    V3 wo = b.frame.ToLocal(woW), wi = b.frame.ToLocal(wiW);
    if (wo.z == 0) return Splat(0);
    return LocalF<kFull>(b, wo, wi);
}
template <bool kFull> inline float EvalPDF(const BSDF &b, V3 woW, V3 wiW) {
    V3 wo = b.frame.ToLocal(woW), wi = b.frame.ToLocal(wiW);
    if (wo.z == 0) return 0;
    return LocalPDF<kFull>(b, wo, wi);
}
template <bool kFull> inline bool SampleF(const BSDF &b, V3 woW, float uc, float u0, float u1, BSample *s) {
    V3 wo = b.frame.ToLocal(woW);
    if (wo.z == 0 || b.black) return false;
    if (!GoodSample(LocalSample<kFull>(b, wo, uc, u0, u1, s), *s)) return false;
    s->wi = b.frame.FromLocal(s->wi);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Scene on the device

// Pixel filter on the device; the CDF lives in a device buffer (dynamically indexed arrays in
// kernel arguments end up in private memory).
struct FilterK {
    uint32_t type;
    float radius, sigma, norm;
    const float *cdf;  // kFilterBins + 1
};

struct DevScene {
    const BVHNode *nodes;
    const int32_t *order;
    const Primitive *prims;
    const Material *mats;
    const Light *lights;
    int32_t lightCount;
    const Spectrum *spectra;
    int32_t envSpectrum;
    float envScale, envPmf;
    V3 worldCenter;
    float worldRadius;
    Camera cam;
    float yIntegral;
    const Medium *media;
    int32_t cameraMedium;
    bool hasInterfaces;
    FilterK filter;
    const float *sheenTable;
};

inline float RayEps(V3 p) {
    float m = sycl::fmax(sycl::fabs(p.x), sycl::fmax(sycl::fabs(p.y), sycl::fabs(p.z)));
    return 5e-5f * sycl::fmax(1.f, m);
}

inline bool HitSphere(const Primitive &pr, V3 o, V3 d, float tMax, float *tHit) {
    if (pr.r <= 0) return false;
    V3 f = o - ToV3(pr.p0);
    float a = Dot(d, d), b = Dot(f, d), c = Dot(f, f) - pr.r * pr.r;
    V3 l = f - d * (b / a);
    float disc = pr.r * pr.r - Len2(l);
    if (disc < 0) return false;
    float q = -b - sycl::copysign(sycl::sqrt(a * disc), b);
    float t0 = c / q, t1 = q / a;
    if (t0 > t1) {
        float t = t0;
        t0 = t1;
        t1 = t;
    }
    float t = t0 > 0 ? t0 : t1;
    if (t <= 0 || t >= tMax) return false;
    *tHit = t;
    return true;
}

inline bool HitTriangle(const Primitive &pr, V3 o, V3 d, float tMax, float *tHit, float *bu, float *bv) {
    V3 p0 = ToV3(pr.p0), e1 = ToV3(pr.p1) - p0, e2 = ToV3(pr.p2) - p0;
    V3 pv = Cross(d, e2);
    float det = Dot(e1, pv);
    if (det == 0) return false;
    float inv = 1 / det;
    V3 tv = o - p0;
    float u = Dot(tv, pv) * inv;
    if (u < 0 || u > 1) return false;
    V3 qv = Cross(tv, e1);
    float v = Dot(d, qv) * inv;
    if (v < 0 || u + v > 1) return false;
    float t = Dot(e2, qv) * inv;
    if (t <= 0 || t >= tMax) return false;
    *tHit = t;
    *bu = u;
    *bv = v;
    return true;
}

struct Hit {
    float t, u, v;
    int prim;
};

inline bool Intersect(const DevScene &s, V3 o, V3 d, float tMax, Hit *hit, bool any) {
    V3 inv{1 / d.x, 1 / d.y, 1 / d.z};
    int neg[3] = {inv.x < 0, inv.y < 0, inv.z < 0};
    int stack[64];
    int sp = 0, cur = 0;
    bool found = false;
    while (true) {
        const BVHNode &n = s.nodes[cur];
        float bmin[3] = {n.bmin.x, n.bmin.y, n.bmin.z}, bmax[3] = {n.bmax.x, n.bmax.y, n.bmax.z};
        float oo[3] = {o.x, o.y, o.z}, ii[3] = {inv.x, inv.y, inv.z};
        float t0 = 0, t1 = tMax;
        bool boxHit = true;
        for (int a = 0; a < 3; ++a) {
            float tn = ((neg[a] ? bmax[a] : bmin[a]) - oo[a]) * ii[a];
            float tf = ((neg[a] ? bmin[a] : bmax[a]) - oo[a]) * ii[a] * (1 + 2 * 1.2e-7f * 3);
            t0 = tn > t0 ? tn : t0;
            t1 = tf < t1 ? tf : t1;
            if (t0 > t1) {
                boxHit = false;
                break;
            }
        }
        if (boxHit) {
            if (n.nPrims > 0) {
                for (int i = 0; i < n.nPrims; ++i) {
                    int pi = s.order[n.offset + i];
                    const Primitive &pr = s.prims[pi];
                    float t, u = 0, v = 0;
                    bool h = pr.type == PrimSphere ? HitSphere(pr, o, d, tMax, &t) : HitTriangle(pr, o, d, tMax, &t, &u, &v);
                    if (h) {
                        if (any) return true;
                        tMax = t;
                        hit->t = t;
                        hit->u = u;
                        hit->v = v;
                        hit->prim = pi;
                        found = true;
                    }
                }
                if (sp == 0) break;
                cur = stack[--sp];
            } else {
                if (neg[n.axis]) {
                    stack[sp++] = cur + 1;
                    cur = n.offset;
                } else {
                    stack[sp++] = n.offset;
                    cur = cur + 1;
                }
            }
        } else {
            if (sp == 0) break;
            cur = stack[--sp];
        }
    }
    return found;
}

struct Surf {
    V3 p, n, ns, dpdu;
    float u, v;
};

inline Surf Shade(const DevScene &s, const Hit &h, V3 o, V3 d) {
    const Primitive &pr = s.prims[h.prim];
    Surf sf;
    if (pr.type == PrimSphere) {
        V3 c = ToV3(pr.p0);
        V3 p = o + d * h.t;
        V3 rel = p - c;
        V3 n = Norm(rel);
        sf.p = c + n * pr.r;
        sf.n = pr.flip ? -n : n;
        sf.ns = sf.n;
        float phi = sycl::atan2(n.z, n.x);
        if (phi < 0) phi += 2 * kPi;
        sf.u = phi * kInv2Pi;
        sf.v = sycl::acos(Clampf(n.y, -1.f, 1.f)) * kInvPi;
        sf.dpdu = V3{-rel.z, 0, rel.x};
        if (Len2(sf.dpdu) < 1e-12f) sf.dpdu = FrameFromZ(n).x;
    } else {
        V3 p0 = ToV3(pr.p0), p1 = ToV3(pr.p1), p2 = ToV3(pr.p2);
        float b1 = h.u, b2 = h.v, b0 = 1 - b1 - b2;
        sf.p = p0 * b0 + p1 * b1 + p2 * b2;
        V3 n = Norm(Cross(p1 - p0, p2 - p0));
        sf.u = pr.u0 * b0 + pr.u1 * b1 + pr.u2 * b2;
        sf.v = pr.v0 * b0 + pr.v1 * b1 + pr.v2 * b2;
        V3 ns = n;
        V3 n0 = ToV3(pr.n0);
        if (Len2(n0) > 0) {
            ns = n0 * b0 + ToV3(pr.n1) * b1 + ToV3(pr.n2) * b2;
            float l = Len(ns);
            ns = l > 0 ? ns * (1 / l) : n;
            n = FaceForward(n, ns);
        }
        sf.n = n;
        sf.ns = ns;
        // Tangent from the uv parameterization (same as the CPU).
        float du02 = pr.u0 - pr.u2, du12 = pr.u1 - pr.u2, dv02 = pr.v0 - pr.v2, dv12 = pr.v1 - pr.v2;
        float det = du02 * dv12 - dv02 * du12;
        V3 dpdu;
        if (sycl::fabs(det) < 1e-9f) dpdu = FrameFromZ(n).x;
        else dpdu = ((p0 - p2) * dv12 - (p1 - p2) * dv02) * (1 / det);
        sf.dpdu = dpdu;
    }
    return sf;
}

inline SS Emit(const DevScene &s, const Light &l, V3 n, V3 w, const Lambda &lam) {
    float c = Dot(n, w);
    if (!l.twoSided && c <= 0) return Splat(0);
    float prof = l.cosPower > 0 ? sycl::pow(sycl::fabs(c), l.cosPower) : 1.f;
    return TabSS(s.spectra, l.spectrum, lam) * (l.scale * prof);
}

// Solid-angle pdf of NEE generating the point p (normal n) on area light l as seen from ref.
inline float AreaPdf(const Light &l, V3 ref, V3 p, V3 n) {
    V3 d = p - ref;
    float d2 = Len2(d);
    float c = sycl::fabs(Dot(n, d * (1 / sycl::sqrt(d2))));
    if (c == 0) return 0;
    return d2 / (c * l.area);
}

inline SS MediumTr(const DevScene &s, int medium, const Lambda &lam, float dist) {
    const Medium &md = s.media[medium];
    SS st = (TabSS(s.spectra, md.sigmaA, lam) + TabSS(s.spectra, md.sigmaS, lam)) * md.scale;
    SS r;
    for (int i = 0; i < 4; ++i) r.v[i] = dist >= kInf ? (st.v[i] > 0 ? 0.f : 1.f) : sycl::exp(-st.v[i] * dist);
    return r;
}

// Transmittance from p to target (a point) through interface surfaces and homogeneous media,
// mirroring pr::Scene::Tr; any other surface blocks. n is the surface normal at p (zero in media).
inline SS Tr(const DevScene &s, V3 p, V3 n, V3 target, int medium, const Lambda &lam) {
    V3 o = p + (Dot(target - p, n) < 0 ? -n : n) * RayEps(p);
    if (!s.hasInterfaces) {
        Hit h;
        if (Intersect(s, o, target - o, 1 - 1e-4f, &h, true)) return Splat(0);
        return medium >= 0 ? MediumTr(s, medium, lam, Len(target - o)) : Splat(1);
    }
    SS T = Splat(1);
    for (int it = 0; it < 256; ++it) {
        V3 d = target - o;
        Hit h;
        bool hit = Intersect(s, o, d, 1 - 1e-4f, &h, false);
        if (hit && s.mats[s.prims[h.prim].material].type != MatInterface) return Splat(0);
        if (medium >= 0) T = T * MediumTr(s, medium, lam, (hit ? h.t : 1.f) * Len(d));
        if (!hit || !NonZero(T)) return T;
        Surf sf = Shade(s, h, o, d);
        const Primitive &pr = s.prims[h.prim];
        medium = Dot(d, sf.n) > 0 ? pr.mediumOutside : pr.mediumInside;
        o = sf.p + (Dot(d, sf.n) < 0 ? -sf.n : sf.n) * RayEps(sf.p);
    }
    return Splat(0);
}

inline TR MakeTR(float ax, float ay) {
    TR mf{ax, ay};
    if (!mf.Smooth()) {
        mf.ax = sycl::fmax(mf.ax, 1e-4f);
        mf.ay = sycl::fmax(mf.ay, 1e-4f);
    }
    return mf;
}

inline SS DiffuseAlbedo(const DevScene &s, const Material &m, const Surf &sf, const Lambda &lam) {
    int id = m.albedo;
    if (m.albedoB >= 0 && m.checkerScale > 0) {
        int odd = (int(sycl::floor(sf.u * m.checkerScale)) + int(sycl::floor(sf.v * m.checkerScale))) & 1;
        if (odd) id = m.albedoB;
    }
    SS R = TabSS(s.spectra, id, lam);
    for (int i = 0; i < 4; ++i) R.v[i] = Clampf(R.v[i], 0.f, 1.f);
    return R;
}

// Conductor lobe; etaScale divides measured n, k (a conductor under a coat sees the coat medium).
inline void ConductorLobe(const DevScene &s, const Material &m, const Lambda &lam, float etaScale, Lobe *l) {
    l->type = LobeConductor;
    l->mf = MakeTR(m.alphaX, m.alphaY);
    if (m.eta >= 0) {
        l->eta = TabSS(s.spectra, m.eta, lam) * (1 / etaScale);
        l->k = TabSS(s.spectra, m.k, lam) * (1 / etaScale);
    } else {
        SS r = TabSS(s.spectra, m.albedo, lam);
        for (int i = 0; i < 4; ++i) {
            float ri = Clampf(r.v[i], 0.f, 0.9999f);
            l->eta.v[i] = 1;
            l->k.v[i] = 2 * sycl::sqrt(ri) / SafeSqrt(1 - ri);
        }
    }
}

template <bool kFull> inline BSDF MakeBSDF(const DevScene &s, const Material &m, const Surf &sf, Lambda &lam) {
    // Only the fields a BSDF kind uses are set (no zero-fill of the whole struct).
    BSDF b;
    b.frame = FrameFromXZ(sf.dpdu, sf.ns);
    b.black = b.layered = false;
    b.sheenWeight = 0;
    b.nSamples = 1;
    switch (m.type) {
    case MatDiffuse:
        b.top.type = LobeDiffuse;
        b.top.R = DiffuseAlbedo(s, m, sf, lam);
        break;
    case MatConductor: ConductorLobe(s, m, lam, 1.f, &b.top); break;
    case MatDielectric:
        b.top.type = m.thin ? LobeThin : LobeDielectric;
        b.top.etaD = Tab(s.spectra, m.eta, lam.l[0]);
        if (m.dispersive) lam.TerminateSecondary();
        if (b.top.etaD == 0) b.top.etaD = 1;
        b.top.mf = MakeTR(m.alphaX, m.alphaY);
        break;
    case MatCoatedDiffuse:
    case MatCoatedConductor: {
        if constexpr (!kFull) {
            b.black = true;  // unreachable: the host selects the full kernel for coated scenes
            break;
        }
        float coatEta = Tab(s.spectra, m.coatEta, lam.l[0]);
        if (m.coatDispersive) lam.TerminateSecondary();
        if (coatEta == 0) coatEta = 1;
        b.layered = true;
        b.top.type = LobeDielectric;
        b.top.etaD = coatEta;
        b.top.mf = MakeTR(m.coatAlpha, m.coatAlpha);
        b.thickness = sycl::fmax(m.thickness, 1.17549435e-38f);
        b.g = m.coatG;
        b.albedo = Splat(0);
        if (m.coatAlbedo >= 0) {
            b.albedo = TabSS(s.spectra, m.coatAlbedo, lam);
            for (int i = 0; i < 4; ++i) b.albedo.v[i] = sycl::fmax(0.f, b.albedo.v[i]);
        }
        b.hasAlbedo = NonZero(b.albedo);
        b.maxDepth = m.coatMaxDepth;
        b.nSamples = sycl::max(1, m.coatSamples);
        if (m.type == MatCoatedDiffuse) {
            b.bottom.type = LobeDiffuse;
            b.bottom.R = DiffuseAlbedo(s, m, sf, lam);
        } else {
            ConductorLobe(s, m, lam, coatEta, &b.bottom);
        }
        break;
    }
    default: b.black = true; break;
    }
    if (kFull && !b.black && m.sheenWeight > 0 && m.sheenColor >= 0) {
        b.sheenColor = TabSS(s.spectra, m.sheenColor, lam);
        for (int i = 0; i < 4; ++i) b.sheenColor.v[i] = sycl::fmax(0.f, b.sheenColor.v[i]);
        b.sheenAlpha = Clampf(m.sheenRoughness, 0.05f, 1.f);
        b.sheenWeight = Clampf(m.sheenWeight, 0.f, 1.f);
        b.sheenColorMax = Clampf(MaxC(b.sheenColor), 0.f, 1.f);
        b.sheenTable = s.sheenTable;
    }
    return b;
}

// ---------------------------------------------------------------------------------------------
// Path tracing (mirrors pr::PathIntegrator::Li)

// Scattering at a surface (bsdf != nullptr) or in a medium (phase function g).
// Scattering vertex. The BSDF is passed separately (by pointer, known non-null exactly when
// kSurface) so that it is never stored in a struct: that would force it into private memory.
struct Vertex {
    V3 p, n, ns, wo;
    float g;
    int mediumIn, mediumOut;  // surface: media on either side; medium vertex: both = current
};

template <bool kFull, bool kSurface> inline SS ScatterF(const Vertex &v, const BSDF *b, V3 wi, float *pdf) {
    if constexpr (kSurface) {
        *pdf = EvalPDF<kFull>(*b, v.wo, wi);
        return EvalF<kFull>(*b, v.wo, wi) * sycl::fabs(Dot(wi, v.ns));
    } else {
        float p = HG(Dot(v.wo, wi), v.g);
        *pdf = p;
        return Splat(p);
    }
}

inline int MediumToward(const Vertex &v, V3 w) { return Dot(w, v.n) > 0 ? v.mediumOut : v.mediumIn; }

// Next-event estimation with MIS (one light chosen by power). Light sampling first produces a
// direction, target point and radiance; the BSDF and shadow ray are then evaluated once, which
// keeps the (large, inlined) BSDF code out of three separate call sites.
template <bool kFull, bool kSurface>
inline SS SampleLd(const DevScene &s, const Vertex &v, const BSDF *b, const Lambda &lam, RNG &rng) {
    float ul = rng.U(), u0 = rng.U(), u1 = rng.U();
    V3 wi, target;
    SS Le;
    float lightPdf;
    bool delta = false;
    if (ul < s.envPmf && s.envSpectrum >= 0) {
        float z = 1 - 2 * u0, r = SafeSqrt(1 - z * z), ph = 2 * kPi * u1;
        wi = V3{r * sycl::cos(ph), r * sycl::sin(ph), z};
        target = v.p + wi * (2 * s.worldRadius);
        Le = TabSS(s.spectra, s.envSpectrum, lam) * s.envScale;
        lightPdf = s.envPmf * kInv4Pi;
    } else {
        if (s.lightCount == 0) return Splat(0);
        float acc = s.envSpectrum >= 0 ? s.envPmf : 0;
        int li = s.lightCount - 1;
        for (int i = 0; i < s.lightCount; ++i) {
            acc += s.lights[i].pmf;
            if (ul < acc) {
                li = i;
                break;
            }
        }
        const Light &l = s.lights[li];
        if (l.pmf <= 0) return Splat(0);
        if (l.type == LightPoint) {
            target = ToV3(l.position);
            V3 dd = target - v.p;
            float d2 = Len2(dd);
            wi = dd * (1 / sycl::sqrt(d2));
            Le = TabSS(s.spectra, l.spectrum, lam) * (l.scale / d2);
            lightPdf = l.pmf;
            delta = true;
        } else {
            const Primitive &lp = s.prims[l.primitive];
            V3 pl, nl;
            if (lp.type == PrimSphere) {
                float z = 1 - 2 * u0, r = SafeSqrt(1 - z * z), ph = 2 * kPi * u1;
                V3 n{r * sycl::cos(ph), r * sycl::sin(ph), z};
                pl = ToV3(lp.p0) + n * lp.r;
                nl = lp.flip ? -n : n;
            } else {
                float b0, b1;
                if (u0 < u1) {
                    b0 = u0 / 2;
                    b1 = u1 - b0;
                } else {
                    b1 = u1 / 2;
                    b0 = u0 - b1;
                }
                float b2 = 1 - b0 - b1;
                V3 p0 = ToV3(lp.p0), p1 = ToV3(lp.p1), p2 = ToV3(lp.p2);
                pl = p0 * b0 + p1 * b1 + p2 * b2;
                nl = Norm(Cross(p1 - p0, p2 - p0));
                V3 n0 = ToV3(lp.n0);
                if (Len2(n0) > 0) nl = FaceForward(nl, n0 * b0 + ToV3(lp.n1) * b1 + ToV3(lp.n2) * b2);
            }
            V3 dd = pl - v.p;
            float d2 = Len2(dd);
            if (!(d2 > 0)) return Splat(0);
            wi = dd * (1 / sycl::sqrt(d2));
            Le = Emit(s, l, nl, -wi, lam);
            lightPdf = l.pmf * AreaPdf(l, v.p, pl, nl);
            if (!NonZero(Le) || !(lightPdf > 0) || !sycl::isfinite(lightPdf)) return Splat(0);
            target = pl + (Dot(-dd, nl) < 0 ? -nl : nl) * RayEps(pl);
        }
    }
    float spdf;
    SS f = ScatterF<kFull, kSurface>(v, b, wi, &spdf);
    if (!NonZero(f)) return Splat(0);
    SS T = Tr(s, v.p, v.n, target, MediumToward(v, wi), lam);
    if (!NonZero(T)) return Splat(0);
    float w = delta ? 1.f : PowerHeuristic(lightPdf, spdf);
    return f * T * Le * (w / lightPdf);
}

// weightScale: factor the throughput will be multiplied by when contributing (spectral MIS).
inline bool RussianRoulette(SS &beta, float etaScale, float weightScale, int depth, RNG &rng) {
    if (depth <= 1) return true;
    float mx = MaxC(beta) * etaScale * weightScale;
    if (mx >= 1) return true;
    float q = sycl::fmax(0.f, 1 - mx);
    if (rng.U() < q) return false;
    beta = beta * (1 / (1 - q));
    return true;
}

template <bool kFull> inline SS Li(const DevScene &s, V3 o, V3 d, Lambda &lam, RNG &rng, int maxDepth) {
    SS L = Splat(0), beta = Splat(1);
    bool specular = false;
    int depth = 0, crossings = 0, medium = s.cameraMedium;
    float etaScale = 1, prevPdf = 1;
    V3 prevP = o;
    // Path-level spectral MIS for media (as pr::PathIntegrator): distances are sampled with the
    // hero wavelength, r[i] = path pdf with wavelength i as hero / hero path pdf, and every
    // contribution is divided by the average of r. Lhero keeps the unweighted sum for paths whose
    // secondary wavelengths are terminated by dispersion.
    SS r = Splat(1), Lhero = Splat(0);
    auto misScale = [&]() {
        float a = 0.25f * (r.v[0] + r.v[1] + r.v[2] + r.v[3]);
        return a > 0 && sycl::isfinite(a) ? 1 / a : 0.f;
    };
    auto terminated = [&]() { return lam.pdf[1] == 0 && lam.pdf[2] == 0 && lam.pdf[3] == 0; };
    auto add = [&](SS c) {
        Lhero = Lhero + c;
        L = L + c * misScale();
    };
    auto rrScale = [&]() { return terminated() ? 1.f : misScale(); };
    for (int guard = 0; guard < 8192; ++guard) {
        Hit h;
        bool hit = Intersect(s, o, d, kInf, &h, false);

        if (medium >= 0) {
            // Homogeneous distance sampling driven by the hero wavelength (index 0).
            const Medium &md = s.media[medium];
            SS ss = TabSS(s.spectra, md.sigmaS, lam) * md.scale;
            SS st = TabSS(s.spectra, md.sigmaA, lam) * md.scale + ss;
            float u = sycl::fmin(rng.U(), kOneMinusEps);
            float tMax = hit ? h.t : kInf;
            float t = st.v[0] > 0 ? -sycl::log(1 - u) / st.v[0] : kInf;
            if (t < tMax) {
                SS T;
                for (int i = 0; i < 4; ++i) T.v[i] = sycl::exp(-st.v[i] * t);
                float pdf0 = st.v[0] * T.v[0];
                if (!(pdf0 > 0)) break;
                beta = beta * T * ss * (1 / pdf0);
                r = r * st * T * (1 / pdf0);
                if (!NonZero(beta)) break;
                if (maxDepth > 0 && depth >= maxDepth) break;
                ++depth;
                Vertex v{o + d * t, V3{0, 0, 0}, V3{0, 0, 0}, -d, md.g, medium, medium};
                add(beta * SampleLd<kFull, false>(s, v, nullptr, lam, rng));
                float u0 = rng.U(), u1 = rng.U(), phasePdf;
                V3 wi = SampleHG(v.wo, md.g, u0, u1, &phasePdf);
                if (!(phasePdf > 0)) break;
                prevPdf = phasePdf;
                prevP = v.p;
                specular = false;
                o = v.p;
                d = wi;
                if (!RussianRoulette(beta, etaScale, rrScale(), depth, rng)) break;
                continue;
            }
            // Passing through has probability T[i] when wavelength i drives the sampling.
            SS T;
            for (int i = 0; i < 4; ++i) T.v[i] = tMax >= kInf ? (st.v[i] > 0 ? 0.f : 1.f) : sycl::exp(-st.v[i] * tMax);
            if (!(T.v[0] > 0)) break;
            beta = beta * T * (1 / T.v[0]);
            r = r * T * (1 / T.v[0]);
            if (!NonZero(beta)) break;
        }

        if (!hit) {
            if (s.envSpectrum >= 0) {
                SS Le = TabSS(s.spectra, s.envSpectrum, lam) * s.envScale;
                float w = 1;
                if (depth > 0 && !specular) w = PowerHeuristic(prevPdf, s.envPmf * kInv4Pi);
                add(beta * Le * w);
            }
            break;
        }
        const Primitive &pr = s.prims[h.prim];
        Surf sf = Shade(s, h, o, d);
        if (pr.light >= 0) {
            const Light &l = s.lights[pr.light];
            SS Le = Emit(s, l, sf.n, -d, lam);
            if (NonZero(Le)) {
                float w = 1;
                if (depth > 0 && !specular) w = PowerHeuristic(prevPdf, l.pmf * AreaPdf(l, prevP, sf.p, sf.n));
                add(beta * Le * w);
            }
        }
        const Material &m = s.mats[pr.material];
        if (m.type == MatInterface) {
            if (++crossings > 1000) break;
            medium = Dot(d, sf.n) > 0 ? pr.mediumOutside : pr.mediumInside;
            o = sf.p + (Dot(d, sf.n) < 0 ? -sf.n : sf.n) * RayEps(sf.p);
            continue;
        }
        if (m.type == MatBlack) break;
        if (maxDepth > 0 && depth >= maxDepth) break;
        ++depth;
        BSDF b = MakeBSDF<kFull>(s, m, sf, lam);
        V3 wo = -d;
        Vertex v{sf.p, sf.n, sf.ns, wo, 0.f, pr.mediumInside, pr.mediumOutside};
        if (b.NonSpecular()) add(beta * SampleLd<kFull, true>(s, v, &b, lam, rng));

        BSample bs;
        float uc = rng.U(), u0 = rng.U(), u1 = rng.U();
        if (!SampleF<kFull>(b, wo, uc, u0, u1, &bs)) break;
        beta = beta * bs.f * (sycl::fabs(Dot(bs.wi, sf.ns)) / bs.pdf);
        // Stochastic (layered) BSDFs only give a proportional pdf; MIS needs the real one.
        prevPdf = bs.proportional ? EvalPDF<kFull>(b, wo, bs.wi) : bs.pdf;
        specular = (bs.flags & FlagSpecular) != 0;
        if (bs.flags & FlagTransmission) etaScale *= bs.eta * bs.eta;
        prevP = sf.p;
        o = sf.p + (Dot(bs.wi, sf.n) < 0 ? -sf.n : sf.n) * RayEps(sf.p);
        d = bs.wi;
        medium = MediumToward(v, d);
        if (!NonZero(beta) || !sycl::isfinite(MaxC(beta))) break;
        if (!RussianRoulette(beta, etaScale, rrScale(), depth, rng)) break;
    }
    return terminated() ? Lhero : L;
}

// ---------------------------------------------------------------------------------------------
// Pixel filter importance sampling (the filter itself matches pr::Filter exactly)

inline float FilterEval1D(const FilterK &f, float x) {
    x = sycl::fabs(x);
    if (x > f.radius) return 0;
    switch (f.type) {
    case FilterGaussian:
        return sycl::fmax(0.f, sycl::exp(-x * x / (2 * f.sigma * f.sigma)) -
                                   sycl::exp(-f.radius * f.radius / (2 * f.sigma * f.sigma)));
    case FilterBlackmanHarris: {
        float t = 0.5f + x / (2 * f.radius);
        return 0.35875f - 0.48829f * sycl::cos(2 * kPi * t) + 0.14128f * sycl::cos(4 * kPi * t) -
               0.01168f * sycl::cos(6 * kPi * t);
    }
    default: return 1;
    }
}

// Samples an offset in [-radius, radius] from the tabulated CDF and returns its density.
inline float FilterSample1D(const FilterK &f, float u, float *pdf) {
    int lo = 0, hi = kFilterBins;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (f.cdf[mid] <= u) lo = mid;
        else hi = mid;
    }
    float w = f.cdf[lo + 1] - f.cdf[lo];
    float t = w > 0 ? (u - f.cdf[lo]) / w : 0.5f;
    float binW = 2 * f.radius / kFilterBins;
    *pdf = w / binW;
    return -f.radius + (float(lo) + t) * binW;
}

// One work-item renders all samples of one pixel (so the film needs no atomics). kFull enables
// coated and sheen materials; scenes without them use a leaner kernel (less register pressure).
template <bool kFull>
inline void RenderPixel(const DevScene &ds, int pix, int w, int spp, int first, uint64_t seed, int maxDepth, float clampY,
                        float *film) {
    const float norm = 1.f / (4 * ds.yIntegral);
    int px = pix % w, py = pix / w;
    const Camera &c = ds.cam;
    float X = 0, Y = 0, Z = 0;
    for (int si = 0; si < spp; ++si) {
        RNG rng;
        rng.Seed(RNG::Mix(uint64_t(pix) * 0x9E3779B97F4A7C15ull ^ seed), RNG::Mix(uint64_t(first + si) + 1));
        Lambda lam = SampleVisible(rng.U());
        // Filter importance sampling: the offset follows (approximately) the pixel
        // filter and fw = filter / pdf keeps the estimate exact for the true filter.
        float pdfX, pdfY;
        float dx = FilterSample1D(ds.filter, rng.U(), &pdfX), dy = FilterSample1D(ds.filter, rng.U(), &pdfY);
        float fw = FilterEval1D(ds.filter, dx) * FilterEval1D(ds.filter, dy) * ds.filter.norm / (pdfX * pdfY);
        if (!(fw != 0) || !sycl::isfinite(fw)) continue;
        float fx = float(px) + 0.5f + dx, fy = float(py) + 0.5f + dy;
        V3 pc{(2 * fx / c.width - 1) * c.tanHalfFov * c.aspect, (1 - 2 * fy / c.height) * c.tanHalfFov, 1};
        V3 dir = Norm(pc), org{0, 0, 0};
        float l0 = rng.U(), l1 = rng.U();
        if (c.lensRadius > 0) {
            float ox = 2 * l0 - 1, oy = 2 * l1 - 1, r = 0, th = 0;
            if (!(ox == 0 && oy == 0)) {
                if (sycl::fabs(ox) > sycl::fabs(oy)) {
                    r = ox;
                    th = 0.78539816f * (oy / ox);
                } else {
                    r = oy;
                    th = 1.57079633f - 0.78539816f * (ox / oy);
                }
            }
            V3 pl{r * sycl::cos(th) * c.lensRadius, r * sycl::sin(th) * c.lensRadius, 0};
            V3 pf = dir * (c.focalDistance / dir.z);
            org = pl;
            dir = Norm(pf - pl);
        }
        const float *M = c.cameraToWorld;
        V3 o{M[0] * org.x + M[1] * org.y + M[2] * org.z + M[3], M[4] * org.x + M[5] * org.y + M[6] * org.z + M[7],
             M[8] * org.x + M[9] * org.y + M[10] * org.z + M[11]};
        V3 dw = Norm(V3{M[0] * dir.x + M[1] * dir.y + M[2] * dir.z, M[4] * dir.x + M[5] * dir.y + M[6] * dir.z,
                        M[8] * dir.x + M[9] * dir.y + M[10] * dir.z});
        SS L = Li<kFull>(ds, o, dw, lam, rng, maxDepth);
        if (!sycl::isfinite(L.v[0] + L.v[1] + L.v[2] + L.v[3])) continue;
        float sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < 4; ++i) {
            if (lam.pdf[i] == 0) continue;
            float v = L.v[i] / lam.pdf[i];
            sx += CieX(lam.l[i]) * v;
            sy += CieY(lam.l[i]) * v;
            sz += CieZ(lam.l[i]) * v;
        }
        // Firefly clamp on the sample's luminance, before the filter weight (as pr::Film::AddSample).
        float k = fw;
        if (clampY > 0 && sy * norm > clampY) k *= clampY / (sy * norm);
        X += sx * k;
        Y += sy * k;
        Z += sz * k;
    }
    film[3 * pix + 0] = X * norm;
    film[3 * pix + 1] = Y * norm;
    film[3 * pix + 2] = Z * norm;
}

// ---------------------------------------------------------------------------------------------
// Device management

struct DeviceState {
    std::unique_ptr<sycl::queue> queue;
    // Scene upload cache.
    const void *key = nullptr;
    int32_t keyPrims = -1, keyNodes = -1, keySpectra = -1, keyMaterials = -1;
    void *buf[9] = {};
    size_t bufSize[9] = {};
    float *film = nullptr;
    size_t filmCount = 0;
};

std::mutex gMutex;
std::vector<sycl::device> gDevices;
std::vector<std::unique_ptr<DeviceState>> gStates;
bool gEnumerated = false;

void Enumerate() {
    if (gEnumerated) return;
    gEnumerated = true;
    try {
        std::vector<sycl::device> lz, other;
        for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
            if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) lz.push_back(d);
            else other.push_back(d);
        }
        // Prefer Level Zero; fall back to other backends (e.g. OpenCL) only if none.
        gDevices = lz.empty() ? other : lz;
    } catch (const sycl::exception &) {
    }
    for (size_t i = 0; i < gDevices.size(); ++i) gStates.push_back(std::make_unique<DeviceState>());
}

void SetErr(char *err, int len, const std::string &m) {
    if (err && len > 0) {
        std::snprintf(err, size_t(len), "%s", m.c_str());
    }
}

template <typename T> T *Upload(DeviceState &st, int slot, const T *data, size_t count) {
    sycl::queue &q = *st.queue;
    size_t bytes = std::max<size_t>(1, count) * sizeof(T);
    if (st.bufSize[slot] < bytes) {
        if (st.buf[slot]) sycl::free(st.buf[slot], q);
        st.buf[slot] = sycl::malloc_device(bytes, q);
        st.bufSize[slot] = bytes;
    }
    if (count) q.memcpy(st.buf[slot], data, count * sizeof(T));
    return static_cast<T *>(st.buf[slot]);
}

} // namespace

PRGPU_API int prgpu_api_version() { return kApiVersion; }

PRGPU_API int prgpu_list_devices(DeviceInfo *out, int max) {
    std::lock_guard<std::mutex> lock(gMutex);
    Enumerate();
    int n = 0;
    for (const auto &d : gDevices) {
        if (n >= max) break;
        DeviceInfo &di = out[n++];
        std::memset(&di, 0, sizeof(di));
        std::snprintf(di.name, sizeof(di.name), "%s", d.get_info<sycl::info::device::name>().c_str());
        di.globalMemory = d.get_info<sycl::info::device::global_mem_size>();
        di.computeUnits = int(d.get_info<sycl::info::device::max_compute_units>());
        di.isGpu = d.is_gpu() ? 1 : 0;
        std::snprintf(di.backend, sizeof(di.backend), "%s",
                      d.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "opencl");
    }
    return n;
}

PRGPU_API int prgpu_render(int deviceIndex, const SceneDesc *sd, const RenderParams *rp, float *outXYZ, char *err, int errLen) {
    std::lock_guard<std::mutex> lock(gMutex);
    Enumerate();
    if (deviceIndex < 0 || deviceIndex >= int(gDevices.size())) {
        SetErr(err, errLen, "no such GPU device");
        return 1;
    }
    if (!sd || sd->apiVersion != kApiVersion) {
        SetErr(err, errLen, "API version mismatch");
        return 1;
    }
    try {
        DeviceState &st = *gStates[size_t(deviceIndex)];
        if (!st.queue) st.queue = std::make_unique<sycl::queue>(gDevices[size_t(deviceIndex)], sycl::property::queue::in_order());
        sycl::queue &q = *st.queue;
        // Upload the scene once per render (keyed on the host buffers).
        if (st.key != sd->prims || st.keyPrims != sd->primCount || st.keyNodes != sd->nodeCount ||
            st.keySpectra != sd->spectrumCount || st.keyMaterials != sd->materialCount) {
            Upload(st, 0, sd->nodes, size_t(sd->nodeCount));
            Upload(st, 1, sd->primOrder, size_t(sd->primCount));
            Upload(st, 2, sd->prims, size_t(sd->primCount));
            Upload(st, 3, sd->materials, size_t(sd->materialCount));
            Upload(st, 4, sd->lights, size_t(sd->lightCount));
            Upload(st, 5, sd->spectra, size_t(sd->spectrumCount));
            Upload(st, 6, sd->media, size_t(sd->mediumCount));
            Upload(st, 7, sd->sheenAlbedo, size_t(kSheenAlpha * kSheenMu));
            Upload(st, 8, sd->filter.cdf, size_t(kFilterBins + 1));
            q.wait();
            st.key = sd->prims;
            st.keyPrims = sd->primCount;
            st.keyNodes = sd->nodeCount;
            st.keySpectra = sd->spectrumCount;
            st.keyMaterials = sd->materialCount;
        }
        DevScene ds;
        ds.nodes = static_cast<const BVHNode *>(st.buf[0]);
        ds.order = static_cast<const int32_t *>(st.buf[1]);
        ds.prims = static_cast<const Primitive *>(st.buf[2]);
        ds.mats = static_cast<const Material *>(st.buf[3]);
        ds.lights = static_cast<const Light *>(st.buf[4]);
        ds.lightCount = sd->lightCount;
        ds.spectra = static_cast<const Spectrum *>(st.buf[5]);
        ds.envSpectrum = sd->envSpectrum;
        ds.envScale = sd->envScale;
        ds.envPmf = sd->envPmf;
        ds.worldCenter = ToV3(sd->worldCenter);
        ds.worldRadius = sd->worldRadius;
        ds.cam = sd->camera;
        ds.yIntegral = sd->cieYIntegral;
        ds.media = static_cast<const Medium *>(st.buf[6]);
        ds.cameraMedium = sd->cameraMedium;
        ds.hasInterfaces = false;
        for (int i = 0; i < sd->materialCount; ++i)
            if (sd->materials[i].type == MatInterface) ds.hasInterfaces = true;
        ds.filter = FilterK{sd->filter.type, sd->filter.radius, sd->filter.sigma, sd->filter.norm,
                            static_cast<const float *>(st.buf[8])};
        ds.sheenTable = static_cast<const float *>(st.buf[7]);

        const int w = sd->camera.width, h = sd->camera.height;
        size_t count = size_t(w) * h * 3;
        if (st.filmCount < count) {
            if (st.film) sycl::free(st.film, q);
            st.film = sycl::malloc_device<float>(count, q);
            st.filmCount = count;
        }
        float *film = st.film;
        const int spp = rp->spp, first = rp->firstSample, maxDepth = rp->maxDepth;
        const float clampY = rp->clampLuminance;
        const uint64_t seed = rp->seed;
        auto launch = [&](auto full) {
            constexpr bool kFull = decltype(full)::value;
            q.parallel_for(sycl::range<1>(size_t(w) * h), [=](sycl::id<1> id) {
                 RenderPixel<kFull>(ds, int(id[0]), w, spp, first, seed, maxDepth, clampY, film);
             }).wait_and_throw();
        };
        bool full = false;
        for (int i = 0; i < sd->materialCount; ++i)
            if (sd->materials[i].type == MatCoatedDiffuse || sd->materials[i].type == MatCoatedConductor ||
                sd->materials[i].sheenWeight > 0)
                full = true;
        if (full) launch(std::true_type{});
        else launch(std::false_type{});
        std::vector<float> host(count);
        q.memcpy(host.data(), film, count * sizeof(float)).wait();
        for (size_t i = 0; i < count; ++i) outXYZ[i] += host[i];
        return 0;
    } catch (const sycl::exception &e) {
        SetErr(err, errLen, std::string("SYCL: ") + e.what());
        return 1;
    } catch (const std::exception &e) {
        SetErr(err, errLen, e.what());
        return 1;
    }
}

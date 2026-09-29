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
// BSDF

enum { FlagReflection = 1, FlagTransmission = 2, FlagSpecular = 16 };

struct BSDF {
    uint32_t type;
    Frame frame;
    SS R;           // diffuse albedo
    SS eta, k;      // conductor
    float etaD;     // dielectric (hero wavelength)
    uint32_t thin;
    TR mf;
    bool NonSpecular() const {
        if (type == MatDiffuse) return true;
        if (type == MatBlack) return false;
        if (type == MatDielectric && thin) return false;
        return !mf.Smooth();
    }
};

struct BSample {
    SS f;
    V3 wi;
    float pdf;
    uint32_t flags;
    float eta;
};

inline SS ConductorF(const BSDF &b, V3 wo, V3 wi) {
    if (wo.z * wi.z <= 0 || b.mf.Smooth()) return Splat(0);
    float co = sycl::fabs(wo.z), ci = sycl::fabs(wi.z);
    if (co == 0 || ci == 0) return Splat(0);
    V3 wm = wi + wo;
    if (Len2(wm) == 0) return Splat(0);
    wm = Norm(wm);
    float d = b.mf.D(wm) * b.mf.Gm(wo, wi) / (4 * ci * co);
    float c = sycl::fabs(Dot(wo, wm));
    SS F;
    for (int i = 0; i < 4; ++i) F.v[i] = FrComplex(c, b.eta.v[i], b.k.v[i]) * d;
    return F;
}

inline float DielectricPDF(const BSDF &b, V3 wo, V3 wi) {
    if (b.etaD == 1 || b.mf.Smooth()) return 0;
    float co = wo.z, ci = wi.z;
    bool refl = ci * co > 0;
    float etap = 1;
    if (!refl) etap = co > 0 ? b.etaD : 1 / b.etaD;
    V3 wm = wi * etap + wo;
    if (ci == 0 || co == 0 || Len2(wm) == 0) return 0;
    wm = FaceForward(Norm(wm), V3{0, 0, 1});
    if (Dot(wm, wi) * ci < 0 || Dot(wm, wo) * co < 0) return 0;
    float R = FrDielectric(Dot(wo, wm), b.etaD), T = 1 - R;
    if (refl) return b.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm))) * R / (R + T);
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
    return b.mf.PDF(wo, wm) * sycl::fabs(Dot(wi, wm)) / denom * T / (R + T);
}

inline float DielectricF(const BSDF &b, V3 wo, V3 wi) {
    if (b.etaD == 1 || b.mf.Smooth()) return 0;
    float co = wo.z, ci = wi.z;
    bool refl = ci * co > 0;
    float etap = 1;
    if (!refl) etap = co > 0 ? b.etaD : 1 / b.etaD;
    V3 wm = wi * etap + wo;
    if (ci == 0 || co == 0 || Len2(wm) == 0) return 0;
    wm = FaceForward(Norm(wm), V3{0, 0, 1});
    if (Dot(wm, wi) * ci < 0 || Dot(wm, wo) * co < 0) return 0;
    float F = FrDielectric(Dot(wo, wm), b.etaD);
    if (refl) return b.mf.D(wm) * b.mf.Gm(wo, wi) * F / sycl::fabs(4 * ci * co);
    float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap) * ci * co;
    float ft = b.mf.D(wm) * (1 - F) * b.mf.Gm(wo, wi) * sycl::fabs(Dot(wi, wm) * Dot(wo, wm) / denom);
    return ft / Sqr(etap);  // radiance transport
}

inline SS EvalF(const BSDF &b, V3 woW, V3 wiW) {
    V3 wo = b.frame.ToLocal(woW), wi = b.frame.ToLocal(wiW);
    if (wo.z == 0) return Splat(0);
    switch (b.type) {
    case MatDiffuse: return wo.z * wi.z > 0 ? b.R * kInvPi : Splat(0);
    case MatConductor: return ConductorF(b, wo, wi);
    case MatDielectric: return b.thin ? Splat(0) : Splat(DielectricF(b, wo, wi));
    default: return Splat(0);
    }
}

inline float EvalPDF(const BSDF &b, V3 woW, V3 wiW) {
    V3 wo = b.frame.ToLocal(woW), wi = b.frame.ToLocal(wiW);
    if (wo.z == 0) return 0;
    switch (b.type) {
    case MatDiffuse: return wo.z * wi.z > 0 ? sycl::fabs(wi.z) * kInvPi : 0;
    case MatConductor: {
        if (wo.z * wi.z <= 0 || b.mf.Smooth()) return 0;
        V3 wm = wo + wi;
        if (Len2(wm) == 0) return 0;
        wm = FaceForward(Norm(wm), V3{0, 0, 1});
        return b.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm)));
    }
    case MatDielectric: return b.thin ? 0 : DielectricPDF(b, wo, wi);
    default: return 0;
    }
}

inline bool SampleF(const BSDF &b, V3 woW, float uc, float u0, float u1, BSample *s) {
    V3 wo = b.frame.ToLocal(woW);
    if (wo.z == 0) return false;
    s->eta = 1;
    switch (b.type) {
    case MatDiffuse: {
        // Concentric disk -> cosine hemisphere.
        float ox = 2 * u0 - 1, oy = 2 * u1 - 1, r, th;
        if (ox == 0 && oy == 0) r = th = 0;
        else if (sycl::fabs(ox) > sycl::fabs(oy)) {
            r = ox;
            th = 0.78539816f * (oy / ox);
        } else {
            r = oy;
            th = 1.57079633f - 0.78539816f * (ox / oy);
        }
        V3 wi{r * sycl::cos(th), r * sycl::sin(th), 0};
        wi.z = SafeSqrt(1 - wi.x * wi.x - wi.y * wi.y);
        if (wo.z < 0) wi.z = -wi.z;
        s->pdf = sycl::fabs(wi.z) * kInvPi;
        s->f = b.R * kInvPi;
        s->wi = b.frame.FromLocal(wi);
        s->flags = FlagReflection;
        return s->pdf > 0;
    }
    case MatConductor: {
        if (b.mf.Smooth()) {
            V3 wi{-wo.x, -wo.y, wo.z};
            float c = sycl::fabs(wi.z);
            for (int i = 0; i < 4; ++i) s->f.v[i] = FrComplex(c, b.eta.v[i], b.k.v[i]) / c;
            s->pdf = 1;
            s->wi = b.frame.FromLocal(wi);
            s->flags = FlagReflection | FlagSpecular;
            return true;
        }
        V3 wm = b.mf.Sample(wo, u0, u1);
        V3 wi = -wo + wm * (2 * Dot(wo, wm));
        if (wo.z * wi.z <= 0) return false;
        s->pdf = b.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm)));
        s->f = ConductorF(b, wo, wi);
        s->wi = b.frame.FromLocal(wi);
        s->flags = FlagReflection;
        return s->pdf > 0;
    }
    case MatDielectric: {
        if (b.thin) {
            float R = FrDielectric(sycl::fabs(wo.z), b.etaD), T = 1 - R;
            if (R < 1) {
                R += T * T * R / (1 - R * R);
                T = 1 - R;
            }
            if (uc < R / (R + T)) {
                V3 wi{-wo.x, -wo.y, wo.z};
                s->f = Splat(R / sycl::fabs(wi.z));
                s->pdf = R / (R + T);
                s->wi = b.frame.FromLocal(wi);
                s->flags = FlagReflection | FlagSpecular;
            } else {
                V3 wi = -wo;
                s->f = Splat(T / sycl::fabs(wi.z));
                s->pdf = T / (R + T);
                s->wi = b.frame.FromLocal(wi);
                s->flags = FlagTransmission | FlagSpecular;
            }
            return true;
        }
        if (b.etaD == 1 || b.mf.Smooth()) {
            float R = FrDielectric(wo.z, b.etaD), T = 1 - R;
            if (uc < R / (R + T)) {
                V3 wi{-wo.x, -wo.y, wo.z};
                s->f = Splat(R / sycl::fabs(wi.z));
                s->pdf = R / (R + T);
                s->wi = b.frame.FromLocal(wi);
                s->flags = FlagReflection | FlagSpecular;
                return true;
            }
            V3 wi;
            float etap;
            if (!Refract(wo, V3{0, 0, 1}, b.etaD, &etap, &wi)) return false;
            s->f = Splat(T / sycl::fabs(wi.z) / (etap * etap));
            s->pdf = T / (R + T);
            s->wi = b.frame.FromLocal(wi);
            s->flags = FlagTransmission | FlagSpecular;
            s->eta = etap;
            return true;
        }
        V3 wm = b.mf.Sample(wo, u0, u1);
        float R = FrDielectric(Dot(wo, wm), b.etaD), T = 1 - R;
        if (uc < R / (R + T)) {
            V3 wi = -wo + wm * (2 * Dot(wo, wm));
            if (wo.z * wi.z <= 0) return false;
            s->pdf = b.mf.PDF(wo, wm) / (4 * sycl::fabs(Dot(wo, wm))) * R / (R + T);
            s->f = Splat(b.mf.D(wm) * b.mf.Gm(wo, wi) * R / (4 * wi.z * wo.z));
            s->wi = b.frame.FromLocal(wi);
            s->flags = FlagReflection;
            return s->pdf > 0;
        }
        V3 wi;
        float etap;
        if (!Refract(wo, wm, b.etaD, &etap, &wi) || wo.z * wi.z > 0 || wi.z == 0) return false;
        float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
        s->pdf = b.mf.PDF(wo, wm) * sycl::fabs(Dot(wi, wm)) / denom * T / (R + T);
        float ft = T * b.mf.D(wm) * b.mf.Gm(wo, wi) * sycl::fabs(Dot(wi, wm) * Dot(wo, wm) / (wi.z * wo.z * denom));
        s->f = Splat(ft / (etap * etap));
        s->wi = b.frame.FromLocal(wi);
        s->flags = FlagTransmission;
        s->eta = etap;
        return s->pdf > 0;
    }
    default: return false;
    }
}

// ---------------------------------------------------------------------------------------------
// Scene on the device

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

// target is a point, or a direction when toInfinity.
inline bool Visible(const DevScene &s, V3 p, V3 n, V3 target, bool toInfinity) {
    V3 d = toInfinity ? target : target - p;
    V3 o = p + (Dot(d, n) < 0 ? -n : n) * RayEps(p);
    Hit h;
    if (toInfinity) return !Intersect(s, o, d, kInf, &h, true);
    return !Intersect(s, o, target - o, 1 - 1e-4f, &h, true);
}

inline BSDF MakeBSDF(const DevScene &s, const Material &m, const Surf &sf, Lambda &lam) {
    BSDF b{};
    b.type = m.type;
    b.frame = FrameFromXZ(sf.dpdu, sf.ns);
    b.mf = TR{m.alphaX, m.alphaY};
    if (!b.mf.Smooth()) {
        b.mf.ax = sycl::fmax(b.mf.ax, 1e-4f);
        b.mf.ay = sycl::fmax(b.mf.ay, 1e-4f);
    }
    if (m.type == MatDiffuse) {
        int id = m.albedo;
        if (m.albedoB >= 0 && m.checkerScale > 0) {
            int odd = (int(sycl::floor(sf.u * m.checkerScale)) + int(sycl::floor(sf.v * m.checkerScale))) & 1;
            if (odd) id = m.albedoB;
        }
        b.R = TabSS(s.spectra, id, lam);
        for (int i = 0; i < 4; ++i) b.R.v[i] = Clampf(b.R.v[i], 0.f, 1.f);
    } else if (m.type == MatConductor) {
        if (m.eta >= 0) {
            b.eta = TabSS(s.spectra, m.eta, lam);
            b.k = TabSS(s.spectra, m.k, lam);
        } else {
            SS r = TabSS(s.spectra, m.albedo, lam);
            for (int i = 0; i < 4; ++i) {
                float ri = Clampf(r.v[i], 0.f, 0.9999f);
                b.eta.v[i] = 1;
                b.k.v[i] = 2 * sycl::sqrt(ri) / SafeSqrt(1 - ri);
            }
        }
    } else if (m.type == MatDielectric) {
        b.etaD = Tab(s.spectra, m.eta, lam.l[0]);
        if (m.dispersive) lam.TerminateSecondary();
        if (b.etaD == 0) b.etaD = 1;
        b.thin = m.thin;
    }
    return b;
}

// ---------------------------------------------------------------------------------------------
// Path tracing (mirrors pr::PathIntegrator::Li without media)

struct Ctx {
    V3 p, n;
};

inline SS Li(const DevScene &s, V3 o, V3 d, Lambda &lam, RNG &rng, int maxDepth) {
    SS L = Splat(0), beta = Splat(1);
    bool specular = false;
    int depth = 0;
    float etaScale = 1, prevPdf = 1;
    Ctx prev{o, {0, 0, 0}};
    for (int guard = 0; guard < 4096; ++guard) {
        Hit h;
        if (!Intersect(s, o, d, kInf, &h, false)) {
            if (s.envSpectrum >= 0) {
                SS Le = TabSS(s.spectra, s.envSpectrum, lam) * s.envScale;
                float w = 1;
                if (depth > 0 && !specular) w = PowerHeuristic(prevPdf, s.envPmf * kInv4Pi);
                L = L + beta * Le * w;
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
                if (depth > 0 && !specular) w = PowerHeuristic(prevPdf, l.pmf * AreaPdf(l, prev.p, sf.p, sf.n));
                L = L + beta * Le * w;
            }
        }
        const Material &m = s.mats[pr.material];
        if (m.type == MatBlack) break;
        if (maxDepth > 0 && depth >= maxDepth) break;
        ++depth;
        BSDF b = MakeBSDF(s, m, sf, lam);
        V3 wo = -d;

        // Next-event estimation.
        if (b.NonSpecular()) {
            float ul = rng.U(), u0 = rng.U(), u1 = rng.U();
            SS Ld = Splat(0);
            if (ul < s.envPmf && s.envSpectrum >= 0) {
                float z = 1 - 2 * u0, r = SafeSqrt(1 - z * z), ph = 2 * kPi * u1;
                V3 wi{r * sycl::cos(ph), r * sycl::sin(ph), z};
                SS f = EvalF(b, wo, wi) * sycl::fabs(Dot(wi, sf.ns));
                if (NonZero(f) && Visible(s, sf.p, sf.n, wi, true)) {
                    float lp = s.envPmf * kInv4Pi;
                    float w = PowerHeuristic(lp, EvalPDF(b, wo, wi));
                    Ld = f * TabSS(s.spectra, s.envSpectrum, lam) * (s.envScale * w / lp);
                }
            } else if (s.lightCount > 0) {
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
                if (l.type == LightPoint) {
                    V3 pl = ToV3(l.position);
                    V3 dd = pl - sf.p;
                    float d2 = Len2(dd);
                    V3 wi = dd * (1 / sycl::sqrt(d2));
                    SS f = EvalF(b, wo, wi) * sycl::fabs(Dot(wi, sf.ns));
                    if (NonZero(f) && Visible(s, sf.p, sf.n, pl, false))
                        Ld = f * TabSS(s.spectra, l.spectrum, lam) * (l.scale / d2 / l.pmf);
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
                    V3 dd = pl - sf.p;
                    float d2 = Len2(dd);
                    if (d2 > 0) {
                        V3 wi = dd * (1 / sycl::sqrt(d2));
                        SS Le = Emit(s, l, nl, -wi, lam);
                        float lpdf = l.pmf * AreaPdf(l, sf.p, pl, nl);
                        if (NonZero(Le) && lpdf > 0 && sycl::isfinite(lpdf)) {
                            SS f = EvalF(b, wo, wi) * sycl::fabs(Dot(wi, sf.ns));
                            V3 target = pl + (Dot(-dd, nl) < 0 ? -nl : nl) * RayEps(pl);
                            if (NonZero(f) && Visible(s, sf.p, sf.n, target, false)) {
                                float w = PowerHeuristic(lpdf, EvalPDF(b, wo, wi));
                                Ld = f * Le * (w / lpdf);
                            }
                        }
                    }
                }
            }
            L = L + beta * Ld;
        }

        // BSDF sampling.
        BSample bs;
        float uc = rng.U(), u0 = rng.U(), u1 = rng.U();
        if (!SampleF(b, wo, uc, u0, u1, &bs)) break;
        beta = beta * bs.f * (sycl::fabs(Dot(bs.wi, sf.ns)) / bs.pdf);
        prevPdf = bs.pdf;
        specular = (bs.flags & FlagSpecular) != 0;
        if (bs.flags & FlagTransmission) etaScale *= bs.eta * bs.eta;
        prev = {sf.p, sf.n};
        o = sf.p + (Dot(bs.wi, sf.n) < 0 ? -sf.n : sf.n) * RayEps(sf.p);
        d = bs.wi;
        if (!NonZero(beta) || !sycl::isfinite(MaxC(beta))) break;
        if (depth > 1) {
            float mx = MaxC(beta) * etaScale;
            if (mx < 1) {
                float q = sycl::fmax(0.f, 1 - mx);
                if (rng.U() < q) break;
                beta = beta * (1 / (1 - q));
            }
        }
    }
    return L;
}

// ---------------------------------------------------------------------------------------------
// Device management

struct DeviceState {
    std::unique_ptr<sycl::queue> queue;
    // Scene upload cache.
    const void *key = nullptr;
    int32_t keyPrims = -1, keyNodes = -1, keySpectra = -1;
    void *buf[8] = {};
    size_t bufSize[8] = {};
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
            st.keySpectra != sd->spectrumCount) {
            Upload(st, 0, sd->nodes, size_t(sd->nodeCount));
            Upload(st, 1, sd->primOrder, size_t(sd->primCount));
            Upload(st, 2, sd->prims, size_t(sd->primCount));
            Upload(st, 3, sd->materials, size_t(sd->materialCount));
            Upload(st, 4, sd->lights, size_t(sd->lightCount));
            Upload(st, 5, sd->spectra, size_t(sd->spectrumCount));
            q.wait();
            st.key = sd->prims;
            st.keyPrims = sd->primCount;
            st.keyNodes = sd->nodeCount;
            st.keySpectra = sd->spectrumCount;
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

        const int w = sd->camera.width, h = sd->camera.height;
        size_t count = size_t(w) * h * 3;
        if (st.filmCount < count) {
            if (st.film) sycl::free(st.film, q);
            st.film = sycl::malloc_device<float>(count, q);
            st.filmCount = count;
        }
        float *film = st.film;
        const int spp = rp->spp, first = rp->firstSample, maxDepth = rp->maxDepth;
        const uint64_t seed = rp->seed;
        q.parallel_for(sycl::range<1>(size_t(w) * h), [=](sycl::id<1> id) {
             int pix = int(id[0]);
             int px = pix % w, py = pix / w;
             const Camera &c = ds.cam;
             float X = 0, Y = 0, Z = 0;
             for (int si = 0; si < spp; ++si) {
                 RNG rng;
                 rng.Seed(RNG::Mix(uint64_t(pix) * 0x9E3779B97F4A7C15ull ^ seed), RNG::Mix(uint64_t(first + si) + 1));
                 Lambda lam = SampleVisible(rng.U());
                 float fx = float(px) + rng.U(), fy = float(py) + rng.U();
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
                 SS L = Li(ds, o, dw, lam, rng, maxDepth);
                 if (!sycl::isfinite(L.v[0] + L.v[1] + L.v[2] + L.v[3])) continue;
                 for (int i = 0; i < 4; ++i) {
                     if (lam.pdf[i] == 0) continue;
                     float v = L.v[i] / lam.pdf[i];
                     X += CieX(lam.l[i]) * v;
                     Y += CieY(lam.l[i]) * v;
                     Z += CieZ(lam.l[i]) * v;
                 }
             }
             float norm = 1.f / (4 * ds.yIntegral);
             film[3 * pix + 0] = X * norm;
             film[3 * pix + 1] = Y * norm;
             film[3 * pix + 2] = Z * norm;
         }).wait_and_throw();
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

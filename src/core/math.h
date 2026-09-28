#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace pr {

constexpr float Pi = 3.14159265358979323846f;
constexpr float InvPi = 0.31830988618379067154f;
constexpr float Inv2Pi = 0.15915494309189533577f;
constexpr float Inv4Pi = 0.07957747154594766788f;
constexpr float PiOver2 = 1.57079632679489661923f;
constexpr float PiOver4 = 0.78539816339744830961f;
constexpr float Sqrt2 = 1.41421356237309504880f;
constexpr float Infinity = std::numeric_limits<float>::infinity();
constexpr float OneMinusEpsilon = 0x1.fffffep-1f;

template <typename T, typename U, typename V>
constexpr T Clamp(T v, U lo, V hi) {
    return v < T(lo) ? T(lo) : (v > T(hi) ? T(hi) : v);
}
inline float Lerp(float t, float a, float b) { return (1 - t) * a + t * b; }
template <typename T> constexpr T Sqr(T v) { return v * v; }
inline float SafeSqrt(float x) { return std::sqrt(std::max(0.f, x)); }
inline float SafeASin(float x) { return std::asin(Clamp(x, -1.f, 1.f)); }
inline float SafeACos(float x) { return std::acos(Clamp(x, -1.f, 1.f)); }
inline float Radians(float deg) { return (Pi / 180.f) * deg; }
inline float Degrees(float rad) { return (180.f / Pi) * rad; }
inline float SmoothStep(float x, float a, float b) {
    if (a == b) return x < a ? 0.f : 1.f;
    float t = Clamp((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

// Evaluate polynomial c0 + c1 t + c2 t^2 + ...
template <typename... Args> inline float EvaluatePolynomial(float, float c) { return c; }
template <typename... Args> inline float EvaluatePolynomial(float t, float c, Args... cs) {
    return std::fma(t, EvaluatePolynomial(t, cs...), c);
}

// Find the largest index i in [0, size-2] such that pred(i) is true (pred must be monotone).
template <typename Pred> int FindInterval(int size, const Pred &pred) {
    int first = 1, sz = size - 2;
    while (sz > 0) {
        int half = sz >> 1, middle = first + half;
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        sz = predResult ? sz - (half + 1) : half;
    }
    return Clamp(first - 1, 0, size - 2);
}

// ---------------------------------------------------------------------------------------------
struct Vec2f {
    float x = 0, y = 0;
    constexpr Vec2f() = default;
    constexpr Vec2f(float x, float y) : x(x), y(y) {}
    float operator[](int i) const { return i == 0 ? x : y; }
    float &operator[](int i) { return i == 0 ? x : y; }
    Vec2f operator+(Vec2f o) const { return {x + o.x, y + o.y}; }
    Vec2f operator-(Vec2f o) const { return {x - o.x, y - o.y}; }
    Vec2f operator*(float s) const { return {x * s, y * s}; }
    Vec2f operator/(float s) const { return {x / s, y / s}; }
};

struct Vec3f {
    float x = 0, y = 0, z = 0;
    constexpr Vec3f() = default;
    constexpr Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    Vec3f operator+(const Vec3f &o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3f operator-(const Vec3f &o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3f operator*(const Vec3f &o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3f operator/(float s) const {
        float inv = 1.f / s;
        return {x * inv, y * inv, z * inv};
    }
    Vec3f operator-() const { return {-x, -y, -z}; }
    Vec3f &operator+=(const Vec3f &o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3f &operator-=(const Vec3f &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3f &operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    bool operator==(const Vec3f &o) const { return x == o.x && y == o.y && z == o.z; }
    bool operator!=(const Vec3f &o) const { return !(*this == o); }
    bool IsZero() const { return x == 0 && y == 0 && z == 0; }
};
inline Vec3f operator*(float s, const Vec3f &v) { return v * s; }

inline float Dot(const Vec3f &a, const Vec3f &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float AbsDot(const Vec3f &a, const Vec3f &b) { return std::abs(Dot(a, b)); }
inline Vec3f Cross(const Vec3f &a, const Vec3f &b) {
    // Use double internally to reduce cancellation (as in pbrt).
    double ax = a.x, ay = a.y, az = a.z, bx = b.x, by = b.y, bz = b.z;
    return {float(ay * bz - az * by), float(az * bx - ax * bz), float(ax * by - ay * bx)};
}
inline float LengthSquared(const Vec3f &v) { return Dot(v, v); }
inline float Length(const Vec3f &v) { return std::sqrt(LengthSquared(v)); }
inline Vec3f Normalize(const Vec3f &v) { return v / Length(v); }
inline float Distance(const Vec3f &a, const Vec3f &b) { return Length(a - b); }
inline float DistanceSquared(const Vec3f &a, const Vec3f &b) { return LengthSquared(a - b); }
inline Vec3f Abs(const Vec3f &v) { return {std::abs(v.x), std::abs(v.y), std::abs(v.z)}; }
inline Vec3f Min(const Vec3f &a, const Vec3f &b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3f Max(const Vec3f &a, const Vec3f &b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
inline float MaxComponent(const Vec3f &v) { return std::max(v.x, std::max(v.y, v.z)); }
inline int MaxDimension(const Vec3f &v) {
    return (v.x > v.y) ? (v.x > v.z ? 0 : 2) : (v.y > v.z ? 1 : 2);
}
inline Vec3f FaceForward(const Vec3f &n, const Vec3f &v) { return Dot(n, v) < 0.f ? -n : n; }
inline Vec3f Lerp(float t, const Vec3f &a, const Vec3f &b) { return (1 - t) * a + t * b; }

// Duff et al. 2017 orthonormal basis.
inline void CoordinateSystem(const Vec3f &v1, Vec3f *v2, Vec3f *v3) {
    float sign = std::copysign(1.f, v1.z);
    float a = -1 / (sign + v1.z);
    float b = v1.x * v1.y * a;
    *v2 = Vec3f(1 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    *v3 = Vec3f(b, sign + v1.y * v1.y * a, -v1.y);
}

struct Frame {
    Vec3f x{1, 0, 0}, y{0, 1, 0}, z{0, 0, 1};
    static Frame FromZ(const Vec3f &z) {
        Frame f;
        f.z = z;
        CoordinateSystem(z, &f.x, &f.y);
        return f;
    }
    // Build a frame from a (not necessarily orthogonal) tangent and a normal.
    static Frame FromXZ(const Vec3f &xIn, const Vec3f &z) {
        Vec3f x = xIn - z * Dot(xIn, z);
        float l2 = LengthSquared(x);
        if (l2 < 1e-12f || !std::isfinite(l2)) return FromZ(z);
        x = x / std::sqrt(l2);
        Frame f;
        f.z = z;
        f.x = x;
        f.y = Cross(z, x);
        return f;
    }
    Vec3f ToLocal(const Vec3f &v) const { return {Dot(v, x), Dot(v, y), Dot(v, z)}; }
    Vec3f FromLocal(const Vec3f &v) const { return x * v.x + y * v.y + z * v.z; }
};

// Local shading-space helpers (z is the normal).
inline float CosTheta(const Vec3f &w) { return w.z; }
inline float Cos2Theta(const Vec3f &w) { return w.z * w.z; }
inline float AbsCosTheta(const Vec3f &w) { return std::abs(w.z); }
inline float Sin2Theta(const Vec3f &w) { return std::max(0.f, 1 - Cos2Theta(w)); }
inline float SinTheta(const Vec3f &w) { return std::sqrt(Sin2Theta(w)); }
inline float TanTheta(const Vec3f &w) { return SinTheta(w) / CosTheta(w); }
inline float Tan2Theta(const Vec3f &w) { return Sin2Theta(w) / Cos2Theta(w); }
inline float CosPhi(const Vec3f &w) {
    float s = SinTheta(w);
    return s == 0 ? 1 : Clamp(w.x / s, -1.f, 1.f);
}
inline float SinPhi(const Vec3f &w) {
    float s = SinTheta(w);
    return s == 0 ? 0 : Clamp(w.y / s, -1.f, 1.f);
}
inline bool SameHemisphere(const Vec3f &a, const Vec3f &b) { return a.z * b.z > 0; }
inline Vec3f SphericalDirection(float sinTheta, float cosTheta, float phi) {
    return {Clamp(sinTheta, -1.f, 1.f) * std::cos(phi), Clamp(sinTheta, -1.f, 1.f) * std::sin(phi),
            Clamp(cosTheta, -1.f, 1.f)};
}
inline float SphericalTheta(const Vec3f &v) { return SafeACos(v.z); }
inline float SphericalPhi(const Vec3f &v) {
    float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

inline Vec3f Reflect(const Vec3f &wo, const Vec3f &n) { return -wo + n * (2 * Dot(wo, n)); }

// Refract wi about n with relative IOR eta (eta = n_t / n_i measured w.r.t. n's side).
// Returns false on total internal reflection. etap receives the effective relative IOR.
inline bool Refract(const Vec3f &wi, Vec3f n, float eta, float *etap, Vec3f *wt) {
    float cosTheta_i = Dot(n, wi);
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }
    float sin2Theta_i = std::max(0.f, 1 - Sqr(cosTheta_i));
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1) return false;
    float cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    *wt = -wi / eta + n * (cosTheta_i / eta - cosTheta_t);
    if (etap) *etap = eta;
    return true;
}

// ---------------------------------------------------------------------------------------------
struct Ray {
    Vec3f o, d;
    Ray() = default;
    Ray(const Vec3f &o, const Vec3f &d) : o(o), d(d) {}
    Vec3f operator()(float t) const { return o + d * t; }
};

struct Bounds3f {
    Vec3f pMin{Infinity, Infinity, Infinity}, pMax{-Infinity, -Infinity, -Infinity};
    Bounds3f() = default;
    explicit Bounds3f(const Vec3f &p) : pMin(p), pMax(p) {}
    Bounds3f(const Vec3f &a, const Vec3f &b) : pMin(Min(a, b)), pMax(Max(a, b)) {}
    bool IsEmpty() const { return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z; }
    Vec3f Diagonal() const { return pMax - pMin; }
    Vec3f Centroid() const { return (pMin + pMax) * 0.5f; }
    float SurfaceArea() const {
        if (IsEmpty()) return 0;
        Vec3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    int MaxExtent() const { return MaxDimension(Diagonal()); }
    Vec3f Offset(const Vec3f &p) const {
        Vec3f o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    bool Inside(const Vec3f &p) const {
        return p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y && p.z >= pMin.z &&
               p.z <= pMax.z;
    }
    void BoundingSphere(Vec3f *c, float *r) const {
        *c = Centroid();
        *r = IsEmpty() ? 0.f : Distance(*c, pMax);
    }
    const Vec3f &operator[](int i) const { return i == 0 ? pMin : pMax; }
};
inline Bounds3f Union(const Bounds3f &b, const Vec3f &p) {
    Bounds3f r;
    r.pMin = Min(b.pMin, p);
    r.pMax = Max(b.pMax, p);
    return r;
}
inline Bounds3f Union(const Bounds3f &a, const Bounds3f &b) {
    Bounds3f r;
    r.pMin = Min(a.pMin, b.pMin);
    r.pMax = Max(a.pMax, b.pMax);
    return r;
}

// ---------------------------------------------------------------------------------------------
// 4x4 affine/projective transform (row-major, column vectors).
struct Mat4 {
    float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    static Mat4 Identity() { return Mat4(); }
    Mat4 operator*(const Mat4 &b) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int k = 0; k < 4; ++k) s += double(m[i][k]) * b.m[k][j];
                r.m[i][j] = float(s);
            }
        return r;
    }
    Mat4 Transposed() const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) r.m[i][j] = m[j][i];
        return r;
    }
    bool Inverse(Mat4 *out) const;
};

class Transform {
  public:
    Transform() = default;
    explicit Transform(const Mat4 &m) : m_(m) { m.Inverse(&mInv_); }
    Transform(const Mat4 &m, const Mat4 &mInv) : m_(m), mInv_(mInv) {}

    static Transform Translate(const Vec3f &d);
    static Transform Scale(const Vec3f &s);
    static Transform Rotate(float degrees, const Vec3f &axis);
    // Camera-to-world transform for a camera at pos looking at target (+z forward).
    static Transform LookAt(const Vec3f &pos, const Vec3f &target, const Vec3f &up);

    Transform operator*(const Transform &t) const { return Transform(m_ * t.m_, t.mInv_ * mInv_); }
    Transform Inverse() const { return Transform(mInv_, m_); }
    const Mat4 &Matrix() const { return m_; }

    Vec3f Point(const Vec3f &p) const {
        const auto &m = m_.m;
        float x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        float y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        float z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        float w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        return w == 1 ? Vec3f(x, y, z) : Vec3f(x, y, z) / w;
    }
    Vec3f Vector(const Vec3f &v) const {
        const auto &m = m_.m;
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    // Normals transform with the inverse transpose.
    Vec3f Normal(const Vec3f &n) const {
        const auto &mi = mInv_.m;
        return {mi[0][0] * n.x + mi[1][0] * n.y + mi[2][0] * n.z,
                mi[0][1] * n.x + mi[1][1] * n.y + mi[2][1] * n.z,
                mi[0][2] * n.x + mi[1][2] * n.y + mi[2][2] * n.z};
    }

  private:
    Mat4 m_, mInv_;
};

} // namespace pr

#pragma once

#include "core/math.h"
#include "core/spectrum.h"

#include <string>

namespace pr {

struct RGB {
    float r = 0, g = 0, b = 0;
    RGB() = default;
    RGB(float r, float g, float b) : r(r), g(g), b(b) {}
    float operator[](int i) const { return i == 0 ? r : (i == 1 ? g : b); }
    float &operator[](int i) { return i == 0 ? r : (i == 1 ? g : b); }
    RGB operator*(float s) const { return {r * s, g * s, b * s}; }
    RGB operator+(const RGB &o) const { return {r + o.r, g + o.g, b + o.b}; }
    float Max() const { return std::max(r, std::max(g, b)); }
};

struct Mat3 {
    float m[3][3];
    RGB Apply(float x, float y, float z) const {
        return {m[0][0] * x + m[0][1] * y + m[0][2] * z, m[1][0] * x + m[1][1] * y + m[1][2] * z,
                m[2][0] * x + m[2][1] * y + m[2][2] * z};
    }
    Mat3 operator*(const Mat3 &b) const;
    Mat3 Inverse() const;
};

enum class ColorSpaceId { sRGB, ACEScg, Rec2020 };
bool ParseColorSpace(const std::string &name, ColorSpaceId *out);
const char *ColorSpaceName(ColorSpaceId id);

// XYZ (D65 white) to linear RGB for the given colour space.
const Mat3 &XYZToRGBMatrix(ColorSpaceId cs);
const Mat3 &RGBToXYZMatrix(ColorSpaceId cs);  // sRGB/Rec2020 only are D65-native

// Chromatic adaptation that maps the renderer's own spectral D65 white (as seen through the
// analytic CMFs) to the exact CIE D65 white point, so a perfect white reflector renders neutral.
const Mat3 &SpectralWhiteAdaptation();
// Bradford adaptation between two XYZ whites.
Mat3 BradfordAdaptation(const XYZ &srcWhite, const XYZ &dstWhite);

inline float SRGBEncode(float v) {
    v = Clamp(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}
inline float SRGBDecode(float v) {
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

enum class ToneMap { None, Clamp, Reinhard, ACES };
bool ParseToneMap(const std::string &name, ToneMap *out);
RGB ApplyToneMap(ToneMap tm, RGB linearSRGB);

} // namespace pr

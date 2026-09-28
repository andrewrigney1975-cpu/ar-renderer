#include "core/color.h"

namespace pr {

Mat3 Mat3::operator*(const Mat3 &b) const {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += double(m[i][k]) * b.m[k][j];
            r.m[i][j] = float(s);
        }
    return r;
}

Mat3 Mat3::Inverse() const {
    double a = m[0][0], b = m[0][1], c = m[0][2], d = m[1][0], e = m[1][1], f = m[1][2], g = m[2][0],
           h = m[2][1], i = m[2][2];
    double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    double det = a * A + b * B + c * C;
    double inv = 1.0 / det;
    Mat3 r{};
    r.m[0][0] = float(A * inv);
    r.m[0][1] = float(-(b * i - c * h) * inv);
    r.m[0][2] = float((b * f - c * e) * inv);
    r.m[1][0] = float(B * inv);
    r.m[1][1] = float((a * i - c * g) * inv);
    r.m[1][2] = float(-(a * f - c * d) * inv);
    r.m[2][0] = float(C * inv);
    r.m[2][1] = float(-(a * h - b * g) * inv);
    r.m[2][2] = float((a * e - b * d) * inv);
    return r;
}

bool ParseColorSpace(const std::string &name, ColorSpaceId *out) {
    if (name == "srgb" || name == "sRGB" || name == "rec709" || name == "linear_srgb") *out = ColorSpaceId::sRGB;
    else if (name == "acescg" || name == "ACEScg" || name == "aces") *out = ColorSpaceId::ACEScg;
    else if (name == "rec2020" || name == "Rec2020") *out = ColorSpaceId::Rec2020;
    else return false;
    return true;
}

const char *ColorSpaceName(ColorSpaceId id) {
    switch (id) {
    case ColorSpaceId::sRGB: return "srgb";
    case ColorSpaceId::ACEScg: return "acescg";
    case ColorSpaceId::Rec2020: return "rec2020";
    }
    return "?";
}

static const Mat3 kXYZToSRGB = {{{3.2404542f, -1.5371385f, -0.4985314f},
                                 {-0.9692660f, 1.8760108f, 0.0415560f},
                                 {0.0556434f, -0.2040259f, 1.0572252f}}};
static const Mat3 kSRGBToXYZ = {{{0.4124564f, 0.3575761f, 0.1804375f},
                                 {0.2126729f, 0.7151522f, 0.0721750f},
                                 {0.0193339f, 0.1191920f, 0.9503041f}}};
// XYZ (D65) -> ACEScg (AP1, D60) including Bradford adaptation D65 -> D60.
static const Mat3 kXYZToACEScg = {{{1.6410234f, -0.3248033f, -0.2364247f},
                                   {-0.6636629f, 1.6153316f, 0.0167563f},
                                   {0.0117219f, -0.0082844f, 0.9883949f}}};
static const Mat3 kXYZToRec2020 = {{{1.7166512f, -0.3556708f, -0.2533663f},
                                    {-0.6666844f, 1.6164812f, 0.0157685f},
                                    {0.0176399f, -0.0427706f, 0.9421031f}}};

const Mat3 &XYZToRGBMatrix(ColorSpaceId cs) {
    switch (cs) {
    case ColorSpaceId::ACEScg: return kXYZToACEScg;
    case ColorSpaceId::Rec2020: return kXYZToRec2020;
    default: return kXYZToSRGB;
    }
}

const Mat3 &RGBToXYZMatrix(ColorSpaceId cs) {
    static const Mat3 acesInv = kXYZToACEScg.Inverse();
    static const Mat3 rec2020Inv = kXYZToRec2020.Inverse();
    switch (cs) {
    case ColorSpaceId::ACEScg: return acesInv;
    case ColorSpaceId::Rec2020: return rec2020Inv;
    default: return kSRGBToXYZ;
    }
}

Mat3 BradfordAdaptation(const XYZ &src, const XYZ &dst) {
    static const Mat3 B = {{{0.8951f, 0.2664f, -0.1614f}, {-0.7502f, 1.7135f, 0.0367f}, {0.0389f, -0.0685f, 1.0296f}}};
    static const Mat3 Binv = B.Inverse();
    RGB s = B.Apply(src.x, src.y, src.z), d = B.Apply(dst.x, dst.y, dst.z);
    Mat3 D = {{{d.r / s.r, 0, 0}, {0, d.g / s.g, 0}, {0, 0, d.b / s.b}}};
    return Binv * (D * B);
}

const Mat3 &SpectralWhiteAdaptation() {
    static const Mat3 m = [] {
        XYZ ours = SpectrumToXYZ(StdIlluminantD65());
        XYZ d65{0.95047f, 1.0f, 1.08883f};
        return BradfordAdaptation(ours, d65);
    }();
    return m;
}

bool ParseToneMap(const std::string &name, ToneMap *out) {
    if (name == "none" || name == "linear") *out = ToneMap::None;
    else if (name == "clamp") *out = ToneMap::Clamp;
    else if (name == "reinhard") *out = ToneMap::Reinhard;
    else if (name == "aces" || name == "filmic") *out = ToneMap::ACES;
    else return false;
    return true;
}

static float RRTAndODTFit(float v) {
    float a = v * (v + 0.0245786f) - 0.000090537f;
    float b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

RGB ApplyToneMap(ToneMap tm, RGB c) {
    switch (tm) {
    case ToneMap::None:
    case ToneMap::Clamp:
        return {Clamp(c.r, 0.f, 1.f), Clamp(c.g, 0.f, 1.f), Clamp(c.b, 0.f, 1.f)};
    case ToneMap::Reinhard: {
        auto f = [](float x) { x = std::max(0.f, x); return x / (1 + x); };
        return {f(c.r), f(c.g), f(c.b)};
    }
    case ToneMap::ACES: {
        // Stephen Hill's fit of the ACES RRT+ODT (sRGB in, sRGB out).
        static const Mat3 in = {{{0.59719f, 0.35458f, 0.04823f}, {0.07600f, 0.90834f, 0.01566f}, {0.02840f, 0.13383f, 0.83777f}}};
        static const Mat3 out = {{{1.60475f, -0.53108f, -0.07367f}, {-0.10208f, 1.10813f, -0.00605f}, {-0.00327f, -0.07276f, 1.07602f}}};
        RGB v = in.Apply(std::max(0.f, c.r), std::max(0.f, c.g), std::max(0.f, c.b));
        v = {RRTAndODTFit(v.r), RRTAndODTFit(v.g), RRTAndODTFit(v.b)};
        v = out.Apply(v.r, v.g, v.b);
        return {Clamp(v.r, 0.f, 1.f), Clamp(v.g, 0.f, 1.f), Clamp(v.b, 0.f, 1.f)};
    }
    }
    return c;
}

} // namespace pr

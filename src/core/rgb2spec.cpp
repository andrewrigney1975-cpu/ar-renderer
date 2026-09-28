#include "core/rgb2spec.h"

#include "core/log.h"
#include "core/parallel.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef RGB
#endif

namespace pr {

float RGBSigmoidPolynomial::MaxValue() const {
    float result = std::max((*this)(LambdaMin), (*this)(LambdaMax));
    if (c0_ != 0) {
        float lambda = -c1_ / (2 * c0_);
        if (lambda >= LambdaMin && lambda <= LambdaMax) result = std::max(result, (*this)(lambda));
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// Table construction (after pbrt-v4's rgb2spec_opt.cpp, Jakob & Hanika 2019).
namespace {

constexpr int kFineSamples = 189;  // 2.5nm spacing over [360, 830]

struct FitTables {
    double lambda[kFineSamples];
    double rgb[3][kFineSamples];  // spectrum -> linear sRGB weights (includes D65 and adaptation)
    double xyzWhite[3];
    double rgbToXyz[3][3];
};

const FitTables &GetFitTables() {
    static const FitTables t = [] {
        FitTables t{};
        const Mat3 &adapt = SpectralWhiteAdaptation();
        const Mat3 toRGB = XYZToRGBMatrix(ColorSpaceId::sRGB) * adapt;
        const Mat3 &toXYZ = RGBToXYZMatrix(ColorSpaceId::sRGB);
        const Spectrum &d65 = StdIlluminantD65();
        double h = (LambdaMax - LambdaMin) / (kFineSamples - 1);
        double norm = 1.0 / CIE_Y_Integral();
        for (int i = 0; i < kFineSamples; ++i) {
            double l = LambdaMin + i * h;
            double w = (i == 0 || i == kFineSamples - 1) ? 0.5 * h : h;  // trapezoid rule
            double I = d65(float(l));
            double xyz[3] = {CIE_X(float(l)), CIE_Y(float(l)), CIE_Z(float(l))};
            t.lambda[i] = l;
            for (int k = 0; k < 3; ++k) {
                double s = 0;
                for (int j = 0; j < 3; ++j) s += toRGB.m[k][j] * xyz[j];
                t.rgb[k][i] = s * I * w * norm;
            }
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t.rgbToXyz[i][j] = toXYZ.m[i][j];
        for (int i = 0; i < 3; ++i) t.xyzWhite[i] = t.rgbToXyz[i][0] + t.rgbToXyz[i][1] + t.rgbToXyz[i][2];
        return t;
    }();
    return t;
}

void CieLab(const FitTables &t, double *p) {
    double X = 0, Y = 0, Z = 0;
    for (int j = 0; j < 3; ++j) {
        X += p[j] * t.rgbToXyz[0][j];
        Y += p[j] * t.rgbToXyz[1][j];
        Z += p[j] * t.rgbToXyz[2][j];
    }
    auto f = [](double v) {
        const double delta = 6.0 / 29.0;
        return v > delta * delta * delta ? std::cbrt(v) : v / (delta * delta * 3.0) + (4.0 / 29.0);
    };
    double Xw = t.xyzWhite[0], Yw = t.xyzWhite[1], Zw = t.xyzWhite[2];
    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

double Sigmoid(double x) { return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5; }

void EvalResidual(const FitTables &t, const double *coeffs, const double *rgb, double *residual) {
    double out[3] = {0, 0, 0};
    for (int i = 0; i < kFineSamples; ++i) {
        double lambda = (t.lambda[i] - LambdaMin) / (LambdaMax - LambdaMin);
        double x = 0;
        for (int j = 0; j < 3; ++j) x = x * lambda + coeffs[j];
        double s = Sigmoid(x);
        for (int j = 0; j < 3; ++j) out[j] += t.rgb[j][i] * s;
    }
    CieLab(t, out);
    double target[3] = {rgb[0], rgb[1], rgb[2]};
    CieLab(t, target);
    for (int j = 0; j < 3; ++j) residual[j] = target[j] - out[j];
}

void EvalJacobian(const FitTables &t, const double *coeffs, const double *rgb, double J[3][3]) {
    const double eps = 1e-5;
    for (int i = 0; i < 3; ++i) {
        double tmp[3] = {coeffs[0], coeffs[1], coeffs[2]};
        double r0[3], r1[3];
        tmp[i] = coeffs[i] - eps;
        EvalResidual(t, tmp, rgb, r0);
        tmp[i] = coeffs[i] + eps;
        EvalResidual(t, tmp, rgb, r1);
        for (int j = 0; j < 3; ++j) J[j][i] = (r1[j] - r0[j]) / (2 * eps);
    }
}

// Solve A x = b for 3x3 with partial pivoting. Returns false if singular.
bool Solve3(double A[3][3], const double b[3], double x[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
        M[i][3] = b[i];
    }
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int r = c + 1; r < 3; ++r)
            if (std::abs(M[r][c]) > std::abs(M[piv][c])) piv = r;
        if (std::abs(M[piv][c]) < 1e-15) return false;
        if (piv != c)
            for (int k = 0; k < 4; ++k) std::swap(M[c][k], M[piv][k]);
        for (int r = c + 1; r < 3; ++r) {
            double f = M[r][c] / M[c][c];
            for (int k = c; k < 4; ++k) M[r][k] -= f * M[c][k];
        }
    }
    for (int i = 2; i >= 0; --i) {
        double s = M[i][3];
        for (int k = i + 1; k < 3; ++k) s -= M[i][k] * x[k];
        x[i] = s / M[i][i];
    }
    return true;
}

void GaussNewton(const FitTables &t, const double rgb[3], double coeffs[3], int it = 15) {
    for (int i = 0; i < it; ++i) {
        double J[3][3], residual[3], x[3];
        EvalResidual(t, coeffs, rgb, residual);
        EvalJacobian(t, coeffs, rgb, J);
        if (!Solve3(J, residual, x)) break;
        double r = 0;
        for (int j = 0; j < 3; ++j) {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }
        double mx = std::max(std::max(coeffs[0], coeffs[1]), coeffs[2]);
        if (mx > 200)
            for (int j = 0; j < 3; ++j) coeffs[j] *= 200 / mx;
        if (r < 1e-6) break;
    }
}

std::string gCacheDir;

std::vector<std::string> CacheCandidates() {
    const char *fname = "prender_rgb2spec_srgb_v1.bin";
    std::vector<std::string> out;
    if (!gCacheDir.empty()) out.push_back((std::filesystem::path(gCacheDir) / fname).string());
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) out.push_back((std::filesystem::path(buf).parent_path() / fname).string());
#endif
    if (const char *lad = std::getenv("LOCALAPPDATA"))
        out.push_back((std::filesystem::path(lad) / "prender" / fname).string());
    return out;
}

} // namespace

void RGBToSpectrumTable::SetCacheDirectory(const std::string &dir) { gCacheDir = dir; }

const RGBToSpectrumTable &RGBToSpectrumTable::Get() {
    // Magic static: thread-safe one-time initialization with no locking on the hot path.
    static const RGBToSpectrumTable *table = [] {
        auto *t = new RGBToSpectrumTable;
        auto candidates = CacheCandidates();
        for (const auto &c : candidates)
            if (t->Load(c)) return t;
        LogInfo("Building RGB-to-spectrum table (one-time)...");
        t->Build();
        for (const auto &c : candidates) {
            std::error_code ec;
            std::filesystem::create_directories(std::filesystem::path(c).parent_path(), ec);
            if (t->Save(c)) break;
        }
        return t;
    }();
    return *table;
}

static const char kMagic[8] = {'P', 'R', 'R', 'G', 'B', '2', 'S', '1'};

bool RGBToSpectrumTable::Load(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    int res = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMagic, 8) == 0 &&
              std::fread(&res, sizeof(int), 1, f) == 1 && res == Res;
    if (ok) {
        zNodes_.resize(Res);
        coeffs_.resize(size_t(3) * Res * Res * Res * 3);
        ok = std::fread(zNodes_.data(), sizeof(float), zNodes_.size(), f) == zNodes_.size() &&
             std::fread(coeffs_.data(), sizeof(float), coeffs_.size(), f) == coeffs_.size();
    }
    std::fclose(f);
    return ok;
}

bool RGBToSpectrumTable::Save(const std::string &path) const {
    std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    int res = Res;
    bool ok = std::fwrite(kMagic, 1, 8, f) == 8 && std::fwrite(&res, sizeof(int), 1, f) == 1 &&
              std::fwrite(zNodes_.data(), sizeof(float), zNodes_.size(), f) == zNodes_.size() &&
              std::fwrite(coeffs_.data(), sizeof(float), coeffs_.size(), f) == coeffs_.size();
    std::fclose(f);
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

void RGBToSpectrumTable::Build() {
    const FitTables &t = GetFitTables();
    auto smoothstep = [](double x) { return x * x * (3.0 - 2.0 * x); };
    zNodes_.resize(Res);
    for (int k = 0; k < Res; ++k) zNodes_[k] = float(smoothstep(smoothstep(double(k) / (Res - 1))));
    coeffs_.assign(size_t(3) * Res * Res * Res * 3, 0.f);
    const double c0 = LambdaMin, c1 = 1.0 / (LambdaMax - LambdaMin);
    ParallelFor(int64_t(3) * Res, [&](int64_t lj, int) {
        int l = int(lj / Res), j = int(lj % Res);
        double y = double(j) / (Res - 1);
        for (int i = 0; i < Res; ++i) {
            double x = double(i) / (Res - 1);
            auto fitRange = [&](int kStart, int kEnd, int step) {
                double coeffs[3] = {0, 0, 0};
                for (int k = kStart; k != kEnd; k += step) {
                    double b = zNodes_[k];
                    double rgb[3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;
                    GaussNewton(t, rgb, coeffs);
                    double A = coeffs[0], B = coeffs[1], C = coeffs[2];
                    size_t idx = ((size_t(l) * Res + k) * Res + j) * Res + i;
                    coeffs_[3 * idx + 0] = float(A * c1 * c1);
                    coeffs_[3 * idx + 1] = float(B * c1 - 2 * A * c0 * c1 * c1);
                    coeffs_[3 * idx + 2] = float(C - B * c0 * c1 + A * c0 * c0 * c1 * c1);
                }
            };
            int start = Res / 5;
            fitRange(start, Res, 1);
            fitRange(start, -1, -1);
        }
    });
}

RGBSigmoidPolynomial RGBToSpectrumTable::operator()(RGB rgb) const {
    for (int c = 0; c < 3; ++c) rgb[c] = Clamp(rgb[c], 0.f, 1.f);
    if (rgb.r == rgb.g && rgb.g == rgb.b) {
        float v = rgb.r;
        if (v <= 0) return RGBSigmoidPolynomial(0, 0, -Infinity);
        if (v >= 1) return RGBSigmoidPolynomial(0, 0, Infinity);
        return RGBSigmoidPolynomial(0, 0, (v - .5f) / std::sqrt(v * (1 - v)));
    }
    int maxc = (rgb.r > rgb.g) ? ((rgb.r > rgb.b) ? 0 : 2) : ((rgb.g > rgb.b) ? 1 : 2);
    float z = rgb[maxc];
    float x = rgb[(maxc + 1) % 3] * (Res - 1) / z;
    float y = rgb[(maxc + 2) % 3] * (Res - 1) / z;
    int xi = std::min(int(x), Res - 2), yi = std::min(int(y), Res - 2);
    int zi = FindInterval(Res, [&](int i) { return zNodes_[i] < z; });
    float dx = x - xi, dy = y - yi, dz = (z - zNodes_[zi]) / (zNodes_[zi + 1] - zNodes_[zi]);
    float c[3];
    auto co = [&](int zz, int yy, int xx, int i) {
        return coeffs_[3 * (((size_t(maxc) * Res + zz) * Res + yy) * Res + xx) + i];
    };
    for (int i = 0; i < 3; ++i) {
        c[i] = Lerp(dz,
                    Lerp(dy, Lerp(dx, co(zi, yi, xi, i), co(zi, yi, xi + 1, i)),
                         Lerp(dx, co(zi, yi + 1, xi, i), co(zi, yi + 1, xi + 1, i))),
                    Lerp(dy, Lerp(dx, co(zi + 1, yi, xi, i), co(zi + 1, yi, xi + 1, i)),
                         Lerp(dx, co(zi + 1, yi + 1, xi, i), co(zi + 1, yi + 1, xi + 1, i))));
    }
    return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

RGBAlbedoSpectrum::RGBAlbedoSpectrum(RGB rgb) : rsp_(RGBToSpectrumTable::Get()(rgb)) {}

RGBUnboundedSpectrum::RGBUnboundedSpectrum(RGB rgb) {
    float m = std::max(0.f, rgb.Max());
    scale_ = 2 * m;
    rsp_ = RGBToSpectrumTable::Get()(scale_ > 0 ? rgb * (1 / scale_) : RGB(0, 0, 0));
}

RGBIlluminantSpectrum::RGBIlluminantSpectrum(RGB rgb) {
    float m = std::max(0.f, rgb.Max());
    scale_ = 2 * m;
    rsp_ = RGBToSpectrumTable::Get()(scale_ > 0 ? rgb * (1 / scale_) : RGB(0, 0, 0));
}

SampledSpectrum RGBAlbedoSample(RGB rgb, const SampledWavelengths &lambda) {
    RGBSigmoidPolynomial rsp = RGBToSpectrumTable::Get()(rgb);
    SampledSpectrum s;
    for (int i = 0; i < NSpectrumSamples; ++i) s[i] = rsp(lambda[i]);
    return s;
}

SampledSpectrum RGBUnboundedSample(RGB rgb, const SampledWavelengths &lambda) {
    float m = std::max(0.f, rgb.Max());
    if (m == 0) return SampledSpectrum(0.f);
    float scale = 2 * m;
    RGBSigmoidPolynomial rsp = RGBToSpectrumTable::Get()(rgb * (1 / scale));
    SampledSpectrum s;
    for (int i = 0; i < NSpectrumSamples; ++i) s[i] = scale * rsp(lambda[i]);
    return s;
}

SampledSpectrum RGBIlluminantSample(RGB rgb, const SampledWavelengths &lambda) {
    SampledSpectrum s = RGBUnboundedSample(rgb, lambda);
    const Spectrum &d65 = StdIlluminantD65();
    for (int i = 0; i < NSpectrumSamples; ++i) s[i] *= d65(lambda[i]);
    return s;
}

} // namespace pr

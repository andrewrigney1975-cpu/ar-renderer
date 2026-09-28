#include "materials/texture.h"

#include "core/noise.h"
#include "core/rgb2spec.h"

namespace pr {

RGB ImageData::Texel(int x, int y) const {
    int w = img_.Width(), h = img_.Height();
    auto wrap = [&](int v, int n) {
        switch (wrap_) {
        case WrapMode::Repeat: v %= n; return v < 0 ? v + n : v;
        case WrapMode::Clamp: return Clamp(v, 0, n - 1);
        case WrapMode::Mirror: {
            int period = 2 * n;
            v %= period;
            if (v < 0) v += period;
            return v < n ? v : period - 1 - v;
        }
        }
        return v;
    };
    return img_.Get(wrap(x, w), wrap(y, h));
}

RGB ImageData::Lookup(Vec2f uv) const {
    if (img_.Empty()) return {};
    float v = flipV_ ? 1 - uv.y : uv.y;
    float x = uv.x * img_.Width() - 0.5f, y = v * img_.Height() - 0.5f;
    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    float dx = x - x0, dy = y - y0;
    RGB a = Texel(x0, y0), b = Texel(x0 + 1, y0), c = Texel(x0, y0 + 1), d = Texel(x0 + 1, y0 + 1);
    RGB r;
    for (int i = 0; i < 3; ++i)
        r[i] = (1 - dx) * (1 - dy) * a[i] + dx * (1 - dy) * b[i] + (1 - dx) * dy * c[i] + dx * dy * d[i];
    return r;
}

static SampledSpectrum Uplift(RGB rgb, SpectrumType type, const SampledWavelengths &lambda) {
    switch (type) {
    case SpectrumType::Albedo:
        return RGBAlbedoSample({Clamp(rgb.r, 0.f, 1.f), Clamp(rgb.g, 0.f, 1.f), Clamp(rgb.b, 0.f, 1.f)}, lambda);
    case SpectrumType::Unbounded: return RGBUnboundedSample(rgb, lambda);
    case SpectrumType::Illuminant: return RGBIlluminantSample(rgb, lambda);
    }
    return SampledSpectrum(0.f);
}

SampledSpectrum ImageSpectrumTexture::Evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const {
    RGB c = img_->Lookup(map_.Map(ctx.uv));
    c = RGB(c.r * scale_.r, c.g * scale_.g, c.b * scale_.b);
    return Uplift(c, type_, lambda);
}

float NoiseFloatTexture::Evaluate(const TextureEvalContext &ctx) const {
    Vec3f p = world_ ? ctx.p : Vec3f(ctx.uv.x, ctx.uv.y, 0);
    p = (p + offset_) * freq_;
    float n = FBm(p, octaves_, 2.f, 0.5f, turbulence_);
    return turbulence_ ? Clamp(n * 1.6f, 0.f, 1.f) : Clamp(0.5f + 0.5f * n * 1.4f, 0.f, 1.f);
}

SpectrumTexturePtr MakeRGBSpectrumTexture(RGB rgb, SpectrumType type) {
    SpectrumPtr s;
    switch (type) {
    case SpectrumType::Albedo:
        s = std::make_shared<RGBAlbedoSpectrum>(RGB(Clamp(rgb.r, 0.f, 1.f), Clamp(rgb.g, 0.f, 1.f), Clamp(rgb.b, 0.f, 1.f)));
        break;
    case SpectrumType::Unbounded: s = std::make_shared<RGBUnboundedSpectrum>(rgb); break;
    case SpectrumType::Illuminant: s = std::make_shared<RGBIlluminantSpectrum>(rgb); break;
    }
    return std::make_shared<ConstantSpectrumTexture>(s, rgb);
}

} // namespace pr

#pragma once

#include "core/color.h"
#include "core/image.h"
#include "core/spectrum.h"

#include <memory>

namespace pr {

struct TextureEvalContext {
    Vec3f p;
    Vec2f uv;
};

enum class SpectrumType { Albedo, Unbounded, Illuminant };

class FloatTexture {
  public:
    virtual ~FloatTexture() = default;
    virtual float Evaluate(const TextureEvalContext &ctx) const = 0;
};

class SpectrumTexture {
  public:
    virtual ~SpectrumTexture() = default;
    virtual SampledSpectrum Evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const = 0;
    // Rough average RGB (for light power estimates and previews).
    virtual RGB AverageRGB() const = 0;
};

using FloatTexturePtr = std::shared_ptr<const FloatTexture>;
using SpectrumTexturePtr = std::shared_ptr<const SpectrumTexture>;

class ConstantFloatTexture : public FloatTexture {
  public:
    explicit ConstantFloatTexture(float v) : v_(v) {}
    float Evaluate(const TextureEvalContext &) const override { return v_; }
    float Value() const { return v_; }

  private:
    float v_;
};

// A spectrum given directly (named data, blackbody, uplifted constant RGB, ...).
class ConstantSpectrumTexture : public SpectrumTexture {
  public:
    ConstantSpectrumTexture(SpectrumPtr s, RGB avg) : s_(std::move(s)), avg_(avg) {}
    SampledSpectrum Evaluate(const TextureEvalContext &, const SampledWavelengths &lambda) const override {
        return s_->Sample(lambda);
    }
    RGB AverageRGB() const override { return avg_; }
    const SpectrumPtr &GetSpectrum() const { return s_; }

  private:
    SpectrumPtr s_;
    RGB avg_;
};

enum class WrapMode { Repeat, Clamp, Mirror };

struct UVMapping {
    Vec2f scale{1, 1}, offset{0, 0};
    Vec2f Map(Vec2f uv) const { return {uv.x * scale.x + offset.x, uv.y * scale.y + offset.y}; }
};

// Bilinearly filtered image lookup; v=0 is the bottom row (OpenGL/glTF-style flip optional).
class ImageData {
  public:
    ImageData(Image img, WrapMode wrap, bool flipV) : img_(std::move(img)), wrap_(wrap), flipV_(flipV) {
        double s[3] = {0, 0, 0};
        for (int y = 0; y < img_.Height(); ++y)
            for (int x = 0; x < img_.Width(); ++x) {
                RGB c = img_.Get(x, y);
                s[0] += c.r; s[1] += c.g; s[2] += c.b;
            }
        double n = std::max(1.0, double(img_.Width()) * img_.Height());
        avg_ = RGB(float(s[0] / n), float(s[1] / n), float(s[2] / n));
    }
    RGB Lookup(Vec2f uv) const;
    RGB Average() const { return avg_; }
    const Image &GetImage() const { return img_; }

  private:
    RGB Texel(int x, int y) const;
    Image img_;
    WrapMode wrap_;
    bool flipV_;
    RGB avg_;
};

class ImageSpectrumTexture : public SpectrumTexture {
  public:
    ImageSpectrumTexture(std::shared_ptr<const ImageData> img, UVMapping map, SpectrumType type, RGB scale)
        : img_(std::move(img)), map_(map), type_(type), scale_(scale) {}
    SampledSpectrum Evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const override;
    RGB AverageRGB() const override {
        RGB a = img_->Average();
        return {a.r * scale_.r, a.g * scale_.g, a.b * scale_.b};
    }

  private:
    std::shared_ptr<const ImageData> img_;
    UVMapping map_;
    SpectrumType type_;
    RGB scale_;
};

class ImageFloatTexture : public FloatTexture {
  public:
    ImageFloatTexture(std::shared_ptr<const ImageData> img, UVMapping map, int channel, float scale)
        : img_(std::move(img)), map_(map), channel_(channel), scale_(scale) {}
    float Evaluate(const TextureEvalContext &ctx) const override {
        RGB c = img_->Lookup(map_.Map(ctx.uv));
        float v = channel_ < 0 ? (c.r + c.g + c.b) / 3 : c[channel_];
        return v * scale_;
    }

  private:
    std::shared_ptr<const ImageData> img_;
    UVMapping map_;
    int channel_;
    float scale_;
};

class CheckerSpectrumTexture : public SpectrumTexture {
  public:
    CheckerSpectrumTexture(SpectrumTexturePtr a, SpectrumTexturePtr b, UVMapping map)
        : a_(std::move(a)), b_(std::move(b)), map_(map) {}
    SampledSpectrum Evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const override {
        Vec2f st = map_.Map(ctx.uv);
        bool odd = (int(std::floor(st.x)) + int(std::floor(st.y))) & 1;
        return odd ? b_->Evaluate(ctx, lambda) : a_->Evaluate(ctx, lambda);
    }
    RGB AverageRGB() const override { return (a_->AverageRGB() + b_->AverageRGB()) * 0.5f; }
    const SpectrumTexturePtr &A() const { return a_; }
    const SpectrumTexturePtr &B() const { return b_; }
    const UVMapping &Mapping() const { return map_; }

  private:
    SpectrumTexturePtr a_, b_;
    UVMapping map_;
};

class CheckerFloatTexture : public FloatTexture {
  public:
    CheckerFloatTexture(FloatTexturePtr a, FloatTexturePtr b, UVMapping map)
        : a_(std::move(a)), b_(std::move(b)), map_(map) {}
    float Evaluate(const TextureEvalContext &ctx) const override {
        Vec2f st = map_.Map(ctx.uv);
        bool odd = (int(std::floor(st.x)) + int(std::floor(st.y))) & 1;
        return odd ? b_->Evaluate(ctx) : a_->Evaluate(ctx);
    }

  private:
    FloatTexturePtr a_, b_;
    UVMapping map_;
};

// Procedural fBm noise in [0,1], evaluated in world space or in uv space (z = 0).
class NoiseFloatTexture : public FloatTexture {
  public:
    NoiseFloatTexture(float frequency, int octaves, bool turbulence, bool worldSpace, Vec3f offset)
        : freq_(frequency), octaves_(octaves), turbulence_(turbulence), world_(worldSpace), offset_(offset) {}
    float Evaluate(const TextureEvalContext &ctx) const override;

  private:
    float freq_;
    int octaves_;
    bool turbulence_, world_;
    Vec3f offset_;
};

// Blend of two spectrum textures driven by a float texture.
class MixSpectrumTexture : public SpectrumTexture {
  public:
    MixSpectrumTexture(SpectrumTexturePtr a, SpectrumTexturePtr b, FloatTexturePtr t)
        : a_(std::move(a)), b_(std::move(b)), t_(std::move(t)) {}
    SampledSpectrum Evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const override {
        float t = Clamp(t_->Evaluate(ctx), 0.f, 1.f);
        return a_->Evaluate(ctx, lambda) * (1 - t) + b_->Evaluate(ctx, lambda) * t;
    }
    RGB AverageRGB() const override { return (a_->AverageRGB() + b_->AverageRGB()) * 0.5f; }

  private:
    SpectrumTexturePtr a_, b_;
    FloatTexturePtr t_;
};

class MixFloatTexture : public FloatTexture {
  public:
    MixFloatTexture(FloatTexturePtr a, FloatTexturePtr b, FloatTexturePtr t) : a_(std::move(a)), b_(std::move(b)), t_(std::move(t)) {}
    float Evaluate(const TextureEvalContext &ctx) const override {
        float t = Clamp(t_->Evaluate(ctx), 0.f, 1.f);
        return a_->Evaluate(ctx) * (1 - t) + b_->Evaluate(ctx) * t;
    }

  private:
    FloatTexturePtr a_, b_, t_;
};

// Spectrum texture from a constant RGB triple, uplifted according to its usage.
SpectrumTexturePtr MakeRGBSpectrumTexture(RGB rgb, SpectrumType type);

} // namespace pr

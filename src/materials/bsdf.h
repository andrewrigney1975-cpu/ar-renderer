#pragma once

#include "materials/bxdf.h"

namespace pr {

// A BxDF placed in the local shading frame of a surface point.
class BSDF {
  public:
    BSDF() = default;
    BSDF(const Vec3f &ns, const Vec3f &dpdus, const BxDF *bxdf)
        : frame_(Frame::FromXZ(dpdus, ns)), bxdf_(bxdf) {}

    explicit operator bool() const { return bxdf_ != nullptr; }
    BxDFFlags Flags() const { return bxdf_->Flags(); }
    bool HasNonSpecular() const { return IsNonSpecular(Flags()); }
    bool PDFIsExact() const { return bxdf_->PDFIsExact(); }

    Vec3f RenderToLocal(const Vec3f &v) const { return frame_.ToLocal(v); }
    Vec3f LocalToRender(const Vec3f &v) const { return frame_.FromLocal(v); }

    SampledSpectrum f(const Vec3f &woRender, const Vec3f &wiRender,
                      TransportMode mode = TransportMode::Radiance) const {
        Vec3f wi = RenderToLocal(wiRender), wo = RenderToLocal(woRender);
        if (wo.z == 0) return SampledSpectrum(0.f);
        return bxdf_->f(wo, wi, mode);
    }

    std::optional<BSDFSample> Sample_f(const Vec3f &woRender, float u, Vec2f u2,
                                       TransportMode mode = TransportMode::Radiance,
                                       BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vec3f wo = RenderToLocal(woRender);
        if (wo.z == 0 || !(bxdf_->Flags() & uint32_t(sampleFlags))) return std::nullopt;
        auto bs = bxdf_->Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0 || bs->f.HasNaNs() || !std::isfinite(bs->pdf))
            return std::nullopt;
        bs->wi = LocalToRender(bs->wi);
        return bs;
    }

    float PDF(const Vec3f &woRender, const Vec3f &wiRender, TransportMode mode = TransportMode::Radiance,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vec3f wo = RenderToLocal(woRender), wi = RenderToLocal(wiRender);
        if (wo.z == 0) return 0;
        return bxdf_->PDF(wo, wi, mode, sampleFlags);
    }

  private:
    Frame frame_;
    const BxDF *bxdf_ = nullptr;
};

} // namespace pr

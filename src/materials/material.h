#pragma once

#include "core/memory.h"
#include "materials/bsdf.h"
#include "materials/texture.h"

#include <string>

namespace pr {

struct MaterialEvalContext : TextureEvalContext {
    Vec3f wo, n, ns, dpdus;
};

struct SheenParams {
    SpectrumTexturePtr color;
    FloatTexturePtr roughness;
    float weight = 0;
};

class Material {
  public:
    virtual ~Material() = default;
    // May terminate secondary wavelengths (dispersion).
    BSDF GetBSDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const;
    void SetSheen(SheenParams s) { sheen_ = std::move(s); }
    std::string name;

  protected:
    virtual const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                ScratchBuffer &buf) const = 0;

  private:
    SheenParams sheen_;
};

class DiffuseMaterial : public Material {
  public:
    explicit DiffuseMaterial(SpectrumTexturePtr r) : reflectance_(std::move(r)) {}

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    SpectrumTexturePtr reflectance_;
};

struct ConductorParams {
    SpectrumTexturePtr eta, k;         // complex IOR; or
    SpectrumTexturePtr reflectance;    // artist reflectance (used when eta/k are null)
    FloatTexturePtr uRoughness, vRoughness;
};

class ConductorMaterial : public Material {
  public:
    explicit ConductorMaterial(ConductorParams p) : p_(std::move(p)) {}
    static const BxDF *MakeBxDF(const ConductorParams &p, const MaterialEvalContext &ctx,
                                const SampledWavelengths &lambda, ScratchBuffer &buf);

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    ConductorParams p_;
};

class DielectricMaterial : public Material {
  public:
    DielectricMaterial(SpectrumPtr eta, FloatTexturePtr uRough, FloatTexturePtr vRough, bool thin)
        : eta_(std::move(eta)), uRough_(std::move(uRough)), vRough_(std::move(vRough)), thin_(thin) {}
    bool IsDispersive() const { return !eta_->IsConstant(); }

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    SpectrumPtr eta_;
    FloatTexturePtr uRough_, vRough_;
    bool thin_;
};

struct CoatParams {
    SpectrumPtr eta;                    // coat IOR
    FloatTexturePtr roughness;          // coat roughness
    FloatTexturePtr thickness;
    SpectrumTexturePtr albedo;          // medium albedo between layers (0 = clear coat)
    float g = 0;
    int maxDepth = 64, nSamples = 1;
};

class CoatedDiffuseMaterial : public Material {
  public:
    CoatedDiffuseMaterial(CoatParams coat, SpectrumTexturePtr reflectance)
        : coat_(std::move(coat)), reflectance_(std::move(reflectance)) {}

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    CoatParams coat_;
    SpectrumTexturePtr reflectance_;
};

class CoatedConductorMaterial : public Material {
  public:
    CoatedConductorMaterial(CoatParams coat, ConductorParams conductor)
        : coat_(std::move(coat)), conductor_(std::move(conductor)) {}

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    CoatParams coat_;
    ConductorParams conductor_;
};

} // namespace pr

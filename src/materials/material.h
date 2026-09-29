#pragma once

#include "core/memory.h"
#include "materials/bsdf.h"
#include "materials/texture.h"
#include "scene/interaction.h"

#include <string>

namespace pr {

struct MaterialEvalContext : TextureEvalContext {
    Vec3f wo, n, ns, dpdus;
    Vec3f dpdu, dpdv;  // surface parameterization (for normal/bump mapping)
    static MaterialEvalContext From(const SurfaceInteraction &si) {
        MaterialEvalContext c;
        c.p = si.p;
        c.uv = si.uv;
        c.wo = si.wo;
        c.n = si.n;
        c.ns = si.shading.n;
        c.dpdus = si.shading.dpdu;
        c.dpdu = si.dpdu;
        c.dpdv = si.dpdv;
        return c;
    }
};

// Tangent-space normal map (OpenGL/glTF convention: +Y up, rgb = n * 0.5 + 0.5).
struct NormalMapParams {
    std::shared_ptr<const ImageData> image;
    UVMapping mapping;
    float strength = 1;
    bool flipGreen = false;  // DirectX / glTF image-space convention
};

// Height field; the shading normal is perturbed by its gradient (scale in scene units).
struct BumpMapParams {
    FloatTexturePtr height;
    float scale = 1;
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
    // Applies normal/bump mapping to the shading frame in ctx. Returns true if it changed.
    bool PerturbShading(MaterialEvalContext &ctx) const;
    void SetSheen(SheenParams s) { sheen_ = std::move(s); }
    void SetNormalMap(NormalMapParams n) { normalMap_ = std::move(n); }
    void SetBumpMap(BumpMapParams b) { bumpMap_ = std::move(b); }
    bool HasSheen() const { return sheen_.weight > 0; }
    bool HasShadingPerturbation() const { return normalMap_.image || bumpMap_.height; }
    std::string name;

  protected:
    virtual const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda,
                                ScratchBuffer &buf) const = 0;

  private:
    SheenParams sheen_;
    NormalMapParams normalMap_;
    BumpMapParams bumpMap_;
};

// Evaluates the material at a surface hit: applies normal/bump mapping (updating si's shading
// frame so integrators use the perturbed normal) and returns the BSDF.
BSDF EvaluateSurface(const Material &m, SurfaceInteraction &si, SampledWavelengths &lambda, ScratchBuffer &buf);

class DiffuseMaterial : public Material {
  public:
    explicit DiffuseMaterial(SpectrumTexturePtr r) : reflectance_(std::move(r)) {}
    const SpectrumTexturePtr &Reflectance() const { return reflectance_; }

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
    const ConductorParams &Params() const { return p_; }
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
    const SpectrumPtr &Eta() const { return eta_; }
    const FloatTexturePtr &URoughness() const { return uRough_; }
    const FloatTexturePtr &VRoughness() const { return vRough_; }
    bool Thin() const { return thin_; }

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
    const SpectrumTexturePtr &Reflectance() const { return reflectance_; }

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
    const ConductorParams &Conductor() const { return conductor_; }

  protected:
    const BxDF *GetBxDF(const MaterialEvalContext &ctx, SampledWavelengths &lambda, ScratchBuffer &buf) const override;

  private:
    CoatParams coat_;
    ConductorParams conductor_;
};

} // namespace pr

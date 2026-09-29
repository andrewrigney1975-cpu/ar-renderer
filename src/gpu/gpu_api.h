#pragma once

// Plain-data interface between prender.exe (MSVC) and prender_gpu.dll (Intel oneAPI DPC++/SYCL).
// Everything here is POD so it can be copied verbatim into device (USM) memory; there are no
// virtual functions, pointers into host objects, or STL types in the device representation.

#include <cstdint>

#ifdef _WIN32
#define PRGPU_API extern "C" __declspec(dllexport)
#else
#define PRGPU_API extern "C"
#endif

namespace prgpu {

constexpr int kApiVersion = 3;

// Spectra are tabulated at 5 nm from 360 to 830 nm (95 samples) and interpolated linearly.
constexpr int kSpectrumSamples = 95;
constexpr float kSpectrumMin = 360.f, kSpectrumStep = 5.f;
// Pixel filter importance sampling: piecewise-constant 1D CDF over [-radius, radius].
constexpr int kFilterBins = 64;
// Sheen directional albedo table (pr::SheenBxDF::Albedo), alpha-major.
constexpr int kSheenMu = 32, kSheenAlpha = 16;

struct Float3 {
    float x, y, z;
};

struct Spectrum {
    float v[kSpectrumSamples];
};

struct BVHNode {  // identical layout to pr::BVH's flattened nodes
    Float3 bmin, bmax;
    int32_t offset;   // leaf: first primitive; interior: second child
    uint16_t nPrims;  // 0 = interior
    uint8_t axis, pad;
};

enum PrimType : uint32_t { PrimTriangle = 0, PrimSphere = 1 };

struct Primitive {
    uint32_t type;
    int32_t material;   // index into materials
    int32_t light;      // index into lights (area emitters) or -1
    int32_t mediumInside, mediumOutside;  // media on either side of the surface (-1 = vacuum)
    uint32_t pad;
    // Triangle: p0, p1, p2 and normals (n0 == 0 -> flat); sphere: p0 = centre, radius in r.
    Float3 p0, p1, p2;
    Float3 n0, n1, n2;
    float u0, v0, u1, v1, u2, v2;  // texture coordinates
    float r;
    uint32_t flip;
};

enum MaterialType : uint32_t { MatDiffuse = 0, MatConductor = 1, MatDielectric = 2, MatBlack = 3, MatInterface = 4,
                                 MatCoatedDiffuse = 5, MatCoatedConductor = 6 };

struct Material {
    uint32_t type;
    int32_t albedo;         // spectrum index (diffuse reflectance / conductor artist reflectance), -1 none
    int32_t albedoB;        // checker second spectrum, -1 none
    float checkerScale;     // checker frequency in uv (0 = no checker)
    int32_t eta, k;         // spectrum indices (conductor n,k; dielectric IOR), -1 none
    float alphaX, alphaY;   // GGX alpha (roughness^2); < 1e-3 = smooth
    uint32_t dispersive;    // dielectric IOR varies with wavelength
    uint32_t thin;
    // Coat (coated_diffuse / coated_conductor): a dielectric layer over the base, with an
    // optional scattering medium in between (pr::LayeredBxDF).
    int32_t coatEta;        // spectrum index
    float coatAlpha, thickness, coatG;
    int32_t coatAlbedo;     // spectrum index, -1 = clear
    uint32_t coatDispersive;
    int32_t coatMaxDepth, coatSamples;
    // Sheen lobe over any base (pr::SheenBxDF); sheenWeight 0 = none.
    int32_t sheenColor;
    float sheenRoughness, sheenWeight;
    uint32_t pad;
};

// Homogeneous medium: sigma_a, sigma_s spectra (indices) times scale, Henyey-Greenstein g.
struct Medium {
    int32_t sigmaA, sigmaS;
    float scale, g;
};

enum FilterType : uint32_t { FilterBox = 0, FilterGaussian = 1, FilterBlackmanHarris = 2 };

struct Filter {
    uint32_t type;
    float radius, sigma, norm;  // norm: 2D normalization, as in pr::Filter
    float cdf[kFilterBins + 1];
};

enum LightType : uint32_t { LightArea = 0, LightPoint = 1 };

struct Light {
    uint32_t type;
    int32_t primitive;      // area light: emitting primitive
    int32_t spectrum;       // emitted radiance (area) or intensity (point)
    float scale;
    Float3 position;        // point light
    uint32_t twoSided;
    float cosPower;
    float area;
    float pmf;              // selection probability (power based)
    uint32_t pad;
};

struct Camera {
    float cameraToWorld[12];  // row-major 3x4
    float tanHalfFov, aspect;
    float lensRadius, focalDistance;
    int32_t width, height, margin;
    uint32_t pad;
};

struct SceneDesc {
    int32_t apiVersion;
    const BVHNode *nodes;
    int32_t nodeCount;
    const int32_t *primOrder;   // BVH leaf slot -> primitive index
    const Primitive *prims;
    int32_t primCount;
    const Material *materials;
    int32_t materialCount;
    const Light *lights;
    int32_t lightCount;
    const Spectrum *spectra;
    int32_t spectrumCount;
    const Medium *media;
    int32_t mediumCount;
    int32_t cameraMedium;       // medium the camera sits in, -1 none
    int32_t envSpectrum;        // constant environment radiance, -1 none
    float envScale;
    float envPmf;               // selection probability of the environment in NEE
    Float3 worldCenter;
    float worldRadius;
    Camera camera;
    Filter filter;
    const float *sheenAlbedo;   // kSheenAlpha * kSheenMu
    const float *cie;           // 3 x kSpectrumSamples colour matching functions (x, y, z)
    float cieYIntegral;
};

struct RenderParams {
    int32_t spp;          // samples per pixel for this call
    int32_t firstSample;  // sample index offset (progressive rendering / determinism)
    uint64_t seed;
    int32_t maxDepth;     // <= 0: unlimited (Russian roulette)
    uint32_t pad;
};

struct DeviceInfo {
    char name[256];
    uint64_t globalMemory;
    int32_t computeUnits;
    int32_t isGpu;
    char backend[32];
};

} // namespace prgpu

// Returns the number of devices written (at most max).
PRGPU_API int prgpu_list_devices(prgpu::DeviceInfo *out, int max);
// Renders spp samples per pixel over the extended film and ADDS the resulting XYZ sums (unnormalized,
// box filter, film raster order including the margin) into outXYZ (3 floats per sample pixel).
// Returns 0 on success; on failure writes a message into err.
PRGPU_API int prgpu_render(int deviceIndex, const prgpu::SceneDesc *scene, const prgpu::RenderParams *params,
                           float *outXYZ, char *err, int errLen);
PRGPU_API int prgpu_api_version();

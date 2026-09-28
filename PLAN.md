# Spectral MLT Renderer: Project Plan

A physically-based, **unbiased**, **spectral** renderer driven by **Metropolis Light Transport**. It ships as an x64 command-line tool (`prender.exe`), with a lightweight WinUI 3 front end that shows four viewports and has a Render button.

---

## 1. Goals and non-goals

### Goals
| Feature | How it's delivered |
|---|---|
| Global illumination | Full path-space integration (BDPT as the path sampler, MLT on top) |
| Ray tracing | Embree 4 BVH (triangles, analytic spheres, instances) |
| Physically-based lighting | Spectral radiometry; lights defined by SPDs (blackbody, CIE illuminants, measured data) with physical units (W, lm, cd, nits) |
| Shadows | Implicit in the light transport. Area lights give physically correct penumbrae |
| Caustics | BDPT light subpaths plus MLT's local exploration of hard-to-find paths |
| Chromatic dispersion | Spectral rendering with **hero-wavelength sampling**. IOR comes from Sellmeier or Cauchy coefficients |
| Sub-surface scattering | **Volumetric random walk** inside closed meshes (unbiased; no diffusion-dipole approximation) |
| Volumetric lighting | Homogeneous and heterogeneous participating media using delta, ratio and spectral tracking under the null-scattering formulation |
| Unbiased | Only unbiased estimators: no photon mapping or VCM merging, no path clamping, no radiance caching. Any denoising is a separate, opt-in post step that is written to a separate file |

### Non-goals (v1)
- GPU rendering (keep the core data-oriented so a later CUDA or DXR port is possible)
- Animation or motion blur (leave hooks in the camera and transform time parameters)
- Interactive in-UI path tracing (the UI's preview is a raster preview)

---

## 2. Technology choices

| Area | Choice | Rationale |
|---|---|---|
| Core language | C++20, MSVC x64, AVX2 baseline | Performance, SIMD, mature tooling |
| Build | CMake + **vcpkg manifest mode** | Reproducible dependencies |
| Ray tracing kernel | **Embree 4** | Production-grade BVH and robust intersection. A custom BVH isn't worth writing for v1 |
| Threading | oneTBB (or a `std::jthread` pool) | Many independent Markov chains |
| Images | OpenImageIO (or stb_image + tinyexr for a lighter build) | EXR/HDR/PNG/JPEG/TIFF input; EXR/PNG output |
| Mesh import | tinyobjloader, happly (PLY), cgltf / tinygltf | Lightweight, header-only |
| Volumes | NanoVDB | Compact read-only VDB grids |
| JSON | nlohmann/json + a JSON Schema for validation | Human-editable scene files |
| Tests | GoogleTest + custom statistical test harness | See section 10 |
| UI | **WinUI 3 (Windows App SDK 1.6+), C# / .NET 8**, Win2D for viewports | The UI is a thin client. C# keeps it small, and the renderer's performance-critical code is all in the C++ process |

> Why a separate process instead of linking the renderer into the UI? Crash isolation, the CLI stays the single source of truth, the renderer can later run on a farm or remote box, and the UI stays lightweight.

---

## 3. Repository layout

```
/cmake/                  toolchain & helper modules
/src/core/               math, spectral, sampling, film, parallel utils        (lib: pr_core)
/src/scene/              scene graph, loaders, asset cache, schema validation  (lib: pr_scene)
/src/materials/          BSDFs, layered models, texture graph                  (lib: pr_materials)
/src/media/              phase functions, media, tracking estimators           (lib: pr_media)
/src/lights/             emitters, light sampling / power distributions        (lib: pr_lights)
/src/cameras/            camera models, sensor response                        (lib: pr_cameras)
/src/integrators/        path tracer, BDPT, PSSMLT, MMLT                       (lib: pr_integrators)
/src/cli/                prender.exe
/src/ui/PRenderUI/       WinUI 3 C# app
/schemas/                prscene.schema.json (+ generated C# model classes)
/scenes/                 sample scenes, incl. sphere-pyramid
/data/spectra/           CIE CMFs, illuminants, Sellmeier tables, metal n/k data
/tests/unit/             math, sampling, BSDF chi², reciprocity
/tests/reference/        scene-level convergence & regression images
```

---

## 4. Renderer architecture

### 4.1 Spectral pipeline
- **Wavelength sampling:** hero wavelength plus 3 stratified companions (Wilkie et al. 2014) in 360–830 nm, importance-sampled against the visual response.
- **Dispersion:** when a path hits a *dispersive* specular or rough dielectric, the companion wavelengths are terminated and the hero weight is scaled accordingly. Non-dispersive paths keep all four wavelengths, so the variance benefit is preserved.
- **RGB uplifting:** RGB textures and colours are converted to smooth reflectance spectra with Jakob & Hanika 2019 (sigmoid-polynomial LUT). Emission colours use an illuminant-scaled variant.
- **Film:** accumulates CIE XYZ, then converts to an output colour space (linear Rec.709, ACEScg or Rec.2020) at write time. Tone mapping (AgX or ACES) applies only to LDR outputs.
- **Wavelength is a primary-sample dimension**, so MLT mutations can explore it. This is critical for converging dispersion caustics.

### 4.2 Integrators (built in this order, each validating the next)
1. **Unidirectional path tracer (PT)** with next-event estimation and MIS. This is the reference and debug integrator.
2. **Bidirectional path tracer (BDPT)** with full MIS, media-aware and supporting the null-scattering path formulation (Miller et al. 2019) for heterogeneous media.
3. **Primary Sample Space MLT (Kelemen 2002)** over BDPT:
   - Large-step probability of about 0.3. Small steps use a Gaussian or exponential perturbation in primary sample space.
   - **Bootstrap:** N candidate paths are sampled with BDPT to estimate the normalization constant *b* and to seed chains by resampling in proportion to contribution. This removes start-up bias, keeping the estimator unbiased in expectation.
   - One chain per hardware thread (hundreds of chains in total), each splatting to a shared film with atomic or per-tile buffers.
4. **Multiplexed MLT (Hachisuka et al. 2014)**, the target default. Each chain also mutates path length and BDPT strategy (s,t), which gives much better stratification than plain PSSMLT.
5. *(Stretch)* Manifold-exploration or half-vector-space mutations for SDS paths, such as caustics seen through glass. These remain unbiased as extra mutation types.

> **Note on unbiasedness and delta distributions:** specular–diffuse–specular paths lit by a *point* light through a *pinhole* camera have zero probability under any unbiased sampler. The sample scene therefore uses area lights and a thin-lens camera by default, and the docs flag this limitation.

### 4.3 Materials (BSDF library)
Parameterized following **OpenPBR Surface** (ASWF). This is a modern, documented uber-shader schema that maps well to glTF and MaterialX.

| Lobe / model | Used for |
|---|---|
| Lambert, Oren–Nayar | Diffuse base |
| GGX / Trowbridge–Reitz microfacet with VNDF sampling, anisotropic | Metal, rough dielectric, satin |
| Spectral conductor Fresnel (complex n, k from tables) | Gold, copper, aluminium |
| Dielectric with Sellmeier or Cauchy IOR (smooth and rough) | Glass, diamond, water, with dispersion |
| Thin dielectric | Soap film, windows |
| Sheen (Conty–Kulla "Charlie" or Zeltner 2022 LTC sheen) | Satin, velvet |
| **Layered BSDF via position-free Monte Carlo (Guo et al. 2018)** | Clear-coat over base: car paint, lacquer. Stochastic but unbiased |
| Emission (SPD × intensity, optional texture) | Self-illumination |
| Interior medium binding | SSS and coloured glass absorption |

Every BSDF implements `eval`, `sample` and `pdf` on the hero wavelength packet, plus `flags` (delta, dispersive, transmissive).

### 4.4 Participating media and SSS
- **Media:** homogeneous and heterogeneous (NanoVDB grids), with a spectrally varying σa/σs.
- **Estimators:** delta tracking for sampling distance, ratio tracking for transmittance (NEE), and spectral tracking for chromatic media.
- **Phase functions:** isotropic, Henyey–Greenstein, and a Draine/HG mix for fog.
- **SSS:** a mesh with `interior_medium` makes the renderer run a *volumetric random walk* inside the surface. Physical parameters are σa, σs and g, but the author can specify albedo and mean free path in the artist-friendly form (Chiang et al. 2016 / Christensen–Burley inversion used only for parameter conversion, not for rendering).
- **Medium stack:** a nested-dielectric priority system, so liquids inside glass and fog around glass behave correctly.

### 4.5 Lights
- **Area lights:** any mesh or sphere whose material is emissive. Internally these become light-list entries.
- **Explicit emitters:** sphere, disc, rectangle (with optional spot falloff), spot, point, directional (delta types allowed but flagged), and IES profile modulation.
- **Environment:** equirectangular HDR, importance-sampled with a 2D piecewise-constant distribution. Also a physical sun and sky (Prague / Hošek–Wilkie).
- **Light selection:** power-based at first, later a light BVH (Conty & Kulla 2018) for scenes with many emitters.
- **Units:** radiant power (W), luminous power (lm), luminance (nits) or irradiance, all converted to spectral radiance at load time.

### 4.6 Cameras
- **Pinhole**, **thin lens** (aperture shape, focus distance), **orthographic** and **equirectangular 360°**.
- *(Stretch)* **Realistic lens** that ray-traces a lens prescription, using the same dispersive glass database as the materials. This produces genuine lens chromatic aberration.
- **Sensor:** resolution, crop, pixel filter (Blackman–Harris by default), ISO, shutter and f-stop exposure, and an optional spectral sensitivity curve (CIE 1931 by default).
- Cameras are also valid endpoints for BDPT connections, so importance evaluation (`We`) and `pdf_We` are required for each camera type.

---

## 5. Scene model: separating the concerns

The key principle is that **every concern is an independently referenceable asset with an ID**. A scene is a *composition* of these assets, much like USD layering but in simpler JSON.

```
Scene
 ├─ geometry      : what shape             (no appearance)
 ├─ textures      : pure data, no BSDF     (images, procedurals)
 ├─ materials     : BSDF params → constants | texture refs
 ├─ media         : participating media definitions
 ├─ objects       : instance = geometry + material binding + transform (+ interior medium)
 ├─ lights        : emitters, grouped into swappable light rigs
 ├─ cameras       : named cameras, any number
 └─ render        : integrator & output settings (overridable from CLI)
```

### 5.1 Lighting separate from models
- Lights live in **light rigs**: separate files or blocks such as `rig.studio.json` and `rig.sunset.json`. A scene lists rigs to include, and the CLI can swap one with `--rig sunset`.
- A model file (geometry plus material bindings) contains **no lights**. The exception is emissive *materials*, which belong with the object because they are part of its appearance, like a lamp bulb. The loader collects emissive instances into the same internal light list so the integrators don't care where a light came from.
- **Environment lighting** is itself a rig entry, so an HDRI can be swapped without touching the scene.
- Each light has an `enabled` flag and an optional per-light scale so rigs can be tuned without editing the source file.

### 5.2 Textures separate from materials
- A **texture** is data only: source (file or procedural), **colour-space tag** (sRGB, linear, ACEScg, raw or "data"), wrap mode, filtering, UV set and transform. It knows nothing about shading.
- A **material** is a set of named BSDF parameters. Each parameter is `constant | {"texture": "id", "channel": "r"} | {"node": ...}`. A small node graph covers mix, multiply, remap and normal-map decode.
- Textures are loaded lazily into a **shared, mip-mapped, tiled texture cache**, so many materials can reference the same image at no extra cost.
- Normal, bump and roughness maps are just textures bound to material parameters. There are no special cases.
- The spectral uplift of an RGB texture happens at lookup time, driven by the colour-space tag, so "data" textures such as roughness are never uplifted.

### 5.3 Cameras
- A scene defines any number of **named cameras**. `render.camera` picks the default and `--camera` overrides it.
- The camera holds only optics and pose. **Film and sensor settings** (resolution, filter, output) sit in `render.film`, so the same camera can render at different resolutions.
- The WinUI perspective viewport edits the active camera's pose and writes it back to the scene file.

### 5.4 Example (abridged)

```jsonc
{
  "$schema": "../schemas/prscene.schema.json",
  "version": 1,
  "units": "meters",
  "include": ["rigs/studio.json", "materials/library.json"],

  "textures": {
    "checker": { "type": "checker", "scale": 8, "a": [0.8,0.8,0.8], "b": [0.2,0.2,0.2], "colorspace": "srgb" }
  },
  "materials": {
    "floor":   { "model": "openpbr", "base_color": { "texture": "checker" }, "specular_roughness": 0.6 },
    "sf11":    { "model": "openpbr", "transmission_weight": 1, "ior": { "sellmeier": "SCHOTT_SF11" } }
  },
  "geometry": {
    "unit_sphere": { "type": "sphere", "radius": 1 },
    "ground":      { "type": "mesh", "file": "meshes/ground.ply" }
  },
  "objects": [
    { "geometry": "ground",      "material": "floor" },
    { "geometry": "unit_sphere", "material": "sf11", "transform": { "translate": [0,1,0], "scale": 0.5 } }
  ],
  "cameras": {
    "main": { "type": "thin_lens", "position": [4,3,6], "look_at": [0,0.8,0], "fov_y": 35, "f_stop": 8, "focus_distance": 7.2 }
  },
  "render": {
    "camera": "main",
    "integrator": { "type": "mmlt", "max_depth": 32, "bootstrap": 400000, "chains": 1024, "large_step": 0.3 },
    "film": { "width": 1280, "height": 720, "filter": "blackman_harris", "colorspace": "acescg" },
    "output": [ { "file": "out.exr" }, { "file": "out.png", "tonemap": "agx" } ]
  }
}
```

---

## 6. File formats

### Input
| Kind | Formats | Notes |
|---|---|---|
| **Native scene** | `.prscene.json` (+ JSON Schema) | Canonical format; everything else converts into it |
| Scenes (import) | **glTF 2.0 / .glb** | Best interchange for PBR: maps `KHR_materials_transmission`, `_volume`, `_ior`, **`_dispersion`**, `_sheen`, `_clearcoat`, `_specular`, `_emissive_strength`, `_anisotropy` |
| | **pbrt-v4** scene files (subset) | Gives direct access to well-known reference scenes for validation |
| | Mitsuba 3 XML (subset, stretch) | Same purpose as pbrt-v4 |
| | OpenUSD `.usda/.usdc` (stretch, phase 2) | Heavy dependency; use UsdPreviewSurface and OpenPBR |
| Meshes | OBJ(+MTL), PLY (binary), glTF | Plus a native binary mesh cache `.prmesh` for fast reload |
| Textures | EXR, HDR, PNG, JPEG, TGA, TIFF, `.tx` (tiled/mipped) | Via OIIO |
| Volumes | NanoVDB `.nvdb` (OpenVDB via offline conversion) | |
| Spectral data | CSV SPD (λ, value), Sellmeier/Cauchy coefficient tables, n/k metal tables | Bundle refractiveindex.info-derived data in `/data/spectra` |
| Lights | IES LM-63 photometric profiles | |

### Output
- **OpenEXR** (float, linear, with metadata: spp or mutations, time, seed, integrator). Optional AOVs are albedo, normal and depth.
- **PNG / JPEG** (tone-mapped, sRGB).
- **PFM** (simple float format for tests).
- **Checkpoint** `.prstate` (film accumulators plus chain states) to support `--resume`.

---

## 7. The CLI: `prender.exe`

```
prender <scene.prscene.json> [options]

  --camera <name>          override render.camera
  --rig <name>             select/override light rig (repeatable)
  --integrator pt|bdpt|pssmlt|mmlt
  --time <dur>             stop after wall time (e.g. 10m)
  --mutations <n>          stop after N mutations per pixel (MLT) / --spp for PT/BDPT
  --threads <n>            default: all logical cores
  --seed <n>               deterministic runs
  --res <w>x<h>            override film resolution
  --out <file>             repeatable; format inferred from extension
  --preview <file.png> --preview-interval <sec>   periodic progressive preview
  --progress json|text|none   machine-readable progress on stdout
  --resume <file.prstate>
  --validate               load + schema-validate scene only
```

**Progress protocol.** The UI consumes this as one JSON object per line on stdout:
```json
{"event":"stage","name":"bootstrap"}
{"event":"progress","fraction":0.42,"elapsed":31.2,"mutations_per_px":118.4,"accept_rate":0.61}
{"event":"preview","file":"C:/…/preview.png"}
{"event":"done","outputs":["out.exr","out.png"]}
{"event":"error","message":"…"}
```
Exit codes: 0 means OK, 1 a scene error, 2 an I/O error, 3 cancelled (Ctrl+C / CTRL_BREAK, which flushes current outputs first).

---

## 8. WinUI 3 front end (PRenderUI)

### Scope: deliberately lightweight
- **Layout:** a 2×2 grid of viewports (**Top**, **Front** and **Right** orthographic in wireframe, and **Perspective** through the active camera with a solid-shaded preview). A toolbar and a right-hand render panel sit around the grid.
- **Viewports:** Win2D `CanvasControl`. Spheres project to circles or ellipses and meshes to wireframe. The perspective view uses a simple CPU- or Win2D-shaded preview that tints each object with the material's base colour. No D3D pipeline is needed for v1. The upgrade path is `SwapChainPanel` with D3D11 if meshes get heavy.
- **Navigation:** pan and zoom in the ortho views. Orbit, pan and dolly in perspective, which updates the camera pose in the scene model. The active camera frustum is drawn in the ortho views.
- **Toolbar:** Open scene, Save, camera selector, integrator selector, time budget, and the **Render** button, which becomes **Cancel** while a render runs.
- **Render panel:** progressive image (reloaded on each `preview` event), a progress bar, elapsed time, mutations per pixel, acceptance rate, and "Open output folder".

### Flow
1. The UI loads `.prscene.json` into C# model classes generated from `prscene.schema.json`, so the schema is shared with the renderer.
2. **Render** saves a temp copy of the scene (including camera edits) and starts `prender.exe --progress json --preview … --preview-interval 2` with `System.Diagnostics.Process`.
3. The UI reads stdout line by line, updates the view model, and swaps the preview bitmap.
4. **Cancel** sends CTRL_BREAK or closes stdin. The renderer flushes its outputs and exits with code 3.

### Project structure
```
PRenderUI/
  App.xaml, MainWindow.xaml         2×2 Grid + toolbar + render panel
  Viewports/OrthoViewport.cs        Win2D wireframe, shared base w/ pan-zoom
  Viewports/PerspectiveViewport.cs  camera-driven preview
  Scene/SceneModel.cs (generated), SceneLoader.cs
  Render/RenderJob.cs               process launch, JSON progress parsing, cancel
  ViewModels/MainViewModel.cs       CommunityToolkit.Mvvm
```

---

## 9. Sample scene: "Sphere Pyramid"

A **square pyramid of 14 equal spheres** (3×3 + 2×2 + 1), radius *r* = 0.5 m, packed so each upper sphere rests on four spheres below it.

- Layer heights: y₀ = r, y₁ = r + r√2, y₂ = r + 2r√2
- Layer 0 centers on a 2r grid (x,z ∈ {−2r, 0, 2r}); layer 1 at (±r, ±r); layer 2 at (0, 0)

| # | Layer | Material | Exercises |
|---|---|---|---|
| 1 | 0 | Clear crown glass **BK7** (Sellmeier) | IOR, refraction, **caustics** on floor |
| 2 | 0 | Dense flint **SF11** (high dispersion) | **Chromatic dispersion**, rainbow caustics |
| 3 | 0 | **Diamond** (n≈2.42, strong dispersion) | High IOR, total internal reflection, "fire" |
| 4 | 0 | Polished **gold** (spectral n,k; roughness 0.02) | Mirror **reflections**, coloured Fresnel |
| 5 | 0 | **Copper**, roughness 0.35 | **Roughness** / glossy reflections |
| 6 | 0 | **Satin aluminium**: anisotropic GGX (αx 0.15, αy 0.45) + sheen | **Satin finish** |
| 7 | 0 | **Jade**: random-walk SSS, green-tinted σs, long mean free path | **Sub-surface scattering** |
| 8 | 0 | **Candle wax / skin-like**: short-MFP SSS with red-shifted scattering | SSS, translucency backlit by apex |
| 9 | 0 | **Water** (n=1.333, faint red absorption) | Low-IOR refraction, Beer–Lambert absorption |
| 10 | 1 | **Frosted glass** (rough dielectric α 0.25) | Rough transmission, blurred caustics |
| 11 | 1 | **Car paint**: clear-coat over metallic-flake base (layered) | Layered BSDF, dual highlights |
| 12 | 1 | **Red satin fabric** (diffuse + sheen) | Sheen / grazing retro-reflection |
| 13 | 1 | **Opal glass**: dielectric boundary + scattering interior medium | Volumetric scattering in glass, blue/orange opalescence |
| 14 | 2 | **Self-illuminated** sphere: 2700 K blackbody, 40 W, apex | **Self-illumination**, lights the pyramid from within |

**Environment and lighting**
- Ground: large plane, light-grey checker with a Lambertian finish (roughness 0.6) so caustics read clearly.
- Key light: a 0.3 m rectangular area light with spot falloff at 45° elevation, 5000 K. Its hard-edged beam produces strong caustics and shadows.
- Fill: a dim equirectangular HDRI or physical sky at low intensity.
- **Volumetric:** a homogeneous low-density fog box (σs ≈ 0.02 m⁻¹, HG g = 0.6) that can be toggled with `--rig fog`. It shows god rays from the key light through the glass spheres.
- Camera: thin lens, 50 mm equivalent, f/8, focus on sphere 7, 3/4 elevated view. Extra `top`, `front` and `side` orthographic cameras are defined for debugging and to match the UI views.

Files: `scenes/sphere-pyramid/scene.prscene.json`, `materials.json`, `rigs/studio.json`, `rigs/fog.json`.

---

## 10. Validation & testing (critical for "unbiased")

1. **Sampling unit tests:** a χ² goodness-of-fit test for every `BSDF::sample` against its `pdf` (as in pbrt/Mitsuba), and tests for light and phase-function sampling.
2. **BSDF physical tests:** reciprocity, energy conservation (white furnace ≤ 1), and Fresnel limits.
3. **Furnace tests:** a white furnace with a unit-albedo sphere must render uniformly 1, for surfaces and for media with albedo 1.
4. **Integrator cross-checks:** PT, BDPT, PSSMLT and MMLT must converge to the same image. The check compares mean images from K independent seeds and requires every pixel to agree within a confidence interval. This is the main unbiasedness test.
5. **Convergence rate:** MSE against a long reference must fall as roughly 1/N, which catches hidden bias that plateaus.
6. **Reference scenes:** Cornell box, Veach MIS scene, glass-caustic scene and dispersive prism, compared to pbrt-v4 or Mitsuba 3 renders of the same (imported) scenes.
7. **Regression suite:** fixed-seed short renders compared with FLIP or relative MSE thresholds in CI (GitHub Actions `windows-latest`).

---

## 11. Milestones

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Foundations** | CMake + vcpkg, CI, test harness, `prender --version` | Green CI |
| **1. Core** | Math, transforms, RNG and samplers, spectral types, CIE data, RGB uplift, XYZ film, EXR/PNG output | Unit tests pass. A spectral colour checker renders to correct sRGB values |
| **2. Geometry** | Embree integration, analytic spheres, meshes, instances, OBJ/PLY loading | Intersection tests; normals image of the pyramid |
| **3. Scene format** | JSON schema, loader, texture cache, materials/lights/cameras separation, `--validate` | Sample scene loads; schema rejects bad files |
| **4. Path tracer** | PT + MIS, Lambert, GGX conductor and dielectric, area and env lights, thin lens | Furnace tests; Cornell box matches reference |
| **5. Materials** | Dispersion (hero wavelength), sheen, anisotropy, layered (position-free MC), emission, normal maps | χ² tests; dispersive prism shows correct spectrum spread |
| **6. Media & SSS** | Homogeneous and heterogeneous media, tracking estimators, nested dielectrics, random-walk SSS | Media furnace; SSS matches PT brute force |
| **7. BDPT** | Media-aware BDPT with MIS | Matches PT across the reference set |
| **8. MLT** | PSSMLT (bootstrap, chains, splatting), then MMLT; wavelength in primary sample space; time and mutation budgets; checkpoints | Unbiasedness test vs BDPT; faster caustic convergence measured |
| **9. CLI polish** | Progress protocol, previews, cancel and flush, resume, glTF import | UI-ready protocol frozen (versioned) |
| **10. WinUI 3 app** | 4-view layout, scene loading, camera editing, Render/Cancel, progressive preview | Renders sample scene end-to-end from the UI |
| **11. Sample scene & docs** | Sphere Pyramid, fog rig, README, material cookbook | Final beauty render + validation report |
| *Stretch* | Manifold mutations, realistic lens, light BVH, USD import, OIDN denoise post-pass (separate output) | |

Phases 4–8 are the bulk of the work. The UI (phase 10) can be built in parallel once the progress protocol in phase 9 is sketched, using a stub renderer that emits fake progress.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MLT's uneven convergence ("splotches") and start-up bias | Large-step mutations, bootstrap resampling, MMLT, many short chains, and a reference comparison in CI |
| Dispersion plus MLT converges slowly | Wavelength as a mutable primary dimension, and hero wavelength for non-dispersive segments |
| SDS caustics (glass → floor → camera through glass) | Area lights and thin lens by default; manifold exploration as a stretch goal |
| Nested dielectric bugs (glass + fog, SSS inside coat) | A priority-based medium stack with dedicated tests |
| The UI grows into a DCC tool | Scope is fixed to view, camera tweak and render. All scene authoring stays in JSON |
| Precision and self-intersection | Embree robust mode, and ray-origin offsetting that scales with the error bound (pbrt-v4 method) |

---

## 13. Key references
- Veach, *Robust Monte Carlo Methods for Light Transport Simulation* (1997): BDPT, MIS, MLT
- Kelemen et al., *A Simple and Robust Mutation Strategy for MLT* (2002): PSSMLT
- Hachisuka et al., *Multiplexed Metropolis Light Transport* (2014)
- Wilkie et al., *Hero Wavelength Spectral Sampling* (2014)
- Jakob & Hanika, *A Low-Dimensional Function Space for Efficient Spectral Upsampling* (2019)
- Miller, Georgiev, Jarosz, *A Null-Scattering Path Integral Formulation of Light Transport* (2019)
- Guo, Hašan, Zhao, *Position-Free Monte Carlo for Layered Materials* (2018)
- Jakob & Marschner, *Manifold Exploration* (2012)
- Pharr, Jakob, Humphreys, *Physically Based Rendering*, 4th ed.
- OpenPBR Surface specification (ASWF)

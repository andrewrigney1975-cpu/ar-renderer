# Rendering acceleration: CPU tuning, oneAPI GPU and device selection

Investigation of backlog items 1–3 in `PLAN.md` §15, carried out on the development machine and the v0.3.0 code base.

## 1. Hardware and toolchain

| Component | Detail | Relevance |
|---|---|---|
| CPU | Intel Core i9-12900KF (Alder Lake): 8 P-cores (with Hyper-Threading) + 8 E-cores = 24 threads | Hybrid scheduling matters. AVX2/FMA are available; **AVX-512 is fused off** on consumer Alder Lake. The KF model has no integrated GPU. |
| GPU | **Intel Arc A770, 16 GB**, driver 32.0.101.9033 | Xe-HPG: 32 Xe cores, 32 hardware ray-tracing units, about 17 TFLOPS FP32. **No native FP64**: double precision is emulated and very slow. The driver includes Level Zero and OpenCL runtimes, which SYCL needs. |
| RAM | 128 GB | Not a constraint. |
| oneAPI | **Not installed**. Available via winget: `Intel.OneAPI.BaseToolkit` 2025.1 | Required for SYCL (the DPC++/icx compiler) and for Embree's GPU path. |

## 2. Current CPU behaviour (measured)

Sample scene (20 primitives, glass, subsurface, media), 480×270. Wall time includes about 0.06 s of startup.

| Integrator | 1 thread | 4 | 8 | 16 | 24 | Speed-up from 8 → 24 |
|---|---|---|---|---|---|---|
| Path tracer, 16 spp | 4.50 s | 1.17 s | 0.67 s | 0.47 s | 0.41 s | 1.63× |
| MMLT, 64 mutations/px | — | — | 3.15 s | — | 1.98 s | 1.59× |

Dense-mesh scene: 4 tessellated spheres (518,400 triangles), path tracer at 480×270.

| Measurement | Time |
|---|---|
| Scene load (OBJ parse + BVH build) | **1.0 s, single-threaded** |
| 64 spp, 8 threads (total) | 2.80 s (render ≈ 1.8 s) |
| 64 spp, 24 threads (total) | 2.14 s (render ≈ 1.14 s, **1.58×**) |

**Reading the numbers.** The ideal gain from adding Hyper-Threading and 8 E-cores to 8 P-cores is about 1.8–1.9×. We reach about 1.6×, which is reasonable but leaves roughly 15% on the table. On mesh-heavy scenes the serial load path is now a large part of the total time.

### Findings in the code

1. **Scheduling is already hybrid-friendly.** `ParallelFor` hands out work from an atomic counter, so E-cores simply take fewer tiles or chains. The remaining loss comes from:
   - **Thread creation on every call.** Each progressive pass and each MLT round starts and joins 24 threads. That's cheap per call, but the MLT loop makes hundreds of calls.
   - **Barrier tails.** Every pass or round ends when its slowest item finishes. When an E-core picks up a late tile or chain, the P-cores idle.
2. **The BVH is binary and scalar.** It tests one box per node against one ray. There is no SIMD traversal, and every primitive test is a virtual call through a heap-allocated `Triangle` object that holds a `shared_ptr`. That's 518k objects in the dense scene, each with refcount traffic when copied.
3. **Loading is serial.** The OBJ parser uses `istringstream`, and the SAH build is recursive and single-threaded.
4. **Compiler setup:** `/O2 /GL /LTCG /arch:AVX2` is already on. There is no profile-guided optimization. `SampledSpectrum` (4 floats) relies on auto-vectorization.
5. **Film accumulation uses `std::atomic<double>`**, which is compare-and-swap based on x64. Contention is low (splats are spread out), so this is not a hotspot today.

## 3. Item 1: use the full capability of 12th-gen+ Intel CPUs

Recommended work, in order of expected payoff per effort:

| # | Change | Expected gain | Effort |
|---|---|---|---|
| 1a | **Adopt Embree 4 (CPU) behind the existing `BVH` interface.** It provides AVX2 BVH8 traversal, parallel builds and compressed triangle storage. It also gives us the GPU path for item 2. | 2–4× on intersection-bound scenes; load time drops sharply | Medium (vcpkg or prebuilt; keep our BVH as a fallback) |
| 1b | Persistent worker pool with work stealing, and smaller work items at the end of each pass or round | 5–10% overall; less idle time at 24 threads | Small |
| 1c | Parallel, allocation-free mesh loading (`std::from_chars`, a parallel binned SAH build if 1a is not taken) | Seconds saved on large assets | Small |
| 1d | Data-oriented primitives (triangle index + mesh pointer instead of heap objects with virtual calls) | 10–30% on mesh scenes; also a prerequisite for GPU work (§4) | Medium |
| 1e | Profile with **Intel VTune** (free) for hotspots, then apply PGO (`/GENPROFILE` → `/USEPROFILE`) | 5–15% | Small |
| 1f | Explicit SIMD for `SampledSpectrum` (SSE, 4 wide) | Small; do it only if VTune shows spectral math as hot | Small |

AVX-512 is not an option on this CPU. Later Intel parts offer **AVX10**. Embree and runtime dispatch (compiling the hot kernels twice and choosing at startup) would pick it up without affecting AVX2-only machines.

## 4. Item 2: GPU rendering with oneAPI on Intel Arc

### Options considered

| Approach | Pros | Cons |
|---|---|---|
| **SYCL (oneAPI DPC++) + Embree 4 SYCL** | Single-source C++ shared with the CPU renderer. Embree 4 supports Xe-HPG with **hardware ray tracing** on Windows. Same BVH/API on CPU and GPU. | Needs the DPC++ compiler (icx) and the SYCL runtime DLLs. Kernel code may not use virtual functions, function pointers, exceptions, RTTI or dynamic allocation. |
| DirectX 12 DXR 1.1 (inline ray queries) | Mature on Arc. No new compiler. Windows-native. | Shaders in HLSL, so the BSDF, light and medium code must be duplicated. |
| Vulkan ray query | Portable across vendors | Same duplication as DXR, plus more boilerplate. |

**Recommendation:** SYCL with Embree 4. It is the only option that lets the *same* C++ material, light and spectral code run on both devices, and it matches the "oneAPI for Intel dGPUs" intent of the backlog item.

### What has to change in the renderer

The current design is object-oriented, which is fine on the CPU but not legal in SYCL kernels:

| Current construct | GPU-compatible replacement |
|---|---|
| Virtual `BxDF`, `Material`, `Light`, `Shape`, `Medium`, `Texture` classes | Tagged-pointer dispatch (as in pbrt-v4): a switch over a small type tag, so no vtables |
| `ScratchBuffer` bump allocation per sample | Fixed-size, stack-resident BSDF storage, or a variant holding every BxDF type |
| `std::shared_ptr`, `std::function`, `std::optional`, `std::vector` inside the scene | Flat device buffers (unified shared memory) built once from the loaded scene |
| **`std::atomic<double>` film** | **Float accumulators** (the A770 has no fast FP64). Accumulate per pass into a float buffer and reduce into the double film on the host, which keeps precision at high sample counts. |
| Recursive loops with deep, divergent control flow | **Wavefront path tracing**: separate kernels and queues for camera rays, intersection, per-material shading, NEE shadow rays and escape handling, to keep execution coherent |

### Integrator order on the GPU

1. **Path tracer (wavefront)** first. It is unbiased and spectral, and it gives the biggest immediate win for previews and final frames.
2. **BDPT**: feasible (pbrt-v4 has no GPU BDPT, but research implementations exist). Subpath storage per pixel is the main memory cost: `maxDepth` × ~200 B × pixels.
3. **MMLT/PSSMLT**: thousands of independent chains suit a GPU, but each chain carries a primary-sample vector and runs one BDPT connection per mutation, so control flow is highly divergent. Treat this as research. A pragmatic hybrid is **GPU bootstrap + CPU chains**: the bootstrap is the embarrassingly parallel, BDPT-heavy part of MLT.

### Expectations

- Path tracing on an A770 compared with the 24-thread CPU: typically **5–20×** for comparable kernels, with hardware RT helping most on mesh-heavy scenes.
- A 16K film in float is about 1.6 GB, which fits comfortably in 16 GB.
- Correctness: the existing cross-integrator agreement tests extend naturally to CPU-vs-GPU mean-image comparisons. Results will not be bit-identical (different RNG streams and float accumulation), but they must agree statistically. That's what "unbiased on both devices" means in practice.

### Build and deployment

- Put GPU code in a separate module (`prender_gpu.dll`) compiled with `icx -fsycl`. The main `prender.exe` stays MSVC-built and **loads the module at runtime**, so machines without oneAPI or an Intel GPU still run and simply fall back to the CPU.
- Ship the SYCL/Level Zero runtime DLLs (redistributable) next to the module.
- CI would need oneAPI installed, and GPU tests are skipped when no device is present.

### Proposed spike (1–2 weeks)

1. Install the oneAPI Base Toolkit and run `sycl-ls` to confirm the A770 appears as a Level Zero GPU.
2. Build a minimal Embree 4 SYCL sample, then a **wavefront path tracer for diffuse + GGX conductor + dielectric + area and environment lights** on flat scene buffers.
3. Compare against the CPU path tracer (mean luminance within 1–2%) and measure rays/s and time-to-noise-level on the sample scenes.
4. Decide go/no-go on the full port, based on speed-up and the size of the data-oriented refactor.

## 5. Item 3: user-selectable render device

This depends on item 2, but the interface can be designed now:

- **CLI:** `--list-devices` (JSON: index, name, type, memory, supported integrators) and `--device cpu|gpu|gpu:N` (default `cpu`).
- **Scene:** `render.device`, overridable from the CLI like every other render setting.
- **Behaviour:** if the requested device is unavailable, or the selected integrator/feature is not implemented on it (initially MLT, realistic lens, heterogeneous media), fall back to the CPU with a warning, and report the chosen device in the `scene` progress event.
- **UI:** a *Device* combo box in the render panel, populated from `--list-devices` at startup, with unsupported integrator/device combinations disabled.
- **Checkpoints** include the device type in their header, since CPU and GPU states are not interchangeable.

## 6. Recommended sequence

1. **CPU quick wins** (1b, 1c, 1e): small, low-risk, and they benefit everyone.
2. **Data-oriented refactor** (1d, plus tagged dispatch for materials, lights and media). This is the key enabler. It also speeds up the CPU and keeps a single code base for both devices.
3. **Embree 4 on the CPU** (1a), validated by the existing tests.
4. **oneAPI spike** (§4), then the GPU wavefront path tracer.
5. **Device selection** (§5), shipped together with the first GPU integrator.
6. GPU BDPT, then MLT (hybrid first).

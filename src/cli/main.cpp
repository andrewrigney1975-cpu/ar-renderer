// prender: spectral, unbiased Metropolis light transport renderer (command-line front end).

#include "core/film.h"
#include "core/image.h"
#include "core/fsutil.h"
#include "core/log.h"
#include "core/parallel.h"
#include "core/rgb2spec.h"
#include "integrators/aov.h"
#include "integrators/integrator.h"
#include "scene/scene_loader.h"

#include "nlohmann/json.hpp"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef RGB
#endif

using namespace pr;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr const char *kVersion = "0.2.0";

enum ExitCode { ExitOK = 0, ExitSceneError = 1, ExitIOError = 2, ExitCancelled = 3 };

enum class ProgressMode { None, Text, Json };

struct Args {
    std::string scene;
    LoadOptions load;
    std::optional<std::string> integrator;
    std::optional<int> spp, maxDepth, threads, chains;
    std::optional<double> mutations, timeLimit;
    std::optional<int64_t> bootstrap;
    std::optional<uint64_t> seed;
    std::vector<std::string> outputs;
    std::vector<std::pair<std::string, std::string>> aovs;  // name, file
    int aovSpp = 16;
    std::string preview;
    double previewInterval = 2.0;
    ProgressMode progress = ProgressMode::Text;
    bool controlStdin = false;
    bool validate = false;
    std::string convert;
    std::string checkpoint, resume;
    double checkpointInterval = 60;
    bool verbose = false;
    bool quiet = false;
};

void PrintUsage() {
    std::printf(
        "prender %s - spectral unbiased renderer (MMLT / PSSMLT / BDPT / path)\n\n"
        "usage: prender <scene.prscene.json> [options]\n\n"
        "  --camera <name>            override render.camera\n"
        "  --rig <name>               select light rig (repeatable; replaces active_rigs)\n"
        "  --add-rig <name>           add a light rig to the active set (repeatable)\n"
        "  --no-rigs                  disable all light rigs\n"
        "  --integrator <t>           mmlt | pssmlt | bdpt | path\n"
        "  --spp <n>                  samples per pixel (path, bdpt)\n"
        "  --mutations <n>            mutations per pixel (mmlt, pssmlt)\n"
        "  --bootstrap <n>            MLT bootstrap samples\n"
        "  --chains <n>               MLT Markov chains\n"
        "  --max-depth <n>            maximum path depth\n"
        "  --time <dur>               stop after a wall-clock budget, e.g. 90s, 10m, 1h\n"
        "  --threads <n>              worker threads (default: all logical cores)\n"
        "  --seed <n>                 random seed\n"
        "  --res <w>x<h>              override film resolution\n"
        "  --out <file>               output (repeatable): .exr .pfm .png .jpg\n"
        "  --aov <name>=<file>        auxiliary output: albedo, normal, depth, position (repeatable)\n"
        "  --aov-spp <n>              samples per pixel for AOVs (default 16)\n"
        "  --preview <file.png>       write a progressive preview periodically\n"
        "  --preview-interval <sec>   preview period (default 2)\n"
        "  --progress json|text|none  progress reporting on stdout (default text)\n"
        "  --control stdin            accept 'cancel' commands on stdin\n"
        "  --checkpoint <file>        save resumable state periodically, on cancel and at the end\n"
        "  --checkpoint-interval <d>  checkpoint period (default 60s)\n"
        "  --resume <file>            continue a checkpointed render (raise --spp/--mutations or --time)\n"
        "  --convert <out.json>       write the scene (incl. glTF/pbrt imports) as native JSON and exit\n"
        "  --validate                 load and validate the scene, then exit\n"
        "  --verbose | --quiet\n"
        "  --version | --help\n\n"
        "exit codes: 0 ok, 1 scene/argument error, 2 I/O error, 3 cancelled\n",
        kVersion);
}

bool ParseArgs(int argc, char **argv, Args *a, std::string *err) {
    auto need = [&](int &i) -> const char * {
        if (i + 1 >= argc) {
            *err = std::string("missing value for ") + argv[i];
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        const char *v = nullptr;
        if (s == "--help" || s == "-h") {
            PrintUsage();
            std::exit(ExitOK);
        } else if (s == "--version") {
            std::printf("prender %s\n", kVersion);
            std::exit(ExitOK);
        } else if (s == "--camera") {
            if (!(v = need(i))) return false;
            a->load.camera = v;
        } else if (s == "--rig") {
            if (!(v = need(i))) return false;
            if (!a->load.rigs) a->load.rigs = std::vector<std::string>();
            a->load.rigs->push_back(v);
        } else if (s == "--no-rigs") {
            a->load.rigs = std::vector<std::string>();
        } else if (s == "--add-rig") {
            if (!(v = need(i))) return false;
            a->load.addRigs.push_back(v);
        } else if (s == "--integrator") {
            if (!(v = need(i))) return false;
            a->integrator = v;
        } else if (s == "--spp") {
            if (!(v = need(i))) return false;
            a->spp = std::atoi(v);
        } else if (s == "--mutations") {
            if (!(v = need(i))) return false;
            a->mutations = std::atof(v);
        } else if (s == "--bootstrap") {
            if (!(v = need(i))) return false;
            a->bootstrap = std::atoll(v);
        } else if (s == "--chains") {
            if (!(v = need(i))) return false;
            a->chains = std::atoi(v);
        } else if (s == "--max-depth") {
            if (!(v = need(i))) return false;
            a->maxDepth = std::atoi(v);
        } else if (s == "--time") {
            if (!(v = need(i))) return false;
            double t;
            if (!ParseDuration(v, &t)) {
                *err = std::string("bad duration: ") + v;
                return false;
            }
            a->timeLimit = t;
        } else if (s == "--threads") {
            if (!(v = need(i))) return false;
            a->threads = std::atoi(v);
        } else if (s == "--seed") {
            if (!(v = need(i))) return false;
            a->seed = std::strtoull(v, nullptr, 10);
        } else if (s == "--res") {
            if (!(v = need(i))) return false;
            if (std::sscanf(v, "%dx%d", &a->load.width, &a->load.height) != 2 || a->load.width <= 0 || a->load.height <= 0) {
                *err = std::string("bad resolution: ") + v;
                return false;
            }
        } else if (s == "--out" || s == "-o") {
            if (!(v = need(i))) return false;
            a->outputs.push_back(v);
        } else if (s == "--aov") {
            if (!(v = need(i))) return false;
            std::string spec = v;
            size_t eq = spec.find('=');
            AOVType t;
            if (eq == std::string::npos || !ParseAOV(spec.substr(0, eq), &t)) {
                *err = "bad --aov (expected albedo|normal|depth|position=<file>): " + spec;
                return false;
            }
            a->aovs.push_back({spec.substr(0, eq), spec.substr(eq + 1)});
        } else if (s == "--aov-spp") {
            if (!(v = need(i))) return false;
            a->aovSpp = std::max(1, std::atoi(v));
        } else if (s == "--preview") {
            if (!(v = need(i))) return false;
            a->preview = v;
        } else if (s == "--preview-interval") {
            if (!(v = need(i))) return false;
            a->previewInterval = std::atof(v);
        } else if (s == "--progress") {
            if (!(v = need(i))) return false;
            std::string m = v;
            if (m == "json") a->progress = ProgressMode::Json;
            else if (m == "text") a->progress = ProgressMode::Text;
            else if (m == "none") a->progress = ProgressMode::None;
            else {
                *err = "bad --progress mode: " + m;
                return false;
            }
        } else if (s == "--control") {
            if (!(v = need(i))) return false;
            if (std::string(v) != "stdin") {
                *err = "only '--control stdin' is supported";
                return false;
            }
            a->controlStdin = true;
        } else if (s == "--checkpoint") {
            if (!(v = need(i))) return false;
            a->checkpoint = v;
        } else if (s == "--checkpoint-interval") {
            if (!(v = need(i))) return false;
            if (!ParseDuration(v, &a->checkpointInterval)) {
                *err = std::string("bad duration: ") + v;
                return false;
            }
        } else if (s == "--resume") {
            if (!(v = need(i))) return false;
            a->resume = v;
        } else if (s == "--convert") {
            if (!(v = need(i))) return false;
            a->convert = v;
        } else if (s == "--validate") {
            a->validate = true;
        } else if (s == "--verbose" || s == "-v") {
            a->verbose = true;
        } else if (s == "--quiet" || s == "-q") {
            a->quiet = true;
        } else if (!s.empty() && s[0] == '-') {
            *err = "unknown option: " + s;
            return false;
        } else if (a->scene.empty()) {
            a->scene = s;
        } else {
            *err = "unexpected argument: " + s;
            return false;
        }
    }
    if (a->scene.empty()) {
        *err = "no scene file given";
        return false;
    }
    return true;
}

ProgressMode gProgressMode = ProgressMode::Text;

void Emit(const json &j) {
    if (gProgressMode == ProgressMode::Json) WriteStdoutLine(j.dump());
}

void EmitError(const std::string &msg) {
    Emit({{"event", "error"}, {"message", msg}});
    LogError("{}", msg);
}

std::atomic<RenderControl *> gControl{nullptr};

#ifdef _WIN32
BOOL WINAPI CtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        if (RenderControl *c = gControl.load()) {
            if (c->cancel) return FALSE;  // second press: default handling (terminate)
            c->cancel = true;
            return TRUE;
        }
    }
    return FALSE;
}
#endif

const char *IntegratorName(IntegratorType t) {
    switch (t) {
    case IntegratorType::Path: return "path";
    case IntegratorType::BDPT: return "bdpt";
    case IntegratorType::MMLT: return "mmlt";
    case IntegratorType::PSSMLT: return "pssmlt";
    }
    return "?";
}

std::string Lower(std::string s) {
    for (char &c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

bool WriteAOV(const Scene &scene, const OutputSpec &spec, int spp, std::string *err) {
    AOVType type;
    if (!ParseAOV(spec.aov, &type)) {
        *err = "unknown AOV '" + spec.aov + "'";
        return false;
    }
    Image img = RenderAOV(scene, type, spp, scene.settings.seed);
    std::string ext = Lower(Utf8Path(spec.file).extension().string());
    if (ext == ".exr") return WriteEXR(spec.file, img, spec.half, err);
    if (ext == ".pfm") return WritePFM(spec.file, img, err);
    // LDR visualization.
    float maxDepth = 0;
    if (type == AOVType::Depth)
        for (float v : img.Data())
            if (v < 1e9f) maxDepth = std::max(maxDepth, v);
    Image disp(img.Width(), img.Height());
    for (int y = 0; y < img.Height(); ++y)
        for (int x = 0; x < img.Width(); ++x) {
            RGB c = img.Get(x, y);
            if (type == AOVType::Normal) c = RGB(0.5f + 0.5f * c.r, 0.5f + 0.5f * c.g, 0.5f + 0.5f * c.b);
            else if (type == AOVType::Depth) {
                float d = c.r < 1e9f && maxDepth > 0 ? 1 - c.r / maxDepth : 0.f;
                c = RGB(d, d, d);
            } else if (type == AOVType::Albedo) {
                c = RGB(SRGBEncode(c.r), SRGBEncode(c.g), SRGBEncode(c.b));
            }
            disp.Set(x, y, c);
        }
    return WriteLDR(spec.file, disp, err);
}

bool WriteOutput(const Film &film, double scale, const OutputSpec &spec, std::string *err) {
    std::error_code ec;
    fs::path p = Utf8Path(spec.file);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    std::string ext = Lower(p.extension().string());
    if (ext == ".exr") return WriteEXR(spec.file, film.Resolve(scale), spec.half, err);
    if (ext == ".pfm") return WritePFM(spec.file, film.Resolve(scale), err);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga")
        return WriteLDR(spec.file, film.ResolveDisplay(scale, spec.toneMap), err);
    *err = "unsupported output format: " + spec.file;
    return false;
}

} // namespace

int main(int argc, char **argv) {
    Args args;
    std::string err;
    if (!ParseArgs(argc, argv, &args, &err)) {
        std::fprintf(stderr, "prender: %s\n(use --help for usage)\n", err.c_str());
        return ExitSceneError;
    }
    gProgressMode = args.progress;
    SetLogLevel(args.verbose ? LogLevel::Verbose : (args.quiet ? LogLevel::Warning : LogLevel::Info));
    if (args.threads) SetThreadCount(*args.threads);

    RenderControl control;
    control.Start();
    gControl = &control;
#ifdef _WIN32
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
#endif
    if (args.controlStdin) {
        std::thread([&control] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line == "cancel" || line == "stop") control.cancel = true;
            }
        }).detach();
    }

    if (!args.convert.empty()) {
        if (!ConvertSceneToJson(args.scene, args.convert, &err)) {
            EmitError(err);
            return ExitSceneError;
        }
        LogInfo("wrote {}", args.convert);
        Emit({{"event", "done"}, {"outputs", json::array({args.convert})}});
        return ExitOK;
    }
    Emit({{"event", "stage"}, {"name", "load"}});
    // Make sure the spectral uplift table exists before timing the render.
    RGBToSpectrumTable::Get();
    auto scene = LoadScene(args.scene, args.load, &err);
    if (!scene) {
        EmitError(err);
        return ExitSceneError;
    }

    RenderSettings &rs = scene->settings;
    if (args.integrator) {
        std::string t = *args.integrator;
        if (t == "path" || t == "pt") rs.integrator.type = IntegratorType::Path;
        else if (t == "bdpt") rs.integrator.type = IntegratorType::BDPT;
        else if (t == "mmlt" || t == "mlt") rs.integrator.type = IntegratorType::MMLT;
        else if (t == "pssmlt") rs.integrator.type = IntegratorType::PSSMLT;
        else {
            EmitError("unknown integrator: " + t);
            return ExitSceneError;
        }
    }
    if (args.spp) rs.integrator.spp = *args.spp;
    if (args.mutations) rs.integrator.mutationsPerPixel = *args.mutations;
    if (args.bootstrap) rs.integrator.bootstrapSamples = *args.bootstrap;
    if (args.chains) rs.integrator.chains = *args.chains;
    if (args.maxDepth) rs.integrator.maxDepth = *args.maxDepth;
    if (args.timeLimit) rs.integrator.timeLimit = *args.timeLimit;
    if (args.seed) rs.seed = *args.seed;
    if (!args.outputs.empty()) {
        std::vector<OutputSpec> outs;
        for (const auto &o : args.outputs) {
            OutputSpec spec;
            spec.file = o;
            // Keep tone mapping settings from the scene for a matching file type, if any.
            for (const auto &s : rs.outputs)
                if (Lower(Utf8Path(s.file).extension().string()) == Lower(Utf8Path(o).extension().string()))
                    spec.toneMap = s.toneMap, spec.half = s.half;
            outs.push_back(spec);
        }
        rs.outputs = outs;
    }
    if (rs.outputs.empty()) {
        std::string stem = Utf8Path(args.scene).stem().string();
        if (stem.size() > 8 && stem.substr(stem.size() - 8) == ".prscene") stem = stem.substr(0, stem.size() - 8);
        rs.outputs.push_back({stem + ".exr", ToneMap::ACES, false});
        rs.outputs.push_back({stem + ".png", ToneMap::ACES, false});
    }

    size_t nLights = scene->lights.size();
    LogInfo("scene: {} primitives, {} lights, {} media, camera '{}', {}x{}, integrator {}", scene->primitives.size(),
            nLights, scene->media.size(), rs.cameraName, rs.film.width, rs.film.height, IntegratorName(rs.integrator.type));
    Emit({{"event", "scene"},
          {"primitives", scene->primitives.size()},
          {"lights", nLights},
          {"media", scene->media.size()},
          {"camera", rs.cameraName},
          {"width", rs.film.width},
          {"height", rs.film.height},
          {"integrator", IntegratorName(rs.integrator.type)},
          {"threads", ThreadCount()}});
    if (args.validate) {
        Emit({{"event", "done"}, {"outputs", json::array()}, {"validated", true}});
        LogInfo("scene is valid");
        return ExitOK;
    }

    Film film(rs.film);
    auto integrator = CreateIntegrator(rs.integrator, rs.seed);

    control.previewInterval = args.preview.empty() ? 0 : std::max(0.25, args.previewInterval);
    control.resumePath = args.resume;
    control.checkpointPath = !args.checkpoint.empty() ? args.checkpoint : args.resume;
    control.checkpointInterval = std::max(1.0, args.checkpointInterval);
    control.fingerprint = scene->Fingerprint();
    control.seed = rs.seed;
    std::string lastStage;
    control.onProgress = [&](const ProgressInfo &p) {
        if (p.stage != lastStage) {
            lastStage = p.stage;
            Emit({{"event", "stage"}, {"name", p.stage}});
        }
        json j = {{"event", "progress"}, {"stage", p.stage}, {"fraction", p.fraction}, {"elapsed", p.elapsed},
                  {"samples_per_pixel", p.samplesPerPixel}};
        if (p.acceptRate >= 0) j["accept_rate"] = p.acceptRate;
        Emit(j);
        if (gProgressMode == ProgressMode::Text) {
            std::fprintf(stderr, "\r%-9s %5.1f%%  %7.1fs  %8.2f %s%s", p.stage.c_str(), 100 * p.fraction, p.elapsed,
                         p.samplesPerPixel,
                         (rs.integrator.type == IntegratorType::MMLT || rs.integrator.type == IntegratorType::PSSMLT)
                             ? "mut/px"
                             : "spp",
                         p.acceptRate >= 0 ? std::format("  accept {:.1f}%", 100 * p.acceptRate).c_str() : "");
            std::fflush(stderr);
        }
    };
    control.onPreview = [&](const Film &f, double scale) {
        std::string tmp = args.preview + ".tmp.png";
        std::string e;
        if (!WriteLDR(tmp, f.ResolveDisplay(scale, ToneMap::ACES), &e)) return;
        std::error_code ec;
        fs::rename(Utf8Path(tmp), Utf8Path(args.preview), ec);
        if (ec) {
            fs::remove(Utf8Path(tmp), ec);
            return;
        }
        Emit({{"event", "preview"}, {"file", PathUtf8(fs::absolute(Utf8Path(args.preview)))}});
    };

    double scale = 0;
    bool ok = integrator->Render(*scene, film, control, &scale, &err);
    if (gProgressMode == ProgressMode::Text) std::fprintf(stderr, "\n");
    if (!ok) {
        EmitError(err);
        return ExitSceneError;
    }
    if (!args.preview.empty()) control.MaybePreview(film, scale, true);

    Emit({{"event", "stage"}, {"name", "write"}});
    json written = json::array();
    int rc = control.cancel ? ExitCancelled : ExitOK;
    for (const auto &[name, file] : args.aovs) {
        OutputSpec spec;
        spec.file = file;
        spec.aov = name;
        rs.outputs.push_back(spec);
    }
    for (const auto &o : rs.outputs) {
        std::string e;
        bool written_ok = o.aov.empty() ? WriteOutput(film, scale, o, &e) : WriteAOV(*scene, o, args.aovSpp, &e);
        if (!written_ok) {
            EmitError(e);
            rc = ExitIOError;
        } else {
            written.push_back(PathUtf8(fs::absolute(Utf8Path(o.file))));
            LogInfo("wrote {}", o.file);
        }
    }
    Emit({{"event", "done"}, {"outputs", written}, {"elapsed", control.Elapsed()}, {"cancelled", bool(control.cancel)}});
    gControl = nullptr;
    return rc;
}

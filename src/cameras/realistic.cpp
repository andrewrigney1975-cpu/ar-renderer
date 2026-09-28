#include "cameras/realistic.h"

#include "core/fsutil.h"
#include "core/sampling.h"

#include <fstream>
#include <sstream>

namespace pr {

std::vector<LensElement> RealisticCamera::DoubleGauss50(float s) {
    // D-GAUSS F/2 22deg HFOV, US patent 2,673,491 (Tronnier), scaled to 50 mm (as shipped with pbrt).
    // radius, thickness, n_d, aperture diameter, Abbe number (typical glasses for these indices).
    const float d[][5] = {{29.475f, 3.76f, 1.67f, 25.2f, 47.2f},  {84.83f, 0.12f, 1.f, 25.2f, 0},
                          {19.275f, 4.025f, 1.67f, 23.f, 47.2f},  {40.77f, 3.275f, 1.699f, 23.f, 30.1f},
                          {12.75f, 5.705f, 1.f, 18.f, 0},         {0.f, 4.5f, 0.f, 17.1f, 0},
                          {-14.495f, 1.18f, 1.603f, 17.f, 38.0f}, {40.77f, 6.065f, 1.658f, 20.f, 57.3f},
                          {-20.385f, 0.19f, 1.f, 20.f, 0},        {437.065f, 3.22f, 1.717f, 20.f, 47.9f},
                          {-39.73f, 5.f, 1.f, 20.f, 0}};
    std::vector<LensElement> e;
    for (const auto &r : d) e.push_back({r[0] * s, r[1] * s, r[2], r[4], r[3] * s * 0.5f});
    return e;
}

bool RealisticCamera::LoadLensFile(const std::string &path, float s, std::vector<LensElement> *out, std::string *err) {
    std::ifstream in(Utf8Path(path));
    if (!in) {
        *err = "cannot open lens file " + path;
        return false;
    }
    std::string line;
    out->clear();
    while (std::getline(in, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        float r, t, n, a, v = 0;
        if (!(ls >> r >> t >> n >> a)) continue;
        ls >> v;
        out->push_back({r * s, t * s, n, v, a * s * 0.5f});
    }
    if (out->empty()) {
        *err = path + ": no lens elements";
        return false;
    }
    return true;
}

RealisticCamera::RealisticCamera(const Transform &cameraToWorld, std::vector<LensElement> elements, float filmDiagonal,
                                 const CameraFilmInfo &film, float focusDistance, bool dispersion)
    : cameraToWorld_(cameraToWorld), elements_(std::move(elements)), film_(film), dispersion_(dispersion) {
    float aspect = float(film.width) / film.height;
    filmW_ = filmDiagonal * aspect / std::sqrt(1 + aspect * aspect);
    filmH_ = filmW_ / aspect;
    elements_.back().thickness = FocusRearThickness(focusDistance);
    filmRadius_ = std::sqrt(filmW_ * filmW_ + filmH_ * filmH_) / 2;
    ComputeExitPupils();
}

void RealisticCamera::ComputeExitPupils() {
    // Bound the region of the rear element through which light reaches each radial film band
    // (after pbrt): rays are sampled only inside it, which is unbiased as long as the bounds are
    // conservative (they are padded by a grid cell).
    const int nBins = 32, grid = 96, nRadii = 4;
    const float rr = elements_.back().apertureRadius * 1.5f;
    const float step = 2 * rr / grid;
    pupils_.assign(nBins, Pupil{Infinity, Infinity, -Infinity, -Infinity});
    float pass0 = 0;
    for (int b = 0; b < nBins; ++b) {
        Pupil &p = pupils_[b];
        int hits = 0, total = 0;
        for (int k = 0; k < nRadii; ++k) {
            float r = filmRadius_ * (b + (k + 0.5f) / nRadii) / nBins;
            for (int gy = 0; gy < grid; ++gy)
                for (int gx = 0; gx < grid; ++gx) {
                    float x = -rr + (gx + 0.5f) * step, y = -rr + (gy + 0.5f) * step;
                    Vec3f pFilm(r, 0, 0), pRear(x, y, LensRearZ());
                    Ray out;
                    ++total;
                    if (!TraceFromFilm(Ray(pFilm, pRear - pFilm), 587.6f, false, &out)) continue;
                    ++hits;
                    p.x0 = std::min(p.x0, x); p.x1 = std::max(p.x1, x);
                    p.y0 = std::min(p.y0, y); p.y1 = std::max(p.y1, y);
                }
        }
        if (!p.Empty()) {
            p.x0 -= step; p.y0 -= step; p.x1 += step; p.y1 += step;
        }
        if (b == 0 && total > 0) pass0 = float(hits) * step * step / std::max(1, nRadii) ;
    }
    // Normalize exposure so an unvignetted on-axis pixel has weight ~1 regardless of the f-stop.
    exposureNorm_ = pass0 > 0 ? 1 / pass0 : 1.f;
}

float RealisticCamera::Eta(const LensElement &e, float lambda, bool useLambda) const {
    if (e.eta == 0 || e.eta == 1) return 1;
    if (!useLambda || !dispersion_ || e.abbe <= 0) return e.eta;
    // Cauchy law through n_d with the given Abbe number V = (n_d - 1) / (n_F - n_C).
    float B = (e.eta - 1) / (e.abbe * (1 / (486.1f * 486.1f) - 1 / (656.3f * 656.3f)));
    float A = e.eta - B / (587.6f * 587.6f);
    return A + B / (lambda * lambda);
}

static bool IntersectSphericalElement(float radius, float zCenter, const Ray &ray, float *t, Vec3f *n) {
    Vec3f o = ray.o - Vec3f(0, 0, zCenter);
    double A = double(ray.d.x) * ray.d.x + double(ray.d.y) * ray.d.y + double(ray.d.z) * ray.d.z;
    double B = 2 * (double(ray.d.x) * o.x + double(ray.d.y) * o.y + double(ray.d.z) * o.z);
    double C = double(o.x) * o.x + double(o.y) * o.y + double(o.z) * o.z - double(radius) * radius;
    double disc = B * B - 4 * A * C;
    if (disc < 0) return false;
    double q = B < 0 ? -0.5 * (B - std::sqrt(disc)) : -0.5 * (B + std::sqrt(disc));
    double t0 = q / A, t1 = C / q;
    if (t0 > t1) std::swap(t0, t1);
    bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
    *t = float(useCloserT ? t0 : t1);
    if (*t < 0) return false;
    *n = Normalize(o + ray.d * *t);
    *n = FaceForward(*n, -ray.d);
    return true;
}

// Rays are given in camera space (+z towards the scene, film at z = 0); internally the lens
// space has the scene towards -z, as in pbrt.
bool RealisticCamera::TraceFromFilm(Ray r, float lambda, bool useLambda, Ray *out) const {
    Ray rl(Vec3f(r.o.x, r.o.y, -r.o.z), Vec3f(r.d.x, r.d.y, -r.d.z));
    float elementZ = 0;
    for (int i = int(elements_.size()) - 1; i >= 0; --i) {
        const LensElement &e = elements_[i];
        elementZ -= e.thickness;
        float t;
        Vec3f n;
        bool isStop = e.curvatureRadius == 0;
        if (isStop) {
            if (rl.d.z >= 0) return false;
            t = (elementZ - rl.o.z) / rl.d.z;
        } else if (!IntersectSphericalElement(e.curvatureRadius, elementZ + e.curvatureRadius, rl, &t, &n)) {
            return false;
        }
        Vec3f pHit = rl(t);
        if (pHit.x * pHit.x + pHit.y * pHit.y > e.apertureRadius * e.apertureRadius) return false;
        rl.o = pHit;
        if (!isStop) {
            float etaI = Eta(e, lambda, useLambda);
            float etaT = (i > 0 && elements_[i - 1].eta != 0) ? Eta(elements_[i - 1], lambda, useLambda) : 1.f;
            Vec3f wi = Normalize(-rl.d), wt;
            float etap;
            // pr::Refract expects eta = n_t / n_i on the side n points to (here: the incident side).
            if (!Refract(wi, n, etaT / etaI, &etap, &wt)) return false;
            rl.d = wt;
        }
    }
    *out = Ray(Vec3f(rl.o.x, rl.o.y, -rl.o.z), Normalize(Vec3f(rl.d.x, rl.d.y, -rl.d.z)));
    return true;
}

float RealisticCamera::FocusRearThickness(float focusDistance) {
    // Secant search on the rear thickness so that a paraxial ray leaving the film centre crosses
    // the optical axis at the focus distance (measured from the film).
    auto crossing = [&](float rear) -> float {
        elements_.back().thickness = rear;
        float h = elements_.back().apertureRadius * 0.05f;
        Ray r(Vec3f(0, 0, 0), Normalize(Vec3f(h, 0, rear)));
        Ray out;
        if (!TraceFromFilm(r, 587.6f, false, &out) || out.d.x == 0) return Infinity;
        float t = -out.o.x / out.d.x;
        if (t <= 0) return Infinity;  // diverging: focus beyond infinity
        return out(t).z;
    };
    float lensLength = 0;
    for (const auto &e : elements_) lensLength += e.thickness;
    float lo = elements_.back().thickness * 0.2f, hi = elements_.back().thickness * 4 + lensLength;
    // Bisection on f(rear) = crossing(rear) - focusDistance; moving the film back focuses closer.
    float best = elements_.back().thickness;
    for (int it = 0; it < 80; ++it) {
        float mid = 0.5f * (lo + hi);
        float z = crossing(mid);
        if (!std::isfinite(z) || z > focusDistance) lo = mid;
        else hi = mid;
        best = mid;
    }
    elements_.back().thickness = best;
    return best;
}

std::optional<Ray> RealisticCamera::GenerateRay(const CameraSample &s) const {
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
    float w;
    return GenerateRay(s, lambda, &w);
}

std::optional<Ray> RealisticCamera::GenerateRay(const CameraSample &s, SampledWavelengths &lambda, float *weight) const {
    // The lens forms an inverted image: flip film coordinates so the render is upright.
    float fx = -(s.pFilm.x / film_.width - 0.5f) * filmW_;
    float fy = (s.pFilm.y / film_.height - 0.5f) * filmH_;
    Vec3f pFilm(fx, fy, 0);
    float rFilmLen = std::sqrt(fx * fx + fy * fy);
    int bin = std::min(int(rFilmLen / filmRadius_ * pupils_.size()), int(pupils_.size()) - 1);
    const Pupil &pupil = pupils_[bin];
    if (pupil.Empty()) return std::nullopt;
    // Uniform sample in the pupil bounds (defined for a film point on +x), rotated to the film point.
    float px = Lerp(s.pLens.x, pupil.x0, pupil.x1), py = Lerp(s.pLens.y, pupil.y0, pupil.y1);
    float sinT = rFilmLen > 0 ? fy / rFilmLen : 0.f, cosT = rFilmLen > 0 ? fx / rFilmLen : 1.f;
    Vec3f pRear(cosT * px - sinT * py, sinT * px + cosT * py, LensRearZ());
    Ray rFilm(pFilm, pRear - pFilm);
    bool useLambda = dispersion_;
    Ray r;
    if (!TraceFromFilm(rFilm, lambda[0], useLambda, &r)) return std::nullopt;
    if (useLambda) lambda.TerminateSecondary();
    float cosTheta = Normalize(rFilm.d).z;
    // Irradiance estimate: L cos^4(theta) A_pupil / z^2, normalized for exposure.
    *weight = Sqr(Sqr(cosTheta)) * pupil.Area() * exposureNorm_;
    return Ray(cameraToWorld_.Point(r.o), Normalize(cameraToWorld_.Vector(r.d)));
}

} // namespace pr

#include "cameras/camera.h"

#include "core/sampling.h"

namespace pr {

PerspectiveCamera::PerspectiveCamera(const CameraParams &p, const CameraFilmInfo &film)
    : cameraToWorld_(p.cameraToWorld), worldToCamera_(p.cameraToWorld.Inverse()), film_(film) {
    tanHalf_ = std::tan(Radians(p.fovY) / 2);
    aspect_ = float(film.width) / float(film.height);
    lensRadius_ = std::max(0.f, p.lensRadius);
    focalDistance_ = std::max(1e-4f, p.focalDistance);
    float sw = float(film.width + 2 * film.margin) / film.width;
    float sh = float(film.height + 2 * film.margin) / film.height;
    A_ = (2 * tanHalf_ * aspect_ * sw) * (2 * tanHalf_ * sh);
    forward_ = Normalize(cameraToWorld_.Vector(Vec3f(0, 0, 1)));
}

Vec3f PerspectiveCamera::RasterToCamera(Vec2f p) const {
    float x = (2 * p.x / film_.width - 1) * tanHalf_ * aspect_;
    float y = (1 - 2 * p.y / film_.height) * tanHalf_;
    return {x, y, 1};
}

Vec2f PerspectiveCamera::CameraToRaster(const Vec3f &pz1) const {
    float px = (pz1.x / (tanHalf_ * aspect_) + 1) * film_.width / 2;
    float py = (1 - pz1.y / tanHalf_) * film_.height / 2;
    return {px, py};
}

bool PerspectiveCamera::InsideSampleBounds(Vec2f p) const {
    float m = float(film_.margin);
    return p.x >= -m && p.x < film_.width + m && p.y >= -m && p.y < film_.height + m;
}

std::optional<Ray> PerspectiveCamera::GenerateRay(const CameraSample &s) const {
    Vec3f pCamera = RasterToCamera(s.pFilm);
    Vec3f dir = Normalize(pCamera);
    Vec3f o(0, 0, 0);
    if (lensRadius_ > 0) {
        Vec2f pLens = SampleUniformDiskConcentric(s.pLens) * lensRadius_;
        float ft = focalDistance_ / dir.z;
        Vec3f pFocus = dir * ft;
        o = Vec3f(pLens.x, pLens.y, 0);
        dir = Normalize(pFocus - o);
    }
    return Ray(cameraToWorld_.Point(o), Normalize(cameraToWorld_.Vector(dir)));
}

SampledSpectrum PerspectiveCamera::We(const Ray &ray, Vec2f *pRaster) const {
    float cosTheta = Dot(ray.d, forward_);
    if (cosTheta <= 0) return SampledSpectrum(0.f);
    float t = (lensRadius_ > 0 ? focalDistance_ : 1.f) / cosTheta;
    Vec3f pFocus = worldToCamera_.Point(ray(t));
    Vec3f pz1 = pFocus / pFocus.z;
    Vec2f pr = CameraToRaster(pz1);
    if (pRaster) *pRaster = pr;
    if (!InsideSampleBounds(pr)) return SampledSpectrum(0.f);
    float lensArea = lensRadius_ != 0 ? (Pi * lensRadius_ * lensRadius_) : 1;
    float cos2Theta = cosTheta * cosTheta;
    return SampledSpectrum(1 / (A_ * lensArea * cos2Theta * cos2Theta));
}

void PerspectiveCamera::PDF_We(const Ray &ray, float *pdfPos, float *pdfDir) const {
    float cosTheta = Dot(ray.d, forward_);
    if (cosTheta <= 0) {
        *pdfPos = *pdfDir = 0;
        return;
    }
    float t = (lensRadius_ > 0 ? focalDistance_ : 1.f) / cosTheta;
    Vec3f pFocus = worldToCamera_.Point(ray(t));
    Vec2f pr = CameraToRaster(pFocus / pFocus.z);
    if (!InsideSampleBounds(pr)) {
        *pdfPos = *pdfDir = 0;
        return;
    }
    float lensArea = lensRadius_ != 0 ? (Pi * lensRadius_ * lensRadius_) : 1;
    *pdfPos = 1 / lensArea;
    *pdfDir = 1 / (A_ * cosTheta * cosTheta * cosTheta);
}

std::optional<CameraWiSample> PerspectiveCamera::SampleWi(const Interaction &ref, Vec2f u) const {
    Vec2f pLens = SampleUniformDiskConcentric(u) * lensRadius_;
    Vec3f pLensWorld = cameraToWorld_.Point(Vec3f(pLens.x, pLens.y, 0));
    Vec3f wi = pLensWorld - ref.p;
    float dist = Length(wi);
    if (dist == 0) return std::nullopt;
    wi = wi / dist;
    float lensArea = lensRadius_ != 0 ? (Pi * lensRadius_ * lensRadius_) : 1;
    float cosLens = AbsDot(forward_, wi);
    if (cosLens == 0) return std::nullopt;
    CameraWiSample cs;
    cs.pdf = (dist * dist) / (cosLens * lensArea);
    cs.Wi = We(Ray(pLensWorld, -wi), &cs.pRaster);
    if (!cs.Wi) return std::nullopt;
    cs.wi = wi;
    cs.pLens.p = pLensWorld;
    cs.pLens.n = forward_;
    cs.pLens.medium = medium;
    return cs;
}

OrthographicCamera::OrthographicCamera(const CameraParams &p, const CameraFilmInfo &film)
    : cameraToWorld_(p.cameraToWorld), film_(film) {
    halfH_ = p.orthoHeight / 2;
    halfW_ = halfH_ * float(film.width) / film.height;
}

std::optional<Ray> OrthographicCamera::GenerateRay(const CameraSample &s) const {
    float x = (2 * s.pFilm.x / film_.width - 1) * halfW_;
    float y = (1 - 2 * s.pFilm.y / film_.height) * halfH_;
    return Ray(cameraToWorld_.Point(Vec3f(x, y, 0)), Normalize(cameraToWorld_.Vector(Vec3f(0, 0, 1))));
}

} // namespace pr

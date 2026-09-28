#include "core/sampling.h"

namespace pr {

Distribution1D::Distribution1D(std::span<const float> f) : func_(f.begin(), f.end()) {
    int n = int(func_.size());
    for (float &v : func_) v = std::abs(v);
    cdf_.resize(n + 1);
    cdf_[0] = 0;
    double acc = 0;
    for (int i = 1; i < n + 1; ++i) {
        acc += double(func_[i - 1]) / n;
        cdf_[i] = float(acc);
    }
    funcInt_ = float(acc);
    if (funcInt_ == 0) {
        for (int i = 1; i < n + 1; ++i) cdf_[i] = float(i) / float(n);
    } else {
        for (int i = 1; i < n + 1; ++i) cdf_[i] = float(cdf_[i] / acc);
    }
    cdf_[n] = 1;
}

float Distribution1D::SampleContinuous(float u, float *pdf, int *off) const {
    int n = Count();
    int offset = FindInterval(int(cdf_.size()), [&](int i) { return cdf_[i] <= u; });
    if (off) *off = offset;
    float du = u - cdf_[offset];
    if (cdf_[offset + 1] - cdf_[offset] > 0) du /= (cdf_[offset + 1] - cdf_[offset]);
    if (pdf) *pdf = funcInt_ > 0 ? func_[offset] / funcInt_ : 1.f;
    return std::min((offset + du) / n, OneMinusEpsilon);
}

int Distribution1D::SampleDiscrete(float u, float *pmf, float *uRemapped) const {
    int offset = FindInterval(int(cdf_.size()), [&](int i) { return cdf_[i] <= u; });
    if (pmf) *pmf = DiscretePMF(offset);
    if (uRemapped) {
        float w = cdf_[offset + 1] - cdf_[offset];
        *uRemapped = w > 0 ? std::min((u - cdf_[offset]) / w, OneMinusEpsilon) : 0.f;
    }
    return offset;
}

Distribution2D::Distribution2D(const float *f, int nu, int nv) {
    conditional_.reserve(nv);
    for (int v = 0; v < nv; ++v) conditional_.emplace_back(std::span<const float>(f + v * nu, nu));
    std::vector<float> marg(nv);
    for (int v = 0; v < nv; ++v) marg[v] = conditional_[v].Integral();
    marginal_ = Distribution1D(marg);
}

Vec2f Distribution2D::SampleContinuous(Vec2f u, float *pdf) const {
    float pdfs[2];
    int v;
    float d1 = marginal_.SampleContinuous(u.y, &pdfs[1], &v);
    float d0 = conditional_[v].SampleContinuous(u.x, &pdfs[0]);
    *pdf = pdfs[0] * pdfs[1];
    return {d0, d1};
}

float Distribution2D::PDF(Vec2f p) const {
    int nu = conditional_[0].Count(), nv = marginal_.Count();
    int iu = Clamp(int(p.x * nu), 0, nu - 1);
    int iv = Clamp(int(p.y * nv), 0, nv - 1);
    if (marginal_.Integral() == 0) return 0;
    return conditional_[iv].Func(iu) / marginal_.Integral();
}

} // namespace pr

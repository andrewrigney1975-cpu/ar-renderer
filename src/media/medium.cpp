#include "media/medium.h"

namespace pr {

Medium::DistanceSample Medium::SampleDistance(float tMax, float u, const SampledWavelengths &lambda) const {
    SampledSpectrum sa = SigmaA(lambda), ss = SigmaS(lambda);
    SampledSpectrum st = sa + ss;
    SampledSpectrum lpdf = lambda.PDF();
    int active[NSpectrumSamples];
    int nActive = 0;
    for (int i = 0; i < NSpectrumSamples; ++i)
        if (lpdf[i] != 0) active[nActive++] = i;
    if (nActive == 0) {
        active[0] = 0;
        nActive = 1;
    }
    // Choose the channel that drives distance sampling, reusing the random number.
    int ci = std::min(int(u * nActive), nActive - 1);
    float uu = std::min(u * nActive - ci, OneMinusEpsilon);
    int c = active[ci];

    DistanceSample ds;
    float t = st[c] > 0 ? -std::log(1 - uu) / st[c] : Infinity;
    if (t < tMax) {
        // Scattering event at t. Mixture pdf over the active channels.
        float pdf = 0;
        for (int k = 0; k < nActive; ++k) {
            int i = active[k];
            pdf += st[i] * std::exp(-st[i] * t);
        }
        pdf /= nActive;
        SampledSpectrum Tr = Exp(-st * t);
        ds.scattered = true;
        ds.t = t;
        ds.weight = pdf > 0 ? Tr * ss / pdf : SampledSpectrum(0.f);
        return ds;
    }
    // Passed through to tMax: probability of no event under the mixture.
    float pSurf = 0;
    for (int k = 0; k < nActive; ++k) {
        int i = active[k];
        pSurf += std::isinf(tMax) ? (st[i] > 0 ? 0.f : 1.f) : std::exp(-st[i] * tMax);
    }
    pSurf /= nActive;
    SampledSpectrum Tr = Transmittance(tMax, lambda);
    ds.scattered = false;
    ds.t = tMax;
    ds.weight = pSurf > 0 ? Tr / pSurf : SampledSpectrum(0.f);
    return ds;
}

} // namespace pr

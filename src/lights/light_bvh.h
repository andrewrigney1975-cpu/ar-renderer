#pragma once

#include "lights/light.h"

#include <unordered_map>
#include <vector>

namespace pr {

// Light BVH sampler (Conty & Kulla 2018, pbrt-v4): picks a light with probability proportional to
// an importance estimate of its contribution at the shading point. PMF() reproduces the exact
// traversal probability, so it can be used for MIS. Infinite and distant lights are sampled
// uniformly in a separate branch.
class BVHLightSampler {
  public:
    BVHLightSampler() = default;
    explicit BVHLightSampler(const std::vector<const Light *> &lights);
    const Light *Sample(const Vec3f &p, const Vec3f &n, float u, float *pmf) const;
    float PMF(const Vec3f &p, const Vec3f &n, const Light *light) const;
    bool Empty() const { return infinite_.empty() && nodes_.empty(); }

  private:
    struct Node {
        LightBounds lb;
        int child1 = -1;  // interior: second child (the first child follows the node)
        int light = -1;   // leaf: index into bounded_
    };
    int Build(std::vector<std::pair<int, LightBounds>> &items, int start, int end, int parent);
    float PInfinite() const {
        return infinite_.empty() ? 0.f : float(infinite_.size()) / float(infinite_.size() + (nodes_.empty() ? 0 : 1));
    }
    std::vector<const Light *> bounded_, infinite_;
    std::vector<Node> nodes_;
    std::vector<int> parent_;                       // per node
    std::unordered_map<const Light *, int> leafOf_;  // light -> leaf node
};

} // namespace pr

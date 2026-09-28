#pragma once

#include "geometry/shape.h"

#include <vector>

namespace pr {

// Bounding volume hierarchy over shapes (binned SAH build, flattened depth-first layout).
class BVH {
  public:
    struct Hit {
        float t = Infinity;
        int prim = -1;
        HitInfo info;
    };

    BVH() = default;
    void Build(std::vector<const Shape *> shapes);

    // Closest hit with t in (0, tMax).
    bool Intersect(const Ray &ray, float tMax, Hit *hit) const;
    // Any hit with t in (0, tMax).
    bool IntersectP(const Ray &ray, float tMax) const;

    Bounds3f Bounds() const { return nodes_.empty() ? Bounds3f() : nodes_[0].bounds; }
    const Shape *GetShape(int i) const { return shapes_[i]; }
    // Index into the original (pre-build) shape list.
    int OriginalIndex(int i) const { return order_[i]; }

  private:
    struct Node {
        Bounds3f bounds;
        int offset = 0;        // leaf: first primitive; interior: second child
        uint16_t nPrims = 0;   // 0 => interior
        uint8_t axis = 0;
    };
    struct BuildPrim {
        Bounds3f b;
        Vec3f c;
        int index;
    };
    int BuildRecursive(std::vector<BuildPrim> &prims, int start, int end);

    std::vector<Node> nodes_;
    std::vector<const Shape *> shapes_;  // reordered
    std::vector<int> order_;
    std::vector<const Shape *> input_;
};

} // namespace pr

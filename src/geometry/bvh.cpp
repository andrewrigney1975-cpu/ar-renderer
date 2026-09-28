#include "geometry/bvh.h"

#include <algorithm>

namespace pr {

void BVH::Build(std::vector<const Shape *> shapes) {
    input_ = std::move(shapes);
    nodes_.clear();
    shapes_.clear();
    order_.clear();
    if (input_.empty()) return;
    std::vector<BuildPrim> prims(input_.size());
    for (size_t i = 0; i < input_.size(); ++i) {
        prims[i].b = input_[i]->Bounds();
        prims[i].c = prims[i].b.Centroid();
        prims[i].index = int(i);
    }
    nodes_.reserve(2 * prims.size());
    BuildRecursive(prims, 0, int(prims.size()));
    shapes_.resize(prims.size());
    order_.resize(prims.size());
    for (size_t i = 0; i < prims.size(); ++i) {
        shapes_[i] = input_[prims[i].index];
        order_[i] = prims[i].index;
    }
}

int BVH::BuildRecursive(std::vector<BuildPrim> &prims, int start, int end) {
    int nodeIndex = int(nodes_.size());
    nodes_.emplace_back();
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        bounds = Union(bounds, prims[i].b);
        centroidBounds = Union(centroidBounds, prims[i].c);
    }
    nodes_[nodeIndex].bounds = bounds;
    int n = end - start;
    auto makeLeaf = [&] {
        nodes_[nodeIndex].offset = start;
        nodes_[nodeIndex].nPrims = uint16_t(n);
        return nodeIndex;
    };
    if (n <= 2) return makeLeaf();
    int dim = centroidBounds.MaxExtent();
    float cmin = centroidBounds.pMin[dim], cmax = centroidBounds.pMax[dim];
    if (cmax == cmin) {
        if (n <= 255) return makeLeaf();
        int mid = (start + end) / 2;
        nodes_[nodeIndex].axis = uint8_t(dim);
        BuildRecursive(prims, start, mid);
        nodes_[nodeIndex].offset = BuildRecursive(prims, mid, end);
        return nodeIndex;
    }

    // Binned SAH.
    constexpr int nBuckets = 16;
    struct Bucket {
        int count = 0;
        Bounds3f b;
    } buckets[nBuckets];
    auto bucketOf = [&](const BuildPrim &p) {
        int b = int(nBuckets * ((p.c[dim] - cmin) / (cmax - cmin)));
        return std::min(b, nBuckets - 1);
    };
    for (int i = start; i < end; ++i) {
        int b = bucketOf(prims[i]);
        buckets[b].count++;
        buckets[b].b = Union(buckets[b].b, prims[i].b);
    }
    float costs[nBuckets - 1];
    {
        int countBelow = 0;
        Bounds3f below;
        for (int i = 0; i < nBuckets - 1; ++i) {
            below = Union(below, buckets[i].b);
            countBelow += buckets[i].count;
            costs[i] = countBelow * below.SurfaceArea();
        }
        int countAbove = 0;
        Bounds3f above;
        for (int i = nBuckets - 1; i >= 1; --i) {
            above = Union(above, buckets[i].b);
            countAbove += buckets[i].count;
            costs[i - 1] += countAbove * above.SurfaceArea();
        }
    }
    int minBucket = 0;
    float minCost = Infinity;
    for (int i = 0; i < nBuckets - 1; ++i)
        if (costs[i] < minCost) {
            minCost = costs[i];
            minBucket = i;
        }
    const float traversalCost = 0.5f;
    float leafCost = float(n);
    minCost = traversalCost + minCost / bounds.SurfaceArea();
    if (n <= 4 && leafCost <= minCost) return makeLeaf();

    auto midIt = std::partition(prims.begin() + start, prims.begin() + end,
                                [&](const BuildPrim &p) { return bucketOf(p) <= minBucket; });
    int mid = int(midIt - prims.begin());
    if (mid == start || mid == end) {
        mid = (start + end) / 2;
        std::nth_element(prims.begin() + start, prims.begin() + mid, prims.begin() + end,
                         [&](const BuildPrim &a, const BuildPrim &b) { return a.c[dim] < b.c[dim]; });
    }
    nodes_[nodeIndex].axis = uint8_t(dim);
    BuildRecursive(prims, start, mid);
    int second = BuildRecursive(prims, mid, end);
    nodes_[nodeIndex].offset = second;
    return nodeIndex;
}

static inline bool IntersectBox(const Bounds3f &b, const Vec3f &o, const Vec3f &invDir, const int dirIsNeg[3],
                                float tMax) {
    float tMin = (b[dirIsNeg[0]].x - o.x) * invDir.x;
    float tMaxX = (b[1 - dirIsNeg[0]].x - o.x) * invDir.x;
    float tyMin = (b[dirIsNeg[1]].y - o.y) * invDir.y;
    float tyMax = (b[1 - dirIsNeg[1]].y - o.y) * invDir.y;
    // Conservative rounding (pbrt): widen far distances slightly.
    tMaxX *= 1 + 2 * 1.2e-7f * 3;
    tyMax *= 1 + 2 * 1.2e-7f * 3;
    if (tMin > tyMax || tyMin > tMaxX) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMaxX) tMaxX = tyMax;
    float tzMin = (b[dirIsNeg[2]].z - o.z) * invDir.z;
    float tzMax = (b[1 - dirIsNeg[2]].z - o.z) * invDir.z;
    tzMax *= 1 + 2 * 1.2e-7f * 3;
    if (tMin > tzMax || tzMin > tMaxX) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMaxX) tMaxX = tzMax;
    return (tMin < tMax) && (tMaxX > 0);
}

bool BVH::Intersect(const Ray &ray, float tMax, Hit *hit) const {
    if (nodes_.empty()) return false;
    Vec3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int toVisit[64];
    int toVisitOffset = 0, current = 0;
    bool found = false;
    while (true) {
        const Node &node = nodes_[current];
        if (IntersectBox(node.bounds, ray.o, invDir, dirIsNeg, tMax)) {
            if (node.nPrims > 0) {
                for (int i = 0; i < node.nPrims; ++i) {
                    int pi = node.offset + i;
                    float t;
                    HitInfo info;
                    if (shapes_[pi]->Intersect(ray, tMax, &t, &info)) {
                        tMax = t;
                        hit->t = t;
                        hit->prim = order_[pi];
                        hit->info = info;
                        found = true;
                    }
                }
                if (toVisitOffset == 0) break;
                current = toVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node.axis]) {
                    toVisit[toVisitOffset++] = current + 1;
                    current = node.offset;
                } else {
                    toVisit[toVisitOffset++] = node.offset;
                    current = current + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            current = toVisit[--toVisitOffset];
        }
    }
    return found;
}

bool BVH::IntersectP(const Ray &ray, float tMax) const {
    if (nodes_.empty()) return false;
    Vec3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int toVisit[64];
    int toVisitOffset = 0, current = 0;
    while (true) {
        const Node &node = nodes_[current];
        if (IntersectBox(node.bounds, ray.o, invDir, dirIsNeg, tMax)) {
            if (node.nPrims > 0) {
                for (int i = 0; i < node.nPrims; ++i) {
                    float t;
                    HitInfo info;
                    if (shapes_[node.offset + i]->Intersect(ray, tMax, &t, &info)) return true;
                }
                if (toVisitOffset == 0) break;
                current = toVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node.axis]) {
                    toVisit[toVisitOffset++] = current + 1;
                    current = node.offset;
                } else {
                    toVisit[toVisitOffset++] = node.offset;
                    current = current + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            current = toVisit[--toVisitOffset];
        }
    }
    return false;
}

} // namespace pr

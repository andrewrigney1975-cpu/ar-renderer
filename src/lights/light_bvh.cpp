#include "lights/light_bvh.h"

#include <algorithm>

namespace pr {

// ---------------------------------------------------------------------------------------------
// Direction cones and light bounds

static float AngleBetween(const Vec3f &a, const Vec3f &b) {
    if (Dot(a, b) < 0) return Pi - 2 * SafeASin(Length(a + b) / 2);
    return 2 * SafeASin(Length(b - a) / 2);
}

DirectionCone Union(const DirectionCone &a, const DirectionCone &b) {
    if (a.IsEmpty()) return b;
    if (b.IsEmpty()) return a;
    float theta_a = SafeACos(a.cosTheta), theta_b = SafeACos(b.cosTheta);
    float theta_d = AngleBetween(a.w, b.w);
    if (std::min(theta_d + theta_b, Pi) <= theta_a) return a;
    if (std::min(theta_d + theta_a, Pi) <= theta_b) return b;
    float theta_o = (theta_a + theta_d + theta_b) / 2;
    if (theta_o >= Pi) return DirectionCone::EntireSphere();
    float theta_r = theta_o - theta_a;
    Vec3f wr = Cross(a.w, b.w);
    if (LengthSquared(wr) == 0) return DirectionCone::EntireSphere();
    Vec3f w = Transform::Rotate(Degrees(theta_r), wr).Vector(a.w);
    return DirectionCone(w, std::cos(theta_o));
}

LightBounds Union(const LightBounds &a, const LightBounds &b) {
    if (a.phi == 0) return b;
    if (b.phi == 0) return a;
    DirectionCone cone = Union(DirectionCone(a.w, a.cosTheta_o), DirectionCone(b.w, b.cosTheta_o));
    LightBounds r;
    r.bounds = Union(a.bounds, b.bounds);
    r.w = cone.w;
    r.phi = a.phi + b.phi;
    r.cosTheta_o = cone.cosTheta;
    r.cosTheta_e = std::min(a.cosTheta_e, b.cosTheta_e);
    r.twoSided = a.twoSided || b.twoSided;
    return r;
}

float LightBounds::Importance(const Vec3f &p, const Vec3f &n) const {
    Vec3f pc = (bounds.pMin + bounds.pMax) * 0.5f;
    float d2 = DistanceSquared(p, pc);
    d2 = std::max(d2, Length(bounds.Diagonal()) / 2);
    auto cosSubClamped = [](float sinA, float cosA, float sinB, float cosB) {
        if (cosA > cosB) return 1.f;
        return cosA * cosB + sinA * sinB;
    };
    auto sinSubClamped = [](float sinA, float cosA, float sinB, float cosB) {
        if (cosA > cosB) return 0.f;
        return sinA * cosB - cosA * sinB;
    };
    Vec3f wi = p - pc;
    float l = Length(wi);
    wi = l > 0 ? wi / l : Vec3f(0, 0, 1);
    float cosTheta_w = Dot(w, wi);
    if (twoSided) cosTheta_w = std::abs(cosTheta_w);
    float sinTheta_w = SafeSqrt(1 - Sqr(cosTheta_w));
    // Cone of directions subtended by the bounds from p.
    Vec3f c;
    float r;
    bounds.BoundingSphere(&c, &r);
    float cosTheta_b = -1;
    float dc2 = DistanceSquared(p, c);
    if (dc2 >= r * r) cosTheta_b = SafeSqrt(1 - r * r / dc2);
    float sinTheta_b = SafeSqrt(1 - Sqr(cosTheta_b));
    float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
    float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float cosThetap = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
    if (cosThetap <= cosTheta_e) return 0;
    float importance = phi * cosThetap / d2;
    if (!n.IsZero()) {
        float cosTheta_i = AbsDot(wi, n);
        float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
        importance *= cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
    }
    return std::max(importance, 0.f);
}

// ---------------------------------------------------------------------------------------------
// BVH

namespace {

float EvaluateCost(const LightBounds &b, const Bounds3f &bounds, int dim) {
    float theta_o = SafeACos(b.cosTheta_o), theta_e = SafeACos(b.cosTheta_e);
    float theta_w = std::min(theta_o + theta_e, Pi);
    float sinTheta_o = SafeSqrt(1 - Sqr(b.cosTheta_o));
    float M_omega = 2 * Pi * (1 - b.cosTheta_o) +
                    Pi / 2 * (2 * theta_w * sinTheta_o - std::cos(theta_o - 2 * theta_w) - 2 * theta_o * sinTheta_o + b.cosTheta_o);
    Vec3f d = bounds.Diagonal();
    float Kr = d[dim] > 0 ? MaxComponent(d) / d[dim] : 1.f;
    return b.phi * M_omega * Kr * b.bounds.SurfaceArea();
}

} // namespace

BVHLightSampler::BVHLightSampler(const std::vector<const Light *> &lights) {
    std::vector<std::pair<int, LightBounds>> items;
    for (const Light *l : lights) {
        auto lb = l->Bounds();
        if (!lb) {
            infinite_.push_back(l);
        } else if (lb->phi > 0) {
            items.push_back({int(bounded_.size()), *lb});
            bounded_.push_back(l);
        }
    }
    if (!items.empty()) Build(items, 0, int(items.size()), -1);
}

int BVHLightSampler::Build(std::vector<std::pair<int, LightBounds>> &items, int start, int end, int parent) {
    if (end - start == 1) {
        int nodeIndex = int(nodes_.size());
        Node node;
        node.lb = items[start].second;
        node.light = items[start].first;
        nodes_.push_back(node);
        parent_.push_back(parent);
        leafOf_[bounded_[node.light]] = nodeIndex;
        return nodeIndex;
    }
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = items[i].second;
        bounds = Union(bounds, lb.bounds);
        centroidBounds = Union(centroidBounds, lb.bounds.Centroid());
    }
    float minCost = Infinity;
    int minBucket = -1, minDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) continue;
        LightBounds bucket[nBuckets];
        for (int i = start; i < end; ++i) {
            Vec3f pc = items[i].second.bounds.Centroid();
            int b = int(nBuckets * centroidBounds.Offset(pc)[dim]);
            b = Clamp(b, 0, nBuckets - 1);
            bucket[b] = Union(bucket[b], items[i].second);
        }
        for (int i = 0; i < nBuckets - 1; ++i) {
            LightBounds b0, b1;
            for (int j = 0; j <= i; ++j) b0 = Union(b0, bucket[j]);
            for (int j = i + 1; j < nBuckets; ++j) b1 = Union(b1, bucket[j]);
            float cost = EvaluateCost(b0, bounds, dim) + EvaluateCost(b1, bounds, dim);
            if (cost > 0 && cost < minCost) {
                minCost = cost;
                minBucket = i;
                minDim = dim;
            }
        }
    }
    int mid;
    if (minDim == -1) {
        mid = (start + end) / 2;
    } else {
        auto midIt = std::partition(items.begin() + start, items.begin() + end, [&](const std::pair<int, LightBounds> &l) {
            int b = int(nBuckets * centroidBounds.Offset(l.second.bounds.Centroid())[minDim]);
            return Clamp(b, 0, nBuckets - 1) <= minBucket;
        });
        mid = int(midIt - items.begin());
        if (mid == start || mid == end) mid = (start + end) / 2;
    }
    int nodeIndex = int(nodes_.size());
    nodes_.push_back(Node());
    parent_.push_back(parent);
    Build(items, start, mid, nodeIndex);
    int child1 = Build(items, mid, end, nodeIndex);
    nodes_[nodeIndex].child1 = child1;
    nodes_[nodeIndex].lb = Union(nodes_[nodeIndex + 1].lb, nodes_[child1].lb);
    return nodeIndex;
}

const Light *BVHLightSampler::Sample(const Vec3f &p, const Vec3f &n, float u, float *pmf) const {
    float pInf = PInfinite();
    if (u < pInf) {
        u /= pInf;
        int index = std::min(int(u * infinite_.size()), int(infinite_.size()) - 1);
        *pmf = pInf / infinite_.size();
        return infinite_[index];
    }
    if (nodes_.empty()) return nullptr;
    u = std::min((u - pInf) / (1 - pInf), OneMinusEpsilon);
    int nodeIndex = 0;
    float prob = 1 - pInf;
    while (true) {
        const Node &node = nodes_[nodeIndex];
        if (node.light >= 0) {
            if (nodeIndex > 0 || node.lb.Importance(p, n) > 0) {
                *pmf = prob;
                return bounded_[node.light];
            }
            return nullptr;
        }
        const Node &c0 = nodes_[nodeIndex + 1], &c1 = nodes_[node.child1];
        float ci0 = c0.lb.Importance(p, n), ci1 = c1.lb.Importance(p, n);
        if (ci0 == 0 && ci1 == 0) return nullptr;
        float p0 = ci0 / (ci0 + ci1);
        if (u < p0) {
            prob *= p0;
            u = std::min(u / p0, OneMinusEpsilon);
            nodeIndex = nodeIndex + 1;
        } else {
            prob *= 1 - p0;
            u = std::min((u - p0) / (1 - p0), OneMinusEpsilon);
            nodeIndex = node.child1;
        }
    }
}

float BVHLightSampler::PMF(const Vec3f &p, const Vec3f &n, const Light *light) const {
    auto it = leafOf_.find(light);
    if (it == leafOf_.end()) {
        for (const Light *l : infinite_)
            if (l == light) return PInfinite() / infinite_.size();
        return 0;
    }
    int leaf = it->second;
    if (leaf == 0) return nodes_[0].lb.Importance(p, n) > 0 ? 1 - PInfinite() : 0.f;
    // Walk from the leaf up to the root, multiplying the child-selection probabilities.
    float pmf = 1 - PInfinite();
    for (int child = leaf, parent = parent_[leaf]; parent >= 0; child = parent, parent = parent_[parent]) {
        const Node &node = nodes_[parent];
        float ci0 = nodes_[parent + 1].lb.Importance(p, n), ci1 = nodes_[node.child1].lb.Importance(p, n);
        if (ci0 == 0 && ci1 == 0) return 0;
        pmf *= (child == node.child1 ? ci1 : ci0) / (ci0 + ci1);
    }
    return pmf;
}

} // namespace pr

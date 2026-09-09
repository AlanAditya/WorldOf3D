#pragma once
#include "FoveaClassify.h"
#include "FoveaRansacGPU.h"

#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

// Object-level segmentation on top of ground classification: voxel-hash +
// flood-fill clustering of non-ground points (a cheap stand-in for DBSCAN -
// no k-d tree needed, same practical effect for a single frame), same
// algorithm as the validated Python prototype. Works without real metric
// unprojection since it only needs local proximity, not a globally
// consistent physical direction (unlike gravity-constrained RANSAC) - but
// voxelSize is tuned in placeholder-space units, empirically, same as
// ransac_ground_plane's distanceThreshold.
namespace fovea {

struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h = std::hash<int32_t>()(k.x);
        h = h * 31 + std::hash<int32_t>()(k.y);
        h = h * 31 + std::hash<int32_t>()(k.z);
        return h;
    }
};

// Clusters every point NOT flagged in `groundMask` (pass ransac's inlier mask
// - skip clustering the floor itself). Returns per-point cluster id in
// `clusterIdOut` (-1 = noise / component smaller than minClusterSize), and
// the number of real clusters found (ids 0..numClusters-1).
//
// `borderMargin`: while still on the placeholder XY range (fixed -5..5),
// points sitting right at that boundary (X or Y within `borderMargin` of
// +-5) are almost always sensor edge-noise, not real geometry - a phantom
// cluster whose position jumps around wildly frame to frame is the tell.
// This is a placeholder-space-specific hack (hardcodes the +-5 assumption)
// and should be dropped once real unprojection is back; pass 0 to disable.
inline int voxel_cluster(
    const std::vector<Vec3>& points,
    const std::vector<uint8_t>& groundMask,
    std::vector<int>& clusterIdOut,
    float voxelSize = 0.4f,
    int minClusterSize = 4,
    float borderMargin = 0.3f
) {
    size_t n = points.size();
    clusterIdOut.assign(n, -1);

    const float placeholderExtent = 5.0f;
    std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> voxelMap;
    for (size_t i = 0; i < n; ++i) {
        if (!groundMask.empty() && groundMask[i]) continue;
        if (borderMargin > 0.0f &&
            (std::fabs(points[i].x) > placeholderExtent - borderMargin ||
             std::fabs(points[i].y) > placeholderExtent - borderMargin)) {
            continue; // sensor edge - skip entirely, not even counted as noise
        }
        VoxelKey key{
            (int32_t)std::floor(points[i].x / voxelSize),
            (int32_t)std::floor(points[i].y / voxelSize),
            (int32_t)std::floor(points[i].z / voxelSize)
        };
        voxelMap[key].push_back(i);
    }

    std::unordered_map<VoxelKey, bool, VoxelKeyHash> visited;
    int nextClusterId = 0;

    for (auto& entry : voxelMap) {
        const VoxelKey& startKey = entry.first;
        if (visited[startKey]) continue;
        visited[startKey] = true;

        std::vector<VoxelKey> stack{startKey};
        std::vector<size_t> componentPoints;

        while (!stack.empty()) {
            VoxelKey cur = stack.back();
            stack.pop_back();
            auto it = voxelMap.find(cur);
            if (it == voxelMap.end()) continue;
            for (size_t idx : it->second) componentPoints.push_back(idx);

            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        VoxelKey nb{cur.x + dx, cur.y + dy, cur.z + dz};
                        if (voxelMap.count(nb) && !visited[nb]) {
                            visited[nb] = true;
                            stack.push_back(nb);
                        }
                    }
        }

        if ((int)componentPoints.size() >= minClusterSize) {
            for (size_t idx : componentPoints) clusterIdOut[idx] = nextClusterId;
            ++nextClusterId;
        }
    }
    return nextClusterId;
}

inline Vec3 cluster_centroid(const std::vector<Vec3>& points, const std::vector<int>& clusterId, int cid) {
    Vec3 sum{0.0f, 0.0f, 0.0f};
    int count = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (clusterId[i] == cid) {
            sum.x += points[i].x; sum.y += points[i].y; sum.z += points[i].z;
            ++count;
        }
    }
    if (count == 0) return sum;
    return {sum.x / count, sum.y / count, sum.z / count};
}

// Frame-to-frame nearest-centroid tracker, tagging clusters dynamic/static by
// displacement since last call - same MVP approach as the Python prototype.
// Caveat that's NOT specific to the placeholder space: this compares
// positions in camera-relative coordinates, so if the phone itself moves
// between frames, a genuinely static object can appear to have moved too -
// real unprojection wouldn't fix this either, it needs actual ego-motion
// compensation (e.g. ARKit's VIO), a separate, bigger piece of work. Keep the
// phone still for a clean dynamic-tag demo for now.
class DynamicTracker {
public:
    explicit DynamicTracker(float matchGate = 3.0f, float dynamicSpeedThreshold = 0.15f)
        : matchGate_(matchGate), dynamicSpeedThreshold_(dynamicSpeedThreshold) {}

    std::vector<uint8_t> update(const std::vector<Vec3>& centroids) {
        std::vector<uint8_t> isDynamic(centroids.size(), 0);
        if (!prevCentroids_.empty()) {
            for (size_t i = 0; i < centroids.size(); ++i) {
                float bestDist = 1e9f;
                for (const auto& prev : prevCentroids_) {
                    float d = length(sub(centroids[i], prev));
                    if (d < bestDist) bestDist = d;
                }
                if (bestDist < matchGate_ && bestDist > dynamicSpeedThreshold_) {
                    isDynamic[i] = 1;
                }
            }
        }
        prevCentroids_ = centroids;
        return isDynamic;
    }

private:
    float matchGate_;
    float dynamicSpeedThreshold_;
    std::vector<Vec3> prevCentroids_;
};

// Colors by QUANTIZED POSITION, not by cluster id - voxel_cluster's ids come
// from unordered_map iteration order, which isn't stable frame to frame, so
// the same real object can get a different id (and, if colored by id, a
// different flickering color) every frame even though it hasn't moved.
// Position is what's actually stable for a real object, so hash that instead.
// quantStep gives some hysteresis against small frame-to-frame jitter -
// still not perfect right at a bucket boundary, but far better than
// per-frame-random.
inline void color_from_position(const Vec3& pos, float quantStep, float& r, float& g, float& b) {
    int32_t qx = (int32_t)std::lround(pos.x / quantStep);
    int32_t qy = (int32_t)std::lround(pos.y / quantStep);
    int32_t qz = (int32_t)std::lround(pos.z / quantStep);
    uint32_t h = (uint32_t)qx * 2654435761u;
    h ^= (uint32_t)qy * 2246822519u;
    h ^= (uint32_t)qz * 3266489917u;
    r = 0.3f + 0.7f * ((h & 0xFF) / 255.0f);
    g = 0.3f + 0.7f * (((h >> 8) & 0xFF) / 255.0f);
    b = 0.3f + 0.7f * (((h >> 16) & 0xFF) / 255.0f);
}

// Full pipeline: RANSAC ground removal (GPU) -> cluster the rest into
// distinct objects -> optionally tag dynamic vs static (pass a persistent
// `tracker`, reused across frames - nullptr skips this and just segments).
// Ground = tan, each static object gets its own distinct color so
// segmentation is visually obvious, dynamic objects = red regardless of
// which cluster they were.
// `gravity` optional, last param, defaults to no constraint - see
// classify_ground_colors's comment in FoveaClassify.h for details.
inline matrix classify_segmented_colors(
    matrix& points,
    DynamicTracker* tracker = nullptr,
    int ransacIterations = 300,
    float ransacDistanceThreshold = 0.06f,
    float voxelSize = 0.4f,
    int minClusterSize = 4,
    float borderMargin = 0.3f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    std::vector<Vec3> pts = extract_points(points);

    Vec3 gravityVec{gravity.x, gravity.y, gravity.z};
    bool hasGravity = length(gravityVec) > 1e-6f;
    std::vector<uint8_t> groundMask;
    ransac_ground_plane_gpu(pts, groundMask, ransacIterations, ransacDistanceThreshold,
                             hasGravity ? &gravityVec : nullptr);

    std::vector<int> clusterId;
    int numClusters = voxel_cluster(pts, groundMask, clusterId, voxelSize, minClusterSize, borderMargin);

    std::vector<Vec3> centroids(numClusters);
    for (int c = 0; c < numClusters; ++c) centroids[c] = cluster_centroid(pts, clusterId, c);

    std::vector<uint8_t> isDynamic(numClusters, 0);
    if (tracker && numClusters > 0) {
        isDynamic = tracker->update(centroids);
    }

    // precompute each cluster's stable color once, not once per point
    std::vector<std::array<float, 3>> clusterColor(numClusters);
    for (int c = 0; c < numClusters; ++c) {
        color_from_position(centroids[c], voxelSize, clusterColor[c][0], clusterColor[c][1], clusterColor[c][2]);
    }

    matrix colors = matrix::withShape({(size_m)pts.size(), 4}, dtype::Float);
    colors.begin_refcount();

    for (size_t i = 0; i < pts.size(); ++i) {
        float r, g, b;
        if (groundMask[i]) {
            r = 0.76f; g = 0.66f; b = 0.47f; // terrain
        } else if (clusterId[i] < 0) {
            r = 0.5f; g = 0.5f; b = 0.5f; // noise / unclustered
        } else if (tracker && isDynamic[clusterId[i]]) {
            r = 0.89f; g = 0.20f; b = 0.15f; // dynamic
        } else {
            const auto& c = clusterColor[clusterId[i]]; // stable color per static object
            r = c[0]; g = c[1]; b = c[2];
        }
        colors.at<float>((size_m)i, 0) = r;
        colors.at<float>((size_m)i, 1) = g;
        colors.at<float>((size_m)i, 2) = b;
        colors.at<float>((size_m)i, 3) = 1.0f;
    }
    return colors;
}

} // namespace fovea

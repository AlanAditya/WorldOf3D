#pragma once
#include "../matrix.h"
#include "FoveaGeometry.h"

#include <cmath>
#include <random>
#include <simd/simd.h>
#include <vector>

// Step 1 of the FoveaMap port: RANSAC ground-plane classification, recoloring
// an existing point cloud (terrain vs everything else) so it can go straight
// into the current PointCloudController - no renderer changes needed. This
// stays plain C++ deliberately: matrix has no comparison ops, argmin/argmax,
// or sorting, and RANSAC's "sample, count inliers, keep the best" loop is
// exactly the kind of branchy per-point logic that isn't worth fighting the
// lazy compute graph for.
namespace fovea {

struct Vec3 {
    float x, y, z;
};

inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }

// Reads a [N,3] float points matrix into a plain buffer. Calls eval() first -
// only safe to cast .buffer directly once the matrix has actually been
// evaluated (matrix is type-erased; this is the fast bulk-read path vs N
// calls to .at<float>()).
inline std::vector<Vec3> extract_points(matrix& points) {
    points.eval();
    size_t n = points.shape()[0];
    const float* raw = static_cast<const float*>(points.buffer);
    std::vector<Vec3> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = {raw[i * 3 + 0], raw[i * 3 + 1], raw[i * 3 + 2]};
    }
    return out;
}

struct PlaneModel {
    Vec3 normal{0.0f, 0.0f, 1.0f};
    float d = 0.0f;
};

// Fits a ground plane via RANSAC: repeatedly sample 3 points, build a plane
// from their cross product (no matrix solve/inverse needed), count inliers,
// keep the best model. distanceThreshold is in whatever unit `points` is
// actually in - see the note in the header of this project about the current
// depth->XYZ conversion possibly not being metric yet; tune this threshold
// empirically against real data for now.
//
// `priorNormal`, if given, is the phone's known "down" direction (e.g. from
// CoreMotion's device-motion gravity vector) expressed in the *same*
// coordinate frame as `points`. Any sampled candidate plane whose normal
// deviates from it by more than `maxAngleDeviationDegrees` is rejected before
// counting inliers - without this, RANSAC has no way to know a wall isn't the
// floor, and wastes samples exploring orientations that are never the floor
// anyway. Pass nullptr to fall back to plain unconstrained RANSAC.
inline PlaneModel ransac_ground_plane(
    const std::vector<Vec3>& points,
    std::vector<uint8_t>& inlierMaskOut,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    const Vec3* priorNormal = nullptr,
    float maxAngleDeviationDegrees = 20.0f
) {
    size_t n = points.size();
    inlierMaskOut.assign(n, 0);
    if (n < 3) return {};

    Vec3 priorNormalUnit{0.0f, 0.0f, 1.0f};
    float cosAngleThreshold = -1.0f; // "no constraint" sentinel
    if (priorNormal) {
        float priorLen = length(*priorNormal);
        if (priorLen > 1e-6f) {
            priorNormalUnit = {priorNormal->x / priorLen, priorNormal->y / priorLen, priorNormal->z / priorLen};
            cosAngleThreshold = std::cos(maxAngleDeviationDegrees * (float)M_PI / 180.0f);
        }
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    PlaneModel best;
    int bestInlierCount = -1;

    for (int iter = 0; iter < iterations; ++iter) {
        size_t i0 = pick(rng), i1 = pick(rng), i2 = pick(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Vec3 normal = cross(sub(points[i1], points[i0]), sub(points[i2], points[i0]));
        float len = length(normal);
        if (len < 1e-6f) continue;
        normal = {normal.x / len, normal.y / len, normal.z / len};

        if (cosAngleThreshold > -1.0f) {
            // abs(): the cross product's sign is arbitrary depending on sample
            // order, so "close to the down direction" also means "close to up".
            float cosAngle = std::fabs(dot(normal, priorNormalUnit));
            if (cosAngle < cosAngleThreshold) continue;
        }

        float d = -dot(normal, points[i0]);

        int inlierCount = 0;
        for (const auto& p : points) {
            if (std::fabs(dot(normal, p) + d) < distanceThreshold) ++inlierCount;
        }
        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            best = {normal, d};
        }
    }

    for (size_t i = 0; i < n; ++i) {
        inlierMaskOut[i] = std::fabs(dot(best.normal, points[i]) + best.d) < distanceThreshold ? 1 : 0;
    }
    return best;
}

// Builds an [N,4] float colors matrix: ground (terrain) points one color,
// everything else another - drop straight into an existing
// PointCloudController::update(points, colors) call, no renderer code needed.
inline matrix colors_from_ground_mask(const std::vector<uint8_t>& inlierMask) {
    constexpr float terrainColor[4] = {0.76f, 0.66f, 0.47f, 1.0f}; // tan
    constexpr float otherColor[4] = {0.31f, 0.51f, 0.74f, 1.0f};   // blue

    matrix colors = matrix::withShape({(size_m)inlierMask.size(), 4}, dtype::Float);
    colors.begin_refcount(); // safe to copy/alias once this leaves this function

    for (size_t i = 0; i < inlierMask.size(); ++i) {
        const float* c = inlierMask[i] ? terrainColor : otherColor;
        for (int k = 0; k < 4; ++k) {
            colors.at<float>((size_m)i, (size_m)k) = c[k];
        }
    }
    return colors;
}

// Convenience: points matrix in, colors matrix out. Everything else in this
// file is exposed too in case the next step (clustering) needs the raw
// points/inlier mask directly instead of going through a matrix round-trip.
//
// `gravity`: phone-orientation prior, same coordinate frame as `points` (e.g.
// straight from CoreMotion's CMDeviceMotion.gravity). Sign doesn't matter -
// see ransac_ground_plane's fabs(dot(...)) check above, "up" and "down" score
// identically, so whichever axis convention your device code uses is fine as
// long as it's the true vertical. Optional, last param, defaults to
// simd_float3{0,0,0} (skips the constraint entirely, same as unconstrained
// RANSAC) - not wired to anything real yet, so leave it out for now.
inline matrix classify_ground_colors(
    matrix& points,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    float maxAngleDeviationDegrees = 20.0f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    Vec3 gravityVec{gravity.x, gravity.y, gravity.z};
    bool hasGravity = length(gravityVec) > 1e-6f;

    std::vector<Vec3> pts = extract_points(points);
    std::vector<uint8_t> inlierMask;
    ransac_ground_plane(
        pts, inlierMask, iterations, distanceThreshold,
        hasGravity ? &gravityVec : nullptr, maxAngleDeviationDegrees
    );
    return colors_from_ground_mask(inlierMask);
}

// Classifies using PROPERLY unprojected points (real intrinsics -
// gravity-constrained RANSAC only makes physical sense here, not on the
// placeholder XY-range point cloud, since that one doesn't preserve real
// angles), but returns just the colors matrix - pair it with whatever points
// you're actually displaying (e.g. the placeholder-based default point
// cloud), since both come from the same depth frame in the same per-pixel
// order and line up index-for-index regardless of which XYZ formula built
// the *displayed* positions.
inline matrix classify_ground_colors_from_depth(
    matrix depth,
    const CameraIntrinsics& camera,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    float maxAngleDeviationDegrees = 20.0f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    matrix unprojectedPoints = points_from_depth_intrinsics(depth, camera);
    return classify_ground_colors(unprojectedPoints, iterations, distanceThreshold, maxAngleDeviationDegrees, gravity);
}

} // namespace fovea

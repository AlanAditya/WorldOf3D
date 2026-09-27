#pragma once
#include "FoveaRansacGPU.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Multi-plane RANSAC segmentation over a [N,3] float points matrix. It returns
// DENSE per-point results aligned with `points` (labels / mask of length N),
// never a copy of the points and never compact index lists. That way the
// result lines up index-for-index with colours, normals and so on, and
// recolouring is a plain lazy broadcast op (see colour_planes).
//
// How it runs, per plane:
//   1. CPU: collect the still-unassigned points and pick a random scoring
//      subset of at most `score_samples` of them.
//   2. CPU: build `iterations` candidate planes from that subset: random
//      triples (cross product, same as ransac_ground_plane), or with normals,
//      one point + its normal.
//   3. GPU: count each candidate's inliers on the SUBSET only
//      (fovea_planes_count_inliers). This is the expensive O(samples x iterations)
//      step, and subsampling is what keeps it cheap at millions of points.
//   4. CPU: argmax over the counts, then a least-squares (PCA) refit of the
//      winner on its inliers in the subset.
//   5. GPU: one pass over all N points, claiming the unassigned inliers of the
//      winner (fovea_planes_assign).
// Stops at max_planes, when fewer than min_inliers points are left, or when
// the best plane claims fewer than min_inliers points (that round is undone).
//
// Planes are (nx, ny, nz, d) with a unit normal: a point p is on the plane when
// |n.p + d| < distance_threshold, in the same units as `points`.
namespace fovea {

struct PlaneSegmentation {
    matrix planes;                        // [K,4] float (nx, ny, nz, d)
    matrix labels;                        // [N] int32: 0..K-1, or -1 = on no plane
    std::vector<uint32_t> inlier_counts;  // [K] points claimed by each plane

    size_t plane_count() const { return inlier_counts.size(); }
};

struct PlaneFit {
    matrix plane;                // [4] float (nx, ny, nz, d); zeros if nothing was found
    matrix mask;                 // [N] float: 1 on the plane, 0 elsewhere
    uint32_t inlier_count = 0;
};

namespace detail {

// The kernels read points / normals as tightly packed float3s.
inline matrix packed_float3_rows(matrix m, const char* what) {
    if (m.dims != 2 || m.shape()[1] != 3) {
        throw std::invalid_argument(std::string(what) + " must be [N,3]");
    }
    if (m.type != dtype::Float || (m.flags & NON_CONTIGUOUS_FLAG)) {
        m = m.astype(dtype::Float, true);
    }
    m.eval();
    return m;
}

// Least-squares plane through points[idx]: centroid + smallest-eigenvalue
// eigenvector of the covariance (closed-form 3x3, same as the Metal
// fovea_smallest_eigenvector). Returns false if there are too few points or the
// points don't define a plane.
inline bool fit_plane_pca(const float* pts, const std::vector<uint32_t>& idx, simd_float4& plane) {
    if (idx.size() < 3) return false;
    double cx = 0, cy = 0, cz = 0;
    for (uint32_t i : idx) { cx += pts[i * 3]; cy += pts[i * 3 + 1]; cz += pts[i * 3 + 2]; }
    cx /= idx.size(); cy /= idx.size(); cz /= idx.size();
    double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
    for (uint32_t i : idx) {
        double x = pts[i * 3] - cx, y = pts[i * 3 + 1] - cy, z = pts[i * 3 + 2] - cz;
        a00 += x * x; a01 += x * y; a02 += x * z; a11 += y * y; a12 += y * z; a22 += z * z;
    }
    double q = (a00 + a11 + a22) / 3.0;
    double p1 = a01 * a01 + a02 * a02 + a12 * a12;
    double p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2.0 * p1;
    double p = std::sqrt(p2 / 6.0);
    if (p <= 0.0) return false;
    double b00 = (a00 - q) / p, b01 = a01 / p, b02 = a02 / p, b11 = (a11 - q) / p, b12 = a12 / p, b22 = (a22 - q) / p;
    double r = 0.5 * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02));
    double phi = std::acos(std::clamp(r, -1.0, 1.0)) / 3.0;
    double smallest = q + 2.0 * p * std::cos(phi + 2.0943951023931953);

    simd_double3 r0 = simd_make_double3(a00 - smallest, a01, a02);
    simd_double3 r1 = simd_make_double3(a01, a11 - smallest, a12);
    simd_double3 r2 = simd_make_double3(a02, a12, a22 - smallest);
    simd_double3 c01 = simd_cross(r0, r1), c02 = simd_cross(r0, r2), c12 = simd_cross(r1, r2);
    double d01 = simd_dot(c01, c01), d02 = simd_dot(c02, c02), d12 = simd_dot(c12, c12);
    simd_double3 n = d01 >= d02 && d01 >= d12 ? c01 : (d02 >= d12 ? c02 : c12);
    double len = simd_length(n);
    if (len <= 0.0) return false;
    n /= len;
    plane = simd_make_float4((float)n.x, (float)n.y, (float)n.z, (float)-(n.x * cx + n.y * cy + n.z * cz));
    return true;
}

// Shared by segment_planes and segment_planar_surfaces. `normals` null or
// cos_max_angle < 0 = distance test only.
inline PlaneSegmentation segment_planes_impl(
    matrix points,
    matrix* normals,
    float cos_max_angle,
    int max_planes,
    float distance_threshold,
    uint32_t min_inliers,
    int iterations,
    uint32_t score_samples
) {
    points = packed_float3_rows(points, "points");
    const size_t n = points.shape()[0];
    const float* nrm = nullptr;
    id<MTLBuffer> normalsBuf = nil;
    if (normals && cos_max_angle >= 0.0f) {
        *normals = packed_float3_rows(*normals, "normals");
        if ((size_t)normals->shape()[0] != n) {
            throw std::invalid_argument("segment_planar_surfaces: normals must have one row per point");
        }
        nrm = static_cast<const float*>(normals->buffer);
        normalsBuf = normals->metalBuffer;
        if (!normalsBuf) {
            normalsBuf = [GlobalGPUManager.metalDevice newBufferWithBytes:nrm length:n * 3 * sizeof(float) options:MTLResourceStorageModeShared];
        }
    } else {
        cos_max_angle = -1.0f;
    }
    id<MTLDevice> device = GlobalGPUManager.metalDevice;

    PlaneSegmentation seg;
    seg.labels = matrix::withShape({(size_m)n}, dtype::Int32);
    seg.labels.begin_refcount();
    if (!seg.labels.metalBuffer) seg.labels.buildMetalBuffer();
    int32_t* labels = static_cast<int32_t*>(seg.labels.buffer);
    std::fill(labels, labels + n, -1);

    std::vector<simd_float4> found;

    id<MTLComputePipelineState> countPipeline = _fovea_gpu_pipeline(@"fovea_planes_count_inliers");
    id<MTLComputePipelineState> assignPipeline = _fovea_gpu_pipeline(@"fovea_planes_assign");
    const bool ready = countPipeline && assignPipeline && seg.labels.metalBuffer && n >= 3 && max_planes > 0 && iterations > 0;

    if (ready) {
        const float* pts = static_cast<const float*>(points.buffer);
        auto point_at = [pts](uint32_t i) { return Vec3{pts[i * 3 + 0], pts[i * 3 + 1], pts[i * 3 + 2]}; };

        id<MTLBuffer> pointsBuf = points.metalBuffer;
        if (!pointsBuf) {
            pointsBuf = [device newBufferWithBytes:pts length:n * 3 * sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (!normalsBuf) normalsBuf = pointsBuf; // dummy binding, never read when cos_max_angle < 0
        const uint32_t sampleCap = (uint32_t)std::min<size_t>(std::max<uint32_t>(score_samples, 3), n);
        id<MTLBuffer> samplesBuf = [device newBufferWithLength:sampleCap * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> candidatesBuf = [device newBufferWithLength:iterations * sizeof(simd_float4) options:MTLResourceStorageModeShared];
        id<MTLBuffer> countsBuf = [device newBufferWithLength:iterations * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> claimedBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        uint32_t* samples = static_cast<uint32_t*>(samplesBuf.contents);
        simd_float4* candidates = static_cast<simd_float4*>(candidatesBuf.contents);
        const uint32_t* counts = static_cast<const uint32_t*>(countsBuf.contents);
        uint32_t* claimed = static_cast<uint32_t*>(claimedBuf.contents);
        const uint32_t pointCountU = (uint32_t)n;

        std::mt19937 rng(std::random_device{}());
        std::vector<uint32_t> unassigned;
        unassigned.reserve(n);

        for (int32_t k = 0; k < max_planes; ++k) {
            unassigned.clear();
            for (uint32_t i = 0; i < n; ++i) {
                if (labels[i] < 0) unassigned.push_back(i);
            }
            if (unassigned.size() < std::max<uint32_t>(min_inliers, 3)) break;

            // --- scoring subset: partial Fisher-Yates, first `s` entries are the sample ---
            const uint32_t s = (uint32_t)std::min<size_t>(sampleCap, unassigned.size());
            for (uint32_t j = 0; j < s; ++j) {
                std::uniform_int_distribution<size_t> pick(j, unassigned.size() - 1);
                std::swap(unassigned[j], unassigned[pick(rng)]);
            }
            memcpy(samples, unassigned.data(), s * sizeof(uint32_t));

            // --- candidate planes from the subset ---
            // Without normals: random triples, which need all 3 points on the
            // same surface, so a surface holding a fraction f of the points is
            // hit with probability f^3 per try (a 5% table: ~1 in 8000).
            // With normals: one point + its normal is already a plane, so the
            // odds are f per try. The candidate's tilt comes from one voxel
            // normal, which is fine because the winner is refit below.
            std::uniform_int_distribution<uint32_t> pick(0, s - 1);
            uint32_t candidateCount = 0;
            for (int iter = 0; iter < iterations; ++iter) {
                uint32_t i0 = pick(rng);
                Vec3 p0 = point_at(samples[i0]);
                Vec3 normal;
                if (nrm) {
                    uint32_t p = samples[i0];
                    normal = {nrm[p * 3 + 0], nrm[p * 3 + 1], nrm[p * 3 + 2]};
                    if (length(normal) < 0.5f) continue; // no normal for this point
                } else {
                    uint32_t i1 = pick(rng), i2 = pick(rng);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    normal = cross(sub(point_at(samples[i1]), p0), sub(point_at(samples[i2]), p0));
                    float len = length(normal);
                    if (len < 1e-6f) continue;
                    normal = {normal.x / len, normal.y / len, normal.z / len};
                }
                candidates[candidateCount++] = simd_make_float4(normal.x, normal.y, normal.z, -dot(normal, p0));
            }
            if (candidateCount == 0) break;
            memset(countsBuf.contents, 0, candidateCount * sizeof(uint32_t));

            // --- GPU pass 1: inliers per candidate, on the subset only ---
            // Threadgroup height must stay 1: simd_sum in the kernel assumes a
            // SIMD group never spans two candidates.
            id<MTLCommandBuffer> cmd = [GlobalGPUManager.gCommandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:countPipeline];
            [enc setBuffer:pointsBuf offset:0 atIndex:0];
            [enc setBuffer:samplesBuf offset:0 atIndex:1];
            [enc setBuffer:candidatesBuf offset:0 atIndex:2];
            [enc setBuffer:countsBuf offset:0 atIndex:3];
            [enc setBytes:&s length:sizeof(s) atIndex:4];
            [enc setBytes:&distance_threshold length:sizeof(distance_threshold) atIndex:5];
            [enc setBuffer:normalsBuf offset:0 atIndex:6];
            [enc setBytes:&cos_max_angle length:sizeof(cos_max_angle) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(s, candidateCount, 1)
              threadsPerThreadgroup:MTLSizeMake(countPipeline.threadExecutionWidth, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint32_t bestIdx = 0;
            for (uint32_t c = 1; c < candidateCount; ++c) {
                if (counts[c] > counts[bestIdx]) bestIdx = c;
            }
            simd_float4 best = candidates[bestIdx];

            // --- least-squares refit (PCL's setOptimizeCoefficients): PCA over the
            // winner's inliers in the subset, twice, since the refit plane's
            // inlier set shifts a little. Keeps a raw candidate's tilt/offset from
            // leaving a parallel sliver of the same surface for a later round. ---
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<uint32_t> inliers;
                inliers.reserve(counts[bestIdx]);
                for (uint32_t j = 0; j < s; ++j) {
                    uint32_t i = samples[j];
                    Vec3 p = point_at(i);
                    if (std::fabs(best.x * p.x + best.y * p.y + best.z * p.z + best.w) >= distance_threshold) continue;
                    if (nrm && std::fabs(best.x * nrm[i * 3] + best.y * nrm[i * 3 + 1] + best.z * nrm[i * 3 + 2]) < cos_max_angle) continue;
                    inliers.push_back(i);
                }
                simd_float4 refit;
                if (!fit_plane_pca(pts, inliers, refit)) break;
                best = refit;
            }

            // --- GPU pass 2: claim the winner's unassigned inliers among all N ---
            *claimed = 0;
            id<MTLCommandBuffer> cmd2 = [GlobalGPUManager.gCommandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
            [enc2 setComputePipelineState:assignPipeline];
            [enc2 setBuffer:pointsBuf offset:0 atIndex:0];
            [enc2 setBytes:&best length:sizeof(best) atIndex:1];
            [enc2 setBuffer:seg.labels.metalBuffer offset:0 atIndex:2];
            [enc2 setBuffer:claimedBuf offset:0 atIndex:3];
            [enc2 setBytes:&pointCountU length:sizeof(pointCountU) atIndex:4];
            [enc2 setBytes:&distance_threshold length:sizeof(distance_threshold) atIndex:5];
            [enc2 setBytes:&k length:sizeof(k) atIndex:6];
            [enc2 setBuffer:normalsBuf offset:0 atIndex:7];
            [enc2 setBytes:&cos_max_angle length:sizeof(cos_max_angle) atIndex:8];
            [enc2 dispatchThreads:MTLSizeMake(n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(assignPipeline.threadExecutionWidth, 1, 1)];
            [enc2 endEncoding];
            [cmd2 commit];
            [cmd2 waitUntilCompleted];

            if (*claimed < min_inliers) {
                // Too small to count as a plane: undo this round's claims and stop.
                for (size_t i = 0; i < n; ++i) {
                    if (labels[i] == k) labels[i] = -1;
                }
                break;
            }
            found.push_back(best);
            seg.inlier_counts.push_back(*claimed);
        }
    }

    seg.planes = matrix::withShape({(size_m)found.size(), 4}, dtype::Float);
    seg.planes.begin_refcount();
    if (!found.empty()) {
        memcpy(seg.planes.buffer, found.data(), found.size() * sizeof(simd_float4));
    }
    return seg;
}

} // namespace detail

// Plain RANSAC: inliers are every point within distance_threshold of the
// (infinite) plane. Mathematically exact, but a plane also claims every other
// surface it slices through, e.g. a band of wall at table-top height. Same
// behaviour as Open3D's segment_plane.
inline PlaneSegmentation segment_planes(
    matrix points,
    int max_planes = 8,
    float distance_threshold = 0.02f,
    uint32_t min_inliers = 1000,
    int iterations = 1000,
    uint32_t score_samples = 50000
) {
    return detail::segment_planes_impl(points, nullptr, -1.0f, max_planes, distance_threshold,
                                       min_inliers, iterations, score_samples);
}

// Surface-aware RANSAC (the Schnabel et al. / CGAL Efficient RANSAC normal
// test): a point is an inlier only if it is within distance_threshold of the
// plane AND its normal is within max_normal_angle_deg of the plane's normal.
// The band of wall a table plane slices through has horizontal normals, so it
// no longer counts, neither when scoring candidates nor when claiming points.
// `normals` is [N,3] (e.g. from estimate_normals); zero normals never match.
//
// Still not handled: disconnected surfaces that are truly coplanar (two
// tables at the same height), which need a connectivity split.
inline PlaneSegmentation segment_planar_surfaces(
    matrix points,
    matrix normals,
    int max_planes = 8,
    float distance_threshold = 0.02f,
    float max_normal_angle_deg = 20.0f,
    uint32_t min_inliers = 1000,
    int iterations = 1000,
    uint32_t score_samples = 50000
) {
    float cos_max_angle = std::cos(max_normal_angle_deg * (float)M_PI / 180.0f);
    return detail::segment_planes_impl(points, &normals, cos_max_angle, max_planes, distance_threshold,
                                       min_inliers, iterations, score_samples);
}

// Single dominant plane (the Open3D segment_plane equivalent), as a float mask
// ready to use in arithmetic, e.g. blending colours:
//   matrix m = fit.mask.reshape(N, 1);
//   matrix out = m * red + (1.0f - m) * colours;
inline PlaneFit segment_plane(
    matrix points,
    float distance_threshold = 0.02f,
    int iterations = 1000,
    uint32_t score_samples = 50000
) {
    PlaneSegmentation seg = segment_planes(points, 1, distance_threshold, 3, iterations, score_samples);

    PlaneFit fit;
    fit.plane = matrix::withShape({4}, dtype::Float);
    fit.plane.begin_refcount();
    float* plane = static_cast<float*>(fit.plane.buffer);
    std::fill(plane, plane + 4, 0.0f);
    if (seg.plane_count() > 0) {
        memcpy(plane, seg.planes.buffer, 4 * sizeof(float));
        fit.inlier_count = seg.inlier_counts[0];
    }
    fit.mask = seg.labels.astype(dtype::Float) + 1.0f; // label -1 -> 0, label 0 -> 1
    return fit;
}

// Lazy, slider-friendly version of segment_planar_surfaces: returns just the [N] int32 labels
// (0..K-1, or -1 = on no plane) as a graph node that re-runs when `points`, `normals` or any
// tracked parameter changes. Every parameter can be a constant or a 1-element matrix (e.g. a
// SliderGroup leaf). max_normal_angle_deg = 90 turns the normal test off (plain RANSAC).
// Colour the result with colour_plane_labels (the plane count isn't known until it runs).
inline matrix segment_planar_surfaces_labels(
    matrix points,
    matrix normals,
    const ScalarArg& max_planes = 8,
    const ScalarArg& distance_threshold = 0.02f,
    const ScalarArg& max_normal_angle_deg = 20.0f,
    const ScalarArg& min_inliers = 1000,
    const ScalarArg& iterations = 1000,
    uint32_t score_samples = 50000
) {
    if (points.dims != 2 || points.shape()[1] != 3) {
        throw std::invalid_argument("segment_planar_surfaces_labels: points must be [N,3]");
    }
    std::vector<matrix> inputs = {points, normals};
    HostArg planes = HostArg::track(max_planes, inputs);
    HostArg dist = HostArg::track(distance_threshold, inputs);
    HostArg angle = HostArg::track(max_normal_angle_deg, inputs);
    HostArg minIn = HostArg::track(min_inliers, inputs);
    HostArg iters = HostArg::track(iterations, inputs);
    const size_m n = points.shape()[0];
    return matrix::host_op({n}, dtype::Int32, [=](matrix& out, std::vector<matrix>& ins) {
        float cos_max_angle = std::cos(std::clamp(angle.value(ins), 0.0f, 90.0f) * (float)M_PI / 180.0f);
        matrix nrm = ins[1];
        PlaneSegmentation seg = detail::segment_planes_impl(
            ins[0], &nrm, cos_max_angle,
            (int)std::lround(planes.value(ins)), std::max(dist.value(ins), 1e-5f),
            (uint32_t)std::max<long>(std::lround(minIn.value(ins)), 3), (int)std::max<long>(std::lround(iters.value(ins)), 1),
            score_samples);
        memcpy(out.buffer, seg.labels.buffer, out.total_size * sizeof(int32_t));
    }, inputs);
}

namespace detail {
// [k+1, 4] RGBA: row 0 for "no plane" (discarded by the blend), then one distinct colour per plane.
inline matrix plane_palette(size_t k) {
    matrix palette = matrix::withShape({(size_m)k + 1, 4}, dtype::Float);
    palette.begin_refcount();
    float* rgba = static_cast<float*>(palette.buffer);
    std::fill(rgba, rgba + 4, 0.0f);
    for (size_t i = 0; i < k; ++i) {
        // Golden-ratio hue steps keep neighbouring plane indices far apart in colour.
        float h = std::fmod(0.1f + i * 0.618034f, 1.0f) * 6.0f;
        float s = 0.75f, v = 0.95f;
        float c = v * s, x = c * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f)), m = v - c;
        float r = 0, g = 0, b = 0;
        switch ((int)h) {
            case 0: r = c; g = x; break;
            case 1: r = x; g = c; break;
            case 2: g = c; b = x; break;
            case 3: g = x; b = c; break;
            case 4: r = x; b = c; break;
            default: r = c; b = x; break;
        }
        float* row = rgba + (i + 1) * 4;
        row[0] = r + m; row[1] = g + m; row[2] = b + m; row[3] = 1.0f;
    }
    return palette;
}
} // namespace detail

// Colours every plane with its own distinct colour and leaves points on no
// plane with their `base_colours` ([N,4] float RGBA). Lazy, runs on the GPU:
// a palette lookup via take, then a blend with a 0/1 mask. No `where` needed.
inline matrix colour_planes(const PlaneSegmentation& seg, const matrix& base_colours) {
    const size_t k = seg.plane_count();
    const size_m n = seg.labels.shape()[0];
    matrix palette = detail::plane_palette(k);

    matrix idx = seg.labels + 1;                                        // [N] int32, 0 = no plane
    matrix plane_rgba = palette.take(idx, 0);                           // [N,4]
    matrix on = idx.astype(dtype::Float).clamp(0.0, 1.0).reshape(n, (size_m)1); // [N,1] 0/1
    return on * plane_rgba + (1.0f - on) * base_colours;
}

// colour_planes for the lazy labels of segment_planar_surfaces_labels: the plane count is only
// known once that node runs, so the palette is fixed at `max_planes` colours (same colours as
// colour_planes; labels past it are clamped to the last one).
inline matrix colour_plane_labels(const matrix& labels, const matrix& base_colours, size_t max_planes = 64) {
    const size_m n = labels.shape()[0];
    matrix palette = detail::plane_palette(max_planes);
    matrix idx = (labels + 1).clamp(0, (int)max_planes);                // [N] int32, 0 = no plane
    matrix plane_rgba = palette.take(idx, 0);                           // [N,4]
    matrix on = idx.astype(dtype::Float).clamp(0.0, 1.0).reshape(n, (size_m)1); // [N,1] 0/1
    return on * plane_rgba + (1.0f - on) * base_colours;
}

} // namespace fovea

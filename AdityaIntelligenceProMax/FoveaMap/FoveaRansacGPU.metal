#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// GPU half of RANSAC: candidate plane SAMPLING (picking random triples,
// cross product) stays on the CPU since it's cheap (a few hundred ops) and
// branchy/serial - not worth a kernel. What actually costs time with real
// point counts is "count how many of N points are inliers to this candidate,
// for every candidate" - an O(N x iterations) loop, which is exactly the kind
// of embarrassingly-parallel work a GPU is for. One thread per
// (point, candidate) pair, one atomic counter per candidate.
kernel void fovea_ransac_count_inliers(
    device const packed_float3* points  [[ buffer(0) ]], // packed: matches the CPU-side tightly-packed 12-byte Vec3, plain `float3` would stride 16 bytes and misread every point
    device const float4* candidates    [[ buffer(1) ]], // xyz = plane normal, w = d
    device atomic_uint* inlierCounts   [[ buffer(2) ]], // one counter per candidate
    constant uint& pointCount          [[ buffer(3) ]],
    constant float& distanceThreshold  [[ buffer(4) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint pointIdx = gid.x;
    uint candidateIdx = gid.y;
    if (pointIdx >= pointCount) return;

    float4 plane = candidates[candidateIdx];
    float dist = fabs(dot(plane.xyz, float3(points[pointIdx])) + plane.w);
    if (dist < distanceThreshold) {
        atomic_fetch_add_explicit(&inlierCounts[candidateIdx], 1, memory_order_relaxed);
    }
}

// Second pass: once the CPU has picked the winning candidate (argmax over
// inlierCounts - only `iterations` values, trivially fast on CPU, no need for
// a GPU reduction there), compute its final per-point inlier mask.
kernel void fovea_ransac_final_mask(
    device const packed_float3* points [[ buffer(0) ]], // packed - see note on the other kernel above
    constant float4& winningPlane     [[ buffer(1) ]],
    device uchar* inlierMaskOut       [[ buffer(2) ]],
    constant uint& pointCount         [[ buffer(3) ]],
    constant float& distanceThreshold [[ buffer(4) ]],
    uint pointIdx [[ thread_position_in_grid ]]
) {
    if (pointIdx >= pointCount) return;
    float dist = fabs(dot(winningPlane.xyz, float3(points[pointIdx])) + winningPlane.w);
    inlierMaskOut[pointIdx] = dist < distanceThreshold ? 1 : 0;
}

// ---- Multi-plane segmentation (FoveaPlaneSegmentation.h) -------------------
//
// Same idea as the two kernels above, but it scores candidates on a random
// SUBSET of the points (`samples`, indices into `points`) instead of all N, and
// labels are written into a dense [N] int buffer (-1 = no plane yet) so every
// round only claims points that are still unassigned.
//
// Optional normal test (segment_planar_surfaces): a point only counts as an
// inlier if its normal is within the max angle of the plane's normal. Without
// it, the infinite plane also picks up every surface it slices through, e.g. a
// band of wall at the height of a table top.
//
// Counting uses simd_sum: one atomic per SIMD group instead of one per inlier.
// With 1000 candidates, most of them hitting the same few counters, per-inlier
// atomics were the bottleneck. This relies on every SIMD group sharing a single
// candidate row (gid.y), which the host guarantees by dispatching threadgroups
// of (threadExecutionWidth, 1, 1). No early return before simd_sum, so every
// lane in the group reaches it.
kernel void fovea_planes_count_inliers(
    device const packed_float3* points [[ buffer(0) ]], // [N,3] float, contiguous
    device const uint* samples         [[ buffer(1) ]], // indices of the points to score against
    device const float4* candidates    [[ buffer(2) ]], // xyz = unit normal, w = d
    device atomic_uint* inlierCounts   [[ buffer(3) ]], // one counter per candidate
    constant uint& sampleCount         [[ buffer(4) ]],
    constant float& distanceThreshold  [[ buffer(5) ]],
    device const packed_float3* normals [[ buffer(6) ]], // [N,3]; ignored when cosMaxAngle < 0
    constant float& cosMaxAngle        [[ buffer(7) ]],  // < 0 = no normal test
    uint2 gid [[ thread_position_in_grid ]]
) {
    bool inlier = false;
    if (gid.x < sampleCount) {
        float4 plane = candidates[gid.y];
        uint idx = samples[gid.x];
        inlier = fabs(dot(plane.xyz, float3(points[idx])) + plane.w) < distanceThreshold
              && (cosMaxAngle < 0.0f || fabs(dot(plane.xyz, float3(normals[idx]))) >= cosMaxAngle);
    }
    uint groupCount = simd_sum(inlier ? 1u : 0u);
    if (simd_is_first() && groupCount > 0) {
        atomic_fetch_add_explicit(&inlierCounts[gid.y], groupCount, memory_order_relaxed);
    }
}

// Claims every still-unassigned point within distanceThreshold of the winning
// plane: labels[i] = label. It also counts how many points it claimed, so the
// host can check min_inliers without reading back all N labels.
kernel void fovea_planes_assign(
    device const packed_float3* points [[ buffer(0) ]],
    constant float4& plane             [[ buffer(1) ]],
    device int* labels                 [[ buffer(2) ]], // [N], -1 = unassigned
    device atomic_uint* claimedCount   [[ buffer(3) ]],
    constant uint& pointCount          [[ buffer(4) ]],
    constant float& distanceThreshold  [[ buffer(5) ]],
    constant int& label                [[ buffer(6) ]],
    device const packed_float3* normals [[ buffer(7) ]], // [N,3]; ignored when cosMaxAngle < 0
    constant float& cosMaxAngle        [[ buffer(8) ]],  // < 0 = no normal test
    uint i [[ thread_position_in_grid ]]
) {
    bool claimed = false;
    if (i < pointCount && labels[i] < 0) {
        claimed = fabs(dot(plane.xyz, float3(points[i])) + plane.w) < distanceThreshold
               && (cosMaxAngle < 0.0f || fabs(dot(plane.xyz, float3(normals[i]))) >= cosMaxAngle);
        if (claimed) labels[i] = label;
    }
    uint groupCount = simd_sum(claimed ? 1u : 0u);
    if (simd_is_first() && groupCount > 0) {
        atomic_fetch_add_explicit(claimedCount, groupCount, memory_order_relaxed);
    }
}

// ---- Normal estimation (FoveaNormals.h) -------------------------------------
//
// Voxel hash grid + summed moments instead of per-point neighbour lists.
// Open3D's GPU path uses a spatial hash grid for neighbours; PCL's integral-image
// normals use summed moments. This combines the two: each voxel accumulates
// count, sum(p) and sum(p p^T), a voxel's covariance is the sum over its 3x3x3
// neighbourhood, and its normal is that covariance's smallest eigenvector.
// Every point takes its voxel's normal. O(N + 27 V), no KD-tree.
//
// Voxel keys are the linear index into the grid's bounding box (host checks it
// fits in 32 bits); 0xFFFFFFFF marks an empty hash slot. Open addressing with
// linear probing; the table is at least 2x the maximum voxel count, so probes
// terminate.

struct FoveaGrid {
    float4 originVoxel; // xyz = grid origin (bbox min), w = voxel size
    uint4 dimsMask;     // xyz = voxels per axis, w = hash table size - 1 (power of two)
};

constant uint FOVEA_EMPTY_KEY = 0xFFFFFFFFu;

inline int3 fovea_voxel_coord(float3 p, constant FoveaGrid& g) {
    return int3(floor((p - g.originVoxel.xyz) / g.originVoxel.w));
}

inline uint fovea_voxel_key(int3 c, constant FoveaGrid& g) {
    return uint(c.x) + g.dimsMask.x * (uint(c.y) + g.dimsMask.y * uint(c.z));
}

inline uint fovea_hash_slot(uint key, uint mask) {
    key *= 0x9E3779B1u; // Fibonacci hashing: linear keys of neighbouring voxels would otherwise cluster
    return (key ^ (key >> 16)) & mask;
}

inline int fovea_voxel_lookup(device const uint* keys, device const uint* ids, uint key, uint mask) {
    uint slot = fovea_hash_slot(key, mask);
    while (true) {
        uint k = keys[slot];
        if (k == key) return int(ids[slot]);
        if (k == FOVEA_EMPTY_KEY) return -1;
        slot = (slot + 1) & mask;
    }
}

// Pass 1: insert every point's voxel key into the hash table.
kernel void fovea_voxel_insert(
    device const packed_float3* points [[ buffer(0) ]],
    constant FoveaGrid& grid           [[ buffer(1) ]],
    device atomic_uint* keys           [[ buffer(2) ]],
    constant uint& pointCount          [[ buffer(3) ]],
    uint i [[ thread_position_in_grid ]]
) {
    if (i >= pointCount) return;
    uint key = fovea_voxel_key(fovea_voxel_coord(float3(points[i]), grid), grid);
    uint slot = fovea_hash_slot(key, grid.dimsMask.w);
    while (true) {
        uint expected = FOVEA_EMPTY_KEY;
        if (atomic_compare_exchange_weak_explicit(&keys[slot], &expected, key, memory_order_relaxed, memory_order_relaxed)) return;
        if (expected == key) return;                 // voxel already inserted
        if (expected == FOVEA_EMPTY_KEY) continue;   // spurious weak-CAS failure, retry this slot
        slot = (slot + 1) & grid.dimsMask.w;
    }
}

// Pass 2: give every occupied slot a dense voxel id 0..V-1.
kernel void fovea_voxel_number(
    device const uint* keys      [[ buffer(0) ]],
    device uint* ids             [[ buffer(1) ]],
    device uint* voxelKeys       [[ buffer(2) ]], // [tableSize] upper bound; first V entries used
    device atomic_uint* counter  [[ buffer(3) ]],
    constant uint& tableSize     [[ buffer(4) ]],
    uint slot [[ thread_position_in_grid ]]
) {
    if (slot >= tableSize) return;
    uint key = keys[slot];
    if (key == FOVEA_EMPTY_KEY) return;
    uint id = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    ids[slot] = id;
    voxelKeys[id] = key;
}

// Pass 3: accumulate each point into its voxel's moments, in voxel-local
// coordinates (p - voxel corner, so values stay in [0, voxel) and the
// E[pp^T] - E[p]E[p]^T subtraction doesn't lose precision). 10 floats per voxel:
// count, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz.
kernel void fovea_voxel_accumulate(
    device const packed_float3* points [[ buffer(0) ]],
    constant FoveaGrid& grid           [[ buffer(1) ]],
    device const uint* keys            [[ buffer(2) ]],
    device const uint* ids             [[ buffer(3) ]],
    device atomic_float* moments       [[ buffer(4) ]],
    device uint* pointVoxel            [[ buffer(5) ]],
    constant uint& pointCount          [[ buffer(6) ]],
    uint i [[ thread_position_in_grid ]]
) {
    if (i >= pointCount) return;
    float3 p = float3(points[i]);
    int3 c = fovea_voxel_coord(p, grid);
    uint id = uint(fovea_voxel_lookup(keys, ids, fovea_voxel_key(c, grid), grid.dimsMask.w));
    pointVoxel[i] = id;

    float3 l = p - (grid.originVoxel.xyz + float3(c) * grid.originVoxel.w);
    device atomic_float* m = moments + id * 10;
    atomic_fetch_add_explicit(&m[0], 1.0f, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[1], l.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[2], l.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[3], l.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[4], l.x * l.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[5], l.x * l.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[6], l.x * l.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[7], l.y * l.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[8], l.y * l.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&m[9], l.z * l.z, memory_order_relaxed);
}

// Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix
// (closed-form eigenvalues, then the largest cross product of two rows of
// A - lambda I). Returns 0 when the neighbourhood has no preferred direction.
inline float3 fovea_smallest_eigenvector(float a00, float a01, float a02, float a11, float a12, float a22) {
    float scale = max(max(max(fabs(a00), fabs(a01)), max(fabs(a02), fabs(a11))), max(fabs(a12), fabs(a22)));
    if (scale <= 0.0f) return float3(0.0f);
    a00 /= scale; a01 /= scale; a02 /= scale; a11 /= scale; a12 /= scale; a22 /= scale;

    float q = (a00 + a11 + a22) / 3.0f;
    float p1 = a01 * a01 + a02 * a02 + a12 * a12;
    float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2.0f * p1;
    float p = sqrt(p2 / 6.0f);
    if (p < 1e-8f) return float3(0.0f); // isotropic

    float b00 = (a00 - q) / p, b01 = a01 / p, b02 = a02 / p;
    float b11 = (a11 - q) / p, b12 = a12 / p, b22 = (a22 - q) / p;
    float r = 0.5f * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02));
    float phi = acos(clamp(r, -1.0f, 1.0f)) / 3.0f;
    float smallest = q + 2.0f * p * cos(phi + 2.0943951f); // + 2*pi/3

    float3 r0 = float3(a00 - smallest, a01, a02);
    float3 r1 = float3(a01, a11 - smallest, a12);
    float3 r2 = float3(a02, a12, a22 - smallest);
    float3 c01 = cross(r0, r1), c02 = cross(r0, r2), c12 = cross(r1, r2);
    float d01 = dot(c01, c01), d02 = dot(c02, c02), d12 = dot(c12, c12);
    float3 best = d01 >= d02 && d01 >= d12 ? c01 : (d02 >= d12 ? c02 : c12);
    float len = length(best);
    return len > 1e-12f ? best / len : float3(0.0f);
}

// Pass 4: one normal per voxel from its 3x3x3 neighbourhood's summed moments,
// shifted into this voxel's local frame. For a neighbour offset by delta:
//   sum(l + delta) = s + n delta,  sum((l+delta)(l+delta)^T) = Q + s delta^T + delta s^T + n delta delta^T
kernel void fovea_voxel_normals(
    constant FoveaGrid& grid     [[ buffer(0) ]],
    device const uint* keys      [[ buffer(1) ]],
    device const uint* ids       [[ buffer(2) ]],
    device const uint* voxelKeys [[ buffer(3) ]],
    device const float* moments  [[ buffer(4) ]],
    device packed_float3* voxelNormals [[ buffer(5) ]],
    constant uint& voxelCount    [[ buffer(6) ]],
    constant uint& minPoints     [[ buffer(7) ]],
    uint v [[ thread_position_in_grid ]]
) {
    if (v >= voxelCount) return;
    uint key = voxelKeys[v];
    int3 c = int3(key % grid.dimsMask.x, (key / grid.dimsMask.x) % grid.dimsMask.y, key / (grid.dimsMask.x * grid.dimsMask.y));
    int3 dims = int3(grid.dimsMask.xyz);
    float size = grid.originVoxel.w;

    float n = 0, sx = 0, sy = 0, sz = 0, xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int3 nc = c + int3(dx, dy, dz);
        if (any(nc < 0) || any(nc >= dims)) continue;
        int id = fovea_voxel_lookup(keys, ids, fovea_voxel_key(nc, grid), grid.dimsMask.w);
        if (id < 0) continue;
        device const float* m = moments + id * 10;
        float cnt = m[0];
        float3 s = float3(m[1], m[2], m[3]);
        float3 d = float3(dx, dy, dz) * size;
        n += cnt;
        sx += s.x + cnt * d.x; sy += s.y + cnt * d.y; sz += s.z + cnt * d.z;
        xx += m[4] + 2.0f * s.x * d.x + cnt * d.x * d.x;
        xy += m[5] + s.x * d.y + s.y * d.x + cnt * d.x * d.y;
        xz += m[6] + s.x * d.z + s.z * d.x + cnt * d.x * d.z;
        yy += m[7] + 2.0f * s.y * d.y + cnt * d.y * d.y;
        yz += m[8] + s.y * d.z + s.z * d.y + cnt * d.y * d.z;
        zz += m[9] + 2.0f * s.z * d.z + cnt * d.z * d.z;
    }

    float3 normal = float3(0.0f);
    if (n >= float(minPoints)) {
        float3 mean = float3(sx, sy, sz) / n;
        normal = fovea_smallest_eigenvector(
            xx / n - mean.x * mean.x, xy / n - mean.x * mean.y, xz / n - mean.x * mean.z,
            yy / n - mean.y * mean.y, yz / n - mean.y * mean.z, zz / n - mean.z * mean.z);
    }
    voxelNormals[v] = packed_float3(normal);
}

// Pass 5: every point takes its voxel's normal.
kernel void fovea_gather_normals(
    device const uint* pointVoxel            [[ buffer(0) ]],
    device const packed_float3* voxelNormals [[ buffer(1) ]],
    device packed_float3* normals            [[ buffer(2) ]],
    constant uint& pointCount                [[ buffer(3) ]],
    uint i [[ thread_position_in_grid ]]
) {
    if (i >= pointCount) return;
    normals[i] = voxelNormals[pointVoxel[i]];
}

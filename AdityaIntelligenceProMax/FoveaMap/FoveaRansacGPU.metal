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

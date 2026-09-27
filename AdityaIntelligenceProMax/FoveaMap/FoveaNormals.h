#pragma once
#include "FoveaRansacGPU.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Per-point normals for an unorganized [N,3] point cloud (PLY scans without
// normals). This does the same job as Open3D's estimate_normals / PCL's
// NormalEstimation (PCA over a neighbourhood, normal = smallest-eigenvalue
// eigenvector), but the neighbourhood is a voxel cube instead of a KD-tree
// radius search:
//   - a GPU voxel hash grid (Open3D's GPU neighbour search is a hash grid too),
//   - with summed moments per voxel (the trick behind PCL's integral-image
//     normals), so a neighbourhood's covariance is 27 lookups + adds, not a
//     neighbour list.
// Cost is O(N + 27 V) for V occupied voxels, all on the GPU. Every point in a
// voxel gets that voxel's normal, computed from the 3x3x3 voxels around it,
// i.e. a neighbourhood about 3 * voxel_size across.
//
// Normals are unoriented (the sign is arbitrary), which is all plane
// segmentation needs since it compares with |dot|. A point gets (0,0,0) when
// its neighbourhood has fewer than `min_points` points or no dominant direction.
//
// Pick voxel_size so a 3-voxel cube holds a couple of dozen points and still
// sits on a single surface: about 3-5 cm for room-scale LiDAR scans.
namespace fovea {

struct FoveaGridParams {
    simd_float4 originVoxel; // xyz = bbox min, w = voxel size
    simd_uint4 dimsMask;     // xyz = voxels per axis, w = hash table size - 1
};

inline matrix estimate_normals(matrix points, float voxel_size = 0.04f, uint32_t min_points = 6) {
    if (points.dims != 2 || points.shape()[1] != 3) {
        throw std::invalid_argument("estimate_normals: points must be [N,3]");
    }
    if (points.type != dtype::Float || (points.flags & NON_CONTIGUOUS_FLAG)) {
        points = points.astype(dtype::Float, true);
    }
    points.eval();

    const size_t n = points.shape()[0];
    matrix normals = matrix::withShape({(size_m)n, 3}, dtype::Float);
    normals.begin_refcount();
    if (!normals.metalBuffer) normals.buildMetalBuffer();
    std::fill_n(static_cast<float*>(normals.buffer), n * 3, 0.0f);
    if (n == 0) return normals;

    id<MTLComputePipelineState> insertPipeline = _fovea_gpu_pipeline(@"fovea_voxel_insert");
    id<MTLComputePipelineState> numberPipeline = _fovea_gpu_pipeline(@"fovea_voxel_number");
    id<MTLComputePipelineState> accumulatePipeline = _fovea_gpu_pipeline(@"fovea_voxel_accumulate");
    id<MTLComputePipelineState> voxelNormalsPipeline = _fovea_gpu_pipeline(@"fovea_voxel_normals");
    id<MTLComputePipelineState> gatherPipeline = _fovea_gpu_pipeline(@"fovea_gather_normals");
    if (!insertPipeline || !numberPipeline || !accumulatePipeline || !voxelNormalsPipeline || !gatherPipeline || !normals.metalBuffer) {
        return normals;
    }

    // --- grid over the bounding box; voxel keys are linear indices into it ---
    const float* pts = static_cast<const float*>(points.buffer);
    simd_float3 lo = simd_make_float3(pts[0], pts[1], pts[2]), hi = lo;
    for (size_t i = 1; i < n; ++i) {
        simd_float3 p = simd_make_float3(pts[i * 3 + 0], pts[i * 3 + 1], pts[i * 3 + 2]);
        lo = simd_min(lo, p);
        hi = simd_max(hi, p);
    }
    simd_uint3 dims;
    for (int a = 0; a < 3; ++a) dims[a] = (uint32_t)std::floor((hi[a] - lo[a]) / voxel_size) + 1;
    if ((uint64_t)dims.x * dims.y * dims.z >= 0xFFFFFFFFull) {
        throw std::invalid_argument("estimate_normals: voxel grid too fine for 32-bit keys, increase voxel_size");
    }

    // Open addressing needs load < 1; at most min(N, cells) voxels, table >= 2x that.
    const uint64_t maxVoxels = std::min<uint64_t>(n, (uint64_t)dims.x * dims.y * dims.z);
    uint32_t tableSize = 1024;
    while (tableSize < 2 * maxVoxels) tableSize <<= 1;

    FoveaGridParams grid;
    grid.originVoxel = simd_make_float4(lo, voxel_size);
    grid.dimsMask = simd_make_uint4(dims, tableSize - 1);

    id<MTLDevice> device = GlobalGPUManager.metalDevice;
    id<MTLBuffer> pointsBuf = points.metalBuffer;
    if (!pointsBuf) {
        pointsBuf = [device newBufferWithBytes:pts length:n * 3 * sizeof(float) options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> keysBuf = [device newBufferWithLength:tableSize * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> idsBuf = [device newBufferWithLength:tableSize * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> voxelKeysBuf = [device newBufferWithLength:maxVoxels * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> counterBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *static_cast<uint32_t*>(counterBuf.contents) = 0;
    const uint32_t pointCount = (uint32_t)n;

    // --- GPU: insert voxel keys, then number the occupied slots 0..V-1 ---
    id<MTLCommandBuffer> cmd = [GlobalGPUManager.gCommandQueue commandBuffer];
    id<MTLBlitCommandEncoder> clear = [cmd blitCommandEncoder];
    [clear fillBuffer:keysBuf range:NSMakeRange(0, tableSize * sizeof(uint32_t)) value:0xFF];
    [clear endEncoding];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:insertPipeline];
    [enc setBuffer:pointsBuf offset:0 atIndex:0];
    [enc setBytes:&grid length:sizeof(grid) atIndex:1];
    [enc setBuffer:keysBuf offset:0 atIndex:2];
    [enc setBytes:&pointCount length:sizeof(pointCount) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(insertPipeline.threadExecutionWidth, 1, 1)];

    [enc setComputePipelineState:numberPipeline];
    [enc setBuffer:keysBuf offset:0 atIndex:0];
    [enc setBuffer:idsBuf offset:0 atIndex:1];
    [enc setBuffer:voxelKeysBuf offset:0 atIndex:2];
    [enc setBuffer:counterBuf offset:0 atIndex:3];
    [enc setBytes:&tableSize length:sizeof(tableSize) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(tableSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(numberPipeline.threadExecutionWidth, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const uint32_t voxelCount = *static_cast<uint32_t*>(counterBuf.contents);
    if (voxelCount == 0) return normals;

    id<MTLBuffer> momentsBuf = [device newBufferWithLength:voxelCount * 10 * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> pointVoxelBuf = [device newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> voxelNormalsBuf = [device newBufferWithLength:voxelCount * 3 * sizeof(float) options:MTLResourceStorageModePrivate];

    // --- GPU: moments per voxel -> normal per voxel -> normal per point ---
    id<MTLCommandBuffer> cmd2 = [GlobalGPUManager.gCommandQueue commandBuffer];
    id<MTLBlitCommandEncoder> zero = [cmd2 blitCommandEncoder];
    [zero fillBuffer:momentsBuf range:NSMakeRange(0, voxelCount * 10 * sizeof(float)) value:0];
    [zero endEncoding];

    id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
    [enc2 setComputePipelineState:accumulatePipeline];
    [enc2 setBuffer:pointsBuf offset:0 atIndex:0];
    [enc2 setBytes:&grid length:sizeof(grid) atIndex:1];
    [enc2 setBuffer:keysBuf offset:0 atIndex:2];
    [enc2 setBuffer:idsBuf offset:0 atIndex:3];
    [enc2 setBuffer:momentsBuf offset:0 atIndex:4];
    [enc2 setBuffer:pointVoxelBuf offset:0 atIndex:5];
    [enc2 setBytes:&pointCount length:sizeof(pointCount) atIndex:6];
    [enc2 dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(accumulatePipeline.threadExecutionWidth, 1, 1)];

    [enc2 setComputePipelineState:voxelNormalsPipeline];
    [enc2 setBytes:&grid length:sizeof(grid) atIndex:0];
    [enc2 setBuffer:keysBuf offset:0 atIndex:1];
    [enc2 setBuffer:idsBuf offset:0 atIndex:2];
    [enc2 setBuffer:voxelKeysBuf offset:0 atIndex:3];
    [enc2 setBuffer:momentsBuf offset:0 atIndex:4];
    [enc2 setBuffer:voxelNormalsBuf offset:0 atIndex:5];
    [enc2 setBytes:&voxelCount length:sizeof(voxelCount) atIndex:6];
    [enc2 setBytes:&min_points length:sizeof(min_points) atIndex:7];
    [enc2 dispatchThreads:MTLSizeMake(voxelCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(voxelNormalsPipeline.threadExecutionWidth, 1, 1)];

    [enc2 setComputePipelineState:gatherPipeline];
    [enc2 setBuffer:pointVoxelBuf offset:0 atIndex:0];
    [enc2 setBuffer:voxelNormalsBuf offset:0 atIndex:1];
    [enc2 setBuffer:normals.metalBuffer offset:0 atIndex:2];
    [enc2 setBytes:&pointCount length:sizeof(pointCount) atIndex:3];
    [enc2 dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(gatherPipeline.threadExecutionWidth, 1, 1)];
    [enc2 endEncoding];
    [cmd2 commit];
    [cmd2 waitUntilCompleted];

    return normals;
}

// Lazy version for live tuning: a graph node that re-runs the estimate when `points` or a
// tracked parameter changes (e.g. voxel_size from a slider's leaf_scalar). Same result as above.
inline matrix estimate_normals(matrix points, const ScalarArg& voxel_size, const ScalarArg& min_points = 6) {
    if (points.dims != 2 || points.shape()[1] != 3) {
        throw std::invalid_argument("estimate_normals: points must be [N,3]");
    }
    std::vector<matrix> inputs = {points};
    HostArg voxel = HostArg::track(voxel_size, inputs);
    HostArg minPts = HostArg::track(min_points, inputs);
    const size_m n = points.shape()[0];
    return matrix::host_op({n, 3}, dtype::Float, [voxel, minPts](matrix& out, std::vector<matrix>& ins) {
        matrix normals = estimate_normals(ins[0], std::max(voxel.value(ins), 1e-4f), (uint32_t)std::max(minPts.value(ins), 1.0f));
        memcpy(out.buffer, normals.buffer, out.total_size * sizeof(float));
    }, inputs);
}

} // namespace fovea

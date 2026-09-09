#pragma once
#include "../matrix.h"
#include "../Mods/GPUManager.h"
#include "FoveaClassify.h"

#include <random>
#include <vector>

// GPU-accelerated ground-plane RANSAC. Needs FoveaMap/FoveaRansacGPU.metal
// added to the Xcode target (it's inside the filesystem-synced
// AdityaIntelligenceProMax tree so it should just get picked up and compiled
// into the default Metal library alongside Shaders.metal/DAGShaders.metal -
// if the pipeline fails to build at runtime, that .metal file not actually
// being in "Compile Sources" is the first thing to check).
//
// Only the O(N x iterations) "count inliers per candidate" loop moves to the
// GPU - candidate sampling (picking random triples, cross product) stays on
// the CPU exactly as in ransac_ground_plane, since it's cheap and branchy/
// serial, not worth a kernel. This mirrors that function's logic closely on
// purpose so the two stay easy to compare; if you change one, check the other.
namespace fovea {

inline id<MTLComputePipelineState> _fovea_gpu_pipeline(NSString* functionName) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = [NSMutableDictionary new];
    id<MTLComputePipelineState> cached = cache[functionName];
    if (cached) return cached;

    NSError* error = nil;
    id<MTLFunction> fn = [GlobalGPUManager.library newFunctionWithName:functionName];
    if (!fn) {
        NSLog(@"FoveaRansacGPU: Metal function '%@' not found - is FoveaRansacGPU.metal in the target's Compile Sources?", functionName);
        return nil;
    }
    id<MTLComputePipelineState> state = [GlobalGPUManager.metalDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!state) {
        NSLog(@"FoveaRansacGPU: failed to build pipeline '%@': %@", functionName, error);
    }
    cache[functionName] = state;
    return state;
}

inline PlaneModel ransac_ground_plane_gpu(
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

    // --- candidate sampling: same logic as ransac_ground_plane, still CPU ---
    Vec3 priorNormalUnit{0.0f, 0.0f, 1.0f};
    float cosAngleThreshold = -1.0f;
    if (priorNormal) {
        float priorLen = length(*priorNormal);
        if (priorLen > 1e-6f) {
            priorNormalUnit = {priorNormal->x / priorLen, priorNormal->y / priorLen, priorNormal->z / priorLen};
            cosAngleThreshold = std::cos(maxAngleDeviationDegrees * (float)M_PI / 180.0f);
        }
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    struct Candidate { float nx, ny, nz, d; };
    std::vector<Candidate> candidates;
    candidates.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        size_t i0 = pick(rng), i1 = pick(rng), i2 = pick(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        Vec3 normal = cross(sub(points[i1], points[i0]), sub(points[i2], points[i0]));
        float len = length(normal);
        if (len < 1e-6f) continue;
        normal = {normal.x / len, normal.y / len, normal.z / len};

        if (cosAngleThreshold > -1.0f) {
            if (std::fabs(dot(normal, priorNormalUnit)) < cosAngleThreshold) continue;
        }

        float d = -dot(normal, points[i0]);
        candidates.push_back({normal.x, normal.y, normal.z, d});
    }
    if (candidates.empty()) return {};

    id<MTLDevice> device = GlobalGPUManager.metalDevice;
    uint32_t pointCountU = (uint32_t)n;
    float threshold = distanceThreshold;

    id<MTLBuffer> pointsBuf = [device newBufferWithBytes:points.data()
                                                   length:points.size() * sizeof(Vec3)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> candidatesBuf = [device newBufferWithBytes:candidates.data()
                                                       length:candidates.size() * sizeof(Candidate)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> countsBuf = [device newBufferWithLength:candidates.size() * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
    memset(countsBuf.contents, 0, candidates.size() * sizeof(uint32_t));

    // --- GPU pass 1: inlier count per candidate, one thread per (point, candidate) ---
    id<MTLComputePipelineState> countPipeline = _fovea_gpu_pipeline(@"fovea_ransac_count_inliers");
    if (!countPipeline) return {}; // .metal not compiled in - see log above

    id<MTLCommandBuffer> cmd = [GlobalGPUManager.gCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:countPipeline];
    [enc setBuffer:pointsBuf offset:0 atIndex:0];
    [enc setBuffer:candidatesBuf offset:0 atIndex:1];
    [enc setBuffer:countsBuf offset:0 atIndex:2];
    [enc setBytes:&pointCountU length:sizeof(pointCountU) atIndex:3];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, candidates.size(), 1)
      threadsPerThreadgroup:MTLSizeMake(countPipeline.threadExecutionWidth, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const uint32_t* counts = static_cast<const uint32_t*>(countsBuf.contents);
    size_t bestIdx = 0;
    uint32_t bestCount = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (counts[i] > bestCount) { bestCount = counts[i]; bestIdx = i; }
    }
    Candidate winning = candidates[bestIdx];

    PlaneModel best;
    best.normal = {winning.nx, winning.ny, winning.nz};
    best.d = winning.d;

    // --- GPU pass 2: final inlier mask for the winning plane only ---
    id<MTLComputePipelineState> maskPipeline = _fovea_gpu_pipeline(@"fovea_ransac_final_mask");
    if (!maskPipeline) return best;

    id<MTLBuffer> maskBuf = [device newBufferWithLength:n * sizeof(uint8_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd2 = [GlobalGPUManager.gCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
    [enc2 setComputePipelineState:maskPipeline];
    [enc2 setBuffer:pointsBuf offset:0 atIndex:0];
    [enc2 setBytes:&winning length:sizeof(winning) atIndex:1];
    [enc2 setBuffer:maskBuf offset:0 atIndex:2];
    [enc2 setBytes:&pointCountU length:sizeof(pointCountU) atIndex:3];
    [enc2 setBytes:&threshold length:sizeof(threshold) atIndex:4];
    [enc2 dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(maskPipeline.threadExecutionWidth, 1, 1)];
    [enc2 endEncoding];
    [cmd2 commit];
    [cmd2 waitUntilCompleted];

    memcpy(inlierMaskOut.data(), maskBuf.contents, n);
    return best;
}

// Drop-in GPU replacement for classify_ground_colors / classify_ground_colors_from_depth.
// `gravity` optional, last param, defaults to no constraint - see
// classify_ground_colors's comment in FoveaClassify.h for details.
inline matrix classify_ground_colors_gpu(
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
    ransac_ground_plane_gpu(
        pts, inlierMask, iterations, distanceThreshold,
        hasGravity ? &gravityVec : nullptr, maxAngleDeviationDegrees
    );
    return colors_from_ground_mask(inlierMask);
}

inline matrix classify_ground_colors_from_depth_gpu(
    matrix depth,
    const CameraIntrinsics& camera,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    float maxAngleDeviationDegrees = 20.0f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    matrix unprojectedPoints = points_from_depth_intrinsics(depth, camera);
    return classify_ground_colors_gpu(unprojectedPoints, iterations, distanceThreshold, maxAngleDeviationDegrees, gravity);
}

} // namespace fovea

#pragma once
#include "../matrix.h"
#include "FoveaRansacGPU.h"

// Colors each depth-grid point from the actual RGB video image, handling the
// color image and depth map being different resolutions. LiDAR depth and the
// paired color camera share the same field of view (that's exactly what
// intrinsicMatrixReferenceDimensions guarantees, same assumption
// FoveaGeometry.h's rescale relies on) - just delivered at different pixel
// densities. So each depth pixel maps to a color pixel by simple proportional
// scaling (nearest-neighbor), not a re-projection.
namespace fovea {

// `colorImage`: [colorH, colorW, 4] (any numeric dtype - e.g. the UInt8
// buffer `cap read_metal:image` already fills). Returns an [depthH*depthW, 4]
// float colors matrix in the same per-pixel order points_from_depth_intrinsics
// produces, so it drops straight into PointCloudController::update's colors
// argument, or gets blended with classification colors afterward.
inline matrix colors_from_image(matrix& colorImage, size_m depthH, size_m depthW) {
    size_m colorH = colorImage.shape()[0];
    size_m colorW = colorImage.shape()[1];

    matrix depthU = matrix::linespace(0.0f, (float)(depthW - 1), depthW);
    matrix depthV = matrix::linespace(0.0f, (float)(depthH - 1), depthH);
    auto mesh = matrix::meshgrid(depthU, depthV); // [depthH, depthW], same convention as points_from_depth_intrinsics
    matrix U = mesh.first;
    matrix V = mesh.second;

    float scaleU = (float)colorW / (float)depthW;
    float scaleV = (float)colorH / (float)depthH;

    // clamp before casting to int - floating point rounding at the last pixel
    // could otherwise push the index one past colorW-1/colorH-1
    matrix colorU = (U * scaleU).clamp(0.0, (double)(colorW - 1)).astype(dtype::Int32);
    matrix colorV = (V * scaleV).clamp(0.0, (double)(colorH - 1)).astype(dtype::Int32);

    matrix flatIndex = colorV * (int)colorW + colorU; // [depthH, depthW]
    matrix colorFlat = colorImage.reshape(colorH * colorW, (size_m)4);
    matrix gathered = colorFlat.take(flatIndex.flatten(), 0); // [depthH*depthW, 4]

    return gathered.astype(dtype::Float) / 255.0f;
}

// Ground/not-ground classification, but colored with the real photo instead
// of a flat tint - `tintWeight` dials between pure photo color (0.0) and
// pure classification tint (1.0), so the demo still reads as a real scene
// with a visible ground/obstacle cue, not a flat-colored diagram.
inline matrix classify_ground_colors_blended(
    matrix& points,
    matrix& colorImage,
    size_m depthH,
    size_m depthW,
    float tintWeight = 0.35f,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    float maxAngleDeviationDegrees = 20.0f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    matrix tintColors = classify_ground_colors_gpu(points, iterations, distanceThreshold, maxAngleDeviationDegrees, gravity);
    matrix photoColors = colors_from_image(colorImage, depthH, depthW);
    return tintColors * tintWeight + photoColors * (1.0f - tintWeight);
}

// Same idea, but asymmetric: ground gets pulled `groundTintWeight` toward
// green, non-ground points stay pure untouched photo color (weight 0) -
// rather than one uniform blend applied everywhere, this builds a per-point
// weight (0.9 where ground, 0 elsewhere) and lets `matrix` broadcast it
// against the [N,4] colors.
inline matrix classify_ground_colors_green_tint(
    matrix& points,
    matrix& colorImage,
    size_m depthH,
    size_m depthW,
    float groundTintWeight = 0.9f,
    int iterations = 300,
    float distanceThreshold = 0.06f,
    float maxAngleDeviationDegrees = 20.0f,
    simd_float3 gravity = simd_float3{0.0f, 0.0f, 0.0f}
) {
    std::vector<Vec3> pts = extract_points(points);
    Vec3 gravityVec{gravity.x, gravity.y, gravity.z};
    bool hasGravity = length(gravityVec) > 1e-6f;
    std::vector<uint8_t> groundMask;
    ransac_ground_plane_gpu(pts, groundMask, iterations, distanceThreshold,
                             hasGravity ? &gravityVec : nullptr, maxAngleDeviationDegrees);

    matrix photoColors = colors_from_image(colorImage, depthH, depthW);

    matrix weight = matrix::withShape({(size_m)pts.size(), 1}, dtype::Float);
    weight.begin_refcount();
    for (size_t i = 0; i < pts.size(); ++i) {
        weight.at<float>((size_m)i, 0) = groundMask[i] ? groundTintWeight : 0.0f;
    }

    matrix green = matrix::of<float>({0.10f, 0.85f, 0.20f, 1.0f}).reshape(1, 4);
    return green * weight + photoColors * (1.0f - weight);
}

} // namespace fovea

#pragma once
#include "../matrix.h"
#include "../Mods/Utils.h"
#include <simd/simd.h>

// Depth -> metric XYZ using the camera's actual intrinsics, replacing the old
// fixed -5..5 linspace guess in depth_streamer. This stays in the matrix
// graph (not plain C++) since it's exactly the bulk elementwise transform
// over a whole image that matrix is built for - unlike the RANSAC/clustering
// code in FoveaClassify.h, there's no branchy per-point logic here.
namespace fovea {

// Bundles a camera's intrinsic matrix with the resolution it was actually
// calibrated at - these two must always travel together (see
// AVDepthData.cameraCalibrationData.intrinsicMatrix /
// .intrinsicMatrixReferenceDimensions, which Apple's own samples never
// separate either), since the matrix alone is meaningless without knowing
// what resolution its fx/fy/cx/cy assume. Populate this straight from
// depthData.cameraCalibrationData - both fields, together, every frame -
// rather than hardcoding referenceWidth/Height, since that's a guess that
// can silently be wrong for a different capture format/device.


// `camera.intrinsicMatrix` is Apple's standard pinhole camera matrix
// (matrix_float3x3, column-major): columns[0] = (fx, 0, 0),
// columns[1] = (0, fy, 0), columns[2] = (cx, cy, 1). Standard formula:
// X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth - this gives Y
// increasing *downward* (standard camera-frame convention), which may look
// upside-down next to the old hack's arbitrary Y-up flip; negate Y after the
// fact if so, it's a one-line change on your end, not worth guessing here.
//
// `camera.referenceWidth/Height` rescale fx/fy/cx/cy from the resolution
// they were calibrated at down to this depth map's actual resolution -
// without it, cx/cy end up so much larger than the depth map's actual
// pixel-index range that (u - cx) is dominated by the constant offset and
// barely varies, collapsing X and Y into near-pure multiples of Z: the whole
// cloud squished onto a diagonal ray through the origin, which is exactly
// the symptom this fixed.
inline matrix points_from_depth_intrinsics(matrix& depth, const CameraIntrinsics& camera) {
    matrix Z = depth.astype(dtype::Float);
    size_m h = Z.shape()[0];
    size_m w = Z.shape()[1];

    float scaleX = (float)w / camera.referenceWidth;
    float scaleY = (float)h / camera.referenceHeight;
    float fx = camera.intrinsicMatrix.columns[0].x * scaleX;
    float fy = camera.intrinsicMatrix.columns[1].y * scaleY;
    float cx = camera.intrinsicMatrix.columns[2].x * scaleX;
    float cy = camera.intrinsicMatrix.columns[2].y * scaleY;

    matrix u_coords = matrix::linespace(0.0f, (float)(w - 1), w);
    matrix v_coords = matrix::linespace(0.0f, (float)(h - 1), h);
    auto mesh = matrix::meshgrid(u_coords, v_coords); // same first/second convention depth_streamer already used
    matrix U = mesh.first;
    matrix V = mesh.second;

    matrix X = (U - cx) * Z / fx;
    matrix Y = (V - cy) * Z / fy;

    return matrix::stack({X.flatten(), Y.flatten(), Z.flatten()}, -1);
}

} // namespace fovea

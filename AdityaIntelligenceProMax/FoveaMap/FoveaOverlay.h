#pragma once
#include "../matrix.h"
#include "FoveaClassify.h"
#include "FoveaGeometry.h"

// Draws ground/path classification as a flat 2D overlay on the depth image
// itself, instead of rendering a 3D point cloud - sidesteps whether the 3D
// reprojection looks geometrically clean, since this only needs the
// classification (which the RANSAC fit, using real 3D points internally, has
// already been verified against). Much easier to get looking clean for a demo.
namespace fovea {

// AVFoundation delivers camera/depth buffers in the sensor's native landscape
// orientation regardless of device/display orientation - nothing rotates
// this for you. A rotated 3D point cloud just looks like "a different
// viewing angle" so it goes unnoticed; a flat 2D image makes it obvious.
// Direction (CW vs CCW) depends on how the sensor is physically mounted
// relative to your display code, which I can't see - try `clockwise=true`
// first, flip to `false` if it rotates the wrong way (still 90 degrees, just
// the other direction, one-word change).
inline matrix rotate90(const matrix& image, bool clockwise) {
    size_m h = image.shape()[0];
    size_m w = image.shape()[1];
    size_m channels = image.shape()[2];

    matrix rotated = matrix::withShape({w, h, channels}, dtype::UInt8);
    rotated.begin_refcount();

    const uint8_t* src = static_cast<const uint8_t*>(image.buffer);
    uint8_t* dst = static_cast<uint8_t*>(rotated.buffer);

    for (size_m row = 0; row < h; ++row) {
        for (size_m col = 0; col < w; ++col) {
            size_m outRow = clockwise ? col : (w - 1 - col);
            size_m outCol = clockwise ? (h - 1 - row) : row;
            const uint8_t* s = src + (size_t)(row * w + col) * channels;
            uint8_t* d = dst + (size_t)(outRow * h + outCol) * channels;
            for (size_m c = 0; c < channels; ++c) d[c] = s[c];
        }
    }
    return rotated;
}

// Returns a [H,W,4] (or [W,H,4] if rotated) UInt8 RGBA image: grayscale depth
// as the base, path/ground pixels tinted green. Matches the UInt8 [H,W,4]
// convention depth_streamer's own `image` variable already uses for a
// displayable frame - if updateBaseImageV2: (or whatever you display this
// with) actually expects a different dtype/layout, tell me and I'll adjust
// rather than guessing twice.
//
// `intrinsics`/`referenceWidth`/`referenceHeight` are unused for now - back
// on the placeholder XY range below (not real metric unprojection) since
// that one's confirmed giving good RANSAC results, per the "unprojecting is
// causing issues right now" call. Left in the signature so nothing else
// needs to change when real intrinsics-based unprojection comes back.
inline matrix ground_overlay_image(
    matrix depth,
    const matrix_float3x3& intrinsics,
    float referenceWidth = 1920.0f,
    float referenceHeight = 1080.0f,
    int iterations = 200,
    float distanceThreshold = 0.06f,
    bool rotateClockwise = true
) {
    size_m h = depth.shape()[0];
    size_m w = depth.shape()[1];
    size_t n = (size_t)h * (size_t)w;

    matrix depthScaled = (depth.astype(dtype::Float) * -5);

    matrix lineY = matrix::linespace(5.0f, -5.0f, h);
    matrix lineX = matrix::linespace(-5.0f, 5.0f, w);
    auto mesh = matrix::meshgrid(lineX, lineY);
    matrix X = mesh.first;
    matrix Y = mesh.second;

    matrix Z_flat = depthScaled.flatten();
    matrix points = matrix::stack({X.flatten(), Y.flatten(), Z_flat}, -1);
    std::vector<Vec3> pts = extract_points(points);
    std::vector<uint8_t> inlierMask;
    ransac_ground_plane(pts, inlierMask, iterations, distanceThreshold);

    // grayscale base uses the original raw depth, not the *-5-scaled copy
    // used above for RANSAC, so brightness isn't computed off flipped values
    matrix depthF = depth.astype(dtype::Float);
    depthF.eval();
    const float* depthRaw = static_cast<const float*>(depthF.buffer);

    float minD = 1e9f, maxD = -1e9f;
    for (size_t i = 0; i < n; ++i) {
        float d = depthRaw[i];
        if (d < minD) minD = d;
        if (d > maxD) maxD = d;
    }
    float range = (maxD - minD) > 1e-6f ? (maxD - minD) : 1.0f;

    matrix image = matrix::withShape({h, w, 4}, dtype::UInt8);
    image.begin_refcount();
    uint8_t* out = static_cast<uint8_t*>(image.buffer);

    for (size_t i = 0; i < n; ++i) {
        uint8_t gray = (uint8_t)(255.0f * (depthRaw[i] - minD) / range);
        uint8_t* px = out + i * 4;
        if (inlierMask[i]) {
            px[0] = gray / 3; px[1] = 200; px[2] = gray / 3; // path/ground: green tint
        } else {
            px[0] = gray; px[1] = gray; px[2] = gray; // everything else: plain grayscale
        }
        px[3] = 255;
    }

    return rotate90(image, rotateClockwise);
}

} // namespace fovea

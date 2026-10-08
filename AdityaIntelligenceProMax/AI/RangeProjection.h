#pragma once
// LiDAR scan <-> SalsaNext's range image.
//
// project(): points -> the network's input, exactly as SalsaNext / RangeNet++ build it
// (train/common/laserscan.py do_range_projection + the parser's normalisation):
//   - spherical projection: column from yaw, row from pitch within [fov_down, fov_up], floor + clamp
//   - when several points land on one pixel, the closest one wins
//   - 5 channels (range, x, y, z, remission), each (value - mean) / std, then 0 where no point landed
// labels_for_points(): the network's per-pixel classes -> one class per point (each point takes
// its own pixel's argmax; that's also what SalsaNext does without its optional KNN clean-up).
//
// macOS only (MLX).

#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE

#include "SalsaNext.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace ai {

struct RangeImage {
    mx::array input = mx::array(0.0f);     // [1, H, W, 5] normalised network input
    int height = 0, width = 0;
    std::vector<int32_t> px, py;           // per point: its pixel (-1 if it has no direction, i.e. at the origin)
    std::vector<int32_t> pixel_point;      // per pixel: index of the point shown there, -1 = empty
};

inline RangeImage project(const float* xyz, const float* remission, size_t n, const SalsaNext& net) {
    const int H = net.height, W = net.width;
    const float fov_up = net.fov_up / 180.0f * (float)M_PI;
    const float fov_down = net.fov_down / 180.0f * (float)M_PI;
    const float fov = std::fabs(fov_down) + std::fabs(fov_up);

    RangeImage ri;
    ri.height = H;
    ri.width = W;
    ri.px.assign(n, -1);
    ri.py.assign(n, -1);
    ri.pixel_point.assign((size_t)H * W, -1);
    std::vector<float> best((size_t)H * W, INFINITY);

    for (size_t i = 0; i < n; ++i) {
        float x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
        float depth = std::sqrt(x * x + y * y + z * z);
        if (!(depth > 0.0f)) continue;              // no direction: can't be projected
        float yaw = -std::atan2(y, x);
        float pitch = std::asin(z / depth);
        float fx = 0.5f * (yaw / (float)M_PI + 1.0f) * W;
        float fy = (1.0f - (pitch + std::fabs(fov_down)) / fov) * H;
        int cx = std::clamp((int)std::floor(fx), 0, W - 1);
        int cy = std::clamp((int)std::floor(fy), 0, H - 1);
        ri.px[i] = cx;
        ri.py[i] = cy;
        size_t p = (size_t)cy * W + cx;
        // Closest wins (the original sorts by decreasing depth and lets later writes overwrite).
        if (depth < best[p]) {
            best[p] = depth;
            ri.pixel_point[p] = (int32_t)i;
        }
    }

    std::vector<float> img((size_t)H * W * 5, 0.0f);
    for (size_t p = 0; p < (size_t)H * W; ++p) {
        int32_t i = ri.pixel_point[p];
        // The original's mask is `proj_idx > 0`, so a pixel held by point 0 counts as empty too.
        if (i <= 0) continue;
        float v[5] = {best[p], xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2], remission[i]};
        for (int c = 0; c < 5; ++c) img[p * 5 + c] = (v[c] - net.means[c]) / net.stds[c];
    }
    ri.input = mx::array(img.data(), {1, H, W, 5}, mx::float32);
    return ri;
}

// probs: the network's [1, H, W, classes] output. Returns one class (0..classes-1) per point;
// points with no pixel get 0 (unlabelled).
inline std::vector<int32_t> labels_for_points(const mx::array& probs, const RangeImage& ri) {
    mx::array am = mx::astype(mx::argmax(probs, -1), mx::int32);   // [1, H, W]
    mx::eval(am);
    const int32_t* pix = am.data<int32_t>();
    std::vector<int32_t> out(ri.px.size(), 0);
    for (size_t i = 0; i < out.size(); ++i) {
        if (ri.px[i] >= 0) out[i] = pix[(size_t)ri.py[i] * ri.width + ri.px[i]];
    }
    return out;
}

// probs: [1, H, W, classes]. Per point, the probability of class `cls` at its pixel (0 if no pixel).
inline std::vector<float> class_probability_for_points(const mx::array& probs, const RangeImage& ri, int cls) {
    mx::array p = mx::slice(probs, {0, 0, 0, cls}, {1, ri.height, ri.width, cls + 1});
    p = mx::flatten(mx::contiguous(p));   // the slice is a strided view: copy before reading it flat
    mx::eval(p);
    const float* pix = p.data<float>();
    std::vector<float> out(ri.px.size(), 0.0f);
    for (size_t i = 0; i < out.size(); ++i) {
        if (ri.px[i] >= 0) out[i] = pix[(size_t)ri.py[i] * ri.width + ri.px[i]];
    }
    return out;
}

// Whether point i is the one its pixel shows. Points that lost their pixel to a closer point only
// inherit that point's class (e.g. the wall behind a car becomes "car"), so per-object work such as
// bounding boxes should use pixel owners only. SalsaNext's optional KNN step fixes those labels.
inline bool owns_pixel(const RangeImage& ri, size_t i) {
    return ri.px[i] >= 0 && ri.pixel_point[(size_t)ri.py[i] * ri.width + ri.px[i]] == (int32_t)i;
}

} // namespace ai

#endif // !TARGET_OS_IPHONE

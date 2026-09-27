// Checks AI/Kitti.h (reading a SemanticKITTI sequence) and AI/RangeProjection.h (scan <-> SalsaNext
// range image) against the reference make_test_sequence.py builds with SalsaNext's own projection
// code, then runs the network on the synthetic scan. Build and run from this folder:
//
//   xcrun clang++ -std=c++20 -O2 -I../../AdityaIntelligenceProMax/AI -I/usr/local/include test_kitti_projection.cpp \
//     -L/usr/local/lib -lmlx -Wl,-rpath,/usr/local/lib -framework Metal -framework Foundation \
//     -framework Accelerate -framework QuartzCore -o /tmp/test_kitti_projection && /tmp/test_kitti_projection <sequence dir>
#include "Kitti.h"
#include "RangeProjection.h"
#include <chrono>
#include <iostream>
#include <map>

namespace mx = mlx::core;
static int g_pass = 0, g_fail = 0;
static void report(const std::string& l, bool ok) { std::cout << (ok ? "[PASS] " : "[FAIL] ") << l << std::endl; (ok ? g_pass : g_fail)++; }

int main(int argc, char** argv) {
    const std::string seq_dir = argc > 1 ? argv[1] : ".";
    const std::string pretrained = "/Users/adityadude/Documents/C++XCODE/pretrained/";

    // ---- loader ----
    kitti::Sequence seq(seq_dir);
    report("sequence: 2 scans, labels and poses found", seq.size() == 2 && seq.has_labels() && seq.has_poses());
    kitti::Scan scan = seq.load(0);
    std::map<int, size_t> ids;
    for (uint16_t l : scan.labels) ids[l]++;
    report("scan 0: ~120k points, labels with the instance bits stripped (40/50/80/252 only)",
           scan.size() > 100000 && ids.size() == 4 && ids.count(40) && ids.count(50) && ids.count(80) && ids.count(252));

    auto ref = mx::load_safetensors(seq_dir + "/reference.safetensors").first;
    mx::array ref_pose = ref.at("lidar_pose1");
    mx::eval(ref_pose);
    simd_float4x4 pose = seq.pose(1);
    float pose_err = 0;
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
        pose_err = std::max(pose_err, std::fabs(pose.columns[c][r] - ref_pose.data<float>()[r * 4 + c]));
    report("pose of scan 1 converted from camera to LiDAR frame (1 m forward)", pose_err < 1e-5f && std::fabs(pose.columns[3][0] - 1.0f) < 1e-5f);

    report("class mapping: road 40 -> 9, building 50 -> 13, pole 80 -> 18, moving-car 252 -> car 1",
           kitti::to_train_class(40) == 9 && kitti::to_train_class(50) == 13 && kitti::to_train_class(80) == 18 &&
           kitti::to_train_class(252) == 1 && kitti::is_moving(252) && !kitti::is_moving(10));
    report("groups: road -> terrain, building -> static, car -> dynamic",
           kitti::group_of(9) == kitti::Group::Terrain && kitti::group_of(13) == kitti::Group::Static &&
           kitti::group_of(1) == kitti::Group::Dynamic && kitti::group_of(0) == kitti::Group::Unknown);

    // ---- projection vs SalsaNext's own laserscan.py ----
    ai::SalsaNext net(pretrained + "salsanext.safetensors");
    auto t0 = std::chrono::steady_clock::now();
    ai::RangeImage ri = ai::project(scan.xyz.data(), scan.remission.data(), scan.size(), net);
    mx::eval(ri.input);
    double proj_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    mx::array ref_in = ref.at("input"), ref_x = ref.at("proj_x"), ref_y = ref.at("proj_y");
    mx::eval(ref_in, ref_x, ref_y);
    float in_err = mx::max(mx::abs(mx::subtract(ri.input, ref_in))).item<float>();
    size_t pix_mismatch = 0;
    for (size_t i = 0; i < scan.size(); i++)
        pix_mismatch += (ri.px[i] != ref_x.data<int32_t>()[i]) || (ri.py[i] != ref_y.data<int32_t>()[i]);
    report("network input matches SalsaNext's projection (max |diff| " + std::to_string(in_err) + ")", in_err < 1e-4f);
    report("every point lands on the same pixel as in SalsaNext's code (" + std::to_string(pix_mismatch) + " differ)", pix_mismatch == 0);

    // ---- back to points: fake per-pixel classes (px + py) % 20 as one-hot probabilities ----
    std::vector<float> onehot((size_t)ri.height * ri.width * 20, 0.0f);
    for (int y = 0; y < ri.height; y++) for (int x = 0; x < ri.width; x++) onehot[((size_t)y * ri.width + x) * 20 + (x + y) % 20] = 1.0f;
    std::vector<int32_t> back = ai::labels_for_points(mx::array(onehot.data(), {1, ri.height, ri.width, 20}, mx::float32), ri);
    size_t back_bad = 0;
    for (size_t i = 0; i < back.size(); i++) back_bad += back[i] != (ri.px[i] + ri.py[i]) % 20;
    report("each point gets its own pixel's class", back_bad == 0);

    // ---- the real network on the synthetic scan (information, not a pass/fail) ----
    mx::array probs = net(ri.input);
    mx::eval(probs);
    t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 5; k++) { probs = net(ri.input); mx::eval(probs); }
    double net_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
    std::vector<int32_t> pred = ai::labels_for_points(probs, ri);
    std::cout << "\nSalsaNext on the synthetic scene (projection " << proj_ms << " ms, network " << net_ms << " ms):\n";
    const char* groups[] = {"unknown", "terrain", "static", "dynamic"};
    for (int truth : {40, 50, 80, 252}) {
        size_t total = 0, same_group = 0;
        std::map<int, size_t> top;
        for (size_t i = 0; i < scan.size(); i++) {
            if (scan.labels[i] != truth) continue;
            total++;
            top[pred[i]]++;
            same_group += kitti::group_of(pred[i]) == kitti::group_of(kitti::to_train_class(truth));
        }
        auto best = std::max_element(top.begin(), top.end(), [](auto& a, auto& b) { return a.second < b.second; });
        std::cout << "  " << net.class_names[kitti::to_train_class(truth)] << " (" << total << " pts): "
                  << 100.0 * same_group / total << "% in the right group ("
                  << groups[(int)kitti::group_of(kitti::to_train_class(truth))] << "), most predicted as "
                  << net.class_names[best->first] << "\n";
    }
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed" << std::endl;
    return g_fail ? 1 : 0;
}

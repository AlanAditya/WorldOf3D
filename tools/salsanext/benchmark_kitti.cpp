// End-to-end benchmark on a SemanticKITTI sequence: the numbers PS 26053 asks for.
//   load scan -> range image -> SalsaNext (MLX) -> class per point -> foveated 2.5D grid -> tiles
// Reports accuracy against the labels (3 groups and 20 classes, IoU, by distance), time per stage and
// FPS, and the grid's memory next to a uniform grid's. Build and run from this folder:
//
//   xcrun clang++ -std=c++20 -O2 -I../../AdityaIntelligenceProMax/AI -I/usr/local/include benchmark_kitti.cpp \
//     -L/usr/local/lib -lmlx -Wl,-rpath,/usr/local/lib -framework Metal -framework Foundation \
//     -framework Accelerate -framework QuartzCore -o /tmp/benchmark_kitti && /tmp/benchmark_kitti <sequence dir> [max scans]
#include "FoveatedGrid.h"
#include "Kitti.h"
#include "RangeProjection.h"
#include <chrono>
#include <cstdio>
#include <iostream>

namespace mx = mlx::core;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t).count() / 1000.0; }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: benchmark_kitti <sequence dir> [max scans]\n"; return 1; }
    kitti::Sequence seq(argv[1]);
    size_t scans = std::min(seq.size(), argc > 2 ? (size_t)std::stoul(argv[2]) : seq.size());
    ai::SalsaNext net("/Users/adityadude/Documents/C++XCODE/pretrained/salsanext.safetensors");
    foveated::Grid grid;

    const int C = net.classes;
    std::vector<uint64_t> cls_inter(C), cls_uni(C);
    uint64_t g_inter[4] = {}, g_uni[4] = {}, g_ok = 0, c_ok = 0, scored = 0;
    const float bands[] = {10, 20, 40, 100};                     // the grid's ring edges
    uint64_t band_ok[5] = {}, band_n[5] = {};
    double t_load = 0, t_proj = 0, t_net = 0, t_back = 0, t_grid = 0, t_tiles = 0;
    size_t occupied = 0, points = 0;

    // Warm-up: the first call compiles the network.
    { kitti::Scan s = seq.load(0); auto ri = ai::project(s.xyz.data(), s.remission.data(), s.size(), net); mx::eval(net(ri.input)); }

    for (size_t k = 0; k < scans; ++k) {
        auto t0 = clk::now();
        kitti::Scan scan = seq.load(k);
        t_load += ms_since(t0);

        t0 = clk::now();
        ai::RangeImage ri = ai::project(scan.xyz.data(), scan.remission.data(), scan.size(), net);
        mx::eval(ri.input);
        t_proj += ms_since(t0);

        t0 = clk::now();
        mx::array probs = net(ri.input);
        mx::eval(probs);
        t_net += ms_since(t0);

        t0 = clk::now();
        std::vector<int32_t> pred = ai::labels_for_points(probs, ri);
        std::vector<uint8_t> cls(scan.size()), grp(scan.size());
        for (size_t i = 0; i < scan.size(); ++i) { cls[i] = (uint8_t)pred[i]; grp[i] = (uint8_t)kitti::group_of(pred[i]); }
        t_back += ms_since(t0);

        t0 = clk::now();
        grid.clear();
        grid.insert(scan.xyz.data(), scan.size(), grp.data(), cls.data());
        t_grid += ms_since(t0);
        t0 = clk::now();
        foveated::Mesh mesh = foveated::tiles(grid, [](const foveated::Cell&, float* c) { c[0] = c[1] = c[2] = c[3] = 1; });
        t_tiles += ms_since(t0);
        occupied += mesh.indices.size() / 6;
        points += scan.size();

        for (size_t i = 0; i < scan.size() && !scan.labels.empty(); ++i) {
            int tc = kitti::to_train_class(scan.labels[i]);
            if (tc == 0) continue;                                   // unlabelled: not scored (SemanticKITTI's rule)
            int pc = pred[i];
            for (int c = 1; c < C; ++c) { cls_inter[c] += (pc == c && tc == c); cls_uni[c] += (pc == c || tc == c); }
            int tg = (int)kitti::group_of(tc), pg = (int)kitti::group_of(pc);
            for (int g = 1; g < 4; ++g) { g_inter[g] += (pg == g && tg == g); g_uni[g] += (pg == g || tg == g); }
            g_ok += pg == tg;
            c_ok += pc == tc;
            ++scored;
            float d = std::max(std::fabs(scan.xyz[i * 3]), std::fabs(scan.xyz[i * 3 + 1]));
            int b = 0;
            while (b < 4 && d >= bands[b]) ++b;
            band_ok[b] += pg == tg;
            band_n[b]++;
        }
        if ((k + 1) % 50 == 0) { std::fprintf(stderr, "  %zu/%zu\n", k + 1, scans); }
    }

    const double n = (double)scans;
    std::printf("\n=== %zu scans, %.0f points/scan, %.0f occupied grid cells/scan ===\n", scans, points / n, occupied / n);
    std::printf("\naccuracy (%llu labelled points)\n", (unsigned long long)scored);
    std::printf("  3 groups: %.2f%%   IoU terrain %.2f%%, static %.2f%%, dynamic %.2f%%\n", 100.0 * g_ok / scored,
                100.0 * g_inter[1] / g_uni[1], 100.0 * g_inter[2] / g_uni[2], 100.0 * g_inter[3] / g_uni[3]);
    double miou = 0;
    int present = 0;
    for (int c = 1; c < C; ++c) if (cls_uni[c]) { miou += (double)cls_inter[c] / cls_uni[c]; ++present; }
    std::printf("  20 classes: accuracy %.2f%%, mIoU %.2f%% (over the %d classes present)\n", 100.0 * c_ok / scored, 100.0 * miou / present, present);
    for (int c = 1; c < C; ++c)
        if (cls_uni[c]) std::printf("    %-14s IoU %6.2f%%\n", net.class_names[c].c_str(), 100.0 * cls_inter[c] / cls_uni[c]);
    const char* band_name[] = {"0-10 m (5 cm cells)", "10-20 m (10 cm)", "20-40 m (20 cm)", "40-100 m (40 cm)", "beyond 100 m"};
    std::printf("  3-group accuracy by distance (grid rings):\n");
    for (int b = 0; b < 5; ++b)
        if (band_n[b]) std::printf("    %-20s %6.2f%%  (%.1f%% of points)\n", band_name[b], 100.0 * band_ok[b] / band_n[b], 100.0 * band_n[b] / scored);

    double total = (t_load + t_proj + t_net + t_back + t_grid + t_tiles) / n;
    std::printf("\ntime per scan (ms, average)\n");
    std::printf("  load %.2f | range image %.2f | SalsaNext %.2f | labels to points %.2f | 2.5D grid %.2f | tiles %.2f\n",
                t_load / n, t_proj / n, t_net / n, t_back / n, t_grid / n, t_tiles / n);
    std::printf("  total %.2f ms = %.1f FPS (the LiDAR runs at 10 Hz)\n", total, 1000.0 / total);

    std::printf("\nmemory\n");
    std::printf("  foveated 2.5D grid: %zu cells x %zu B = %.2f MB\n", grid.cell_count(), sizeof(foveated::Cell), grid.bytes() / 1e6);
    std::printf("  uniform 5 cm 2.5D grid, same 200 m x 200 m: %zu cells = %.1f MB (%.1fx more)\n", grid.uniform_cell_count(),
                grid.uniform_cell_count() * sizeof(foveated::Cell) / 1e6, (double)grid.uniform_cell_count() / grid.cell_count());
    std::printf("  uniform 5 cm 3D voxels, 10 m tall, 1 B each: %.2f GB (%.0fx more)\n", grid.uniform_cell_count() * 200 / 1e9,
                grid.uniform_cell_count() * 200.0 / grid.bytes());
    return 0;
}

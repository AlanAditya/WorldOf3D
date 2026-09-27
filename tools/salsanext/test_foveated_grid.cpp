// Checks AI/FoveatedGrid.h (the variable-resolution 2.5D grid), then bins real KITTI scans labelled
// by SalsaNext into it. Build and run from this folder:
//
//   xcrun clang++ -std=c++20 -O2 -I../../AdityaIntelligenceProMax/AI -I/usr/local/include test_foveated_grid.cpp \
//     -L/usr/local/lib -lmlx -Wl,-rpath,/usr/local/lib -framework Metal -framework Foundation \
//     -framework Accelerate -framework QuartzCore -o /tmp/test_foveated_grid && /tmp/test_foveated_grid [kitti sequence dir]
#include "FoveatedGrid.h"
#include "Kitti.h"
#include "RangeProjection.h"
#include <chrono>
#include <iostream>
#include <random>

static int g_pass = 0, g_fail = 0;
static void report(const std::string& l, bool ok) { std::cout << (ok ? "[PASS] " : "[FAIL] ") << l << std::endl; (ok ? g_pass : g_fail)++; }
static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count() / 1000.0;
}

int main(int argc, char** argv) {
    foveated::Grid grid;

    // ---- layout ----
    report("4 rings, 610,000 cells (vs 16,000,000 for uniform 5 cm over 200 m x 200 m)",
           grid.rings().size() == 4 && grid.cell_count() == 610000 && grid.uniform_cell_count() == 16000000);
    double area = 0;
    bool roundtrip = true;
    for (size_t i = 0; i < grid.cell_count(); ++i) {
        float x0, y0, s;
        grid.cell_bounds(i, x0, y0, s);
        area += (double)s * s;
        if (grid.cell_index(x0 + s * 0.5f, y0 + s * 0.5f) != (long)i) { roundtrip = false; break; }
    }
    // Relative tolerance: 0.05f etc. aren't exact in binary, so summing 610,000 squares rounds a little.
    report("cells tile the 200 m square exactly (total area " + std::to_string(area) + " m^2)", std::fabs(area - 40000.0) / 40000.0 < 1e-6);
    report("every cell's centre maps back to that cell (no two cells overlap, none is unreachable)", roundtrip);

    // ---- every point goes into a cell that contains it, including points on ring edges ----
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-99.999f, 99.999f);
    std::vector<float> probes;
    for (int i = 0; i < 1000000; ++i) { probes.push_back(u(rng)); probes.push_back(u(rng)); }
    for (float e : {10.0f, 20.0f, 40.0f}) for (float sx : {-1.0f, 1.0f}) for (float t : {-e, -e * 0.5f, 0.0f, e * 0.3f, e}) {
        probes.insert(probes.end(), {sx * e, t, t, sx * e, std::nextafter(sx * e, 0.0f), t});
    }
    size_t outside = 0;
    for (size_t k = 0; k < probes.size(); k += 2) {
        float x = probes[k], y = probes[k + 1], x0, y0, s;
        long c = grid.cell_index(x, y);
        if (c < 0) { ++outside; continue; }
        grid.cell_bounds((size_t)c, x0, y0, s);
        const float eps = 1e-4f;
        if (x < x0 - eps || x > x0 + s + eps || y < y0 - eps || y > y0 + s + eps) ++outside;
    }
    report("1,000,000 random points + ring-edge points each land in a cell that contains them", outside == 0);
    report("cell size grows with distance: 5 cm at 3 m, 10 cm at 15 m, 20 cm at 30 m, 40 cm at 90 m", [&] {
        float sizes[4], at[4] = {3, 15, 30, 90};
        for (int k = 0; k < 4; k++) { float x0, y0; grid.cell_bounds((size_t)grid.cell_index(at[k], 0), x0, y0, sizes[k]); }
        return std::fabs(sizes[0] - 0.05f) < 1e-6f && std::fabs(sizes[1] - 0.1f) < 1e-6f && std::fabs(sizes[2] - 0.2f) < 1e-6f && std::fabs(sizes[3] - 0.4f) < 1e-6f;
    }());
    report("points beyond 100 m are dropped, not wrapped", grid.cell_index(100.5f, 0) == -1 && grid.cell_index(0, -150) == -1);

    // ---- cell contents ----
    grid.clear();
    float pts[] = {5.02f, 1.01f, -1.7f,   5.03f, 1.02f, 0.4f,   5.04f, 1.03f, -0.2f};   // one 5 cm cell
    uint8_t grp[] = {1, 3, 2}, cls[] = {9, 6, 13};                                        // road, person, building
    grid.insert(pts, 3, grp, cls);
    const foveated::Cell& c = grid.cells()[(size_t)grid.cell_index(5.02f, 1.01f)];
    report("cell keeps min/max height, count, and the highest-priority class (person over building over road)",
           c.count == 3 && c.zmin == -1.7f && c.zmax == 0.4f && c.group == 3 && c.cls == 6 && grid.occupied() == 1);

    // ---- real scans: SalsaNext labels -> grid ----
    if (argc > 1) {
        kitti::Sequence seq(argv[1]);
        ai::SalsaNext net("/Users/adityadude/Documents/C++XCODE/pretrained/salsanext.safetensors");
        std::cout << "\nreal scans from " << argv[1] << ":\n";
        for (size_t s = 0; s < seq.size(); ++s) {
            kitti::Scan scan = seq.load(s);
            ai::RangeImage ri = ai::project(scan.xyz.data(), scan.remission.data(), scan.size(), net);
            std::vector<int32_t> pred = ai::labels_for_points(net(ri.input), ri);
            std::vector<uint8_t> g(scan.size()), k(scan.size());
            for (size_t i = 0; i < scan.size(); ++i) { k[i] = (uint8_t)pred[i]; g[i] = (uint8_t)kitti::group_of(pred[i]); }

            auto t0 = std::chrono::steady_clock::now();
            grid.clear();
            size_t inside = grid.insert(scan.xyz.data(), scan.size(), g.data(), k.data());
            double bin_ms = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            foveated::Mesh mesh = foveated::tiles(grid, [](const foveated::Cell&, float* rgba) { rgba[0] = rgba[1] = rgba[2] = rgba[3] = 1; });
            double mesh_ms = ms_since(t0);

            size_t occ = grid.occupied(), per_ring[4] = {};
            for (size_t i = 0; i < grid.cell_count(); ++i) {
                if (grid.cells()[i].empty()) continue;
                for (size_t r = 0; r < grid.rings().size(); ++r)
                    if (i < grid.rings()[r].offset + grid.rings()[r].count) { per_ring[r]++; break; }
            }
            std::cout << "  scan " << s << ": " << scan.size() << " points, " << inside << " inside 100 m -> " << occ
                      << " occupied cells (5 cm: " << per_ring[0] << ", 10 cm: " << per_ring[1] << ", 20 cm: " << per_ring[2]
                      << ", 40 cm: " << per_ring[3] << "); binning " << bin_ms << " ms, tiles " << mesh_ms << " ms ("
                      << mesh.indices.size() / 6 << ")\n";
            size_t within = 0;
            for (size_t i = 0; i < scan.size(); ++i)
                within += std::max(std::fabs(scan.xyz[i * 3]), std::fabs(scan.xyz[i * 3 + 1])) < grid.extent();
            report("scan " + std::to_string(s) + ": every point within 100 m binned (" + std::to_string(within) +
                   "), one tile per occupied cell", inside == within && mesh.indices.size() / 6 == occ);
        }
        std::cout << "  memory: grid " << grid.bytes() / 1e6 << " MB (" << grid.cell_count() << " cells x 12 B) vs uniform 5 cm 2.5D grid "
                  << grid.uniform_cell_count() * 12 / 1e6 << " MB, vs uniform 5 cm 3D voxels (10 m tall, 1 byte each) "
                  << grid.uniform_cell_count() * 200 / 1e9 << " GB\n";
    }

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed" << std::endl;
    return g_fail ? 1 : 0;
}

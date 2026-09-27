// Checks AdityaIntelligenceProMax/AI/SalsaNext.h (C++ MLX) against the PyTorch golden output that
// convert_salsanext.py writes, and times the forward pass. Build and run from this folder:
//
//   xcrun clang++ -std=c++20 -O2 -I../../AdityaIntelligenceProMax/AI -I/usr/local/include test_salsanext.cpp \
//     -L/usr/local/lib -lmlx -Wl,-rpath,/usr/local/lib -framework Metal -framework Foundation \
//     -framework Accelerate -framework QuartzCore -o /tmp/test_salsanext && /tmp/test_salsanext
#include "SalsaNext.h"
#include <chrono>
#include <iostream>

namespace mx = mlx::core;

int main() {
    const std::string dir = "/Users/adityadude/Documents/C++XCODE/pretrained/";
    ai::SalsaNext net(dir + "salsanext.safetensors");
    std::cout << "loaded: " << net.height << "x" << net.width << ", " << net.classes << " classes, first "
              << net.class_names[1] << " / last " << net.class_names.back() << "\n";

    auto golden = mx::load_safetensors(dir + "salsanext_golden.safetensors").first;
    mx::array x = golden.at("input"), ref = golden.at("output");

    mx::array out = net(x);
    mx::eval(out);
    float diff = mx::max(mx::abs(mx::subtract(out, ref))).item<float>();
    float agree = mx::mean(mx::equal(mx::argmax(out, -1), mx::argmax(ref, -1))).item<float>();
    std::cout << "compiled C++ vs PyTorch: max |diff| " << diff << ", argmax agreement " << agree * 100 << "%\n";

    mx::array out2 = net.forward(x);
    mx::eval(out2);
    std::cout << "uncompiled C++ vs PyTorch: max |diff| " << mx::max(mx::abs(mx::subtract(out2, ref))).item<float>() << "\n";

    // Timing: warm up, then average a few runs at the full 64 x 2048 input.
    for (int i = 0; i < 3; i++) mx::eval(net(x));
    const int runs = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; i++) mx::eval(net(x));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;
    std::cout << "forward pass: " << ms << " ms (" << 1000.0 / ms << " FPS)\n";
    return diff < 1e-3f && agree > 0.9999f ? 0 : 1;
}

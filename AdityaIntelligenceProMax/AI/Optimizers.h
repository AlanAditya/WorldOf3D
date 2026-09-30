#pragma once
// Optimizers for the matrix engine, mirroring mlx.optimizers:
//
//     auto loss_and_grad_fn = matrix::value_and_grad(loss_fn);
//     optim::Adam optimizer(0.01f);
//
//     auto [loss, grads] = loss_and_grad_fn(params);
//     optimizer.update(params, grads);
//     matrix::eval(params, optimizer.state);
//
// update() only builds the new parameters and state (new matrices, nothing written in place);
// matrix::eval materialises them and drops the graph that made them.
#include "matrix.h"
#include <cmath>
#include <vector>

namespace optim {

// Adam, same update and defaults as mlx.optimizers.Adam (bias correction off unless asked for):
//     m = b1 * m + (1 - b1) * g
//     v = b2 * v + (1 - b2) * g^2
//     p = p - lr * m / (sqrt(v) + eps)
class Adam {
public:
    float learning_rate;
    float beta1, beta2, eps;
    bool bias_correction;
    std::vector<matrix> state;    // first moments, then second moments, one per parameter
    int step = 0;

    explicit Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                  bool bias_correction = false)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), bias_correction(bias_correction) {}

    void update(std::vector<matrix>& params, const std::vector<matrix>& grads) {
        size_t n = params.size();
        if (state.size() != 2 * n) {
            state.clear();
            for (auto& p : params) state.push_back(p * 0.0f);   // m
            for (auto& p : params) state.push_back(p * 0.0f);   // v
        }
        step++;
        float lr = learning_rate;
        if (bias_correction) {
            lr *= std::sqrt(1.0f - std::pow(beta2, (float)step)) / (1.0f - std::pow(beta1, (float)step));
        }
        for (size_t i = 0; i < n; i++) {
            matrix& m = state[i];
            matrix& v = state[n + i];
            m = m * beta1 + grads[i] * (1.0f - beta1);
            v = v * beta2 + grads[i] * grads[i] * (1.0f - beta2);
            params[i] = params[i] - m * lr / (matrix::sqrt(v) + eps);
        }
    }
};

} // namespace optim

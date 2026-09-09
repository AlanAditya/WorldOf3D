//
// Created by Aditya Dudeja on 13/08/26.
//

#include <random>

#include "matrix.h"

matrix Ed(matrix x, int iter) {
    matrix y = x.zeros();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < iter; i++) {
        matrix temp = x.zeros();
        int* temp_buff = (int*)temp.buffer;

        for (int k = 0; k < x.shape()[0]; k++) {
            for (int j = 0; j < x.at<int>(k); j++) {
                temp_buff[k] += dis(gen) == 0 ? -1 : 1;
            }

        }
        y = y + temp;
    }

    return (y.astype(dtype::Float) / iter);
}

matrix Ed2(matrix x, int iter) {
    matrix y = x.zeros();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < iter; i++) {
        matrix temp = x.zeros();
        int* temp_buff = (int*)temp.buffer;

        for (int k = 0; k < x.shape()[0]; k++) {
            for (int j = 0; j < x.at<int>(k); j++) {
                temp_buff[k] += dis(gen) == 0 ? -1 : 1;
            }

        }
        y = y + (temp * temp);
    }

    return (y.astype(dtype::Float) / iter);
}

matrix Ed3(matrix x, float start) {
    matrix y = x.zeros();

    y.at<float>(0) = start;

    for (int i = 1; i < x.shape()[0]; i++) {
        y.at<float>(i) = 2.0f - 1.0f / y.at<float>(i - 1);
    }

    return y;
}

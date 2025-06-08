//
//  Utilities.cpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 23/12/24.
//

#include "Utilities.hpp"
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <simd/simd.h>


void generateMatrixConst(float* mat, int m, int n, float num) {
    for (int i = 0; i < m * n; ++i) {
        mat[i] = num;
    }
}

void generateMatrixRange(float* mat, int m, int n, float num) {
    for (int i = 0; i < m * n; ++i) {
        mat[i] = num + i;
    }
}


void generateMatrixRandFloat(float* mat, int m, int n, float minVal, float maxVal) {
    std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (int i = 0; i < m * n; ++i) {
        mat[i] = dis(gen);
    }
}

void transpose(float* mat, int m, int n) {
    float temp[m * n];
    memcpy(temp, mat, sizeof(float) * m * n);
    for (int i = 0; i < m*n; i++) {
        temp[i] = mat[i];
    }
    
    for (int i =0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[j * m + i] = temp[i * n + j];
        }
    }
}

std::ostream& operator<<(std::ostream& os, const simd::float3& matrix) {
    std::cout << "{ " << matrix.x << ", " << matrix.y << ", " << matrix.z << " }";
    return os;
}
std::ostream& operator<<(std::ostream& os, const simd::float2& matrix) {
    std::cout << "{ " << matrix.x << ", " << matrix.y  << " }";
    return os;
}

//template void generateMatrixRandInt<int>(int* mat, int m, int n, int minVal, int maxVal) ;
//template void generateMatrixRandInt<float>(float* mat, int m, int n, int minVal, int maxVal) ;

void transposeThreaded(float* src, float* dst, int rows, int cols, int numThreads) {
    auto worker = [&](int startRow, int endRow) {
        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    };

    std::vector<std::thread> threads;
    int rowsPerThread = rows / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? rows : startRow + rowsPerThread;
        threads.emplace_back(worker, startRow, endRow);
    }

    for (auto& th : threads) {
        th.join();
    }
}

void print(float* matrix, int m, int n) {
    std::cout << "\nMatrix is \n";
    for (int i= 0; i < m; i++) {
        for (int j= 0; j< n; j++) {
            std::cout << matrix[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}

simd_float4x4 arrayToSIMDMatrix4x4(float* array) {
    simd_float4x4 matrix;
    
    // Fill the simd_float4x4 with values from the 4x4 array
    matrix.columns[0] = simd_make_float4(array[0], array[4], array[8], array[12]);
    matrix.columns[1] = simd_make_float4(array[1], array[5], array[9], array[13]);
    matrix.columns[2] = simd_make_float4(array[2], array[6], array[10], array[14]);
    matrix.columns[3] = simd_make_float4(array[3], array[7], array[11], array[15]);

    return matrix;
}

std::ostream& operator<<(std::ostream& os, const simd::float4x4& matrix) {
    int lenC = 4;
    int lenR = 4;
    for (int row = 0; row < lenR; ++row) {
        for (int col = 0; col < lenC; ++col) {
            os << matrix.columns[col][row] << ", ";
            if (col == lenC-1) {
                os << "\n";
            }
        }
    }
    return os;
}

simd::float4x4 Identity() {
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Translation(simd::float3 dPos) {
    
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {dPos[0], dPos[1], dPos[2], 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationZ(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, sin, 0.0f, 0.0f};
    simd_float4 row1 = {-sin, cos, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationY(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, 0.0f, sin, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {-sin, 0.0f, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationX(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, cos, -sin, 0.0f};
    simd_float4 row2 = {0.0f, sin, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Scale(simd::float3 scale) {
    simd_float4 row0 = {scale.x, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, scale.y, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, scale.z, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

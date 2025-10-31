//
//  Utilities.hpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 23/12/24.
//

#ifndef Utilities_hpp
#define Utilities_hpp

#include <stdio.h>
#include <iostream>
#include <random>
#include <simd/simd.h>
#include <chrono>

void generateMatrixConst(float* mat, int m, int n, float num);
void generateMatrixRandFloat(float* mat, int m, int n, float minVal, float maxVal);
void transpose(float* mat, int m, int n);
void transposeThreaded(float* src, float* dst, int rows, int cols, int numThreads);
void print(float* matrix, int m, int n);
void generateMatrixRange(float* mat, int m, int n, float num);
simd_float4x4 arrayToSIMDMatrix4x4(float* array);

template <typename T>
void generateMatrixRandInt(T* mat, int m, int n, int minVal, int maxVal) {
    std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_int_distribution<int> dis(minVal, maxVal);

    for (int i = 0; i < m * n; ++i) {
        mat[i] = (dis(gen));
    }
}

void matrix_multiply_neon(float* A, float* B, float* C, int m, int p, int n);

std::ostream& operator<<(std::ostream& os, const simd::float3& matrix);
std::ostream& operator<<(std::ostream& os, const simd::float4x4& matrix);
std::ostream& operator<<(std::ostream& os, const simd::float2& matrix);

class Timer {
public:
     Timer() {
        m_startTime = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        Stop();
    }
    
    void Stop() {
        auto endPointTime = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_startTime).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endPointTime).time_since_epoch().count();
        auto duration = end - start;
        double ms = duration * 0.001;
        std::cout << "It took " << duration << " ns " << ms << " ms " <<  "\n";
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

simd::float4x4 Identity();

simd::float4x4 Translation(simd::float3 dPos);

simd::float4x4 RotationZ(float theta);

simd::float4x4 RotationY(float theta);

simd::float4x4 RotationX(float theta);

simd::float4x4 Scale(simd::float3 scale);


#endif /* Utilities_hpp */

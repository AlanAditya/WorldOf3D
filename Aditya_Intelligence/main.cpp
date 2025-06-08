//
//  main.cpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 20/12/24.
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <iostream>
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <arm_neon.h>
#include <type_traits>
#include "Utilities.hpp"
#include <random>
#include <Accelerate/Accelerate.h>
#include "Algebro.hpp"

//size_t total_alloc = 0;
//void* operator new(size_t size) {
//    total_alloc += size;
//    printf("Allocated %l \n", size);
//    return malloc(size);
//}

void matrixMultiplyAccelerate(const float* A, const float* B, float* C, int m, int k, int n);
simd::float4x4 MatMal(simd::float4x4 A, simd::float4x4 B);
float dot(const float* a, const float* b, int n);


std::ostream& operator<<(std::ostream& os, const simd::float3x3& matrix) {
    int lenC = 3;
    int lenR = 3;
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

//std::ostream& operator<<(std::ostream& os, float matrix[3][3]) {
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            os << matrix[i][j] << " ";
//        }
//        os << "\n";  // Newline after each row
//    }
//    return os;
//}


template <size_t Rows, size_t Cols>
std::ostream& operator<<(std::ostream& os, float (&matrix)[Rows][Cols]) {
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            os << matrix[i][j] << " ";
        }
        os << "\n";  // Newline after each row
    }
    return os;
}



template <std::size_t N>
//typename std::enable_if<!std::is_same<T, char>::value, std::ostream&>::type
std::ostream& operator<<(std::ostream& os, const float (&arr)[N]) {
    os << "[";
    for (std::size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i < N - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}





namespace simd_extensions {
    // Converts simd::float3x3 to a normal 2D array using a provided pointer to the normal array
    inline void to2DArray(const simd::float4x4& matrix, float (*normalArray)[4]) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                normalArray[row][col] = matrix.columns[col][row];
            }
        }
    }
}
void Matmul_2(float (*A)[4], float (*B)[4], float (*result)[4]) {
    int lenR = sizeof(*result) / sizeof(*result[0]);
    int lenC = sizeof(result[0]) / sizeof(result[0][0]);
    int CommonDim = 4;
    std::cout << lenC << lenR << "\n";
    for (int row = 0; row < lenR; row++) {
        for (int col = 0; col < lenC; col++) {
            for (int k = 0; k < CommonDim; k++) {
                result[row][col] += A[row][k] * B[k][col];
            }
//            result[row][col] = dot(A[row], A[row], CommonDim);
        }
    }
}

void matrix_multiply_neon(float* A, float* B, float* C, int m, int p, int n) {
    for (int i = 0; i < m; i++) {
        // Iterate over columns of B
        for (int j = 0; j < n; j++) {
            // Initialize sum to zero
            float32x4_t sum = vdupq_n_f32(0.0f);

            // Compute dot product for 4 elements at a time
            int k;
            for (k = 0; k <= p - 4; k += 4) {
                // Load 4 elements from row of A
                float32x4_t a_vec = vld1q_f32(&A[i * p + k]);

                // Load 4 elements from column of B
                float32x4_t b_vec = vld1q_f32(&B[j * p + k]);

                // Multiply and accumulate
                sum = vmlaq_f32(sum, a_vec, b_vec);
            }
            
            // Horizontally add the vector sum to get scalar result
            float temp[4];
            vst1q_f32(temp, sum);
            C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle remaining elements
            for (; k < p; k++) {
                C[i * n + j] += A[i * p + k] * B[i * p + k];
            }
        }
    }
}

void matrix_multiply_neonFaster(float* A, float* B, float* C, int m, int p, int n) {
    for (int i = 0; i < m; i++) {
        // Iterate over columns of B
        for (int j = 0; j < n; j++) {
            // Initialize sum to zero
            float32x4_t sum = vdupq_n_f32(0.0f);

            // Compute dot product for 4 elements at a time
            int k;
            for (k = 0; k <= p - 4; k += 4) {
                // Load 4 elements from row of A
                float32x4_t a_vec = vld1q_f32(&A[i * p + k]);

                // Load 4 elements from column of B
                float32x4_t b_vec = vld1q_f32(&B[j * p + k]);

                // Multiply and accumulate
                sum = vmlaq_f32(sum, a_vec, b_vec);
            }
            
            // Horizontally add the vector sum to get scalar result
            C[i * n + j] = vaddvq_f32(sum);

            // Handle remaining elements
            for (; k < p; k++) {
                C[i * n + j] += A[i * p + k] * B[i * p + k];
            }
        }
    }
}

void matrix_multiply_neonStride(float* A, float* B, float* C, int m, int p, int n) {
    // Iterate over row of A
    for (int i = 0; i < m; i++) {
        // Iterate over columns of B
        for (int j = 0; j < n; j++) {
            // Initialize sum to zero
            float32x4_t sum = vdupq_n_f32(0.0f);

            // Compute dot product for 4 elements at a time
            int k;
            for (k = 0; k <= p - 4; k += 4) {
                // Load 4 elements from row of A
                float32x4_t a_vec = vld1q_f32(&A[i * p + k]);
                
                
                // Load 4 elements from column of B
                float32x4_t b_vec = vdupq_n_f32(0.0f);
                b_vec = vld1q_lane_f32(&B[k * p + j], b_vec, 0);
                b_vec = vld1q_lane_f32(&B[k * p + j + n], b_vec, 1);
                b_vec = vld1q_lane_f32(&B[k * p + j + 2*n], b_vec, 2);
                b_vec = vld1q_lane_f32(&B[k * p + j + 3*n], b_vec, 3);
                
                // Multiply and accumulate
                sum = vmlaq_f32(sum, a_vec, b_vec);
            }
            
            // Horizontally add the vector sum to get scalar result
//            float temp[4];
//            vst1q_f32(temp, sum);
//            C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];
            C[i * n + j] = vaddvq_f32(sum);

            // Handle remaining elements
            for (; k < p; k++) {
                C[i * n + j] += A[i * p + k] * B[k * p + j];
            }
        }
    }
}


void Matmul_4(float (*A)[3], float (*B)[3], float (*result)[3]) {
    int lenR = sizeof(*result) / sizeof(*result[0]);
    int lenC = sizeof(result[0]) / sizeof(result[0][0]);
    int CommonDim = 3;
    
    std::cout << lenC << lenR << "\n";
    for (int row = 0; row < lenR; row++) {
        for (int col = 0; col < lenC; col++) {
            for (int k = 0; k < CommonDim; k++) {
                result[row][col] += A[row][k] * B[k][col];
            }
        }
    }
}

// https://stackoverflow.com/questions/62729814/matrix-multiplication-using-simd-vectors-in-c


int main(int argc, const char * argv[]) {
//    float* k = new float[30];
    trial();
//    memset(k, 4, 30);
//    simd::float4x4 A = simd::float4x4(
//        simd::make_float4(1.0f, 2.0f, 3.0f, 3.0f),
//        simd::make_float4(4.0f, 5.0f, 6.0f, 6.0f),
//        simd::make_float4(7.0f, 8.0f, 9.0f, 9.0f),
//        simd::make_float4(7.0f, 8.0f, 9.0f, 9.0f)
//    );
//    
//    simd::float4x4 B = simd::float4x4(
//        simd::make_float4(9.0f, 8.0f, 7.0f, 7.0f),
//        simd::make_float4(6.0f, 5.0f, 4.0f, 4.0f),
//        simd::make_float4(3.0f, 2.0f, 1.0f, 1.0f),
//        simd::make_float4(3.0f, 2.0f, 1.0f, 1.0f)
//    );
//    
//    constexpr int m = 40;
//    constexpr int p = 40;
//    constexpr int n = 40;
//    
//
////    Matrix mat = Matrix<m, p>::constant(1);
//    
//    
//    float a[m][p];
//    float b[p][n];
//    
//
//    
////    simd_extensions::to2DArray(A, a);
////    simd_extensions::to2DArray(B, b);
//    float c[5][5];
//    generateMatrixConst(c[0], 5, 5, 1.0f);
//    generateMatrixRandInt(c[0], 5, 5, 0, 10);
//    
//    float flattenedA[m*p];
//    float flattenedB[p*n];
//    float flattenedB_T[p*n];
//    float flattenedC[m*n];
//    float flattenedD[m*n];
//    
//    
//    
//    
//    generateMatrixRandInt(flattenedA, m, p, 0, 100);
//    generateMatrixRandInt(flattenedB, p, n, 0, 100);
//    generateMatrixRange(flattenedB, p,n, 0);
//    {
//        Timer time;
//        matrix_multiply_neonStride(flattenedA, flattenedB, flattenedC, m, p, n);
//    }
////    print(flattenedC, m, n);
//    
//    {
//        Timer time;
//        transpose(flattenedB, p, n);
//        matrix_multiply_neon(flattenedA, flattenedB, flattenedC, m, p, n);
//    }
//    
//    {
//        Timer time;
//        transpose(flattenedB, p, n);
//        matrix_multiply_neonFaster(flattenedA, flattenedB, flattenedC, m, p, n);
//    }
////    print(flattenedC, m, n);
//    {
//        Timer time;
//        matrixMultiplyAccelerate(flattenedA, flattenedB, flattenedD, m, p, n);
//    }
    
//    A = arrayToSIMDMatrix4x4(flattenedA);
//    B = arrayToSIMDMatrix4x4(flattenedB);
//    simd_float4x4 C;
//    {
//        Timer time;
//        C = A * B;
//    }
    
    
//    generateMatrixRandInt(flattenedB, 4, 4, 0, 100);
//    generateMatrixConst(flattenedC, 4, 4, 0);
//    int index = 0;
//
//    for (int i = 0; i < 4; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            flattenedA[index++] = a[i][j];
//            flattenedB[index++] = b[i][j];
//            flattenedC[index++] = c[i][j];
//        }
//    }
////    std::cout << ax << "\n";
//    std::cout << flattenedA << "\n";
//    std::cout << flattenedB << "\n";
//    std::cout << flattenedC << "\n";
//    {
//        Timer time;
//        Matmul_2(a, b, c);
//    }
//
//    {
//        Timer time;
//        simd::float4x4 C = MatMal(A, B);
////        std::cout << C << "\n";
//    }
//
//
//    {
//        Timer time;
//        transpose(&flattenedB[0], 4, 5);
//    }
//   
//    {
//        Timer time;
//        matrix_multiply_neon(flattenedA, flattenedB, flattenedC, 5, 4, 5);
//    }
//    std::cout << flattenedC << "\n";
//    std::cout << flattenedB << "\n";
////    transpose(&flattenedB[0], 5, 4);
//    
//    {
//        Timer time;
//        matrixMultiplyAccelerate(flattenedA, flattenedB, &c[0][0], 5, 4, 5);
//    }
//    std::cout << c << "\n";
//
//
//
////    std::cout << C;
//    std::cout << c;
    
    
    return 0;
}

simd::float4x4 MatMal(simd::float4x4 A, simd::float4x4 B) {
    return simd_mul(A, B);
}

//simd::Matrix<float, 3, 3> MatMal_3(
//    const simd::Matrix<float, 3, 3> A,
//    const simd::Matrix<float, 3, 3> B) {
//
//    simd::Matrix<float, 3, 3> result = {}; // Initialize result to 0
//    for (size_t row = 0; row < 9; ++row) {
//        for (size_t col = 0; col < 9; ++col) {
//            float sum = 0.0f;
//            A.Matrix::
//            for (size_t k = 0; k < 3; ++k) {
//
//                sum += A[row][k] * B[k][col];
//            }
//            result[row][col] = sum;
//        }
//    }
//    return result;
//}

float dot(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i <= n-4; i +=4 ) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }
    float result[4];
    vst1q_f32(result, sum);
    float dot_product = result[0] + result[1] + result[2] + result[3];
    for (; i < n; i++) {
        dot_product += a[i] * b[i];
    }

    return dot_product;
}

void matrixMultiplyAccelerate(const float* A, const float* B, float* C, int m, int k, int n) {
    /*
     * A: Input matrix of size m x k (row-major)
     * B: Input matrix of size k x n (row-major)
     * C: Output matrix of size m x n (row-major)
     * m: Number of rows in A
     * k: Number of columns in A (and rows in B)
     * n: Number of columns in B
     */

    // Since Accelerate uses column-major format, we need to pass transposed matrices.
    // To multiply row-major matrices with Accelerate, use C = A × B as:
    // C = Bᵀ × Aᵀ and then transpose the result to get the correct order.

    // Perform matrix multiplication: C = A × B
    // Inputs:
    // - m: Rows in A
    // - k: Columns in A
    // - n: Columns in B
    // - Transpose options:
    //   CblasNoTrans (A and B are used as-is in column-major order)
    // - Leading dimensions (lda, ldb, ldc):
    //   - lda = max(1, rows in A) = m
    //   - ldb = max(1, rows in B) = k
    //   - ldc = max(1, rows in C) = m

    cblas_sgemm(CblasRowMajor,        // Row-major matrix layout
                CblasNoTrans,         // A is not transposed
                CblasNoTrans,         // B is not transposed
                m, n, k,              // Dimensions: A (m x k), B (k x n), C (m x n)
                1.0f,                 // Alpha: Scalar multiplier for A × B
                A, k,                 // Matrix A and its leading dimension
                B, n,                 // Matrix B and its leading dimension
                0.0f,                 // Beta: Scalar multiplier for C (initial values)
                C, n);                // Matrix C and its leading dimension
}

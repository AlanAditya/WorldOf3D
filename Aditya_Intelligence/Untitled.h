
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
#include <cstring> // For memset



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
std::ostream& operator<<(std::ostream& os, float matrix[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            os << matrix[i][j] << " ";
        }
        os << "\n";  // Newline after each row
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, float matrix[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
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
        std::cout << "It took " << duration << " us" << "\n";
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

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
    // Iterate over rows of A
    for (int i = 0; i < m; i++) {
        // Iterate over columns of B
        for (int j = 0; j < n; j++) {
            // Initialize sum to zero
            float32x4_t sum = vdupq_n_f32(0.0f);

            // Compute dot product for 4 elements at a time
            int k;
            for (k = 0; k <= p - 4; k += 4) {
                
                float tmp[4]; // Temporary array to hold col elements
                for (int col = 0; col < 4; ++col) { // Assuming 4 columns to fit into vector
                    tmp[col] = B[(k+col)*n + j]; // Access column-major layout
                }

                
                // Load 4 elements from col of B
                float32x4_t b_vec = vld1q_f32(&tmp[0]);

                // Load 4 elements from row of A
                float32x4_t a_vec = vld1q_f32(&A[i*p + k]);

                // Multiply and accumulate
                sum = vmlaq_f32(sum, a_vec, b_vec);
                
            }

            // Horizontally add the vector sum to get scalar result
            float tmp[4];
            vst1q_f32(tmp, sum);
            C[i * m + j] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            // Handle remaining elements
            for (; k < p; k++) {
                C[i * m + j] += A[i * p + k] * B[k * n + j];
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
    
    
    
//    float a[4][4];
//    float b[4][4];
//
//    A = simd::transpose(A);
//    B = simd::transpose(B);
//    simd_extensions::to2DArray(A, a);
//    simd_extensions::to2DArray(B, b);
    float c[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},{0.0f, 0.0f, 0.0f, 0.0f},{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.f}};
    
    float flattenedA[] = { 1,  8,  6,  9,  9,  5,  1,  0,  8,  9,  0, 10,  5,  5,  7,  3,  7,
                           0,  8,  6,  7,  0,  6,  9, 10,  8,  0,  0,  7,  8, 10,  3,  3,  0,
                           3,  9,  6, 10,  2,  0,  4,  8, 10,  9,  8,  9,  3,  4,  7,  8,  7,
                           4,  5,  3,  8,  0,  2,  9, 10,  8,  3, 10,  0,  4,  8,  1, 10,  4,
                           0,  5,  9,  3,  7,  4,  7,  7,  4,  0,  9,  0,  5};
    float flattenedB[] = { 1,  8,  6,  9,  9,  5,  1,  0,  8,  9,  0, 10,  5,  5,  7,  3,  7,
                           0,  8,  6,  7,  0,  6,  9, 10,  8,  0,  0,  7,  8, 10,  3,  3,  0,
                           3,  9,  6, 10,  2,  0,  4,  8, 10,  9,  8,  9,  3,  4,  7,  8,  7,
                           4,  5,  3,  8,  0,  2,  9, 10,  8,  3, 10,  0,  4,  8,  1, 10,  4,
                           0,  5,  9,  3,  7,  4,  7,  7,  4,  0,  9,  0,  5};
    float flattenedC[81];
    int index = 0;

//    for (int i = 0; i < 4; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            flattenedA[i * 4 + j] = a[i][j];
//            flattenedB[i * 4 + j] = b[i][j];
//            flattenedC[i * 4 + j] = c[i][j];
//        }
//    }
//    std::cout << a << "\n";
//    std::cout << b << "\n";
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
//        std::cout << C << "\n";
//    }

    
    {
        Timer time;
        matrix_multiply_neon_optimized(flattenedA, flattenedB, flattenedC, 9, 9, 9);
    }
    std::cout << flattenedC << "\n";
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

//for (int l = 0; l< 4; l++) {
//    std::cout << A[i * p + k + l] << " ";
//    
//}
//std::cout << "      ";
//for (int l = 0; l< 4; l++) {
//    std::cout << B[j * p + k + l] << " ";
//    
//}
//std::cout  << "   ";

template <typename T>
MatrixH<T> convC(MatrixH<T> kernel) {
    if (kernel.shape[0] % 2 == 0 | kernel.shape[1] % 2 == 0) {
        std::cerr << "Kernel of wrong shape" << std::endl;
    }
    int paddingX = (kernel.shape[1] - 1) / 2;
    int paddingY = (kernel.shape[0] - 1) / 2;
    
    int newW = shape[1] + 2*paddingX;
    int newH = shape[0] + 2*paddingY;
    T* paddedValues = new T[newH * newW * 4];
    
    for (int i = 0; i < paddingY; i++) {
        for (int j = 0; j < newW; j++) {
            paddedValues[4 * (i*newW + j) + 0] = 255;
            paddedValues[4 * (i*newW + j) + 1] = 0;
            paddedValues[4 * (i*newW + j) + 2] = 0;
            paddedValues[4 * (i*newW + j) + 3] = 255;
            
            paddedValues[4 * ((shape[0] + paddingY + i)*newW + j) + 0] = 255;
            paddedValues[4 * ((shape[0] + paddingY + i)*newW + j) + 1] = 0;
            paddedValues[4 * ((shape[0] + paddingY + i)*newW + j) + 2] = 0;
            paddedValues[4 * ((shape[0] + paddingY + i)*newW + j) + 3] = 255;
        }
    }
    
    for (int i = 0; i < paddingX; i++) {
        for (int j = 0; j < shape[0] + 2*paddingY; j++) {
            paddedValues[4 * (j*newW + i) + 0] = 0;
            paddedValues[4 * (j*newW + i) + 1] = 255;
            paddedValues[4 * (j*newW + i) + 2] = 0;
            paddedValues[4 * (j*newW + i) + 3] = 255;
            
            paddedValues[4 * (j*newW + (paddingX + shape[1]) + i) + 0] = 0;
            paddedValues[4 * (j*newW + (paddingX + shape[1]) + i) + 1] = 255;
            paddedValues[4 * (j*newW + (paddingX + shape[1]) + i) + 2] = 0;
            paddedValues[4 * (j*newW + (paddingX + shape[1]) + i) + 3] = 255;
        }
    }
    
    for (int i = 0; i < shape[0]; i++) {
        memcpy(paddedValues + (4 * (paddingY+i) * newW) + 4*paddingX, values + 4*i*shape[1], shape[1] * 4);
    }
    
    auto result = MatrixH<uint8_t>();
    result.values = paddedValues;
    result.shape = std::vector<size_t>({(shape[0] + 2*paddingY) , (shape[1] + 2*paddingX), 4});
    result.total_size = (shape[0] + 2*paddingY) * (shape[1] + 2*paddingX) * 4;
    return result;
}

//void swapD(int dim, int s0, int s1) {
//    if (s0 > shape[dim] && s1 > shape[dim]) {
//        fprintf(stderr, "MATRIX: exceding dimension of  %lu not able to use.\n", shape[dim]);
//        return;
//    }
////        int stride = std::accumulate(shape.begin() + dim, shape.end(), 1, std::multiplies<size_t>());
//    int stride = accMul(shape, dim, shape.size());
////        int blockSize = std::accumulate(shape.begin() + dim + 1, shape.end(), 1, std::multiplies<size_t>());
//    int blockSize = accMul(shape, dim+1, shape.size());
//    int noOfOpp = accMul(shape, 0, dim);
//    if (dim == 0) {stride  = blockSize; }
//    Type* temp = new Type[blockSize];
////        Type temp;
////        std::cout << "noOfop " << std::accumulate(shape.begin(), shape.begin() + dim, 1, std::multiplies<size_t>()) <<" strdide " << stride << " size" << blockSize << " " << sizeof(Type) << " size"<<total_size <<"\n";
//    
//    for (int i = 0; i < noOfOpp; i++) {
//        memcpy(temp, values + (i * stride + blockSize*s0), sizeof(Type)* blockSize);
//        memcpy(values + (i * stride + blockSize*s0), values + (i * stride + blockSize*s1), sizeof(Type) * blockSize);
//        memcpy(values + (i * stride + blockSize*s1), temp, sizeof(Type) * blockSize);
////            temp = values[i * stride + s0];
////            values[i * stride + s0] = values[i * stride + s1];
////            values[i * stride + s1] = temp;
//    }
//    delete [] temp;
//}

//void cvtColor_BGRA2RGBA() {
//    uint8_t* src;
//    int height = shape[0]; int width = shape[1];
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            src = values + 4*(i * width + j);
////                for (int i = 0; i < n; i++) {
//                uint8_t b = *(src);
//                uint8_t g = *(src+1);
//                uint8_t r = *(src+2);
//                uint8_t a = *(src+3);
//                *(src) = r;
//                *(src+1) = g;
//                *(src+2) = b;
//                *(src+3) = a;
////                }
//        }
//    }
//}

//void cvtColor_BGRA2RGBA() {
//    for (int i = 0; i < total_size / 4; i++) {
//        uint8_t b = values[4*i];
//        uint8_t g = values[4*i+1];
//        uint8_t r = values[4*i+2];
//        uint8_t a = values[4*i+3];
//        values[4*i] = r;
//        values[4*i+1] = g;
//        values[4*i+2] = b;
//        values[4*i+3] = a;
//    }
//}

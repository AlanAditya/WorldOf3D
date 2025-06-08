//
//  test.cpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 27/12/24.
//

#include "test.hpp"

int Hello(){
    return 0;
}
//void matrix_multiply_neon(float* A, float* B, float* C, int m, int p, int n) {
////    std::cout << "A: "<< A[4];
//    // Iterate over rows of A
//    for (int i = 0; i < m; i++) {
//        // Iterate over columns of B
//        for (int j = 0; j < n; j++) {
//            // Initialize sum to zero
//            float32x4_t sum = vdupq_n_f32(0.0f);
//
//            // Compute dot product for 4 elements at a time
//            int k;
//            for (k = 0; k <= p - 4; k += 4) {
//                float tmp[4]; // Temporary array to hold col elements
//                for (int col = 0; col < 4; ++col) { // Assuming 4 columns to fit into vector
//                    tmp[col] = B[(k+col)*n + j]; // Access column-major layout
//                }
//                // Load 4 elements from row of A
//                float32x4_t a_vec = vld1q_f32(&A[i * p + k]);
//
//                // Load 4 elements from column of B
//                float32x4_t b_vec = vld1q_f32(tmp);
//
//                // Multiply and accumulate
//                sum = vmlaq_f32(sum, a_vec, b_vec);
//            }
//
//            // Horizontally add the vector sum to get scalar result
//            float temp[4];
//            vst1q_f32(temp, sum);
//            C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];
//
//            // Handle remaining elements
//            for (; k < p; k++) {
//                C[i * n + j] += A[i * p + k] * B[k * n + j];
//            }
//        }
//    }
//}

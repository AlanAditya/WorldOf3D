//
//  Algebro.cpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 03/02/25.
//

#include "Algebro.hpp"
#include <iostream>
#include <arm_neon.h>
//#include "Utilities.hpp"
#include "AlgebroHeap.hpp"
#include <simd/simd.h>
#include <dispatch/dispatch.h>
#include "MetalLayerHPPextension.hpp"
#include <CoreMedia/CoreMedia.h>



template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::Range(float start) {
    Matrix<M, N> mat;
    for (int i = 0; i < M*N; i++) {
        mat.values[i] = start + i;
    }
    return mat; // RVO ensures no copy
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator+(const Matrix<M, N>& other) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] + other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        float32x4_t b_vec = vld1q_f32(&other.values[i]);

        // Add the vectors element-wise
        float32x4_t sum = vaddq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }

//     Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] + other.values[i];
    }
    std::cout << result.rows << result.cols    << "lenth" << "\n";
    return result;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator-(const Matrix<M, N>& other) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] - other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        float32x4_t b_vec = vld1q_f32(&other.values[i]);

        // Add the vectors element-wise
        float32x4_t sum = vsubq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }

    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] - other.values[i];
    }
    return result;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator*(const Matrix<M, N>& other) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] * other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        float32x4_t b_vec = vld1q_f32(&other.values[i]);

        // Add the vectors element-wise
        float32x4_t sum = vmulq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }
    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] * other.values[i];
    }
    return result;
}


template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator/(const Matrix<M, N>& other) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] / other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        float32x4_t b_vec = vld1q_f32(&other.values[i]);

        // Add the vectors element-wise
        float32x4_t sum = vdivq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }
    
    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] / other.values[i];
    }
    return result;
}


template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator+(const float scalar) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] * other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    float32x4_t b_vec = vdupq_n_f32(scalar);
    
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        

        // Add the vectors element-wise
        float32x4_t sum = vaddq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }
    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] + scalar;
    }
    return result;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator-(const float scalar) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] * other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    float32x4_t b_vec =  vdupq_n_f32(scalar);
    
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        

        // Add the vectors element-wise
        float32x4_t sum = vsubq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }
    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] - scalar;
    }
    return result;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator*(const float scalar) const {
    Matrix<M, N> result;
//    for (size_t i = 0; i < M * N; i++) {
//        result.values[i] = values[i] * other.values[i];
//    }
//    return result;
    // Use NEON to add 4 elements at a time
    float32x4_t b_vec =  vdupq_n_f32(scalar);
    
    size_t i;
    for (i = 0; i +4 < M * N; i += 4) {
        // Load 4 elements from the current matrix and the other matrix
        float32x4_t a_vec = vld1q_f32(&values[i]);
        

        // Add the vectors element-wise
        float32x4_t sum = vmulq_f32(a_vec, b_vec);
        // Store the result back to the result matrix
        vst1q_f32(&result.values[i], sum);

    }
    // Handle remaining elements if M * N is not divisible by 4
    for (; i < M * N; i++) {
        result.values[i] = values[i] * scalar;
    }
    return result;
}

template <size_t m, size_t p>
template <size_t n>
Matrix<m, n> Matrix<m, p>::dot(Matrix<p, n>& B) {
    Matrix<m, n> C;
    for (int i = 0; i < m; i++) {
        // Iterate over columns of B
        for (int j = 0; j < n; j++) {
            // Initialize sum to zero
            float32x4_t sum = vdupq_n_f32(0.0f);

            // Compute dot product for 4 elements at a time
            int k;
            for (k = 0; k <= p - 4; k += 4) {
                // Load 4 elements from row of A
                float32x4_t a_vec = vld1q_f32(&values[i * p + k]);

                // Load 4 elements from column of B
                float32x4_t b_vec = vld1q_f32(&B.values[j * p + k]);

                // Multiply and accumulate
                sum = vmlaq_f32(sum, a_vec, b_vec);
            }
            
            // Horizontally add the vector sum to get scalar result
            float temp[4];
            vst1q_f32(temp, sum);
            C.values[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle remaining elements
            for (; k < p; k++) {
                C.values[i * n + j] += values[i * p + k] * B.values[i * p + k];
            }
        }
    }
    return C;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::operator/(const float scalar) const {
    Matrix<M, N> result;
//        for (size_t i = 0; i < M * N; i++) {
//            result.values[i] = values[i] * other.values[i];
//        }
//        return result;
        // Use NEON to add 4 elements at a time
        float32x4_t b_vec =  vdupq_n_f32(scalar);
        
        size_t i;
        for (i = 0; i +4 < M * N; i += 4) {
            // Load 4 elements from the current matrix and the other matrix
            float32x4_t a_vec = vld1q_f32(&values[i]);
            

            // Add the vectors element-wise
            float32x4_t sum = vdivq_f32(a_vec, b_vec);
            // Store the result back to the result matrix
            vst1q_f32(&result.values[i], sum);

        }
        // Handle remaining elements if M * N is not divisible by 4
        for (; i < M * N; i++) {
            result.values[i] = values[i] / scalar;
        }
        return result;
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::ones() {
    Matrix<M, N> mat;
    std::fill(mat.values, mat.values + M * N, 1.0f);
    return mat; // RVO ensures no copy
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::zeros() {
    Matrix<M, N> mat;
    std::fill(mat.values, mat.values + M * N, 0.0f);
    return mat; // RVO ensures no copy
}

template <size_t M, size_t N>
Matrix<M*N, 1> Matrix<M, N>::flatten() {
    Matrix<M*N, 1> mat;
    memcpy(mat.values, values, M*N*sizeof(float));
    return mat; // RVO ensures no copy
}

template <size_t M, size_t N>
template <size_t P, size_t Q>
Matrix<P, Q> Matrix<M, N>::reshape() {
    Matrix<P, Q> mat;
    static_assert(M * N == P * Q, "Matrix dimensions must match for reshape!");
    memcpy(mat.values, values, M*N*sizeof(float));
    return mat; // RVO ensures no copy
}

template <size_t M, size_t N>
void Matrix<M, N>::print() {
    std::cout << "\nMatrix is \n";
    for (int i= 0; i < rows; i++) {
        for (int j= 0; j< cols; j++) {
            std::cout << values[i*cols + j] << " ";
        }
        std::cout << "\n";
    }
}

template <size_t M, size_t N>
Matrix<M, N> Matrix<M, N>::constant(float constant) {
    Matrix<M, N> mat;
    std::fill(mat.values, mat.values + M * N, constant);
    return mat; // RVO ensures no copy
}


template <typename T>
auto make_matrix(std::initializer_list<T> list) {
//    return MatrixH<typename std::underlying_type<T>::type>(list);
}

void printPos(simd_float2 pos) {
    std::cout << pos << "\n";
}

void trial() {
    constexpr int m = 5;
    constexpr int p = 4;
    constexpr int n = 5;
    
    
    
//    Matrix mat = Matrix<m, p>::constant(1);
//    float matOLD[m*n];
//    generateMatrixConst(matOLD, m, p, 1);
//    Matrix mat2 = Matrix<p, n>::constant(3);
//    float matOLD1[p*n];
//    generateMatrixConst(matOLD1, p, n, 3);
//    mat2.print();
//    mat.print();
//    auto mat5 = Matrix<30, 60>::constant(4);
//    auto mat4 = mat5.reshape<mat5.cols, mat5.rows>();
//    mat4.print();
    
    
//    float buffer[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
//    MatrixH Mat = MatrixH<float>::fromBuffer(buffer, std::vector<size_t>({3,3}));
//    
//    Mat.print();
    
//    std::initializer_list<std::initializer_list<float>> num  = {
//        {0.000035, 0.000049, 0.000065, 0.000082, 0.000100, 0.0117, 0.0130, 0.0137, 0.000130, 0.000117, 0.000100, 0.000082, 0.000065, 0.000049, 0.000035},
//        {0.000049, 0.000070, 0.000093, 0.000117, 0.000144, 0.0168, 0.0187, 0.0197, 0.000187, 0.000168, 0.000144, 0.000117, 0.000093, 0.000070, 0.000049},
//        {0.000065, 0.000093, 0.000123, 0.000155, 0.000191, 0.0222, 0.0247, 0.0260, 0.000247, 0.000222, 0.000191, 0.000155, 0.000123, 0.000093, 0.000065},
//        {0.000082, 0.000117, 0.000155, 0.000196, 0.000240, 0.0278, 0.0309, 0.0325, 0.000309, 0.000278, 0.000240, 0.000196, 0.000155, 0.000117, 0.000082},
//        {0.000100, 0.000144, 0.000191, 0.000240, 0.000294, 0.0340, 0.0377, 0.0397, 0.000377, 0.000340, 0.000294, 0.000240, 0.000191, 0.000144, 0.000100},
//        {0.000117, 0.000168, 0.000222, 0.000278, 0.000340, 0.0393, 0.0436, 0.0459, 0.000436, 0.000393, 0.000340, 0.000278, 0.000222, 0.000168, 0.000117},
//        {0.000130, 0.000187, 0.000247, 0.000309, 0.000377, 0.0436, 0.0484, 0.0510, 0.000484, 0.000436, 0.000377, 0.000309, 0.000247, 0.000187, 0.000130},
//        {0.000137, 0.000197, 0.000260, 0.000325, 0.000397, 0.0459, 0.0510, 0.0537, 0.000510, 0.000459, 0.000397, 0.000325, 0.000260, 0.000197, 0.000137},
//        {0.000130, 0.000187, 0.000247, 0.000309, 0.000377, 0.0436, 0.0484, 0.0510, 0.000484, 0.000436, 0.000377, 0.000309, 0.000247, 0.000187, 0.000130},
//        {0.000117, 0.000168, 0.000222, 0.000278, 0.000340, 0.0393, 0.0436, 0.0459, 0.000436, 0.000393, 0.000340, 0.000278, 0.000222, 0.000168, 0.000117},
//        {0.000100, 0.000144, 0.000191, 0.000240, 0.000294, 0.0340, 0.0377, 0.0397, 0.000377, 0.000340, 0.000294, 0.000240, 0.000191, 0.000144, 0.000100},
//        {0.000082, 0.000117, 0.000155, 0.000196, 0.000240, 0.0278, 0.0309, 0.0325, 0.000309, 0.000278, 0.000240, 0.000196, 0.000155, 0.000117, 0.000082},
//        {0.000065, 0.000093, 0.000123, 0.000155, 0.000191, 0.0222, 0.0247, 0.0260, 0.000247, 0.000222, 0.000191, 0.000155, 0.000123, 0.000093, 0.000065},
//        {0.000049, 0.000070, 0.000093, 0.000117, 0.000144, 0.0168, 0.0187, 0.0197, 0.000187, 0.000168, 0.000144, 0.000117, 0.000093, 0.000070, 0.000049},
//        {0.000035, 0.000049, 0.000065, 0.000082, 0.000100, 0.0117, 0.0130, 0.0137, 0.000130, 0.000117, 0.000100, 0.000082, 0.000065, 0.000049, 0.000035}
//    };
    
//    sleep(2);
//    MatrixH<uint8_t> frame = MatrixH<uint8_t>::fromImage();
//    {
//        Timer time;
//        frame.cvtColor_BGRA2RGBA();
//    }
//    {
//        Timer time;
//        frame.swapD(2, 0, 2);
//    }
//    frame.showImage("Image");
//    frame.cvWaitKey(0);
    
    MatrixH<uint8_t> frame = MatrixH<uint8_t>::fromImage();
    frame.showImage("fr");
    frame.cvWaitKey(0);
//    frame.test();
    
    auto cap = CamReaderWrapper(0);
    auto rend = RendererHPP();
////    auto rend = MetalWrapper();
//    
    
//    MatrixH<uint8_t> renderframe;
    int i = 0;
    MatrixH<int> detections;
//    
//    auto rectangle = NGon(0.1, 40);
//    rend.objectQueue.push_back(rectangle);
//    
//    
//    rend.objectQueue.push_back(rectangle);
//    
//    auto tt = TriangleH(1);
//    rend.objectQueue.push_back(tt);
//    
//    
//    tt.position += 0.5;
    

    
    while (true) {
        cap.retrieveFrame(frame);
        frame.FaceDetect(detections);
        frame.HandsDetect(detections, true);
//
//        
        if (detections.values) {
            if (detections.shape[0] > 0) {
                simd_float2 normInd = frame.Normalise(simd_make_int2(*detections(0, 1, 0), *detections(0, 1, 1)));

                normInd.y = 1-normInd.y;
                
                normInd = (normInd * 2) - 1;
                
                simd_float2 normThumb = frame.Normalise(simd_make_int2(*detections(0, 0, 0), *detections(0, 0, 1)));
                normThumb.y = 1-normThumb.y;
                
                normThumb = (normThumb * 2) - 1;
                
                frame.drawLine(0, simd_make_int2(*detections(0, 1, 0), *detections(0, 1, 1)), 0, RED);
//
//                rend.objectQueue[rend.objectQueue.size()-1].position = simd_make_float3(normInd, 1.0);
//                rend.objectQueue[rend.objectQueue.size()-1].scale =  simd_length(normInd - normThumb);
                
            }
//            
            if (detections.shape[0] == 2) {

                frame.drawRect(simd_make_int2(*detections(0, 1, 0), *detections(0, 1, 1)), simd_make_int2(*detections(1, 1, 0), *detections(1, 1, 1)), {255, 255, 255, 255});
                
            }
//            
            frame.drawHands(detections, true);
//            
        }
////        
////        
        rend.Render(frame);
        frame.cvtColor_BGRA2RGBA();
////        
        frame.flip(1);
////        renderframe.flip(1);
////        frame.Translate2D_H(0, 400 * sin(i* 0.1), 300 * sin(i * 0.1));
////        frame.drawRectStroked({100, 100, 300, 300}, {(uint8_t)(255 * sin(i* 0.1)) , (uint8_t)(255 * sin(i* 0.1)), 255, 255}, 0.01);
////        frame.drawText("Dil Hareya", {100, 100}, {255, 255, 255, 255}, 200);
//        
        frame.showImage("W1");
        
//        renderframe.showImage("W2");
//        renderframe.flip(1);
        frame.cvWaitKey(1);
        i++;
    }

    
    
//    MatrixH<float> matrix1 = MatrixH<float>::Range(10, {2, 3, 3, 4});
//    MatrixH<uint8_t> matrix4 = MatrixH<uint8_t>::constant((uint8_t)255.0, { 3, 3, 2});
//    const char* outputsPath = "/Users/adityadude/Downloads/savess.mp4";
//    const char* videoPath = "/Users/adityadude/Downloads/IMG_7567.mov";
//    matrix1.print();
//    
//    matrix4.showImage("Helll");
//    matrix4.cvWaitKey(0);
//    
////    matrix1.PadnTrim2D(1, 10, 0, 0, 0, 0, 0, 0, 0);
//    matrix1.print();
//    for (int i = 0; i < 10; i++) {
//        matrix4.values[i * matrix4.shape[1] * matrix4.shape[2] * matrix4.shape[3] + (i%4) * matrix4.shape[3] + 0] = 0;
//        matrix4.values[i * matrix4.shape[1] * matrix4.shape[2] * matrix4.shape[3] + (i%4) * matrix4.shape[3] + 1] = 0;
//        matrix4.values[i * matrix4.shape[1] * matrix4.shape[2] * matrix4.shape[3] + (i%4) * matrix4.shape[3] + 2] = 0;
//        matrix4.values[i * matrix4.shape[1] * matrix4.shape[2] * matrix4.shape[3] + (i%4) * matrix4.shape[3] + 3] = 0;
//    }

//    MatrixH<uint8_t> vid = MatrixH<uint8_t>::fromVideo(videoPath);
//    std::cout << vid.shape;
//    MatrixH<uint8_t> frame;
////    frame = vid[250];
////    frame.PadnTrim2D(0, 100, 0, 0, 100, 100, 0, 0, 100);
////    frame.shape = {1, frame.shape[0], frame.shape[1], frame.shape[2]};
////    frame.play();
//    
//    for (int i = 7; i < 10; i++) {
//        
//        
//        frame = vid[i];
//        
////        frame.Translate2D_H(0,100 * sin(i * 0.1) , 100 * sin(i * 0.1));
//        frame.PadnTrim2D(0, 100, 0, 0, 0, 100, 0, 0, 0);
//        std::cout << frame.shape << "\n";
//        
//        std::cout << "frame" << i << "\n" ;
//        vid[i] = frame;
//        std::cout << "\n" ;
//        
//    }
//    
    
    
//    vid.print();
//    matrix4.print();
//    std::cout << vid.shape << "\n";
//    vid.Pad(2, 20, 20);
//    vid.PadnTrim(2, 0, 0, 0, 0);
//    vid.PadnTrim(1, 100, 0, 0, 0);
    
//    matrix4.Pad(1, 2, 0);
//    matrix4.play();
//    matrix4.saveImage();
//    matrix4.imshow();
//    matrix4.print();
    
//    auto group = vid.saveVideo(outputsPath);
//    vid.play();
//
//    MatrixH<uint8_t> img = vid[0];
//    vid.play();
    
//    std::cout << "data " << img.shape[0] <<" "<< img.shape[1] <<" " <<img.shape[2]<< " ";
//    img.swapD(2, 0, 2);
//    img.imshow();
//    for (int i = 0; i < vid.shape[0]; i++) {
//        vid[i] = vid[i].ConvG(Mat);
////        vid[i].drawRect({50, 50, 100, 100}, {0, 0, 255, 255});
//        vid[i].drawText("Yaar Na Mille", {100-i*10, 100}, {255, 255, 255, 255}, 300);
//    }
    
//    auto img = MatrixH<uint8_t>::fromImage();
//    auto img2 = MatrixH<uint8_t>::fromImage();
//    img2.swapD(2, 0, 2);
//    auto vid2 = MatrixH<uint8_t>::fromBuffer(new uint8_t[200 * 960 * 1280 * 4], {200, 960, 1280, 4});
//    for (int i = 0; i < vid2.shape[0]; i++) {
//        if ((int)floor(i / 2.0) % 2) {
//            
//            vid2[i] = img2;
//        } else {
//            vid2[i] = img;
//        }
//        
//        std::cout << vid2[i].shape << img.shape << "\n";
////        vid[i].drawRect({50, 50, 100, 100}, {0, 0, 255, 255});
//        vid2[i].drawText("Yaar Na Mille", {100-i*10, 100}, {255, 255, 255, 255}, 300);
//    }
//    
//    vid2.flip(0);
//    vid2.flip(1);
//    auto group = vid.saveVideo(outputsPath);
//    vid.play();
//    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    
//    MatrixH<uint8_t> img = MatrixH<uint8_t>::fromCam(0, IMGFormat::BGRA);
//    img.swapD(2, 0, 2);
//    img = img.ConvG(matrix4);
//    img.imshow();
//    
//    img.flip(1);
    
//    img[0][0] = {0, 0, 0, 0};
//    img[0][0].print();
    
//    img.transpose();
//    std::cout << "img capture successful "<<img.shape[1];
//    img.imshow();
    
//    img.imshow();
    
//    for (int i = 0; i < 500; i++) {
//        for (int j = 0; j < img.shape[1]; j++) {
//            img.values[4 * (i * img.shape[1] + j)] = 255;
//            img.values[4 * (i * img.shape[1] + j) + 1] = 1;
//            img.values[4 * (i * img.shape[1] + j) + 2] = 1 ;
//            img.values[4 * (i * img.shape[1] + j) + 3 ] = 255;
//        }
//    }
//    MatrixH<uint8_t> padded = img.convC<uint8_t>(matrix4);
    
//    padded.saveImage();
    
//    MatrixH<uint8_t> matrix2 = MatrixH<uint8_t>::constant(255, {5, 5, 4});
//    
//    MatrixH<float> matrix4 = MatrixH<float>::constant(1.0/9, {3, 3});
//    matrix4.print();
//    std::cout << matrix2.shape[2] << "\n";
//    auto finalimg = matrix2.convC<uint8_t>(matrix4);
//    matrix2.padding<uint8_t>(1, 1, 1, 1).print();
//    finalimg.print();
//    finalimg.saveImage();
//    matrix2;
    
}

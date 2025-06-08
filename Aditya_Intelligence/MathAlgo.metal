//
//  MathAlgo.metal
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 20/12/24.
//

#include <metal_stdlib>
using namespace metal;




kernel void AddGPU_F(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] + B[gid];
}

kernel void AddGPU_I(device const int* A [[buffer(0)]], device const int* B [[buffer(1)]], device int* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] + B[gid];
}

kernel void AddGPU_C(device const uint8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]], device uint8_t* C [[buffer(2)]],uint gid [[thread_position_in_grid]]){
    C[gid] = A[gid] + B[gid];
}


//

kernel void convGPU(device const uint8_t* A [[buffer(0)]], device const float* B [[buffer(1)]], device uint8_t* C [[buffer(2)]], constant int2& shape [[buffer(3)]], constant int2& kshape [[buffer(4)]] ,uint2 gid [[thread_position_in_grid]]) {
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
    float sum4 = 0;
    sum1 = sum2 = sum3 = sum4 = 0;
    for (int k = 0; k < static_cast<int>(kshape[0]); k++) {
        for (int l = 0; l < static_cast<int>(kshape[1]); l++) {
            
            int offX = (l-static_cast<int>((kshape[1]-1)/2));
            int offY = (k-static_cast<int>((kshape[0]-1)/2));
            
            if ((gid.y + offY) > shape.x || (gid.y + offY) < 0 || (gid.x + offX) > shape.y || (gid.x + offX) < 0) {
                continue;
            }
            
            
            sum1 += static_cast<float> (A[4*((gid.y +offX) * shape[1] + gid.x + offY) + 0]) * B[ k * kshape[1] + l];
            sum2 += static_cast<float> (A[4*((gid.y +offX) * shape[1] + gid.x + offY) + 1]) * B[ k * kshape[1] + l];
            sum3 += static_cast<float> (A[4*((gid.y +offX) * shape[1] + gid.x + offY) + 2]) * B[ k * kshape[1] + l];
            sum4 += static_cast<float> (A[4*((gid.y +offX) * shape[1] + gid.x + offY) + 3]) * B[ k * kshape[1] + l];
        }
    }
    C[4 * (gid.y*shape[1] + gid.x) + 0] = sum1;
    C[4 * (gid.y*shape[1] + gid.x) + 1] = sum2;
    C[4 * (gid.y*shape[1] + gid.x) + 2] = sum3;
    C[4 * (gid.y*shape[1] + gid.x) + 3] = 255;
}


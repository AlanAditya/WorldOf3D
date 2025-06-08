//
//  Add.metal
//  WorldOf3D
//
//  Created by Manoj Kumar on 04/06/24.
//

#include <metal_stdlib>
using namespace metal;



kernel void addArraysGPU(device const float* inA, device const float* inB, device float* result, uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}


kernel void addArraysGPUV2(device const float* inpALL, device float* result, device int* rows, device int* col, uint index [[thread_position_in_grid]]) {
    result[index] = 0;
    for (int i = 0; i < *rows; i++) {
        result[index] += inpALL[i * (*col) +index];
    }
    
}


struct Vertex {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

vertex float4 vertex_main(const device Vertex* vertex_array [[buffer(0)]], uint vertex_id [[vertex_id]]) {
    return vertex_array[vertex_id].position;
}

fragment float4 fragment_main() {
    return float4(1.0, 0.0, 0.0, 1.0); // Red color
}

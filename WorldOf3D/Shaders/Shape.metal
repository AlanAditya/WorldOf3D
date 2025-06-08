//
//  Shape.metal
//  WorldOf3D
//
//  Created by Aditya Dudeja on 15/06/24.
//

#include <metal_stdlib>
using namespace metal;


struct VertexInput {
    float3 position;
    float3 colour;
};

struct VertexOutput {
    float4 position [[position]];
    half3 colour;
};

VertexOutput vertex VertexMainShape(uint vertextID [[vertex_id]], device const VertexInput* verticies [[buffer(0)]], constant float4x4& transform [[buffer(1)]]) {
    VertexOutput payload;
    VertexInput package = verticies[vertextID];
    half3 pos = half3(package.position);
    payload.position = float4(half4x4(transform) * half4(pos, 1.0));
    // payload.position = float4(package.position, 0.0, 1.0);
    payload.colour = half3(package.colour);
    return payload;
}
    
half4 fragment FragmentMainShape(VertexOutput frag [[stage_in]]) {
//    stdout << "hello";
//    if (frag.position[0] > 0) {
//        return half4(0, 1, 0, 0.5);
//    }
    return half4(frag.colour, 0.5);
}

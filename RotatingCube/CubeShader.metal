//
//  CubeShader.metal
//  RotatingCube
//
//  Created by Manoj Kumar on 17/06/24.
//

#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
    float3 normals [[attribute(1)]];
    float3 colour [[attribute(2)]];
};

struct VertexOutput {
    float4 position [[position]];
    half3 colour;
};

VertexOutput vertex VertexMainCube(uint vertextID [[vertex_id]], device const VertexInput* verticies [[buffer(0)]], constant float4x4& transform [[buffer(1)]]) {
    VertexOutput payload;
    VertexInput package = verticies[vertextID];
    half3 pos = half3(package.position);
    
    
    float4x4 normZ = float4x4(1.0f);
    const float nearPlane = 0.1f;
    const float farPlane = 10.0f;
    normZ[2][2] = (1 + nearPlane) / farPlane;
    normZ[2][2] = nearPlane;
    
    payload.position =  normZ * float4(half4x4(transform) * half4(pos, 1.0));
    
    // payload.position = float4(package.position, 0.0, 1.0);
    payload.colour = half3(package.colour);
    return payload;
}
    
half4 fragment FragmentMainCube(VertexOutput frag [[stage_in]]) {
//    stdout << "hello";
//    if (frag.position[0] > 0) {
//        return half4(0, 1, 0, 0.5);
//    }
    return half4(frag.colour, 0.5);
}

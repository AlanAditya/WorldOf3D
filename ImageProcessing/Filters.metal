//
//  Filters.metal
//  ImageProcessing
//
//  Created by Manoj Kumar on 04/08/24.
//

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float3 colour [[attribute(2)]];
};

struct VertexInput {
    float3 position [[attribute(0)]];
    float3 normals [[attribute(1)]];
    float3 colour [[attribute(2)]];
};

vertex VertexOut vertex_main(const device VertexInput *vertexArray [[buffer(0)]],
                             uint vertexID [[vertex_id]], constant float4x4& transform [[buffer(1)]]) {
    VertexOut out;
    VertexInput inp = vertexArray[vertexID];
    
    float4x4 normZ = float4x4(1.0f);
    const float nearPlane = 0.1f;
    const float farPlane = 10.0f;
    normZ[2][2] = (1 + nearPlane) / farPlane;
    normZ[2][2] = nearPlane;
    
    half3 pos = half3(inp.position);
    out.position = normZ * float4(half4x4(transform) * half4(pos, 1.0));
    out.texCoord = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    out.colour = (inp.colour);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> texture [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return texture.sample(s, in.texCoord);
//    return float4(in.colour, 0.5);
}

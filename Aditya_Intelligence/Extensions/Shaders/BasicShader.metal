//
//  BasicShader.metal
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 01/03/25.
//

#include <metal_stdlib>
using namespace metal;

struct vertexOut {
    float4 position [[position]];
    float4  color;
};
struct Vertex3D {
    simd_float3 position;
    simd_float4 colour;
    simd_float3 normal;
    simd_float2 textureCoordinates;
};

simd_float4 constant favCol[3] = {{0.8687, 0.4099, 0.9302, 1.0}, {0.4412, 0.2952, 0.9600, 1.0}, {0.21545,  0.74170,  0.64551,  1.00000}};

vertex vertexOut basicVertexShader(uint vertexID  [[ vertex_id ]], const device Vertex3D* verticies [[buffer(0)]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]],uint instanceId [[instance_id]]) {
    vertexOut vertOut = vertexOut();
    vertOut.color = verticies[vertexID].colour;
    vertOut.position = cam * transform * float4(verticies[vertexID].position + instanceId * 0.1f, 1.0);
    return vertOut;
}

vertex vertexOut instanceVertexShader(uint vertexID  [[ vertex_id ]], const device Vertex3D* verticies [[buffer(0)]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], uint instanceId [[instance_id]]) {
    vertexOut vertOut = vertexOut();
    vertOut.color = verticies[vertexID].colour;
    vertOut.position = cam * instanceData[instanceId] * transform * float4(verticies[vertexID].position, 1.0);
    return vertOut;
}

fragment float4 basicFragmentShader(vertexOut inColor  [[ stage_in ]])
{
    return inColor.color;
}

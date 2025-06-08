//
//  Shaders.metal
//  ParticleSystem
//
//  Created by Manoj Kumar on 13/02/25.
//

#include <metal_stdlib>
using namespace metal;


struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    simd_float4 colour;
};

struct vertexOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
    half4  color;
};
struct Vertex3D {
    simd_float3 position;
    simd_float4 colour;
    simd_float3 normal = {0.0, 0.0, 1.0};
    simd_float2 textureCoordinates = {0.0, 0.0};
};


simd_float4 constant favCol[3] = {{0.8687, 0.4099, 0.9302, 1.0}, {0.4412, 0.2952, 0.9600, 1.0}, {0.21545,  0.74170,  0.64551,  1.00000}};

vertex vertexOut vertexShader(uint vertexID  [[ vertex_id ]], const device Particle* particles [[buffer(0)]], const device Vertex3D* position [[buffer(1)]], uint instanceId [[instance_id]]) {
    vertexOut vertOut = vertexOut();
    vertOut.pointSize = 0.1;
    vertOut.color = half4(favCol[vertexID]);
    vertOut.position = float4(particles[instanceId].position + (position[vertexID].position.xy * 0.1),0.5,1.0);
    return vertOut;
}

fragment half4 fragmentShader(vertexOut inColor  [[ stage_in ]])
{
    return inColor.color;
}

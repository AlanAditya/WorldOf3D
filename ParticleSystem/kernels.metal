//
//  kernels.metal
//  ParticleSystem
//
//  Created by Manoj Kumar on 11/02/25.
//

#include <metal_stdlib>
using namespace metal;

struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    simd_float4 colour;
};

kernel void clearPassFunc(texture2d<float, access::write> outputTexture [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    outputTexture.write(float4(0.0, 0.0, 0.0, 1.0), gid);
}

void drawCircle(texture2d<float, access::write> outputTexture, float2 position, float radius, float4 colour, int2 size);
constant float4 RED = {1.0, 0.0, 0.0, 1.0};
constant float dt = 0.01;
constant float damp = 1;
constant float lowerBoundry = -1;
kernel void drawKernel(device Particle* particles [[buffer(0)]], constant int &count [[buffer(1)]], constant float2 &offsst [[buffer(2)]],texture2d<float, access::write> outputTexture [[texture(0)]], uint gid [[thread_position_in_grid]]) {
    

    int2 size = int2(outputTexture.get_width(), outputTexture.get_height());
    drawCircle(outputTexture, particles[gid].position, 0.01, particles[gid].colour, size);
//    particles[gid].velocity += float2(0, 1) * dt;
//    for (int i = 0; i < count; i++) {
//        if (uint(i) != gid) {
//            float2 diff = particles[gid].position - particles[i].position;
//            particles[gid].velocity += 0.00001* normalize(-diff) * pow(length(diff), -2);
//            
//        }
//    }
    
    particles[gid].velocity += float2(0, 1) * dt;
    if (particles[gid].position.x >= 1 ) {
        particles[gid].velocity.x *= -1;
        particles[gid].position.x = 1-0.01;
        
    }
    if (particles[gid].position.x < lowerBoundry) {
        particles[gid].velocity.x *= -1;
        particles[gid].position.x = lowerBoundry+0.01;
        
    }
    if (particles[gid].position.y >= 1 ) {
        particles[gid].velocity.y *= -1 * damp;
        particles[gid].position.y = 1-0.01;
    }
    if (particles[gid].position.y <= lowerBoundry) {
        particles[gid].velocity.y *= -1 * damp;
        particles[gid].position.y = lowerBoundry+0.01;
    }
    
    for (int i = 0; i < count; i++) {
        particles[gid].position += (offsst / float2(size));
        if (uint(i) != gid) {
            float2 diff = particles[gid].position - particles[i].position;
            if (length(diff) < 0.01) {
//                particles[gid].velocity = 0;
                particles[gid].velocity = reflect(particles[gid].velocity, normalize(diff));
            }
        }
    }
    particles[gid].position += particles[gid].velocity * dt;

}

void drawCircle(texture2d<float, access::write> outputTexture, float2 position, float radius, float4 colour, int2 size) {
    for (uint i = 0; i < 2*radius*outputTexture.get_height(); i++) {
        for (uint j = 0; j < 2*radius*outputTexture.get_width(); j++) {
            float2 normXY = float2(j, i) - ((radius) * float2(size));
            float S1 = dot(normXY, normXY) - pow(radius*outputTexture.get_height(), 2);
            if (S1 < 0) {
                outputTexture.write(colour, uint2(position.x*outputTexture.get_width() + j, position.y*outputTexture.get_height() + i));
            }
        }
    }
}
void drawSquare(texture2d<float, access::write> outputTexture, float2 position, float radius, float4 colour, float2 size) {
    for (uint i = 0; i < radius*outputTexture.get_height(); i++) {
        for (uint j = 0; j < radius*outputTexture.get_width(); j++) {
            float2 XY = ((float2(uint2(j, i)) / size) * 2 ) - 1;
            float2 normXY = XY - position;
            float S1 = dot(normXY, normXY) - pow(radius, 2);
            if (S1 < 0) {
                outputTexture.write(colour, uint2(j, i));
            }
//            outputTexture.write(colour, uint2(position.x*outputTexture.get_width() + j, position.y*outputTexture.get_height() + i));
        }
    }
}




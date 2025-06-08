//
//  ViewController.h
//  ParticleSystem
//
//  Created by Manoj Kumar on 11/02/25.
//

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <random>
#include "Shape.h"

struct Particle {
    simd_float2 position;
    simd_float2 velocity;
    simd_float4 colour;
};


class ParticleSystemRenderer {
public:
    Particle* particleList;
    int noOfParticles;
    id<MTLBuffer> buffer;
    id<MTLBuffer> buff;
    Triangle tri = Triangle(0.1);
    
    float randomFloat(float min, float max) {
        // Create a random number generator
        std::random_device rd;  // 🌟 1. Non-deterministic random number source
        std::mt19937 gen(rd()); // 🌟 2. Mersenne Twister 19937 generator (high-quality PRNG)
        
        // Define the distribution for floats between min and max
        std::uniform_real_distribution<float> dist(min, max); // 🌟 3. Ensures uniform distribution
        
        return dist(gen); // 🌟 4. Generate a random float
    }
    
    ParticleSystemRenderer(int count, id<MTLDevice> metalDevice): noOfParticles(count) {
        particleList = new Particle[count];
        for (int i = 0; i < count; i++) {
            particleList[i].position = {randomFloat(-1.0, 1.0), randomFloat(-1.0, 1.0)};
            particleList[i].velocity = {randomFloat(-1.0, 1.0), randomFloat(-1.0, 1.0)};
            particleList[i].colour = {randomFloat(0.0, 1.0), randomFloat(0.0, 1.0), randomFloat(0.0, 1.0), 1.0f};
        }
        buffer = [metalDevice newBufferWithLength:count*sizeof(Particle) options:MTLResourceStorageModeShared];
        memcpy([buffer contents], particleList, count*sizeof(Particle));
        
        simd_float2 triangle[3] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0}
        };
        
        buff = [metalDevice newBufferWithLength:3 *sizeof(simd_float2) options:MTLResourceStorageModeShared];
        memcpy([buff contents], triangle, 3*sizeof(simd_float2));
        
        
    }
};


@interface MetalViewController : NSObject<MTKViewDelegate>
{
    @public
        id<MTLDevice> metalDevice;
        id<MTLCommandQueue> CommandQueue;
        id<MTLComputePipelineState> BasicComputePipelineState;
        id<MTLComputePipelineState> ClearPassPipelineState;
        id<MTLRenderPipelineState> BasicRenderPipelineState;
        NSPoint WinOrigin;
        ParticleSystemRenderer *particleSys;
    }
- (instancetype)initWithDevice:(id<MTLDevice>)device origin:(NSPoint)origin;
@end


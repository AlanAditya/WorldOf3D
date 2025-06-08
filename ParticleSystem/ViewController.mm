//
//  ViewController.m
//  ParticleSystem
//
//  Created by Manoj Kumar on 11/02/25.
//

#import "ViewController.h"
#import "simd/simd.h"
#import <Metal/Metal.h>
#include <iostream>





@implementation MetalViewController
- (instancetype)initWithDevice:(id<MTLDevice>)device origin:(NSPoint)origin
{
    self = [super init];
    if (self) {
        NSError *error = nil;
        metalDevice = device;
        id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
        id<MTLFunction> func1 = [lib newFunctionWithName:@"drawKernel"];
        BasicComputePipelineState = [metalDevice newComputePipelineStateWithFunction:func1 error:&error];
        
        id<MTLFunction> func2 = [lib newFunctionWithName:@"clearPassFunc"];
        ClearPassPipelineState = [metalDevice newComputePipelineStateWithFunction:func2 error:&error];
        
        id<MTLFunction> func3 = [lib newFunctionWithName:@"vertexShader"];
        id<MTLFunction> func4 = [lib newFunctionWithName:@"fragmentShader"];
        MTLRenderPipelineDescriptor *desc =  [[MTLRenderPipelineDescriptor alloc] init];
        [desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
        [[desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
        [desc setVertexFunction:func3];
        [desc setFragmentFunction:func4];
        BasicRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:desc error:&error];
        
        
          // Get the containing window
        WinOrigin = origin;

        particleSys = new ParticleSystemRenderer(600, metalDevice);
    }
    return self;
}

- (void)drawInMTKView:(nonnull MTKView *)view { 
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [CommandQueue commandBuffer];
        id<MTLComputeCommandEncoder> cmdEncoder = [cmdBuffer computeCommandEncoder];
        id<CAMetalDrawable> drawable = [view currentDrawable];
        if (!cmdEncoder) {
            NSLog(@"Drawable is nil! Nothing will be displayed.");
            return;
        }
        [cmdEncoder setComputePipelineState:ClearPassPipelineState];
        [cmdEncoder setTexture:[drawable texture] atIndex:0];
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(drawable.texture.width, drawable.texture.height, 1);
        
        [cmdEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        
        [cmdEncoder setComputePipelineState:BasicComputePipelineState];
        [cmdEncoder setBuffer:particleSys->buffer offset:0 atIndex:0];
        [cmdEncoder setBytes:&(particleSys->noOfParticles) length:sizeof(int) atIndex:1];
        
        simd_float2 offset = -simd_make_float2(view.window.frame.origin.x - WinOrigin.x, view.window.frame.origin.y - WinOrigin.y);
        [cmdEncoder setBytes:&(offset) length:sizeof(simd_float2) atIndex:2];
//        [cmdEncoder setTexture:[drawable texture] atIndex:0];
//        [cmdEncoder setBytes:&time length:sizeof(int) atIndex:2];
        
        _threadsPerThreadgroup = MTLSizeMake(BasicComputePipelineState.threadExecutionWidth, 1, 1);
        _dispatchExecutionSize =  MTLSizeMake(particleSys->noOfParticles, 1, 1);
        
        [cmdEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        [cmdEncoder endEncoding];
        

        
//        id<MTLRenderCommandEncoder> RendercmdEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:[view currentRenderPassDescriptor]];
//        [RendercmdEncoder setRenderPipelineState:BasicRenderPipelineState];
//        [RendercmdEncoder setVertexBuffer:particleSys->buffer offset:0 atIndex:0];
//        
//        particleSys->tri.buildBuffers(metalDevice);
////        [RendercmdEncoder setVertexBuffer:particleSys->buff offset:0 atIndex:1];
//        [RendercmdEncoder setVertexBuffer:particleSys->tri.vertexBuffer offset:0 atIndex:1];
//        
//        [RendercmdEncoder drawIndexedPrimitives:particleSys->tri.drawType indexCount:particleSys->tri.indexCount indexType:MTLIndexTypeUInt16 indexBuffer:particleSys->tri.indexBuffer indexBufferOffset:0 instanceCount:particleSys->noOfParticles];
////        printArr(particleSys->tri.indexBuffer.contents, 5);
//        [RendercmdEncoder endEncoding];
        
        [cmdBuffer presentDrawable:drawable];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        WinOrigin = view.window.frame.origin;
    }
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size { 
    
}

@end




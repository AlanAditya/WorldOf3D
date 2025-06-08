//
//  MetalView.m
//  Video Editor
//
//  Created by Manoj Kumar on 28/01/25.
//

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import "Header.h"
#include <iostream>

@implementation MetalViewDelegate
- (void)drawInMTKView:(MTKView *)view {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [CommandQueue commandBuffer];
        id<MTLCommandEncoder> cmdEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:[view currentRenderPassDescriptor]];
        id<CAMetalDrawable> drawable = [view currentDrawable];
        
        
        
        [cmdEncoder endEncoding];
        [cmdBuffer presentDrawable:drawable];
        [cmdBuffer commit];
        
    }
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    std::cout << size.width << std::endl;
}

@end

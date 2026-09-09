#ifdef OS_TARGET_IPHONE
#pragma once
#import <ARKit/ARKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import "../matrix.h"

@interface ARSessionCapture : NSObject <ARSessionDelegate> {
    CVMetalTextureCacheRef _textureCache;
}

@property (nonatomic, strong) ARSession *session;

- (instancetype)init;
- (void)start;
- (void)stop;

- (void)getPointCloudTensor:(matrix&)out_matrix;

// Extracts the scene depth map directly using CVMetalTextureCache and an MTLBlitCommandEncoder
- (void)getDepthMapTensor:(matrix&)out_matrix;

@end
#endif

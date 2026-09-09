#ifdef OS_TARGET_IPHONE
#import "ARSessionCapture.h"
#import "../Mods/Utils.h"
@import GPUManager;

@implementation ARSessionCapture

- (instancetype)init {
    self = [super init];
    if (self) {
        _session = [[ARSession alloc] init];
        _session.delegate = self;
        
        CVReturn status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GlobalGPUManager.metalDevice, nil, &_textureCache);
        if (status != kCVReturnSuccess) {
            NSLog(@"ARSessionCapture: Failed to create CVMetalTextureCache");
        }
    }
    return self;
}

- (void)dealloc {
    if (_textureCache) {
        CFRelease(_textureCache);
    }
}

- (void)start {
    ARWorldTrackingConfiguration *configuration = [[ARWorldTrackingConfiguration alloc] init];
    if ([ARWorldTrackingConfiguration supportsFrameSemantics:ARFrameSemanticSceneDepth]) {
        configuration.frameSemantics = ARFrameSemanticSceneDepth;
    }
    [_session runWithConfiguration:configuration];
}

- (void)stop {
    [_session pause];
}

- (void)getPointCloudTensor:(matrix&)out_matrix {
    ARFrame *currentFrame = self.session.currentFrame;
    if (!currentFrame || !currentFrame.rawFeaturePoints) {
        out_matrix = matrix::zeros({0, 3}, dtype::Float);
        return;
    }
    
    ARPointCloud *pointCloud = currentFrame.rawFeaturePoints;
    NSUInteger numPoints = pointCloud.count;
    const vector_float3 *points = pointCloud.points;
    
    if (numPoints == 0) {
        out_matrix = matrix::zeros({0, 3}, dtype::Float);
        return;
    }
    
    if (out_matrix.total_size != numPoints * 3 || out_matrix.dims != 2 || out_matrix.buffer == nullptr) {
        out_matrix = matrix::zeros({(size_m)numPoints, 3}, dtype::Float);
    }
    
    matrix view;
    view.dims = 2;
    view.type = dtype::Float;
    view.buffer = (void *)points;
    view.total_size = numPoints * 3;
    
    view.shape()[0] = numPoints;
    view.shape()[1] = 3;
    
    view.strides()[0] = 4;
    view.strides()[1] = 1;
    
    view.flags = NON_OWNERSHIP_FLAG | NON_CONTIGUOUS_FLAG;
    
    out_matrix.flags |= NON_OWNERSHIP_FLAG;
    out_matrix = view;
    out_matrix.flags &= ~NON_OWNERSHIP_FLAG;
}

- (void)getDepthMapTensor:(matrix&)out_matrix {
    ARFrame *currentFrame = self.session.currentFrame;
    if (!currentFrame || !currentFrame.sceneDepth || !currentFrame.sceneDepth.depthMap) {
        out_matrix = matrix::zeros({0, 0}, dtype::Float);
        return;
    }
    
    CVPixelBufferRef depthMap = currentFrame.sceneDepth.depthMap;
    
    size_t width = CVPixelBufferGetWidth(depthMap);
    size_t height = CVPixelBufferGetHeight(depthMap);
    OSType format = CVPixelBufferGetPixelFormatType(depthMap);
    
    dtype type = dtype::Float;
    MTLPixelFormat mtlFormat = MTLPixelFormatR32Float;
    size_t bytesPerPixel = 4;
    
    if (format == kCVPixelFormatType_DepthFloat32) {
        type = dtype::Float;
        mtlFormat = MTLPixelFormatR32Float;
        bytesPerPixel = 4;
    } else if (format == kCVPixelFormatType_DepthFloat16) {
        type = dtype::Float16;
        mtlFormat = MTLPixelFormatR16Float;
        bytesPerPixel = 2;
    } else {
        NSLog(@"ARSessionCapture: Unsupported depth format");
        return;
    }
    
    if (out_matrix.buffer == nullptr || out_matrix.dims != 2 || out_matrix.shape()[0] != height || out_matrix.shape()[1] != width || out_matrix.type != type) {
        matrix new_frame(2, height * width, type);
        new_frame.shape()[0] = height;
        new_frame.shape()[1] = width;
        new_frame.calcStrides();
        new_frame.buildMetalBuffer();
        out_matrix = new_frame;
    }
    
    CVMetalTextureRef textureRef = NULL;
    CVReturn status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                                _textureCache,
                                                                depthMap,
                                                                nil,
                                                                mtlFormat,
                                                                width,
                                                                height,
                                                                0,
                                                                &textureRef);
    
    if (status == kCVReturnSuccess) {
        id<MTLTexture> mtlTexture = CVMetalTextureGetTexture(textureRef);
        
        GlobalGPUManager.endCommandEncoding();
        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        NSUInteger bytesPerRow = width * bytesPerPixel;
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        
        [blitEncoder copyFromTexture:mtlTexture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:region.origin
                          sourceSize:region.size
                            toBuffer:out_matrix.metalBuffer
                   destinationOffset:0
              destinationBytesPerRow:bytesPerRow
            destinationBytesPerImage:bytesPerRow * height];
            
        [blitEncoder endEncoding];
        GlobalGPUManager.commitCommandBuffer();
        
        CFRelease(textureRef);
    } else {
        NSLog(@"ARSessionCapture: CVMetalTextureCacheCreateTextureFromImage failed %d", status);
    }
}

@end
#endif

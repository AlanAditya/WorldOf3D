////
////  MetalLayerExtension.m
////  Aditya_Intelligence
////
////  Created by Manoj Kumar on 01/03/25.
////
//
//#import <Foundation/Foundation.h>
//#import <Metal/Metal.h>
//#include "AlgebroHeap.hpp"
//#include "shapeExtension.hpp"
//
//
//class MetalWrapper::Renderer {
//    id<MTLDevice> metalDevice;
//    id<MTLCommandQueue> CommandQueue;
//    id<MTLRenderPipelineState> BasicRenderPipelineState;
//    ShapeWrapper<uint16>::Shape triangle = Triangle(1);
//    id<MTLTexture> offscreenTexture;
//    size_t width = 1920;
//    size_t height = 1080;
//    
//public:
//    Renderer() {
//        metalDevice = MTLCreateSystemDefaultDevice();
//        CommandQueue = [metalDevice newCommandQueue];
//        
//        id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
//        NSError* error = nil;
//        
//        id<MTLFunction> func3 = [lib newFunctionWithName:@"basicVertexShader"];
//        id<MTLFunction> func4 = [lib newFunctionWithName:@"basicFragmentShader"];
//        MTLRenderPipelineDescriptor *desc =  [[MTLRenderPipelineDescriptor alloc] init];
////        [desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
//        [[desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
//        [desc setVertexFunction:func3];
//        [desc setFragmentFunction:func4];
//        BasicRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:desc error:&error];
//    }
//    
//
//    void render(MatrixH<uint8_t>& layer) {
//
//        
//        if (layer.shape.size() < 1 || !layer.values) {
//            layer.values = new uint8_t[width * height * 4];
//            layer.shape = {height, width, 4};
//            layer.total_size = height * width * 4;
//        }
//        
////        MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
////        MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
////        captureDescriptor.captureObject = metalDevice;
////        
////        NSError *error = nil;
////        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
////            NSLog(@"Capture start failed: %@", error);
////        }
//        if (!offscreenTexture || layer.shape[0] != height || layer.shape[1] != width) {
//            width = layer.shape[1];
//            height = layer.shape[0];
//            MTLTextureDescriptor* drawableDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
//                                                                                                    width:(NSUInteger)width
//                                                                                                   height:(NSUInteger)height
//                                                                                                mipmapped:NO];
//            drawableDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
//            // Use shared storage so the CPU can read the texture data.
//            drawableDesc.storageMode = MTLStorageModeShared;
//            
//            
//            id<MTLTexture> offscreenTexture = [metalDevice newTextureWithDescriptor:drawableDesc];
//        }
//
////        
//        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
//        NSUInteger bytesPerRow = width * 4;  // 4 bytes per pixel for BGRA8
//        [offscreenTexture replaceRegion:region
//                            mipmapLevel:0
//                              withBytes:layer.values
//                            bytesPerRow:bytesPerRow];
//        
//        MTLRenderPassDescriptor *passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
//        passDescriptor.colorAttachments[0].texture = offscreenTexture;
//        passDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
//        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 0.0, 0.0, 1.0);
//        passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
//        
//        id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
//        id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
//        [encoder setRenderPipelineState:BasicRenderPipelineState];
//        
//        
//        triangle.buildBuffers(metalDevice);
//        [encoder setVertexBuffer:triangle.vertexBuffer offset:0 atIndex:0];
//        
//        [encoder drawPrimitives:triangle.drawType vertexStart:0 vertexCount:3 instanceCount:10];
//        
//        
//        // Here you would bind your self.texture and encode drawing commands to render it.
//        // For example, you might use a simple textured quad shader.
//        // (The complete rendering pipeline setup is beyond this example.)
//        
//        [encoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
////        [captureManager stopCapture];
//        
//        size_t imageByteCount = height * bytesPerRow;
//        uint8_t* tempBuffer = new uint8_t[imageByteCount];
//        
//        [offscreenTexture getBytes:tempBuffer bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
//        
//        // Copy the data from tempBuffer into outBuffer.
//        // (Assuming outBuffer.values was allocated to hold imageByteCount bytes.)
//        memcpy(layer.values, tempBuffer, imageByteCount);
//        
//        // Clean up.
//        delete[] tempBuffer;
//        
//    }
//};
//
//MetalWrapper::MetalWrapper() {
//    pRenderer = new Renderer();
//}
//
//void MetalWrapper::Render(MatrixH<uint8_t>& layer) {
//    pRenderer->render(layer);
//}

//
//  MetalLayerExtensionCPP.cpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 07/03/25.
//

#include <stdio.h>

#include "MetalLayerHPPextension.hpp"

void RendererHPP::Render(MatrixH<uint8_t>& layer) {

    
    if (layer.shape.size() < 1 || !layer.values) {
        
        layer.values = new uint8_t[width * height * 4];
        layer.shape = {height, width, 4};
        layer.total_size = height * width * 4;
        
    }

    
    if (metalCapture) {
        captureManager = MTL::CaptureManager::sharedCaptureManager();
        MTL::CaptureDescriptor *captureDescriptor = MTL::CaptureDescriptor::alloc()->init();
        captureDescriptor->setCaptureObject(metalDevice);
        NS::Error *error = nil;
        if (!captureManager->startCapture(captureDescriptor, &error)) {
        }
    }


    
    if (!offscreenTexture || layer.shape[0] != height || layer.shape[1] != width) {
        MTL::TextureDescriptor* drawableDesc = MTL::TextureDescriptor::alloc()->init();
        drawableDesc->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        drawableDesc->setDepth(1.0);
        drawableDesc->setWidth(width);
        drawableDesc->setHeight(height);
        drawableDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);

        drawableDesc->setStorageMode(MTL::StorageModeShared);
        
        
        offscreenTexture = metalDevice->newTexture(drawableDesc);
    }

    MTL::Region region = MTL::Region(0, 0, (NS::UInteger)width, (NS::UInteger)height);
    NS::UInteger bytesPerRow = width * 4;  // 4 bytes per pixel for BGRA8
    offscreenTexture->replaceRegion(region, 0, layer.values, bytesPerRow);
    
    MTL::RenderPassDescriptor *passDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    
    

    
    passDescriptor->depthAttachment()->setTexture(depthTexture);
    passDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionClear);
    passDescriptor->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);
    passDescriptor->depthAttachment()->setClearDepth(1.0);
    
    passDescriptor->colorAttachments()->object(0)->setTexture(offscreenTexture);
    if (clear) {
        passDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    } else {
        passDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
    }
    
    passDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(1.0, 0.0, 0.0, 1.0));
    passDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    
    MTL::CommandBuffer* commandBuffer = CommandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(passDescriptor);
    encoder->setDepthStencilState(DepthStencilState);
    encoder->setRenderPipelineState(BasicRenderPipelineState);
    simd_float4x4 transform;
    camera.updateMatrix();
    simd_float4x4 cameraMatrix = camera.viewMatrix;
    encoder->setVertexBytes(&cameraMatrix, sizeof(simd_float4x4), 2);

    

//        cubeMesh.buildBuffers(metalDevice);
////        if (!cubeMesh.vertexBuffer) {
//            transform = Transformer(cubeMesh);
//    //            objectQueue[i].print();
//
//            encoder->setVertexBuffer(cubeMesh.vertexBuffer, 0, 0);
//            encoder->setVertexBytes(&transform, sizeof(simd_float4x4), 1);
//            encoder->drawIndexedPrimitives(cubeMesh.drawType, cubeMesh.indexCount, MTL::IndexTypeUInt16, cubeMesh.indexBuffer, 0, cubeMesh.instanceCount);
////        }
//
    
    for (int i = 0; i < objectQueue.size(); i++) {
        Shape<uint16> shape = objectQueue[i];
        objectQueue[i].buildBuffers(metalDevice);
        transform = Transformer(objectQueue[i]);
//            objectQueue[i].print();
        
        encoder->setVertexBuffer(objectQueue[i].vertexBuffer, 0, 0);
        encoder->setVertexBytes(&transform, sizeof(simd_float4x4), 1);
        encoder->drawIndexedPrimitives(objectQueue[i].drawType, shape.indexCount, MTL::IndexTypeUInt16, objectQueue[i].indexBuffer, 0, shape.instanceCount);
        
    }

    encoder->setRenderPipelineState(InstanceRenderPipelineState);
    
    for (int i = 0; i < objectQueueArray.size(); i++) {
        Shape<uint16> shape = objectQueueArray[i].shape;
        objectQueueArray[i].shape.buildBuffers(metalDevice);
        objectQueueArray[i].buildBuffer(metalDevice);
        transform = Transformer(objectQueueArray[i].shape);
//            objectQueue[i].print();
        
        encoder->setVertexBuffer( objectQueueArray[i].shape.vertexBuffer, 0, 0);
        encoder->setVertexBytes(&transform, sizeof(simd_float4x4), 1);
        encoder->setVertexBuffer(objectQueueArray[i].transformBuffer, 0, 3);
        encoder->drawIndexedPrimitives(objectQueueArray[i].shape.drawType, shape.indexCount, MTL::IndexTypeUInt16, objectQueue[i].indexBuffer, 0, shape.instanceCount);
        
    }

    
    
    // Here you would bind your self.texture and encode drawing commands to render it.
    // For example, you might use a simple textured quad shader.
    // (The complete rendering pipeline setup is beyond this example.)
    
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    if (metalCapture) {
        captureManager->stopCapture();
    }
    
    size_t imageByteCount = height * bytesPerRow;
    uint8_t* tempBuffer = new uint8_t[imageByteCount];
    

    offscreenTexture->getBytes(tempBuffer, bytesPerRow, region, 0);
    
    // Copy the data from tempBuffer into outBuffer.
    // (Assuming outBuffer.values was allocated to hold imageByteCount bytes.)
    memcpy(layer.values, tempBuffer, imageByteCount);
    
    // Clean up.
    delete[] tempBuffer;
}

//
//  MetalLayerHPPextension.hpp
//  WorldOf3D
//
//  Created by Manoj Kumar on 01/03/25.
//

#ifndef MetalLayerHPPextension_h
#define MetalLayerHPPextension_h
//#define NS_PRIVATE_IMPLEMENTATION
//#define MTL_PRIVATE_IMPLEMENTATION
//#define MTK_PRIVATE_IMPLEMENTATION
//#define CA_PRIVATE_IMPLEMENTATION

#include "AlgebroHeap.hpp"
#include <Metal/Metal.hpp>
#include "ShapeHPPExtension.h"
#include <iostream>
#include <simd/simd.h>
#include <Foundation/Foundation.hpp>
#include "3DCameraExtension.h"


#include <CoreGraphics/CGColorSpace.h>

class RendererHPP {
    MTL::Device* metalDevice = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* CommandQueue;
    MTL::RenderPipelineState* BasicRenderPipelineState;
    MTL::RenderPipelineState* InstanceRenderPipelineState;
    MTL::DepthStencilState* DepthStencilState;
    MTL::Texture* offscreenTexture = nil;
    MTL::Texture* depthTexture = nil;
    MTL::CaptureManager *captureManager;
    
    size_t width = 1920;
    size_t height = 1080;
    
public:
    Camera3D camera = Camera3D();
    std::vector<Shape<uint16>> objectQueue;
    std::vector<ArrayShape> objectQueueArray;
    bool metalCapture = false;
    bool clear = false;
    Cube cubeMesh;
    
    
    RendererHPP() :CommandQueue(nil), BasicRenderPipelineState(nil), InstanceRenderPipelineState(nil), DepthStencilState(nil), offscreenTexture(nil),depthTexture(nil), captureManager(nil),cubeMesh(0.5) {
        
        CommandQueue = metalDevice->newCommandQueue();
        MTL::Library* library = metalDevice->newDefaultLibrary();
        MTL::Function* vertexFunction = library->newFunction(NS::String::string("basicVertexShader", NS::StringEncoding::UTF8StringEncoding));
        MTL::Function* fragmentFunction = library->newFunction(NS::String::string("basicFragmentShader", NS::StringEncoding::UTF8StringEncoding));
        
        MTL::RenderPipelineDescriptor* pipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        pipelineDescriptor->setVertexFunction(vertexFunction);
        pipelineDescriptor->setFragmentFunction(fragmentFunction);
        pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pipelineDescriptor->setDepthAttachmentPixelFormat( MTL::PixelFormat::PixelFormatDepth32Float );
        
        NS::Error* error = nullptr;
        BasicRenderPipelineState = metalDevice->newRenderPipelineState(pipelineDescriptor, &error);
        
        
        MTL::Function* instanceVertexFunction = library->newFunction(NS::String::string("instanceVertexShader", NS::StringEncoding::UTF8StringEncoding));
        MTL::RenderPipelineDescriptor* InstancePipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        InstancePipelineDescriptor->setVertexFunction(instanceVertexFunction);
        InstancePipelineDescriptor->setFragmentFunction(fragmentFunction);
        InstancePipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        InstancePipelineDescriptor->setDepthAttachmentPixelFormat( MTL::PixelFormat::PixelFormatDepth32Float );
        InstanceRenderPipelineState = metalDevice->newRenderPipelineState(InstancePipelineDescriptor, &error);

        MTL::DepthStencilDescriptor* pDsDesc = MTL::DepthStencilDescriptor::alloc()->init();
        pDsDesc->setDepthCompareFunction( MTL::CompareFunction::CompareFunctionLess );
        pDsDesc->setDepthWriteEnabled( true );
        DepthStencilState = metalDevice->newDepthStencilState( pDsDesc );
        pDsDesc->release();
        
        MTL::TextureDescriptor* depthDesc = MTL::TextureDescriptor::alloc()->init();
        depthDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
        depthDesc->setWidth(width);
        depthDesc->setHeight(height);
        depthDesc->setUsage(MTL::TextureUsageRenderTarget);
        depthTexture = metalDevice->newTexture(depthDesc);
        depthDesc->release();
        
    }
    
    template <typename Type>
    simd_float4x4 Transformer(Shape<Type> shape) {
        return Translation(shape.position) * RotationX( shape.rotation.x) * RotationY( shape.rotation.y) * RotationZ( shape.rotation.z) * Scale( shape.scale);

    }
    
    void Render(MatrixH<uint8_t>& layer);
};



#endif /* MetalLayerHPPextension_h */

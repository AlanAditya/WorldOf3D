//
//  main.cpp
//  WorldOf3D
//
//  Created by Aditya Dudeja on 03/06/24.
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <iostream>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <string>
#include "StringConverter.hpp"
#include <random>
#include <vector>
#include <chrono>
#include "GPU_Compute.cpp"
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>
#include <simd/simd.h>
#include "MathOperations.cpp"
//#include "view_delegate.h"
//#include <Cocoa/Cocoa.h>




simd::float4x4 Identity() {
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Translation(simd::float3 dPos) {
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {dPos[0], dPos[1], dPos[2], 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationZ(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, sin, 0.0f, 0.0f};
    simd_float4 row1 = {-sin, cos, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationY(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, 0.0f, sin, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {-sin, 0.0f, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationX(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, cos, -sin, 0.0f};
    simd_float4 row2 = {0.0f, sin, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Scale(float scale) {
    simd_float4 row0 = {scale, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, scale, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, scale, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}


struct Vertex { simd::float2 pos; simd::float3 colour; };
struct Vertex3d { simd::float3 pos; simd::float3 colour; };
struct Mesh { MTL::Buffer* vertexBuffer, *indexBuffer; };

MTL::Buffer* BuildPoints(MTL::Device* device, int noOfPoints) {
    Vertex3d points[(2 * noOfPoints + 1) * (2 * noOfPoints + 1)];
    float space = 1 / (2 * noOfPoints + 1);
    for (int i= -(noOfPoints); i <= noOfPoints; i++) { points[i].pos[0] = i * space; }
    for (int i=0; i < noOfPoints; i++) { points[i].pos[1] = i * space; }
    for (int i=0; i < noOfPoints; i++) {
        for (int j=0; j < noOfPoints; j++) {
            points[i * noOfPoints + j].pos[2] = points[i * noOfPoints + j].pos[0] * points[i * noOfPoints + j].pos[1];
        }
    }
    MTL::Buffer* buffer = device->newBuffer(&points, sizeof(Vertex3d) * (2 * noOfPoints + 1) * (2 * noOfPoints + 1), MTL::ResourceStorageModeShared);
    return buffer;
}

MTL::Buffer* BuildTriangle(MTL::Device* device) {
    Vertex3d verticies[3] = {
        {{-0.6, -0.6, 0.0}, {1.0, 0.0, 0.4}},
        {{0.6, -0.6, 0.0}, {1.0, 0.0, 0.0}},
        {{0, 0.6, 0.0}, {1.0, 0.0, 100}},
    };
    
    MTL::Buffer* buffer = device->newBuffer(3 * sizeof(Vertex), MTL::ResourceStorageModeShared);
    memcpy(buffer->contents(), verticies, 3 * sizeof(Vertex));
    return buffer;
}

Mesh BuildQuad(MTL::Device* device) {
    Mesh mesh;
    Vertex3d verticies[4] = {
        {{-0.75, -0.75, 0.0}, {1.0, 0.0, 0.0}},
        {{0.75, -0.75, 0.0}, {0.0, 1.0, 0.0}},
        {{0.75, 0.75, 0.0}, {0.0, 0.0, 1.0}},
        {{-0.75, 0.75, 0.0}, {0.0, 0.0, 1.0}}
    };
    
    ushort indicies[6] = {0, 1, 2, 2, 3, 0};
    
    mesh.vertexBuffer = device->newBuffer(4 * sizeof(Vertex), MTL::ResourceStorageModeShared);
    mesh.indexBuffer = device->newBuffer(6 * sizeof(ushort), MTL::ResourceStorageModeShared);
    memcpy(mesh.vertexBuffer->contents(), verticies, 4 * sizeof(Vertex));
    memcpy(mesh.indexBuffer->contents(), indicies, 6 * sizeof(ushort));
    return mesh;
}

Mesh BuildCube(MTL::Device* device, float side) {
    Mesh mesh;
    Vertex3d verticies[24] = {
        // BACK FACE
        {{ -side, -side, side }, {0.0, 1.0, 0.0}},
        {{  side, -side, side }, {0.0, 1.0, 0.0}},
        {{  side,  side, side }, {0.0, 1.0, 0.0}},
        {{ -side,  side, side }, {0.0, 1.0, 0.0}},
        
        // LEFT FACE
        {{ side, -side,  side }, {0.0, 0.0, 1.0}},
        {{ side, -side, -side }, {0.0, 0.0, 1.0}},
        {{ side,  side, -side }, {0.0, 0.0, 1.0}},
        {{ side,  side,  side }, {0.0, 0.0, 1.0}},

        // FRONT FACE
        {{  side, -side, -side }, {0.0, 1.0, 0.0}},
        {{ -side, -side, -side }, {0.0, 1.0, 0.0}},
        {{ -side,  side, -side }, {0.0, 1.0, 0.0}},
        {{  side,  side, -side }, {0.0, 1.0, 0.0}},
        
        // RIGHT FACE
        {{ -side, -side, -side }, {0.0, 0.0, 1.0}},
        {{ -side, -side,  side }, {0.0, 0.0, 1.0}},
        {{ -side,  side,  side }, {0.0, 0.0, 1.0}},
        {{ -side,  side, -side }, {0.0, 0.0, 1.0}},
        
        // TOP FACE
        {{ -side, side,  side }, {1.0, 0.0, 0.0}},
        {{  side, side,  side }, {1.0, 0.0, 0.0}},
        {{  side, side, -side }, {1.0, 0.0, 0.0}},
        {{ -side, side, -side }, {1.0, 0.0, 0.0}},
        
        // BOTTOM FACE
        {{ -side, -side, -side}, {1.0, 0.0, 0.0}},
        {{  side, -side, -side}, {1.0, 0.0, 0.0}},
        {{  side, -side,  side}, {1.0, 0.0, 0.0}},
        {{ -side, -side,  side}, {1.0, 0.0, 0.0}}
    };
    
    ushort indicies[36] = {
        0, 1, 2, 2, 3, 0, // top face
        4, 5, 6, 6, 7, 4, // back face
        8, 9, 10, 10, 11, 8, // front face
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20, // bottom face
    };
    
    mesh.vertexBuffer = device->newBuffer(24*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
    mesh.indexBuffer = device->newBuffer(36*sizeof(ushort), MTL::ResourceStorageModeShared);
    memcpy(mesh.vertexBuffer->contents(), verticies, 24*sizeof(Vertex3d));
    memcpy(mesh.indexBuffer->contents(), indicies, 36*sizeof(ushort));
    return mesh;
}

class ViewDelegate : public MTK::ViewDelegate {
public:
    ViewDelegate(MTL::Device* device)
    : _device(device) {
        _commandQueue = _device->newCommandQueue();
        RenderLib = device->newDefaultLibrary();
        triangleMesh = BuildTriangle(device);
        QuadMesh = BuildQuad(device);
        CubeMesh = BuildCube(device, 0.4);
        trianglePipeline = BuildPipeline("vertexMain", "fragmentMain");
        shapePipeline = BuildPipeline("VertexMainShape", "FragmentMainShape");
        
    }

    ~ViewDelegate() {
        triangleMesh->release();
        trianglePipeline->release();
        
        shapePipeline->release();
        _commandQueue->release();
        _device->release();
    }
    
    MTL::RenderPipelineState* BuildPipeline(const char* vertexName, const char* fragName) {
        MTL::Function* vertexMain = RenderLib->newFunction(NS::String::string(vertexName, NS::StringEncoding::UTF8StringEncoding));
        MTL::Function* fragmentMain = RenderLib->newFunction(NS::String::string(fragName, NS::StringEncoding::UTF8StringEncoding));
        if (vertexMain == nil) {std::cout << "Failed to find the adder function."; }
        MTL::RenderPipelineDescriptor* descriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        descriptor->setVertexFunction(vertexMain);
        descriptor->setFragmentFunction(fragmentMain);
        descriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
//        descriptor->setDepthAttachmentPixelFormat(MTL::PixelFormat::PixelFormatDepth32Float);
        MTL::RenderPipelineState* polygonPipeline = _device->newRenderPipelineState(descriptor, &error);
        
//        MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init();
//        depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLess);
//        depthStencilDescriptor->setDepthWriteEnabled(true);
//        depthStencilState = _device->newDepthStencilState(depthStencilDescriptor);
        return polygonPipeline;
    }

    void drawInMTKView(MTK::View* view) override {
        time += 1.0f;
        if (time > 3000) { time = 0; }
        NS::AutoreleasePool* autoReleasePool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = _commandQueue->commandBuffer();

        MTL::RenderPassDescriptor* passDescriptor = view->currentRenderPassDescriptor();
//        passDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(1.0, 0.0, 0.0, 1.0));
//        passDescriptor->setDepthAttachment(MTL::RenderPassDepthAttachmentDescriptor(1));
        simd::float4x4 transform = Identity();
        simd::float3 dPos = {0.0f, 0.0f, 0.3f};
        simd::float4x4 translate = Translation(dPos) * RotationZ(time) * Scale(2 * sinf(time / 100));
        if (passDescriptor) {
            MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(passDescriptor);
            encoder->setRenderPipelineState(shapePipeline);
//            encoder->setDepthStencilState(depthStencilState);
            
//            encoder->setVertexBytes(&transform, sizeof(simd::float4x4), 1);
//            encoder->setVertexBuffer(QuadMesh.vertexBuffer, 0, 0);
//            encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(6), MTL::IndexType::IndexTypeUInt16, QuadMesh.indexBuffer, NS::UInteger(0), NS::UInteger(1));
//            
//            encoder->setVertexBytes(&transform, sizeof(simd::float4x4), 1);
//            encoder->setVertexBuffer(triangleMesh, 0, 0);
//            encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            
            simd::float4x4 cubeTransform = Translation(dPos) * RotationX(-45) * RotationY(-45)  ;
            encoder->setVertexBytes(&cubeTransform, sizeof(simd::float4x4), 1);
            encoder->setVertexBuffer(CubeMesh.vertexBuffer, 0, 0);
            encoder->setCullMode( MTL::CullMode::CullModeBack );
            encoder->setFrontFacingWinding( MTL::Winding::WindingCounterClockwise );
            encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(36), MTL::IndexType::IndexTypeUInt16, CubeMesh.indexBuffer, NS::UInteger(0), NS::UInteger(1));
            
            encoder->endEncoding();
            commandBuffer->presentDrawable(view->currentDrawable());
        }
        
        commandBuffer->commit();
        autoReleasePool->release();
    }

private:
    MTL::Device* _device;
    MTL::CommandQueue* _commandQueue;
    MTL::Library* RenderLib;
    NS::Error* error = nullptr;
    MTL::RenderPipelineState* trianglePipeline;
    MTL::RenderPipelineState* shapePipeline;
    MTL::DepthStencilState* depthStencilState;
    MTL::Buffer* triangleMesh;
    Mesh QuadMesh;
    Mesh CubeMesh;
    float time=0;
};

int main(int argc, const char * argv[]) {
    NS::AutoreleasePool* autoReleasePool = NS::AutoreleasePool::alloc()->init();
    NS::Application* app = NS::Application::sharedApplication();
    app->setActivationPolicy(NS::ActivationPolicyRegular);
    
    CGRect frame = (CGRect){ {100.0, 100.0}, {640.0, 640.0} };
    NS::Window* window = NS::Window::alloc()->init(frame, NS::WindowStyleMaskTitled | NS::WindowStyleMaskClosable | NS::WindowStyleMaskResizable, NS::BackingStoreBuffered, false);
    
    const NS::String* WinName = NS::String::string("FUCK YOU", NS::UTF8StringEncoding);
    window->setTitle(WinName);
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    NS::View ContentView = NS::View->ContentView;
    MTK::View* contentView = MTK::View::alloc()->init(frame, device);
    contentView->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
    contentView->setClearColor(MTL::ClearColor::Make(0, 0, 0, 1.0));
//    contentView->setDepthStencilPixelFormat( MTL::PixelFormat::PixelFormatDepth32Float );
//    contentView->setClearDepth( 1.0f );
    
    ViewDelegate* viewDelegate = new ViewDelegate(device);
    contentView->setDelegate(viewDelegate);
    window->setContentView(contentView);
    window->makeKeyAndOrderFront(nullptr);
    app->activateIgnoringOtherApps(true);
    app->run();
    
    autoReleasePool->release();
    return 0;
}




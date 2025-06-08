//
//  main.cpp
//  RotatingCube
//
//  Created by Manoj Kumar on 17/06/24.
//
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <iostream>
#include <MetalKit/MetalKit.hpp>
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <simd/simd.h>

struct Vertex3d { simd::float3 position; simd::float3 normal; simd::float3 colour; };
struct Mesh { MTL::Buffer* VertexBuffer; MTL::Buffer* IndexBuffer; };
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

class Renderer {
public: 
    Renderer( MTL::Device* pDevic ) {
        pDevice = pDevic;
        pCommandQueue = pDevice->newCommandQueue();
        buildShaders();
        buildBuffers();
        buildDepthStencilStates();
        std::cout << sizeof(simd::float3);
    }
    ~Renderer() {
        pShaderLibrry->release();
        pDepthStencilState->release();
        CubeMesh.VertexBuffer->release();
        CubeMesh.IndexBuffer->release();
        pPipelineState->release();
        pCommandQueue->release();
        pDevice->release();
    }
    void buildShaders() {
        NS::Error* pError = nullptr;
        MTL::Library* pLibrary = pDevice->newDefaultLibrary();
        MTL::Function* pVertexFunc = pLibrary->newFunction(NS::String::string("VertexMainCube", NS::StringEncoding::UTF8StringEncoding));
        MTL::Function* pFragFunc = pLibrary->newFunction(NS::String::string("FragmentMainCube", NS::StringEncoding::UTF8StringEncoding));
        
        MTL::RenderPipelineDescriptor* pPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        pPipelineDescriptor->setVertexFunction(pVertexFunc);
        pPipelineDescriptor->setFragmentFunction(pFragFunc);
        pPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pPipelineDescriptor->setDepthAttachmentPixelFormat( MTL::PixelFormat::PixelFormatDepth16Unorm );
        
        pPipelineState = pDevice->newRenderPipelineState(pPipelineDescriptor, &pError);
        pVertexFunc->release();
        pFragFunc->release();
        pPipelineDescriptor->release();
        pShaderLibrry = pLibrary;
    }
    void buildDepthStencilStates() {
        MTL::DepthStencilDescriptor* pDsDesc = MTL::DepthStencilDescriptor::alloc()->init();
        pDsDesc->setDepthCompareFunction( MTL::CompareFunction::CompareFunctionLess );
        pDsDesc->setDepthWriteEnabled( true );
        pDepthStencilState = pDevice->newDepthStencilState( pDsDesc );
        pDsDesc->release();
    }
    void buildBuffers() {
        Mesh mesh;
        const float s = 0.4f;

        Vertex3d verts[] = {
            //   Positions          Normals
            { { -s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},
            { { +s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},
            { { +s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},
            { { -s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},

            { { +s, -s, +s }, { 1.f,  0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { +s, -s, -s }, { 1.f,  0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { +s, +s, -s }, { 1.f,  0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { +s, +s, +s }, { 1.f,  0.f,  0.f }, {0.0, 0.0, 1.0}},

            { { +s, -s, -s }, { 0.f,  0.f, -1.f }, {0.0, 1.0, 0.0}},
            { { -s, -s, -s }, { 0.f,  0.f, -1.f }, {0.0, 1.0, 0.0}},
            { { -s, +s, -s }, { 0.f,  0.f, -1.f }, {0.0, 1.0, 0.0}},
            { { +s, +s, -s }, { 0.f,  0.f, -1.f }, {0.0, 1.0, 0.0}},

            { { -s, -s, -s }, { -1.f, 0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { -s, -s, +s }, { -1.f, 0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { -s, +s, +s }, { -1.f, 0.f,  0.f }, {0.0, 0.0, 1.0}},
            { { -s, +s, -s }, { -1.f, 0.f,  0.f }, {0.0, 0.0, 1.0}},

            { { -s, +s, +s }, { 0.f,  1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { +s, +s, +s }, { 0.f,  1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { +s, +s, -s }, { 0.f,  1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { -s, +s, -s }, { 0.f,  1.f,  0.f }, {1.0, 0.0, 0.0}},

            { { -s, -s, -s }, { 0.f, -1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { +s, -s, -s }, { 0.f, -1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { +s, -s, +s }, { 0.f, -1.f,  0.f }, {1.0, 0.0, 0.0}},
            { { -s, -s, +s }, { 0.f, -1.f,  0.f }, {1.0, 0.0, 0.0}}
        };
        uint16_t indices[] = {
             0,  1,  2,  2,  3,  0, /* front */
             4,  5,  6,  6,  7,  4, /* right */
             8,  9, 10, 10, 11,  8, /* back */
            12, 13, 14, 14, 15, 12, /* left */
            16, 17, 18, 18, 19, 16, /* top */
            20, 21, 22, 22, 23, 20, /* bottom */
        };
        mesh.VertexBuffer = pDevice->newBuffer(24*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        mesh.IndexBuffer = pDevice->newBuffer(36*sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(mesh.VertexBuffer->contents(), verts, 24*sizeof(Vertex3d));
        memcpy(mesh.IndexBuffer->contents(), indices, 36*sizeof(uint16_t));
        CubeMesh = mesh;
    }
    void draw( MTK::View* pView ) {
        time += 1;
        if (360 < time) { time = 0; }
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        simd::float3 dPos = {0.0f, 0.0f, 0.7f};
        simd::float4x4 transform = Translation(dPos) * RotationX(time) * RotationY(time);
        
        MTL::RenderPassDescriptor* pPassDescriptor = pView->currentRenderPassDescriptor();

        pPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadAction::LoadActionClear);
        pPassDescriptor->depthAttachment()->setClearDepth(1.0f);
        
        MTL::RenderCommandEncoder* encoder = pCommandBuffer->renderCommandEncoder(pPassDescriptor);
        encoder->setRenderPipelineState(pPipelineState);
        encoder->setDepthStencilState(pDepthStencilState);
        encoder->setVertexBytes(&transform, sizeof(simd::float4x4), 1);
        encoder->setVertexBuffer(CubeMesh.VertexBuffer, 0, 0);
        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(36), MTL::IndexType::IndexTypeUInt16, CubeMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
//        encoder->setCullMode( MTL::CullMode::CullModeBack );
//        encoder->setFrontFacingWinding( MTL::Winding::WindingCounterClockwise );
        encoder->endEncoding();
        pCommandBuffer->presentDrawable(pView->currentDrawable());
        pCommandBuffer->commit();
        pPool->release();
    }
    
private:
    MTL::Device* pDevice;
    MTL::CommandQueue* pCommandQueue;
    MTL::Library* pShaderLibrry;
    MTL::RenderPipelineState* pPipelineState;
    MTL::DepthStencilState* pDepthStencilState;
    Mesh CubeMesh;
    float time = 0;
};

class MyMtkViewDelegate: public MTK::ViewDelegate {
public:
    MyMtkViewDelegate(MTL::Device* pDevice)
    : pRenderer( new Renderer( pDevice ) ) {
        
    }
    
    ~MyMtkViewDelegate() { delete pRenderer; }
    
    virtual void drawInMTKView(class MTK::View *pView) override { pRenderer->draw(pView); }
private:
    Renderer* pRenderer;
};

class MyAppDelegate: public NS::ApplicationDelegate {
public:
    ~MyAppDelegate() {
        pMtkView->release();
        pWindow->release();
        pDevice->release();
        delete pViewDelegate;
    }
    virtual void applicationWillFinishLaunching(NS::Notification *pNotification) override {
        NS::Application* pApp = reinterpret_cast< NS::Application* >(pNotification->object());
        pApp->setActivationPolicy(NS::ActivationPolicy::ActivationPolicyRegular);
    }
    
    virtual void applicationDidFinishLaunching(NS::Notification *pNotification) override {
        CGRect frame = (CGRect) {{100, 100}, {640.0, 640.0}};
        pWindow = NS::Window::alloc()->init(
            frame, NS::WindowStyleMaskClosable | NS::WindowStyleMaskTitled | NS::WindowStyleMaskResizable, NS::BackingStoreBuffered, false);
        pDevice = MTL::CreateSystemDefaultDevice();
        
        pMtkView = MTK::View::alloc()->init(frame, pDevice);
        pMtkView->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pMtkView->setClearColor(MTL::ClearColor::Make(0, 0, 0, 1.0));
        pMtkView->setDepthStencilPixelFormat(MTL::PixelFormat::PixelFormatDepth16Unorm);
        pMtkView->setClearDepth(1.0f);
        
        pViewDelegate = new MyMtkViewDelegate( pDevice );
        pMtkView->setDelegate(pViewDelegate);
        
        pWindow->setContentView(pMtkView);
        pWindow->setTitle(NS::String::string("Rotating Cube", NS::UTF8StringEncoding));
        pWindow->makeKeyAndOrderFront( nullptr );
        NS::Application* pApp = reinterpret_cast< NS::Application* >(pNotification->object());
        pApp->activateIgnoringOtherApps(true);
    }
    
    virtual bool applicationShouldTerminateAfterLastWindowClosed(class NS::Application *pSender) override { return true; }
    
private:
    NS::Window* pWindow;
    MTK::View* pMtkView;
    MTL::Device* pDevice;
    MyMtkViewDelegate* pViewDelegate = nullptr;
};

int main(int argc, const char * argv[]) {
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    MyAppDelegate delegate;
    
    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate(&delegate);
    pSharedApplication->run();
    
    pAutoreleasePool->release();
    return 0;
}

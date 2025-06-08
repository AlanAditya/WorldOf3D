//
//  main.cpp
//  ImageProcessing
//
//  Created by Manoj Kumar on 04/08/24.
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <iostream>
#include <MetalKit/MetalKit.hpp>
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#import <QuartzCore/CAMetalLayer.h>
#include <Foundation/Foundation.hpp>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreGraphics/CGImage.h>
#include <ImageIO/ImageIO.h>
#include <simd/simd.h>
//#include "ImageProcessing-Swift.h"
//#import "MySwiftUIViewControllerBridge.h"
#include "Test.hpp"
//#import <Cocoa/Cocoa.h>
#include "ViewImporterBridge.hpp"
//#include "Test.hpp"



struct Vertex3d { simd::float3 position; simd::float3 normal; simd::float3 colour;};
struct Mesh { MTL::Buffer* VertexBuffer; MTL::Buffer* IndexBuffer; int SizeOfIndex = 0;};

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

const Vertex3d vertexData[] = {
    {{-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}},
    {{-1.0f, -1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}},
    {{1.0f, -1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}},
    {{1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 0.0f}}
};

const ushort indicies[6] = {
    0, 1, 2, 2, 3, 0, // top face
};

class Renderer {
public:
    Renderer( MTL::Device* pDevic, Object3D* package ) {
        pDevice = pDevic;
        pObjects = package;
        pCommandQueue = pDevice->newCommandQueue();
        buildShaders();
        getImgTexture();
        VertexBuffer = pDevice->newBuffer(vertexData, sizeof(vertexData), MTL::ResourceStorageModeShared);
        buildBuffersQuad();
        buildDepthStencilStates();
        buildCube();
        buildSphere({0.0f, 0.0f, 0.0f}, 0.5f, 300);
        BuildDonut({0.0f, 0.0f, 0.0f}, 0.25f, 0.5f, 30, 100);
    }
    ~Renderer() {
        pShaderLibrry->release();
        pPipelineState->release();
        pCommandQueue->release();
        pDevice->release();
    }
    void buildBuffersQuad() {
        Mesh mesh;
        const float s = 1;

        Vertex3d verts[] = {
            //   Positions          Normals
            { { -s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { +s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { +s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { -s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
        };
        uint16_t indices[] = {
             0,  1,  2,  2,  3,  0, /* front */
        };
        mesh.VertexBuffer = pDevice->newBuffer(4*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        mesh.IndexBuffer = pDevice->newBuffer(6*sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(mesh.VertexBuffer->contents(), verts, 4*sizeof(Vertex3d));
        memcpy(mesh.IndexBuffer->contents(), indices, 6*sizeof(uint16_t));
        QuadMesh = mesh;
    }
    
    void buildCube() {
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
    void buildSphere(simd::float3 centre, float radius, const int Subdivisions) {
        int radialDivisions = Subdivisions;
        int lengthDivisions = (Subdivisions - 2) / 2;
        float rad = 0;
        simd::float3 radiusVec;
        int indexBufferSIZE = 2 * (radialDivisions+1) * (lengthDivisions + 1);
        uint16_t indices[indexBufferSIZE];
        float Angle = 0;
        float Height = 0;
        float CD=0;
        Vertex3d* points = new Vertex3d[1 + (radialDivisions * lengthDivisions) + 1];
        
        points[0] = {{centre.x, centre.y - radius, centre.z}, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}};
        for (int k = 0; k< radialDivisions; k++) {
            indices[2*k] = 0;
            indices[2*k + 1] = 1 + k;
        }
        indices[2*radialDivisions] = 0;
        indices[2*radialDivisions+1] = 1;
        
        points[1 + (radialDivisions * lengthDivisions)] = {{centre.x, centre.y + radius, centre.z}, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}};
        for (int j = 0; j < radialDivisions; j++) {
            indices[ 2*((radialDivisions+1)*(lengthDivisions)) + 2*j + 1] = 1 + ((radialDivisions) * lengthDivisions);
            indices[ 2*((radialDivisions+1)*(lengthDivisions)) + 2*j ] = 1 + ((radialDivisions) * (lengthDivisions-1)) + (j);
        }
        indices[2*(radialDivisions+1)*(lengthDivisions+1)-2] = 1 + ((radialDivisions) * (lengthDivisions-1));
        indices[2*(radialDivisions+1)*(lengthDivisions+1) -1] = 1 + (radialDivisions * lengthDivisions);

        for (int i =0; i < lengthDivisions; i++) {
            CD = (M_PI / (lengthDivisions+1));
            Height = - radius * cos(CD * (i+1));
            
            
            for (int j = 0; j < radialDivisions; j++) {
                Angle = j * (2 * M_PI / radialDivisions);
//                Height = (- radius) + (2*radius / (lengthDivisions + 1)) * (i+1);
                rad = sqrt((radius * radius) - (Height * Height));
                radiusVec = {rad * cos(Angle), Height, rad * sin(Angle)};
                points[1 + (i)*radialDivisions + j] = {centre + radiusVec, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}};
                if (i < lengthDivisions-1) {
                    indices[2*(radialDivisions+1) + 2*(i)*(radialDivisions+1) +(2*j)] = 1 + radialDivisions*(i) + (j);
                    indices[2*(radialDivisions+1) + 2*(i)*(radialDivisions+1) +(2*j) + 1] = 1 + radialDivisions*(i+1) + (j);
                }
            }
            if (i < lengthDivisions-1) {
                indices[2*(radialDivisions+1) + 2*(i+1)*(radialDivisions+1) - 2] = 1 + radialDivisions*(i) ;
                indices[2*(radialDivisions+1) + 2*(i+1)*(radialDivisions+1) - 1] = 1 + radialDivisions*(i+1) ;
            }
        }
        Sphere.VertexBuffer = pDevice->newBuffer((1 + (radialDivisions * lengthDivisions) + 1) * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        Sphere.IndexBuffer = pDevice->newBuffer(indexBufferSIZE * sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(Sphere.VertexBuffer->contents(), points, (1 + (radialDivisions * lengthDivisions) + 1) * sizeof(Vertex3d));
        memcpy(Sphere.IndexBuffer->contents(), indices, indexBufferSIZE * sizeof(uint16_t));
        Sphere.SizeOfIndex = indexBufferSIZE;
    }
    float magnitude(simd::float3 v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    void BuildDonut(simd::float3 centre, float iRad, float oRad, int iDiv, int oDiv) {
        simd::float3 oRadVec;
        simd::float3 b2;
        simd::float3 b3;
        simd::float3 iRadVec;
        float len = 0;
        int indexBufferSIZE = 2 * (oDiv) * (iDiv + 1);
        simd::float3x3 basisChangeMatrix;
        uint16_t indices[indexBufferSIZE];
        Vertex3d* points = new Vertex3d[iDiv * oDiv];
        simd::float3* CenPoints = new simd::float3[oDiv];
        float oAng = 0;
        float iAng = 0;
        
        for (int k = 0; k < oDiv; k ++) {
            oAng = (2* M_PI / oDiv) * k;
            oRadVec = {cos(oAng), 0.0f, sin(oAng)};
            CenPoints[k] = oRadVec;
        }
        
        len = 1 / (magnitude(simd::normalize(CenPoints[2] - CenPoints[1]) + simd::normalize(CenPoints[1] - CenPoints[0])) / 2);
        
        
        for (int i = 0; i < oDiv; i ++) {
            oAng = (2* M_PI / oDiv) * i;
            oRadVec = {cos(oAng), 0.0f, sin(oAng)};
            b2 ={0.0f, 1.0f, 0.0f};
            b3 = {0.0f, 0.0f, 0.0f};
            basisChangeMatrix = simd_matrix(-oRad, b2, b3);
//            basisChangeMatrix = simd::transpose(basisChangeMatrix);
            
            oRadVec = oRad * oRadVec;
            
            for (int j=0; j < iDiv; j++) {
                iAng = (2* M_PI / iDiv) * j;
                iRadVec = simd::normalize((-oRadVec / oRad)*cos(iAng) + b2*sin(iAng));
                
                points[iDiv*i + j] = {centre + oRadVec + ( len * iRad * iRadVec), { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}};
                if (i < oDiv-1) {
                    indices[2*(iDiv+1)*i + 2*j] = iDiv*i + j;
                    indices[2*(iDiv+1)*i + 2*j + 1] = iDiv*(i+1) + j;
                } else {
                    indices[2*(iDiv+1)*i + 2*j] = iDiv*i + j;
                    indices[2*(iDiv+1)*i + 2*j + 1] = j;
                }
            }
            if (i < oDiv-1) {
                indices[2*(iDiv+1)*(i+1) - 2] = iDiv*i;
                indices[2*(iDiv+1)*(i+1) - 1] = iDiv*(i+1);
            } else {
                indices[2*(iDiv+1)*(i+1) - 2] = iDiv*i;
                indices[2*(iDiv+1)*(i+1) - 1] = 0;
            }
        }
        
        Donut.VertexBuffer = pDevice->newBuffer(iDiv * oDiv * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        Donut.IndexBuffer = pDevice->newBuffer(indexBufferSIZE * sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(Donut.VertexBuffer->contents(), points, iDiv * oDiv * sizeof(Vertex3d));
        memcpy(Donut.IndexBuffer->contents(), indices, indexBufferSIZE * sizeof(uint16_t));
        Donut.SizeOfIndex = indexBufferSIZE;
    }
    void buildShaders() {
        NS::Error* pError = nullptr;
        MTL::Library* pLibrary = pDevice->newDefaultLibrary();
        MTL::Function* pVertexFunc = pLibrary->newFunction(NS::String::string("vertex_main", NS::StringEncoding::UTF8StringEncoding));
        MTL::Function* pFragFunc = pLibrary->newFunction(NS::String::string("fragment_main", NS::StringEncoding::UTF8StringEncoding));
        
        MTL::RenderPipelineDescriptor* pPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        pPipelineDescriptor->setVertexFunction(pVertexFunc);
        pPipelineDescriptor->setFragmentFunction(pFragFunc);
        pPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pPipelineDescriptor->setDepthAttachmentPixelFormat( MTL::PixelFormat::PixelFormatDepth32Float );
        pPipelineState = pDevice->newRenderPipelineState(pPipelineDescriptor, &pError);
        pVertexFunc->release();
        pFragFunc->release();
        pPipelineDescriptor->release();
        pShaderLibrry = pLibrary;
    }
    void draw( MTK::View* pView ) {
        time += 1.0f;
        if (time > 3000) { time = 0; }
        if (!pTexture) return;
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        MTL::RenderPassDescriptor* pPassDescriptor = pView->currentRenderPassDescriptor();
        if (!pPassDescriptor) return;
        MTL::RenderCommandEncoder* encoder = pCommandBuffer->renderCommandEncoder(pPassDescriptor);
        
        
        encoder->setRenderPipelineState(pPipelineState);
        encoder->setDepthStencilState(pDepthStencilState);
        
//        encoder->setFragmentTexture(pTexture, 0);
//        encoder->setVertexBuffer(CubeMesh.VertexBuffer, 0, 0);
//        simd::float3 dPos = {0.0f, 0.0f, 0.7f};
//        simd::float4x4 cubeTransform = Translation(dPos) * RotationX(time ) * RotationY(time ) * RotationZ(time ) ;
//        encoder->setVertexBytes(&cubeTransform, sizeof(simd::float4x4), 1);
//        encoder->setFragmentTexture( pTexture, /* index */ 0 );
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(36), MTL::IndexType::IndexTypeUInt16, CubeMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
//        encoder->setVertexBuffer(Sphere.VertexBuffer, 0, 0);
//        simd::float3 dPos = {0.0f, 0.0f, 0.7f};
//        simd::float4x4 cubeTransform = Translation(dPos) * RotationX(time ) * RotationY(time ) * RotationZ(time ) ;
//        encoder->setVertexBytes(&cubeTransform, sizeof(simd::float4x4), 1);
//        encoder->setFragmentTexture( pTexture, /* index */ 0 );
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(Sphere.SizeOfIndex), MTL::IndexType::IndexTypeUInt16, Sphere.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
        encoder->setVertexBuffer(Donut.VertexBuffer, 0, 0);
        simd::float3 dPos = {pObjects[0].location.x, 0.0f, 0.8f};
        simd::float4x4 cubeTransform = Translation(dPos) * RotationX(time ) * RotationY(time ) * Scale(pObjects[0].scale.x) ;
//        cubeTransform = Identity();
        encoder->setVertexBytes(&cubeTransform, sizeof(simd::float4x4), 1);
        encoder->setFragmentTexture( pTexture, /* index */ 0 );
        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(Donut.SizeOfIndex), MTL::IndexType::IndexTypeUInt16, Donut.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
        
        encoder->endEncoding();
        pCommandBuffer->presentDrawable(pView->currentDrawable());
        pCommandBuffer->commit();
        pPool->release();
    }
    void buildDepthStencilStates() {
        MTL::DepthStencilDescriptor* pDsDesc = MTL::DepthStencilDescriptor::alloc()->init();
        pDsDesc->setDepthCompareFunction( MTL::CompareFunction::CompareFunctionLess );
        pDsDesc->setDepthWriteEnabled( true );
        pDepthStencilState = pDevice->newDepthStencilState( pDsDesc );
        pDsDesc->release();
    }
    void getImgTexture() {
        CFStringRef path = CFStringCreateWithCString(NULL, "/Users/adityadude/Documents/IMG_1278.JPG", kCFStringEncodingUTF8);
        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
        CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(url);
        CFRelease(path);

        if (!cgImage) {
            std::cerr << "Failed to create CGImage" << std::endl;
            return;
        }
        size_t width = CGImageGetWidth(cgImage);
        size_t height = CGImageGetHeight(cgImage);

        MTL::TextureDescriptor *desc = MTL::TextureDescriptor::alloc()->init();
        desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        desc->setWidth(width);
        desc->setHeight(height);
        desc->setUsage(MTL::TextureUsageShaderRead);
        pTexture = pDevice->newTexture(desc);

        size_t bytesPerRow = 4 * width;
        void *data = malloc(bytesPerRow * height);
        CGContextRef context = CGBitmapContextCreate(data, width, height, 8, bytesPerRow,
                                                     CGImageGetColorSpace(cgImage),
                                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
        CGContextRelease(context);

        MTL::Region region = MTL::Region(0, 0, width, height);
        pTexture->replaceRegion(region, 0, data, bytesPerRow);
        free(data);
        CGImageRelease(cgImage);
        CFRelease(source);
    }
private:
    MTL::Device* pDevice;
    MTL::CommandQueue* pCommandQueue;
    MTL::Library* pShaderLibrry;
    MTL::RenderPipelineState* pPipelineState;
    MTL::DepthStencilState* pDepthStencilState;
    MTL::Texture* pTexture;
    MTL::Buffer* VertexBuffer;
    Mesh QuadMesh;
    Mesh CubeMesh;
    Mesh Sphere;
    Mesh Donut;
    Object3D* pObjects;
    float time;
};

class MyMtkViewDelegate: public MTK::ViewDelegate {
public:
    MyMtkViewDelegate(MTL::Device* pDevice, Object3D* objs)
    : pRenderer( new Renderer( pDevice, objs ) ) {
        
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
        pMtkView->setDepthStencilPixelFormat(MTL::PixelFormat::PixelFormatDepth32Float);
        pMtkView->setClearDepth(1.0f);
        

        
        float A = 0.0f;
//        NS::View *pSwiftView = createHelloWorldView();
        
//        NS::Object Class = SwiftUI_C_Plus_plus();
//        createHelloWorldSwiftViewMTK(pMtkView);
        ExchangePackage package = SwiftCPP_Bridge(pMtkView);
        objs = package.objec;
        pViewDelegate = new MyMtkViewDelegate( pDevice,  objs );
        pMtkView->setDelegate(pViewDelegate);
//        NS::View *pSwiftView = SwiftCPP_Bridge(pMtkView);
        pWindow->setContentView(package.view);
        
        std::cout << "ID" << objs[0].location.x << std::endl;
//        pWindow->setContentView(pMtkView);
        pWindow->setTitle(NS::String::string("Img Processing", NS::UTF8StringEncoding));
        pWindow->makeKeyAndOrderFront( nullptr );
        NS::Application* pApp = reinterpret_cast< NS::Application* >(pNotification->object());
        pApp->activateIgnoringOtherApps(true);
    }
    
    virtual bool applicationShouldTerminateAfterLastWindowClosed(class NS::Application *pSender) override { return true; }
    
private:
    NS::Window* pWindow;
    MTK::View* pMtkView;
    MTL::Device* pDevice;
    Object3D* objs;
    MyMtkViewDelegate* pViewDelegate = nullptr;
};

int main(int argc, const char * argv[]) {
    
//    CFStringRef path = CFStringCreateWithCString(NULL, "/Users/adityadude/Documents/IMG_1278.JPG", kCFStringEncodingUTF8);
//    CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
//    CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
//    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
//    CFRelease(url);
//    CFRelease(path);
//    
//    ImageProcessing::PhotoPickerView::init(&cgImage, false);
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    MyAppDelegate delegate;
    
    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate(&delegate);
    pSharedApplication->run();
    
    pAutoreleasePool->release();
    return 0;
}

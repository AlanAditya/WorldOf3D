//
//  main.cpp
//  Scene2D
//
//  Created by Manoj Kumar on 19/06/24.
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
#include <VisionKit/VisionKit.h>

MTL::PixelFormat depth = MTL::PixelFormat::PixelFormatDepth32Float ;

struct Vertex3d { simd::float3 position; simd::float3 normal; simd::float3 colour; };
struct Mesh { MTL::Buffer* VertexBuffer; MTL::Buffer* IndexBuffer; int SizeOfIndex = 0; };
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

void numpyArrange(float lowerLimit, float upperLimit, int noOfPoints, simd::float1* array) {
    for (int i = 0; i < noOfPoints; i++) {
        array[i] = lowerLimit + (i * ((upperLimit - lowerLimit) / (noOfPoints - 1)));
    }
}

float sigmoid(float x) {
    if (0.0001 > x > 0) {
        return 0;
    } else if (x < 0) {
        return -1;
    } else if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}


class Renderer {
public:
    Renderer( MTL::Device* pDevic ) {
        pDevice = pDevic;
        pCommandQueue = pDevice->newCommandQueue();
        buildShaders();
        buildBuffers();
        buildBuffersQuad();
        BuildGraph(601, 2.0f);
        BuildCircle(0.5f, {0.0f, 0.0f, 0.0f}, 60);
        buildSphere({0.0f, 0.0f, 0.0f}, 0.5f, 20);
        const float s = 0.6f;

        Vertex3d verts[] = {
            //   Positions          Normals
            { { -s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { +s, -s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { +s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { -s, +s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            { { -s + 0.2f, 2 * s, +s }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},
        };
        BuildLine(verts, 5, 2);
        Vertex3d pipePts[700] = {
            //   Positions          Normals
//            { { 0.4f, 0.6f, 0.7f }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
//            { { 0.0f, 0.6f, 0.7f }, { 0.f,  0.f,  1.f }, {0.0f, 1.0f, 0.0f}},
//            { { 0.0f, 0.0f, 0.7f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 1.0f}},
//            { { 0.0f, 0.0f, 0.3f }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}}
//            { { 0.0f, -0.2f, 0.0f }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}},
            
            { { 0.4f, 0.0f, 0.0f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.4f, 0.61f, 0.0f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { -0.4f, 0.6f, 0.1f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { -0.41f, 0.61f, 0.8f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            
            { { 0.4f, 0.6f, 0.81f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.41f, 0.0f, 0.80f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { -0.41f, 0.0f, 0.81f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { -0.71f, -0.4f, 0.81f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.71f, -0.4f, 0.81f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.71f, -0.4f, 0.4f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.71f, 0.41f, -0.0f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 1.0f, 0.91f, 0.5f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.8f, 0.51f, 0.9f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.81f, 0.21f, 0.89f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { 0.8f, 0.01f, 0.59f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
            { { -0.8f, 0.0f, 0.59f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}}
//
//            { { -0.4f, 0.6f, 0.7f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { 0.4f, 0.61f, 0.7f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { 0.41f, 0.0f, 0.71f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { 0.41f, 0.3f, 0.91f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}}
            
//            { { 0.4f, 0.4f, 0.0f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { 0.0, 0.0, 0.0f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { 0.0f, 0.0f, 0.7f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}},
//            { { -0.4f, 0.4f, 0.7f }, { 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}}
        };
//        dataGenHelix(pipePts, 500);
        buildPipeSimpleV2(pipePts, 16, 40, 0.1f);
//        buildPipeCapped(pipePts, 13, 60, 0.1f);
        buildDepthStencilStates();
        
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
    
    void dataGenHelix(Vertex3d *pts, int noOfPts) {
        float height = 1.0f;
        float time = 0.0;
        for (int i = 0; i < noOfPts; i++) {
            time++;
            pts[i] = {{cos(time*0.03f), sin(time*0.03f), time*(height / noOfPts)} ,{ 0.f,  0.f,  1.f }, {1.0f, 0.0f, 0.0f}};
        }
    }
    
    void buildShaders() {
        NS::Error* pError = nullptr;
        MTL::Library* pLibrary = pDevice->newDefaultLibrary();
        MTL::Function* pVertexFunc = pLibrary->newFunction(NS::String::string("VertexMainScene", NS::StringEncoding::UTF8StringEncoding));
        MTL::Function* pFragFunc = pLibrary->newFunction(NS::String::string("FragmentMainScene", NS::StringEncoding::UTF8StringEncoding));
        
        MTL::RenderPipelineDescriptor* pPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
        pPipelineDescriptor->setVertexFunction(pVertexFunc);
        pPipelineDescriptor->setFragmentFunction(pFragFunc);
        pPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pPipelineDescriptor->setDepthAttachmentPixelFormat( depth );
        
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
        };
        uint16_t indices[] = {
             0,  1,  2,  2,  3,  0, /* front */
        };
        mesh.VertexBuffer = pDevice->newBuffer(4*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        mesh.IndexBuffer = pDevice->newBuffer(6*sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(mesh.VertexBuffer->contents(), verts, 4*sizeof(Vertex3d));
        memcpy(mesh.IndexBuffer->contents(), indices, 6*sizeof(uint16_t));
        CubeMesh = mesh;
    }
    
    void buildBuffersQuad() {
        Mesh mesh;
        const float s = 0.6f;

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
    
    void BuildGraph(int noOfPoints, float time) {
        Mesh mesh;
        simd::float1 xCoord[noOfPoints];
        simd::float1 yCoord[noOfPoints];
        numpyArrange(-1, 1, noOfPoints, &xCoord[0]);
        memcpy(yCoord, xCoord, noOfPoints * sizeof(simd::float1));
        
        Vertex3d verts[2 * noOfPoints];
        
        for (int i = 0; i < noOfPoints; i++) {
            simd::float1 y = exp(-5 * abs(yCoord[i])) * 0.3f * (sin(0.1f * time) + 1.1f) * cos(50 * yCoord[i]);
            simd::float1 dy = exp(-5 * abs(yCoord[i])) * 0.3f * (sin(0.1f * time) + 1.1f) * 50 * -sin(50 * yCoord[i]) +                                                         (-5 * sigmoid(xCoord[i])) * exp(-5 * abs(yCoord[i])) * 0.3f * (sin(0.1f * time) + 1.1f) * cos(50 * yCoord[i]);
            simd::float1 normaliser = 100.0f * sqrt((1) + (dy * dy));
            verts[2 * i] = {{ xCoord[i], y, 0.0f }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}};
            
            verts[2 * i + 1] = {{ xCoord[i] - (dy / normaliser), y + (1 / normaliser), 0.0f }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}};
            
//            verts[2 * i] = {{ xCoord[i], sin(xCoord[i]), 0 }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}};
//            verts[(2 * i) + 1] = {{ xCoord[i], sin(xCoord[i]) + 0.01f, 0 }, { 0.f,  0.f,  1.f }, {0.0f, 0.0f, 1.0f}};
        }
        
        mesh.VertexBuffer = pDevice->newBuffer(2 * noOfPoints*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        memcpy(mesh.VertexBuffer->contents(), verts, 2 * noOfPoints*sizeof(Vertex3d));
        GraphMesh = mesh;
        
    }
    
    
    void BuildLine(Vertex3d* pts, const int noOfPoints, float thick) {
        Mesh line;
        simd::float2 der;
        simd::float2 der1;
        simd::float2 der2;
        float newX, newY;
        float normaliser;
        Vertex3d verts[2 * noOfPoints];
        for (int i = 0; i < noOfPoints; i++) {
            if (i == 0) {
                der = { (pts[i+1].position.x - pts[i].position.x), (pts[i+1].position.y - pts[i].position.y) };
//                std::cout << der.x << " " << pts[i+1].position.x << " "<< pts[i].position.x << std::endl;
            }
            else if (i == noOfPoints-1){
                der = { (pts[i].position.x - pts[i-1].position.x), (pts[i].position.y - pts[i-1].position.y) };
            }
            else {
                der1 = { (pts[i].position.x - pts[i-1].position.x), (pts[i].position.y - pts[i-1].position.y) };
                der2 = { (pts[i+1].position.x - pts[i].position.x), (pts[i+1].position.y - pts[i].position.y) };
                float nrm1 = sqrt((der1.x * der1.x) + (der1.y * der1.y));
                float nrm2 = sqrt((der2.x * der2.x) + (der2.y * der2.y));
                der = ((der1 / nrm1) + (der2 / nrm2))/2;
            }
            verts[2 * i] = pts[i];
            normaliser = 100.0f * sqrt((der.y * der.y) + (der.x * der.x));
//            std::cout << " Iteration: " << i << " " << der.x << " " << der.y << std::endl;
            newX = pts[i].position.x - thick * (der.y / normaliser);
            newY = pts[i].position.y + thick * (der.x / normaliser) ;
            verts[2 * i + 1] = { { newX, newY, pts[i].position.z}, pts[i].normal, pts[i].colour};
        }
        line.VertexBuffer = pDevice->newBuffer(2 * noOfPoints * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        memcpy(line.VertexBuffer->contents(), verts, 2 * noOfPoints*sizeof(Vertex3d));
        LineMesh = line;
    }
    
    void BuildCircle(float radius, simd::float3 centre, int noOfPoints) {
        Mesh Circle;
        Vertex3d points[noOfPoints + 1];
        uint16_t indices[2 * noOfPoints + 2];
        float CD = 2 * M_PI / noOfPoints;
        points[0] = {{centre.x, centre.y, centre.z}, {0, 0, 0}, {1, 0, 0}};
        
        for (int i=1; i < noOfPoints+1; i++) {
            float Ang = 0 + (CD * (i));
            points[i] = {{centre.x + (radius * cos(Ang)), centre.y + (radius * sin(Ang)), centre.z}, {0, 0, 0}, {1, 0, 0}};
            indices[2 * i - 2 ] = 0;
            indices[2 * i - 1] = i;
        }
        
        indices[2 * noOfPoints] = 0;
        indices[2 * noOfPoints + 1] = 1;
        
        Circle.VertexBuffer = pDevice->newBuffer((noOfPoints + 1) * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        Circle.IndexBuffer = pDevice->newBuffer((2 * noOfPoints + 2) * sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(Circle.VertexBuffer->contents(), points, (noOfPoints + 1) * sizeof(Vertex3d));
        memcpy(Circle.IndexBuffer->contents(), indices, (2 * noOfPoints + 2) * sizeof(uint16_t));
        CircleMesh = Circle;
    }
    
    void buildPipeSimpleV1(Vertex3d* pts, const int noOfPoints, int CirclePts, float radius) {
        Mesh Pipe;
        simd::float3 der;
        simd::float3 der1;
        simd::float3 der2;
        simd::float3 centre;
        Vertex3d points[(noOfPoints * CirclePts) + noOfPoints];
        float newX, newY;
        float normaliser;
        float CD = 2 * M_PI / CirclePts;
        for (int i = 0; i < CirclePts; i++) {
            float Ang = 0 + (CD * (i));
            for (int j = 0; j < noOfPoints; j++) {
                centre = pts[j].position;
                points[i * noOfPoints + j] = {{centre.x + (radius * cos(Ang)), centre.y + (radius * sin(Ang)), centre.z}, {0, 0, 0}, {1, 0, 0}};
            }
        }
        points[CirclePts * noOfPoints + 0] = points[0];
        points[CirclePts * noOfPoints + 1] = points[1];
        
        Pipe.VertexBuffer = pDevice->newBuffer(((CirclePts * noOfPoints) + noOfPoints) * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        memcpy(Pipe.VertexBuffer->contents(), points, ((CirclePts * noOfPoints) + noOfPoints) * sizeof(Vertex3d));
        PipeMesh = Pipe;
    }
    
    simd::float3 derivative(int i, Vertex3d* pts, int noOfPoints) {
        simd::float3 der;
        simd::float3 der1;
        simd::float3 der2;
        if (i == 0) {
            der = { (pts[i+1].position.x - pts[i].position.x), (pts[i+1].position.y - pts[i].position.y),  (pts[i+1].position.z - pts[i].position.z)};
            der = der / magnitude(der);
//                std::cout << der.x << " " << pts[i+1].position.x << " "<< pts[i].position.x << std::endl;
        }
        else if (i == noOfPoints-1) {
            der = { (pts[i].position.x - pts[i-1].position.x), (pts[i].position.y - pts[i-1].position.y), (pts[i].position.z - pts[i-1].position.z) };
            der = der / magnitude(der);
        }
        else {
            der1 = { (pts[i].position.x - pts[i-1].position.x), (pts[i].position.y - pts[i-1].position.y), (pts[i].position.z - pts[i-1].position.z) };
            der2 = { (pts[i+1].position.x - pts[i].position.x), (pts[i+1].position.y - pts[i].position.y), (pts[i+1].position.z - pts[i].position.z) };
            float nrm1 = magnitude(der1);
            float nrm2 = magnitude(der2);
            der = ((der1 / nrm1) + (der2 / nrm2)) / 2;
        }
        if (magnitude(der) == 0){
            return 0;
        }
        return der;
    }
    
    float magnitude(simd::float3 v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    void caser(simd::float3 der) {
        
        if (std::abs(der.y) > std::abs(der.x)) {
            std::cout << "y > x" << std::endl;
        } else {
            std::cout << "y < x" << std::endl;
        }
        
        if (std::abs(der.y) > std::abs(der.z)) {
            std::cout << "y > z" << std::endl;
        } else {
            std::cout << "y < z" << std::endl;
        }
        
        if (std::abs(der.z) > std::abs(der.x)) {
            std::cout << "z > x" << std::endl;
        } else {
            std::cout << "z < x" << std::endl;
        }
    }
    void selectBasis(int k, simd::float3 *basis1, simd::float3 *basis2, simd::float3 tangent) {
        simd::float3 b1 = simd::normalize(simd::float3{-tangent.y, tangent.x, 0.0f});
        simd::float3 b2 = simd::normalize(simd::cross(b1, tangent));
        std::cout << "Magnitude" << magnitude(b1) << magnitude(b2) << std::endl;
        basis1 = &b1;
        basis2 = &b2;
        
    }
    
    void buildPipeSimpleV2(Vertex3d* pts, const int noOfPoints, int CirclePts, float radius) {
        Mesh Pipe;
        simd::float3 der;
        simd::float3 centre;
        uint16_t indices[2 * (CirclePts + 1) * (noOfPoints - 1)];
        Vertex3d points[(noOfPoints * CirclePts) + noOfPoints];
        float newX, newY;
        float normaliser;
        simd::float3 previousBasis1 = {-1.0f, 0.0f, 0.0f}; // Initial arbitrary basis vector
        simd::float3 previousBasis2 {0.0f, 0.0f, -1.0f};

        int k = 1;
        float prevX = 1;
        float prevY = 1;
        float flipped = 0;
        float CD = 2 * M_PI / CirclePts;
        for (int j = 0; j < noOfPoints; j++) {
            simd::float3 basis1;
            simd::float3 basis2;
            centre = pts[j].position;
            der = derivative(j, pts, noOfPoints);
            simd::float3 der1 = pts[j].position - pts[j-1].position;
            simd::float3 der2 = pts[j+1].position - pts[j].position;
            simd::float3 tan = der2 - der1;
            if (std::abs(der.y) >= std::abs(der.z) && std::abs(der.x) >= std::abs(der.z)) {
                std::cout << "In XY plane" << std::endl;
            }
            else {
                if (j == noOfPoints - 1) {
                    if (std::abs(der1.x) > std::abs(der1.y)) {
                        if (der1.x * prevX < 0) {
                            k *= -1;
                        }
                    }
                }
                else if (std::abs(der2.x) > std::abs(der2.y)) {
                    if (der.x * prevX < 0) {
                        k *= -1;
                    }
                } else {
                    if (der.y * prevY < 0) {
                        k *= -1;
                    }
                }
            }

            
            std::cout << "der1 " << j << " "<< der1.x << ", "<< der1.y << ", "<< der1.z << std::endl;
            std::cout << "der2 " << j << " "<< der2.x << ", "<< der2.y << ", "<< der2.z << std::endl;
            std::cout << "der " << j << " "<< der.x << ", "<< der.y << ", "<< der.z << std::endl;
            std::cout << "tan " << j << " "<< tan.x << ", "<< tan.y << ", "<< tan.z << std::endl;

//            
//            if (std::abs(der.z) > 0) {
//                if ((std::abs(tan.y) >= std::abs(tan.x))) {
//                    std::cout << "flipped" << std::endl;
//                    basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
//                    basis1 = simd::normalize(simd::cross(basis2, der));
//                    basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                    flipped = 1;
//                }
//                else {
//                    basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                    basis2 = simd::normalize(simd::cross(basis1, der));
//                }
//            } else {
//                basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                basis2 = simd::normalize(simd::cross(basis1, der));
//                flipped = 0;
//            }
            
//            if (std::abs(der.z) > 0) {
//                if (std::abs(der2.x) >= std::abs(der1.x) && std::abs(der2.y) >= std::abs(der1.y)) {
//                    std::cout << "flipped" << std::endl;
//                    basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
//                    basis1 = simd::normalize(simd::cross(basis2, der));
//                    basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                    flipped = 1;
//                } else if (std::abs(der2.x) < std::abs(der1.x) && std::abs(der2.y) < std::abs(der1.y)) {
//                    std::cout << "flipped" << std::endl;
//                    basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
//                    basis1 = simd::normalize(simd::cross(basis2, der));
//                    basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                    flipped = 1;
//                }
//                else {
//                    basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                    basis2 = simd::normalize(simd::cross(basis1, der));
//                }
//            } else {
//                basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                basis2 = simd::normalize(simd::cross(basis1, der));
//                flipped = 0;
//            }
            
            std::cout << "k " << k << std::endl;
            if (std::abs(der2.z) > 0) {
                if (j == noOfPoints - 1) {
                    if (std::abs(der1.y) >= std::abs(der1.x)) {
                        std::cout << "flipped" << std::endl;
                        basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
                        basis1 = simd::normalize(simd::cross(basis2, der));
                        basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                    } else {
                        basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                        basis2 = simd::normalize(simd::cross(basis1, der));
                    }

                } else if (std::abs(der2.y) >= std::abs(der2.x) && std::abs(der1.y) >= std::abs(der1.x) && std::abs(der.y) > std::abs(der.x)) {
                    std::cout << "flipped" << std::endl;
                    basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
                    basis1 = simd::normalize(simd::cross(basis2, der));
                    basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                    flipped = 1;
                    
                } else if (std::abs(der2.y) < std::abs(der2.x) && std::abs(der1.y) < std::abs(der1.x) && std::abs(der.y) > std::abs(der.x)) {
                    std::cout << "flipped" << std::endl;
                    basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
                    basis1 = simd::normalize(simd::cross(basis2, der));
                    basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                    flipped = 1;
                    
                }
                else {
                    std::cout << "not flipped" << std::endl;
                    basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                    basis2 = simd::normalize(simd::cross(basis1, der));
                }
            } else {
                basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                basis2 = simd::normalize(simd::cross(basis1, der));
                flipped = 0;
            }
            
            if (j == 14) {
//                basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//                basis1 = simd::normalize(simd::cross(basis2, der));
//                basis2 = simd::normalize(-k * simd::float3{-der.y, der.x, 0.0f});
            }
            std::cout << "      " << std::endl;
            std::cout << "      " << std::endl;
            prevX = 1 * der.x;
            prevY = 1 * der.y;
            
//            basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
//            basis2 = simd::normalize(simd::cross(basis1, der));
            
//            selectBasis(k, &basis1, &basis2, der);

//            simd::float3 tangent = derivative(j, pts, noOfPoints);
//            std::cout << j << std::endl;
//            if (std::fabs(der.z) > 1e-6 && std::fabs(der.y) < 1e-6) {
//                std::cout << "In Plane" << std::endl;
//            } else if (std::fabs(der.y) > 1e-6 && std::fabs(der.z) < 1e-6) {
//                std::cout << "In Plane" << std::endl;
//            } else {
//                std::cout << "Not Plane" << std::endl;
//            }
//            caser(der);
////            if (std::abs(tangent.z) > 1e-6) {
//                basis1 = simd::normalize(simd::float3{-tangent.y, tangent.x, 0.0f});
////            } else {
////                basis1 = simd::normalize(simd::float3{0.0f, tangent.z, -tangent.y});
////            }
//
//            // Generate the second basis vector
//            
//            // Ensure the orientation is consistent with the previous segment
//            if (simd::dot(previousBasis1, basis1) < 0) {
//                basis1 = -basis1;
//            }
//            basis2 = simd::normalize(simd::cross(tangent, basis1));
            simd::float3 dirOFturn = simd::normalize(simd::cross(der, simd::cross(der1, der2)));
            if (std::abs(simd::dot(basis1, dirOFturn)) > std::abs(simd::dot(basis2, dirOFturn))) {
                basis1 = std::abs(simd::dot(basis1, dirOFturn)) * (1 /  magnitude(der)) * basis1;
            } else {
                basis2 = std::abs(simd::dot(basis2, dirOFturn)) * (1 /  magnitude(der)) * basis2;
            }
////            if (simd::dot(previousBasis2, basis2) < 0) {
////                basis2 = -basis2;
////            }
//            previousBasis1 = basis1;
//            previousBasis2 = basis2;
            
            for (int i = 0; i < CirclePts; i++) {
                float Ang = 0 + (CD * (i));
//                centre = pts[j].position;
//                der = derivative(j, pts, noOfPoints);

//                std::cout << "der " << j << " "<< der.x << ", "<< der.y << ", "<<der.z << std::endl;
//                if (der.x == 0 & der.y == 0) {
//                    std::cout << "expt" << j << std::endl;
//                    basis1 = {0, 1, 0};
//                    basis1 = basis1 / magnitude(basis1);
//                    basis2 = simd_cross(basis1, der);
//                    
//                    basis2 = basis2 / magnitude(basis2);
//                } else {
//                    if (j == 0 or j == noOfPoints-1) {
//                        if (std::abs(der.x) < std::abs(der.y)) {
//                            basis1 = simd::normalize(simd::float3{0, -der.z, der.y});
//                            std::cout << "x < y for " << j << std::endl;
//                        } else {
//                    std::cout << "cmon" << (std::fabs(der.z) > 1e-6) << std::endl;
//                    if (std::fabs(der.z) > 1e-6 && std::fabs(der.y) < 1e-1) {
//                        int k = -1;
//                        std::cout << "cmon" << k << (std::fabs(der.z) > 1e-6) << std::endl;
//                        if (der.x > 0) {
//                            k=-1;
//                        } else {
////                            k = 1;
//                            if (der.y <= 0) {
//                                k=-1;
//                            }
//                        }
                        
//                    if (std::abs(der.y) >= std::abs(der.z) && std::abs(der.x) >= std::abs(der.z)) {
//                        
//                    } else {
//
//                            if ((der.x < 0 && der.y > 0) || (der.x > 0 && der.y < 0)) {
//                                k = 1;
//                            } else {
//                                k =-1;
//                            }
////
//                    }
//                    
//                    
//                        basis1 = simd::normalize(k * simd::float3{-der.y, (der.x), 0});
//                    if (std::abs(der.y) >= std::abs(der.z) && std::abs(der.x) >= std::abs(der.z)) {
//                        basis1 = simd::normalize(simd::float3{-der.y, der.x, 0.0f});
//                    } else {
//                        basis1 = simd::normalize(simd::float3{0.0f, der.z, -der.y});
//                    }
//                    
//                    // Calculate basis2 as the cross product
//                    basis2 = simd::normalize(simd::cross(basis1, der));
//                    
//                    // Ensure consistent orientation
//                    if (simd::dot(simd::cross(basis1, basis2), der) < 0) {
//                        basis1 = -basis1;
//                    }
                    
                            
//                    } else if (std::fabs(der.y) > 1e-6 && std::fabs(der.z) < 1e-1) {
//                        int k = 1;
//                        std::cout << "cunt" << (std::fabs(der.z) > 1e-1) << std::endl;
//                        if (der.y < 0) {
//                            k=-1;
//                        }
//                        basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0});
//                        
//                    } else if (std::fabs(der.x) < 1e-6) {
//                        std::cout << "bitch" << (std::fabs(der.z) > 1e-6) << std::endl;
//                        basis1 = simd::cross(der, simd::normalize(simd::float3{-der.y, der.x, 0}));
//                    } else {
//                        std::cout << "dome" << (std::fabs(der.z) > 1e-1) << std::endl;
//                        basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                    }
//                    if (std::fabs(der.z) > 1e-6 && std::fabs(der.y) < 1e-6) {
//                        
//                        basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                    } else {
//                        basis1 = simd::cross(der, simd::normalize(simd::float3{-der.y, der.x, 0}));
//                    }
//                    basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                    if (der.x >= 0){
//                        basis1 = - basis1;
//                    }
                        
//                    if (std::abs(der.y) >= std::abs(der.z)) {
//                        
//                        std::cout << "y > z for " << j << std::endl;
//                    } else {
////                        basis1 = simd::normalize(simd::float3{-der.z, 0, der.x});
//                        std::cout << "y < z for " << j << std::endl;
//                    }
//
//                            
//                        }
//                        basis1 = basis1 / magnitude(basis1);
////                        std::cout << basis1.x << ", "<< basis1.y <<", " <<basis1.z << "\n" << std::endl;
//                        basis2 = simd_cross(basis1, der);
//                        basis2 = basis2 / magnitude(basis2);
//                    if (j != 0 or j != noOfPoints-1) {
//                        simd::float3 der1 = pts[j].position - pts[j-1].position;
//                        simd::float3 der2 = pts[j+1].position - pts[j].position;
//                        simd::float3 dirOFturn = simd::normalize(simd::cross(der, simd::cross(der1, der2)));
//                        if (std::abs(simd::dot(basis1, dirOFturn)) > std::abs(simd::dot(basis2, dirOFturn))) {
//                            basis1 = std::abs(simd::dot(basis1, dirOFturn)) * (1 /  magnitude(der)) * basis1;
//                        } else {
//                            basis2 = std::abs(simd::dot(basis2, dirOFturn)) * (1 /  magnitude(der)) * basis2;
//                        }
//                    }
                        
//                    } else {
//                        simd::float3 der1 = pts[j].position - pts[j-1].position;
//                        simd::float3 der2 = pts[j+1].position - pts[j].position;
//                        if (std::abs(der.x) > std::abs(der.y)) {
//                            basis1 =  simd_cross(der1, der2);
//                            basis1 = basis1 / magnitude(basis1);
//                            basis2 = simd_cross(basis1, der);
//                        } else {
//                            basis2 =  simd_cross(der1, der2);
//                            basis2 = basis2 / magnitude(basis2);
//                            basis1 = -simd_cross(basis2, der);
//                        }
//                        simd::float3 normal = simd::normalize(simd_cross(der1, der2));
                        
//                        if (std::abs(der.x) >= std::abs(der.y)) {
//                            std::cout << "True" << std::endl;
//                            basis1 = -simd::normalize(simd_cross(normal, der));
//                            basis2 = simd::normalize(simd_cross(der, basis1));
//                        } else {
//                            std::cout << "False" << std::endl;
//                            basis2 = simd::normalize(simd_cross(normal, der));
//                            basis1 = -simd::normalize(simd_cross(der, basis1));
//                        }
//                    }
//                }
                simd::float3 radiusVec = centre + (basis1 * radius * cos(Ang)) + (basis2 * radius * sin(Ang));
                points[i * noOfPoints + j] = {radiusVec, {0, 0, 0}, {1, 0, 0}};
            }
        }
        
        for (int j = 0; j < noOfPoints - 1; j++) {
            for (int i = 0; i < CirclePts; i++) {
                indices[2 * ((j * (CirclePts + 1)) + i)] = (i * noOfPoints) + j;
                indices[2 * ((j * (CirclePts + 1)) + i) + 1] = (i * noOfPoints) + j + 1;
            }
            indices[2 * ((j * (CirclePts + 1)) + CirclePts)] = (0 * noOfPoints) + j;
            indices[2 * ((j * (CirclePts + 1)) + CirclePts) + 1] = (0 * noOfPoints) + j + 1;
        }
        

        
        Pipe.VertexBuffer = pDevice->newBuffer((CirclePts) * noOfPoints * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        Pipe.IndexBuffer = pDevice->newBuffer(2 * (CirclePts + 1) * (noOfPoints - 1) * sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(Pipe.VertexBuffer->contents(), points, (CirclePts) * noOfPoints * sizeof(Vertex3d));
        memcpy(Pipe.IndexBuffer->contents(), indices, 2 * (CirclePts + 1) * (noOfPoints - 1) * sizeof(uint16_t));
        PipeMesh = Pipe;
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
            for (int j = 0; j < radialDivisions; j++) {
                Angle = j * (2 * M_PI / radialDivisions);
                Height = (- radius) + (2*radius / (lengthDivisions + 1)) * (i+1);
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
    
    void buildPipeCapped(Vertex3d* pts, const int noOfPoints, int CirclePts, float radius) {
        Mesh Pipe;
        simd::float3 der;
        simd::float3 centre;
        simd::float3 basis1;
        simd::float3 basis2;
        int offset = 2 * (CirclePts + 1);
        uint16_t indices[offset + (2 * (CirclePts + 1) * (noOfPoints - 1)) + offset];
        Vertex3d points[1 + ((noOfPoints * CirclePts)) + 1];
//        float newX, newY;
//        float normaliser;
        float CD = 2 * M_PI / CirclePts;
        std::cout << pts[0].colour.x << pts[0].colour.y << pts[0].colour.z << std::endl;
        points[0] = pts[0];
        points[1 + ((noOfPoints * CirclePts))] = pts[noOfPoints - 1];
        
        for (int i=1; i < CirclePts+1; i++) {
//            float Ang = 0 + (CD * (i));
//            points[i] = {{centre.x + (radius * cos(Ang)), centre.y + (radius * sin(Ang)), centre.z}, {0, 0, 0}, {1, 0, 0}};
            indices[2 * i - 2 ] = 0;
            indices[2 * i - 1] = 1 + ((i-1) * noOfPoints);
        }
        indices[2 * CirclePts] = 0;
        indices[2 * CirclePts + 1] = 1;
        
        int index = offset/2 + (CirclePts+1)*(noOfPoints-1);
        for (int i=1; i < CirclePts+1; i++) {
            
            std::cout << 2 * index << " : " << std::endl;
            indices[2 * index + 1] = 1 + ((noOfPoints * CirclePts));
            indices[2 * index ] =  ((i) * noOfPoints);
            index++;
        }
        indices[2 * index +1 ] = 1 + ((noOfPoints * CirclePts));
        indices[2 * index ] = noOfPoints;
        int k = 1;
        float prevX = 1;
        float prevY = 1;
        for (int j = 0; j < noOfPoints; j++) {
            centre = pts[j].position;
            der = derivative(j, pts, noOfPoints);
            if (std::abs(der.y) >= std::abs(der.z) && std::abs(der.x) >= std::abs(der.z)) {
            }
            else {
                if (std::abs(der.x) > std::abs(der.y)) {
                    if (der.x * prevX < 0) {
                        k *= -1;
                    }
                } else {
                    if (der.y * prevY < 0) {
                        k *= -1;
                    }
                }
            }
            if ((std::abs(prevX) > std::abs(prevY)) && (std::abs(der.y) > std::abs(der.x))) {
                basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                basis1 = simd::normalize(simd::cross(basis2, der));
            } else if ((std::abs(prevX) < std::abs(prevY)) && (std::abs(der.y) < std::abs(der.x))) {
                basis2 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                basis1 = simd::normalize(simd::cross(basis2, der));
            }
            else {
                basis1 = simd::normalize(k * simd::float3{-der.y, der.x, 0.0f});
                basis2 = simd::normalize(simd::cross(basis1, der));
            }
            prevX = 1 * der.x;
            prevY = 1 * der.y;

            
//            selectBasis(k, &basis1, &basis2, der);
            simd::float3 der1 = pts[j].position - pts[j-1].position;
            simd::float3 der2 = pts[j+1].position - pts[j].position;
            simd::float3 dirOFturn = simd::normalize(simd::cross(der, simd::cross(der1, der2)));
            if (std::abs(simd::dot(basis1, dirOFturn)) > std::abs(simd::dot(basis2, dirOFturn))) {
                basis1 = std::abs(simd::dot(basis1, dirOFturn)) * (1 /  magnitude(der)) * basis1;
            } else {
                basis2 = std::abs(simd::dot(basis2, dirOFturn)) * (1 /  magnitude(der)) * basis2;
            }
            
            for (int i = 0; i < CirclePts; i++) {
                float Ang = 0 + (CD * (i));


                if (der.x == 0 & der.y == 0) {
                    basis1 = {0, 1, 0};
                    basis1 = basis1 / magnitude(basis1);
                    basis2 = simd_cross(basis1, der);
                    
                    basis2 = basis2 / magnitude(basis2);
                } else {
//                    if (j == 0 or j == noOfPoints-1) {
//                        if (std::abs(der.x) < std::abs(der.y)) {
//                            basis1 = simd::normalize(simd::float3{0, -der.z, der.y});
//                        } else {
//                            basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                        }
//                        basis1 = basis1 / magnitude(basis1);
//                        basis2 = simd_cross(basis1, der);
//                        basis2 = basis2 / magnitude(basis2);
//                        
//                    } else {
//                        simd::float3 der1 = pts[j].position - pts[j-1].position;
//                        simd::float3 der2 = pts[j+1].position - pts[j].position;
//                        if (std::abs(der.x) > std::abs(der.y)) {
//                            basis1 =  simd_cross(der1, der2);
//                            basis1 = basis1 / magnitude(basis1);
//                            basis2 = simd_cross(basis1, der);
//                        } else {
//                            basis2 =  simd_cross(der1, der2);
//                            basis2 = basis2 / magnitude(basis2);
//                            basis1 = - simd_cross(basis2, der);
//                        }
//                    }
//                    if (std::fabs(der.z) > 1e-6 && std::fabs(der.y) < 1e-6) {
//                        basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                    } else if (std::fabs(der.y) > 1e-6 && std::fabs(der.z) < 1e-6) {
//                        basis1 = simd::normalize(simd::float3{-der.y, der.x, 0});
//                    } else {
//                        basis1 = simd::cross(der, simd::normalize(simd::float3{-der.y, der.x, 0}));
//                    }
//                    basis1 = basis1 / magnitude(basis1);
////                        std::cout << basis1.x << ", "<< basis1.y <<", " <<basis1.z << "\n" << std::endl;
//                    basis2 = simd_cross(basis1, der);
//                    basis2 = basis2 / magnitude(basis2);
//                    basis2 = (1 / magnitude(der)) * basis2;
                }
                simd::float3 radiusVec = centre + (basis1 * radius * cos(Ang)) + (basis2 * radius * sin(Ang));
                points[1 + (i * noOfPoints + j)] = {radiusVec,pts[j].normal, pts[j].colour};
            }
        }
        
        for (int j = 0; j < noOfPoints - 1; j++) {
            for (int i = 0; i < CirclePts; i++) {
                std::cout << offset + (2 * ((j * (CirclePts + 1)) + i)) << ": " << 1 + ((i * noOfPoints) + j) << std::endl;
                indices[offset + (2 * ((j * (CirclePts + 1)) + i))] = 1 + ((i * noOfPoints) + j);
                indices[offset + (2 * ((j * (CirclePts + 1)) + i) + 1)] = 1 + ((i * noOfPoints) + j + 1);
            }
            std::cout << offset + (2 * ((j * (CirclePts + 1)) + CirclePts)) << ": " << 1 + ((0 * noOfPoints) + j) << std::endl;
            indices[offset + (2 * ((j * (CirclePts + 1)) + CirclePts))] = 1+((0 * noOfPoints) + j);
            indices[offset + (2 * ((j * (CirclePts + 1)) + CirclePts) + 1)] = 1+((0 * noOfPoints) + j + 1);
        }
        
        uint16_t indexBufferSIZE = offset + (2 * (CirclePts + 1) * (noOfPoints - 1)) + offset;
        
        Pipe.VertexBuffer = pDevice->newBuffer((1 + (CirclePts * noOfPoints) + 1) * sizeof(Vertex3d), MTL::ResourceStorageModeShared);
        Pipe.IndexBuffer = pDevice->newBuffer(indexBufferSIZE * sizeof(uint16_t), MTL::ResourceStorageModeShared);
        memcpy(Pipe.VertexBuffer->contents(), points, (1 + (CirclePts * noOfPoints) + 1) * sizeof(Vertex3d));
        memcpy(Pipe.IndexBuffer->contents(), indices, indexBufferSIZE * sizeof(uint16_t));
        PipeMeshCapped = Pipe;
    }
    
    
    void draw( MTK::View* pView ) {
        time += 1;
        if (3600 < time) { time = 0; }
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();

        
        MTL::RenderPassDescriptor* pPassDescriptor = pView->currentRenderPassDescriptor();

//        pPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadAction::LoadActionClear);
//        pPassDescriptor->depthAttachment()->setClearDepth(1.0f);
        
        MTL::RenderCommandEncoder* encoder = pCommandBuffer->renderCommandEncoder(pPassDescriptor);
        encoder->setRenderPipelineState(pPipelineState);
        encoder->setDepthStencilState(pDepthStencilState);
        

        
        simd::float3 dPosV2 = {0.0f, 0.0f, 0.2f};
        simd::float4x4 transformV2 =  Translation(dPosV2);
//        encoder->setVertexBytes(&transformV2, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(QuadMesh.VertexBuffer, 0, 0);
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(6), MTL::IndexType::IndexTypeUInt16, CubeMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
        simd::float3 dPosV1 = {0.0f, 0.0f, 0.8f};
        simd::float4x4 transformV1 =  Translation(dPosV1) * RotationY(time * 0.5);
        transformV1 = Translation(dPosV1) * RotationX(90);
        transformV1 = Identity();
//
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(CubeMesh.VertexBuffer, 0, 0);
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(6), MTL::IndexType::IndexTypeUInt16, CubeMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
//        encoder->setCullMode( MTL::CullMode::CullModeFront );
//        encoder->setFrontFacingWinding( MTL::Winding::WindingCounterClockwise );
        
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(GraphMesh.VertexBuffer, 0, 0);
//        encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(1202));
        
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(LineMesh.VertexBuffer, 0, 0);
//        encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(10));
        
////        PipeMesh encorder setup
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(PipeMesh.VertexBuffer, 0, 0);
////        encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger((4*3) + 2));
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(2 * (40+1)*(16-1)), MTL::IndexType::IndexTypeUInt16, PipeMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(PipeMeshCapped.VertexBuffer, 0, 0);
//        int CirclePts = 60;
//        int noOfPoints = 13;
//        int indexBufferSIZE = (2 * (CirclePts+1)) + (2 * (CirclePts + 1) * (noOfPoints - 1)) + (2 * (CirclePts+1));
////        encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger((4*3) + 2));
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(indexBufferSIZE), MTL::IndexType::IndexTypeUInt16, PipeMeshCapped.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
//        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
//        encoder->setVertexBuffer(CircleMesh.VertexBuffer, 0, 0);
//        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(120 + 2), MTL::IndexType::IndexTypeUInt16, CircleMesh.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
        
        //        PipeMesh encorder setup
        encoder->setVertexBytes(&transformV1, sizeof(simd::float4x4), 1);
        encoder->setVertexBuffer(Sphere.VertexBuffer, 0, 0);
        //        encoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger((4*3) + 2));
        
        encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(Sphere.SizeOfIndex), MTL::IndexType::IndexTypeUInt16, Sphere.IndexBuffer, NS::UInteger(0), NS::UInteger(1));
//
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
    Mesh QuadMesh;
    Mesh GraphMesh;
    Mesh CircleMesh;
    Mesh LineMesh;
    Mesh PipeMesh;
    Mesh PipeMeshCapped;
    Mesh Sphere;
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
        pMtkView->setDepthStencilPixelFormat(depth);
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

/*
 0  0.6 0.7
 0  -1 0

 0  -0.1 0.7
 0  -0.615412 -0.788205

 0  0 0.3
 0  0.242536 -0.970142
 */

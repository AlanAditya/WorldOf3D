//
//  ShapeHPPExtension.h
//  WorldOf3D
//
//  Created by Manoj Kumar on 01/03/25.
//

#ifndef ShapeHPPExtension_h
#define ShapeHPPExtension_h
#include <Metal/Metal.hpp>
#include <simd/simd.h>

template<typename T>
void printArr(T* arr, int count) {
    for (int i = 0; i < count; i++) {
        std::cout << ((uint16*)arr)[i] << ", ";
    }
    std::cout << " }\n";
}
struct Vertex3D {
    simd_float3 position;
    simd_float4 colour;
    simd_float3 normal = {0.0, 0.0, 1.0};
    simd_float2 textureCoordinates = {0.0, 0.0};
};

//std::ostream& operator<<(std::ostream& os, const simd_float3 vector) {
//    os << vector.x << " " << vector.y << " " << vector.z;
//    return os;
//}

template<typename T>
class Shape {
public:
    Vertex3D* Verticies;
    int vertexCount;
    
    T* indices;
    int indexCount;
    
    MTL::Buffer* vertexBuffer = nil;
    MTL::Buffer* indexBuffer = nil;
    
    size_t instanceCount = 1;
    
    
    simd_float3 position= {0.0, 0.0, 0.0};
    simd_float3 scale = {1.0, 1.0, 1.0};
    simd_float3 rotation = {0.0, 0.0, 0.0};
    
    bool update = true;
    
    MTL::PrimitiveType drawType = MTL::PrimitiveTypeTriangle;
    simd_float4x4* instanceMatrix;
    
    Shape(Vertex3D* Verticies, int vertexCount, T* indices, int indexCount) {
        this->Verticies = Verticies;
        this->vertexCount = vertexCount;
        this->indices = indices;
        this->indexCount = indexCount;
    }

    void print() {
        for (int i = 0; i < vertexCount; i++) {
            std::cout << Verticies[i].position << "\n";
        }
    }
    
    void buildBuffers(MTL::Device* metalDevice) {
        if (!vertexBuffer || !indexBuffer || update) {
            vertexBuffer =  metalDevice->newBuffer(vertexCount * sizeof(Vertex3D),MTL::ResourceStorageModeShared );
            memcpy(vertexBuffer->contents(), Verticies, vertexCount*sizeof(Vertex3D));
            
            indexBuffer = metalDevice->newBuffer(indexCount * sizeof(T), MTL::ResourceStorageModeShared );
            
            memcpy(indexBuffer->contents(), indices, indexCount*sizeof(T));
            std::cout << "buffer built" << "\n";
        }
        update = false;
    }
    
};

class TriangleH: public Shape<uint16> {
public:
    Vertex3D verticesL[3] = {
        {{0,0,0}, {1,1,1,1}},
        {{1,0,0}, {1,1,1,1}},
        {{0,1,0}, {1,1,1, 1}}
    };
    uint16 indicesL[3] = {0, 1, 2};
    TriangleH(float side): Shape<uint16>(new Vertex3D[3], 3, indicesL, 3) {
        Verticies[0] = {{0,0,0}, {1,1,1,1}};
        Verticies[1] = {{side,0,0}, {1,1,1,1}};
        Verticies[2] = {{0,side,0}, {1,1,1, 1}};
    }
};

class NGon: public Shape<uint16> {
public:
    NGon(float radius, int sides): Shape<uint16>(new Vertex3D[sides + 1], sides + 1, new uint16[2 * sides + 2], 2 * sides + 2) {
        
        Verticies[0] = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0}};
        float Ang = 0.0f;
        for (int i =0; i<sides; i++) {
            Ang = i * (2* M_PI / sides);
            Verticies[1 + i] = {{radius * cos(Ang), radius * sin(Ang), 0.5}, {1.0, 1.0, 1.0, 1.0}};
            
            indices[2*i] = 0;
            indices[2*i + 1] = i+1;
            
        }
        indices[2*sides] = 0;
        indices[2*sides + 1] = 1;
        printArr(indices, 2 * sides);
        drawType = MTL::PrimitiveTypeTriangleStrip;
        std::cout << "Initialised";
    }
};

class Cube: public Shape<uint16> {
public:
    Cube(float s): Shape<uint16>(new Vertex3D[24], 24, new uint16[36], 36) {

        
        Verticies[0]  = { { -s, -s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f,  1.f } };
        Verticies[1]  = { { +s, -s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f,  1.f } };
        Verticies[2]  = { { +s, +s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f,  1.f } };
        Verticies[3]  = { { -s, +s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f,  1.f } };
        Verticies[4]  = { { +s, -s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, { 1.f,  0.f,  0.f } };
        Verticies[5]  = { { +s, -s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, { 1.f,  0.f,  0.f } };
        Verticies[6]  = { { +s, +s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, { 1.f,  0.f,  0.f } };
        Verticies[7]  = { { +s, +s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, { 1.f,  0.f,  0.f } };
        Verticies[8]  = { { +s, -s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f, -1.f } };
        Verticies[9]  = { { -s, -s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f, -1.f } };
        Verticies[10] = { { -s, +s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f, -1.f } };
        Verticies[11] = { { +s, +s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, { 0.f,  0.f, -1.f } };
        Verticies[12] = { { -s, -s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, { -1.f, 0.f,  0.f } };
        Verticies[13] = { { -s, -s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, { -1.f, 0.f,  0.f } };
        Verticies[14] = { { -s, +s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, { -1.f, 0.f,  0.f } };
        Verticies[15] = { { -s, +s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, { -1.f, 0.f,  0.f } };
        Verticies[16] = { { -s, +s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f,  1.f,  0.f } };
        Verticies[17] = { { +s, +s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f,  1.f,  0.f } };
        Verticies[18] = { { +s, +s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f,  1.f,  0.f } };
        Verticies[19] = { { -s, +s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f,  1.f,  0.f } };
        Verticies[20] = { { -s, -s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f, -1.f,  0.f } };
        Verticies[21] = { { +s, -s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f, -1.f,  0.f } };
        Verticies[22] = { { +s, -s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f, -1.f,  0.f } };
        Verticies[23] = { { -s, -s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, { 0.f, -1.f,  0.f } };
        
        uint16_t indicesC[] = {
             0,  1,  2,  2,  3,  0, /* front */
             4,  5,  6,  6,  7,  4, /* right */
             8,  9, 10, 10, 11,  8, /* back */
            12, 13, 14, 14, 15, 12, /* left */
            16, 17, 18, 18, 19, 16, /* top */
            20, 21, 22, 22, 23, 20, /* bottom */
        };
        memcpy(indices, indicesC, sizeof(uint16_t) * 36);
//        drawType = MTL::PrimitiveTypeTriangleStrip;
        std::cout << "Initialised";
    }
};

struct ArrayShape {
    Shape<uint16> shape;
    simd_float4x4* transform;
    bool update = true;
    MTL::Buffer* transformBuffer = nil;
    
    void buildBuffer(MTL::Device* metalDevice) {
        if (!transformBuffer || update) {
            transformBuffer =  metalDevice->newBuffer(shape.instanceCount * sizeof(simd_float4x4),MTL::ResourceStorageModeShared );
            memcpy(transformBuffer->contents(), transform, shape.instanceCount * sizeof(simd_float4x4));
            
            std::cout << "buffer built" << "\n";
        }
        update = false;
    }
    ~ArrayShape() {
        std::cout << "deleted shape array" << "\n";
        delete[] transform;
    }
};

#endif /* ShapeHPPExtension_h */

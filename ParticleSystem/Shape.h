//
//  Shape.hpp
//  ParticleSystem
//
//  Created by Manoj Kumar on 13/02/25.
//

#ifndef Shape_hpp
#define Shape_hpp

#include <stdio.h>
#include "Shape.h"
#import <Metal/Metal.h>
#include <iostream>
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

template<typename T>
class Shape {
public:
    Vertex3D* Verticies;
    int vertexCount;
    
    T* indices;
    int indexCount;
    
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    
    simd_float3 position= {0.0, 0.0, 0.0};
    simd_float3 scale = {0.0, 0.0, 0.0};
    simd_float3 rotation = {0.0, 0.0, 0.0};
    
    bool update = true;
    
    MTLPrimitiveType drawType = MTLPrimitiveTypeTriangle;
    
    Shape(Vertex3D* Verticies, int vertexCount, T* indices, int indexCount) {
        this->Verticies = Verticies;
        this->vertexCount = vertexCount;
        this->indices = indices;
        this->indexCount = indexCount;
    }
    void buildBuffers(id<MTLDevice> metalDevice) {
        if (!vertexBuffer || !indexBuffer) {
            vertexBuffer = [metalDevice newBufferWithLength:vertexCount * sizeof(Vertex3D) options:MTLResourceStorageModeShared];
            memcpy([vertexBuffer contents], Verticies, vertexCount*sizeof(Vertex3D));
            
            indexBuffer = [metalDevice newBufferWithLength:indexCount * sizeof(T) options:MTLResourceStorageModeShared];
            
            memcpy([indexBuffer contents], indices, indexCount*sizeof(T));
            std::cout << "buffer built" << "\n";
        }
        update = false;
    }

};

class Triangle: public Shape<uint16> {
public:
    Vertex3D verticesL[3] = {
        {{0,0,0}, {1,1,1,1}},
        {{1,0,0}, {1,1,1,1}},
        {{0,1,0}, {1,1,1, 1}}
    };
    uint16 indicesL[3] = {0, 1, 2};
    Triangle(float side): Shape<uint16>(new Vertex3D[3], 3, indicesL, 3) {
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
        drawType = MTLPrimitiveTypeTriangleStrip;
        std::cout << "Initialised";
    }
};


#endif /* Shape_hpp */

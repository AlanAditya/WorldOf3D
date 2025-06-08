//
//  GPU_Compute.cpp
//  WorldOf3D
//
//  Created by Manoj Kumar on 11/06/24.
//

#include <stdio.h>
#include <iostream>
#include <initializer_list>
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

using namespace std;

template<typename T>
T* flatten(size_t row, size_t col, T** arrays) {
    T* flattenedARRAY = new T[row * col];
    for (size_t i=0; i < row; i++) {
        for (size_t j=0; j < col; j++) {
            flattenedARRAY[(i*col) + j] = arrays[i][j];
        }
    }
    return flattenedARRAY;
}

template<typename T>
float* AddArraysGPU(size_t arraySize, std::initializer_list<T*> arrays) {
    size_t rows = arrays.size();
    T** param = new T*[rows];
    int index = 0;
    for (const auto& elem : arrays) {
        param[index] = elem;
        ++index;
    }
    
    T* flattenedARRAY = flatten(rows, arraySize, param);
    NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    NS::UInteger bufferSize = arraySize * sizeof(T);
    NS::Error* error = nil;
    const char* cppString = "addArraysGPUV2";
    NS::String* funcName = NS::String::string(cppString, NS::StringEncoding::UTF8StringEncoding);
    
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::Library* GPUfuncLibrary = device->newDefaultLibrary();
    MTL::Function* additionGPUfunc = GPUfuncLibrary->newFunction(funcName);
    if (additionGPUfunc == nil)
    {
        std::cout << "Failed to find the adder function.";
    }
    MTL::ComputePipelineState* additionComputePipelineState = device->newComputePipelineState(additionGPUfunc, &error);
    if (additionComputePipelineState == nil)
    {
        std::cout << "Failed to created pipeline state object, error: " << error;
    }
    MTL::Buffer* BufferData = device->newBuffer(flattenedARRAY, bufferSize * arrays.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* BufferResult = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* BufferRow = device->newBuffer(&rows, sizeof(size_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* BufferCol = device->newBuffer(&arraySize, sizeof(size_t), MTL::ResourceStorageModeShared);
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
    commandEncoder->setComputePipelineState(additionComputePipelineState);
    commandEncoder->setBuffer(BufferData, 0, 0);
    commandEncoder->setBuffer(BufferResult, 0, 1);
    commandEncoder->setBuffer(BufferRow, 0, 2);
    commandEncoder->setBuffer(BufferCol, 0, 3);
    MTL::Size threadsPerGrid = MTL::Size(arraySize, 1, 1);
    NS::UInteger maxThreadsPerThreadGroup = additionComputePipelineState->maxTotalThreadsPerThreadgroup();
    MTL::Size ThreadsPerThreadGroup = MTL::Size(maxThreadsPerThreadGroup, 1, 1);
    commandEncoder->dispatchThreads(threadsPerGrid, ThreadsPerThreadGroup);
    commandEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    float* resultsGPU = ((float*)BufferResult->contents());
//    for (int j = 0; j < arraySize; j++) {
//        std::cout << resultsGPU[j] << std::endl;
//    }
//    
    autoreleasePool->release();
    return resultsGPU;
}






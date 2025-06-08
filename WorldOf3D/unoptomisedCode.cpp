//
//  unoptomisedCode.cpp
//  WorldOf3D
//
//  Created by Manoj Kumar on 14/06/24.
//

#include <stdio.h>
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

void addArraysCPU(float* inA, float* inB, float* res, size_t len) {
    for (int i = 0; i < len; i++) {
        *(res+i) = *(inA+i) + *(inB+i);
    }
}

void assignBuffer(MTL::Buffer* buffer, float* inp) {
    float* dataPtr = (float*)buffer->contents();
    size_t numElements = buffer->length() / 4;
    for (int i = 0; i < numElements; i++) {
        dataPtr[i] = inp[i];
    }
}

void generateRandomFloatArray(float* arr, size_t size, float min = 0.0f, float max = 1.0f) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (size_t i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
}

int ADD_ARRAYS_EXAMPLE(int argc, const char * argv[]) {
    size_t arraySize = 500000000;
    float* randomFloatsA = new float[arraySize];
    float* randomFloatsB = new float[arraySize];
    generateRandomFloatArray(randomFloatsA, arraySize, 0, 100);
    generateRandomFloatArray(randomFloatsB, arraySize, 0, 100);
    auto startFunc = std::chrono::high_resolution_clock::now();
    AddArraysGPU<float>(arraySize, {randomFloatsA, randomFloatsB, randomFloatsB, randomFloatsB});
    auto endFunc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationFunc = endFunc - startFunc;
    std::cout << "func took " << durationFunc.count() << " milliseconds to execute." << std::endl;
    return 0;
}

int mainUNOPTOMISED(int argc, const char * argv[]) {
    auto startFunc = std::chrono::high_resolution_clock::now();
    size_t arraySize = 500000000;
    float* randomFloatsA = new float[arraySize];
    float* randomFloatsB = new float[arraySize];
    generateRandomFloatArray(randomFloatsA, arraySize, 0, 100);
    generateRandomFloatArray(randomFloatsB, arraySize, 0, 100);
    
//    float inA[4] = {0.0, 0.1, 0.5, 0.3};
//    float inB[4] = {0.0, 0.1, 0.2, 0.3};
    float* res = new float[arraySize];
    
    NS::Error* error = nil;
    NS::UInteger bufferSize = arraySize * sizeof(float);
    NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
    const char* cppString = "addArraysGPU";
//    NS::String* funcName = stringToNSString(cppString);
    NS::String* funcName = NS::String::string(cppString, NS::StringEncoding::UTF8StringEncoding);
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::Library* GPUfuncLibrary = device->newDefaultLibrary();
    MTL::Function* additionGPUfunc = GPUfuncLibrary->newFunction(funcName);
    if (additionGPUfunc == nil)
    {
        std::cout << "Failed to find the adder function.";
        return 1;
    }
    MTL::ComputePipelineState* additionComputePipelineState = device->newComputePipelineState(additionGPUfunc, &error);
    if (additionComputePipelineState == nil)
    {
        std::cout << "Failed to created pipeline state object, error: " << error;
        return 1;
    }
//    auto startAssign = std::chrono::high_resolution_clock::now();
    MTL::Buffer* BufferA = device->newBuffer(randomFloatsA, bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* BufferB = device->newBuffer(randomFloatsB, bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* BufferC = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

// Assigning Buffers Manually, Avoid as it takes time
//    assignBuffer(BufferA, randomFloatsA);
//    assignBuffer(BufferB, randomFloatsB);
//    auto endAssign = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> durationAssign = endAssign - startAssign;
//    std::cout << durationAssign << std::endl;
//    void* cont = BufferA->contents();
//    float* dataArray = (float*)cont;
    
//    std::cout << "Its kind working" << dataArray[3] << std::endl;
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
    commandEncoder->setComputePipelineState(additionComputePipelineState);
    commandEncoder->setBuffer(BufferA, 0, 0);
    commandEncoder->setBuffer(BufferB, 0, 1);
    commandEncoder->setBuffer(BufferC, 0, 2);
    MTL::Size threadsPerGrid = MTL::Size(arraySize, 1, 1);
    NS::UInteger maxThreadsPerThreadGroup = additionComputePipelineState->maxTotalThreadsPerThreadgroup();
    MTL::Size ThreadsPerThreadGroup = MTL::Size(maxThreadsPerThreadGroup, 1, 1);
    commandEncoder->dispatchThreads(threadsPerGrid, ThreadsPerThreadGroup);
    commandEncoder->endEncoding();
//    auto startGPU = std::chrono::high_resolution_clock::now();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();


//    auto endGPU = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> durationGPU = endGPU - startGPU;
//    std::cout << "GPU took " << durationGPU.count() << " milliseconds to execute." << std::endl;
    float* resultsGPU = ((float*)BufferC->contents());
//    for (int j = 0; j < arraySize; j++) {
//        std::cout << resultsGPU[j] << std::endl;
//    }
    autoreleasePool->release();
    
    auto endFunc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationFunc = endFunc - startFunc;
    std::cout << "func took " << durationFunc.count() << " milliseconds to execute." << std::endl;
    
    auto startCPU = std::chrono::high_resolution_clock::now();
    addArraysCPU(&randomFloatsA[0], &randomFloatsB[0], res, arraySize);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;
    std::cout << "CPU took " << durationCPU.count() << " milliseconds to execute." << std::endl;
    
    for (int i =0; i < arraySize; i++) {
        if (res[i] != resultsGPU[i]) {
            std::cout << randomFloatsA[i]<<" + " << randomFloatsB[i] <<" Hello, World! new array " << res[i] << " Not Equal  "<< *(resultsGPU+i) <<"\n";
        }
        
    }
    return 0;
}



int main2() {
    // Create Metal device
    auto startFunc = std::chrono::high_resolution_clock::now();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device." << std::endl;
        return -1;
    }

    // Create arrays and fill them with data
    const size_t arraySize = 500000000; // 1 million elements
    std::vector<float> inA(arraySize, 1.0f);
    std::vector<float> inB(arraySize, 2.0f);
    std::vector<float> result(arraySize, 0.0f);
    generateRandomFloatArray(&inA[0], arraySize);
    generateRandomFloatArray(&inB[0], arraySize);
    

    // Create buffers for the input and output data
    MTL::Buffer* bufferA = device->newBuffer(inA.data(), arraySize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(inB.data(), arraySize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = device->newBuffer(result.data(), arraySize * sizeof(float), MTL::ResourceStorageModeShared);

    // Load the compute shader
    NS::Error* error = nullptr;
    // Shader source
    const char* shaderSource = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_arrays(device const float *inA [[buffer(0)]],
                               device const float *inB [[buffer(1)]],
                               device float *result [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
            result[id] = inA[id] + inB[id];
        }
    )";

    NS::String* nsShaderSource = NS::String::string(shaderSource, NS::StringEncoding::UTF8StringEncoding);
//    NS::String* shaderSource = NS::String::string(u8R"(
//        #include <metal_stdlib>
//        using namespace metal;
//
//        kernel void add_arrays(device const float *inA [[buffer(0)]],
//                               device const float *inB [[buffer(1)]],
//                               device float *result [[buffer(2)]],
//                               uint id [[thread_position_in_grid]]) {
//            result[id] = inA[id] + inB[id];
//        }
//    )", NS::StringEncoding::UTF8StringEncoding);

    MTL::Library* library = device->newLibrary(nsShaderSource, nullptr, &error);
    if (!library) {
        std::cerr << "Failed to create library: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    MTL::Function* function = library->newFunction(NS::String::string("add_arrays", NS::StringEncoding::UTF8StringEncoding));
    if (!function) {
        std::cerr << "Failed to create function." << std::endl;
        return -1;
    }

    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    // Create command queue
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    computeEncoder->setComputePipelineState(pipelineState);

    // Set the buffers
    computeEncoder->setBuffer(bufferA, 0, 0);
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferResult, 0, 2);

    // Dispatch threads
    MTL::Size gridSize = MTL::Size(arraySize, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);
    computeEncoder->dispatchThreads(gridSize, threadGroupSize);

    // End encoding and commit the command buffer
    computeEncoder->endEncoding();
//    auto startGPU = std::chrono::high_resolution_clock::now();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
//    auto endGPU = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> durationGPU = endGPU - startGPU;
//    std::cout << "GPU took " << durationGPU.count() << " milliseconds to execute." << std::endl;
    // Read back the results
    memcpy(result.data(), bufferResult->contents(), arraySize * sizeof(float));

//     Verify the result

    
//    float* res = new float[arraySize];
//    std::cout << "All values are correct!" << std::endl;
//    auto startCPU = std::chrono::high_resolution_clock::now();
//    addArraysCPU(&inA[0], &inB[0], res, arraySize);
//    auto endCPU = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;
//    std::cout << "CPU took " << durationCPU.count() << " milliseconds to execute." << std::endl;
//    // Clean up
//
//    for (size_t i = 0; i < arraySize; ++i) {
//        if (result[i] != res[i]) {
//            std::cerr << "Mismatch at index " << i << ": " << result[i] << std::endl;
//            return -1;
//        }
//    }
    
    bufferA->release();
    bufferB->release();
    bufferResult->release();
    function->release();
    pipelineState->release();
    library->release();
    commandQueue->release();
    device->release();
    
    auto endFunc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationFunc = endFunc - startFunc;
    std::cout << "func took " << durationFunc.count() << " milliseconds to execute." << std::endl;
    return 0;
}




//Mesh BuildCube(MTL::Device* device, float side) {
//    Mesh mesh;
//    Vertex3d verticies[8] = {
//        {{-side, side, side}, {1.0, 0.0, 0.0}},
//        {{side, side, side}, {0.0, 1.0, 0.0}},
//        {{side, side, -side}, {0.0, 0.0, 1.0}},
//        {{-side, side, -side}, {1.0, 0.0, 0.0}},
//        
//        {{-side, -side, side}, {0.0, 1.0, 0.0}},
//        {{side, -side, side}, {0.0, 0.0, 0.0}},
//        {{side, -side, -side}, {1.0, 0.0, 0.0}},
//        {{-side, -side, -side}, {1.0, 0.0, 0.0}}
//    };
//    
//    ushort indicies[36] = {
//        0, 1, 2, 2, 3, 0,
//        4, 5, 6, 6, 7, 4,
//        0, 1, 5, 5, 4, 0,
//        3, 2, 6, 6, 7, 3,
//        0, 3, 7, 7, 4, 0,
//        1, 2, 6, 6, 5, 1
//    };
//    
//    mesh.vertexBuffer = device->newBuffer(8*sizeof(Vertex3d), MTL::ResourceStorageModeShared);
//    mesh.indexBuffer = device->newBuffer(36*sizeof(ushort), MTL::ResourceStorageModeShared);
//    memcpy(mesh.vertexBuffer->contents(), verticies, 8*sizeof(Vertex3d));
//    memcpy(mesh.indexBuffer->contents(), indicies, 36*sizeof(ushort));
//    return mesh;
//}

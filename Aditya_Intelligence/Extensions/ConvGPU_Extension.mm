//
//  ConvGPU_Extension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 22/02/25.
//

#import <Foundation/Foundation.h>
#import "AlgebroHeap.hpp"
#import <Metal/Metal.h>
#import <simd/simd.h>
template <typename Type>
template <typename T>
MatrixH<Type> MatrixH<Type>::ConvG(MatrixH<T> &other) {
    MatrixH<Type> result = MatrixH<Type>();
    result.values = new Type[total_size];
    result.total_size = total_size;
    result.shape = shape;
    
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    
    id<MTLBuffer> buffer1 = [metalDevice newBufferWithBytesNoCopy:values length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    id<MTLBuffer> buffer2 = [metalDevice newBufferWithBytesNoCopy:other.values length:other.total_size*sizeof(T) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    id<MTLBuffer> buffer3 = [metalDevice newBufferWithBytesNoCopy:result.values length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    
    id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
    id<MTLFunction> func1;
    
    if constexpr (std::is_integral<Type>::value) {
        func1 = [lib newFunctionWithName:@"convGPU"];
    } else if constexpr (std::is_floating_point<Type>::value) {
        func1 = [lib newFunctionWithName:@"convGPU"];
    } else {
        func1 = [lib newFunctionWithName:@"convGPU"];
    }
    
    
    NSError *error = nil;
    id<MTLComputePipelineState> computeState = [metalDevice newComputePipelineStateWithFunction:func1 error:&error];
    
    if (error) {
        NSLog(@"convGPU: %@", error.localizedDescription);
    }
    
    id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    
    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    auto _dispatchExecutionSize =  MTLSizeMake(shape[1], shape[0], 1);
    
    
    simd_int2 shapes = simd::make_int2(shape[0], shape[1]);
    simd_int2 kernel_shape = simd::make_int2(other.shape[0], other.shape[1]);
    
    [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
    [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
    [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
    [commandEncoder setBytes: &shapes length:sizeof(simd_int2) atIndex:3];
    [commandEncoder setBytes: &kernel_shape length:sizeof(simd_int2) atIndex:4];
    
    [commandEncoder setComputePipelineState:computeState];
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];
    
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return result;
}

//template MatrixH<float> MatrixH<float>::ConvG(MatrixH<float> &other);
//template MatrixH<int> MatrixH<int>::ConvG(MatrixH<int> &other);
template MatrixH<uint8_t> MatrixH<uint8_t>::ConvG(MatrixH<float> &kernel);
//template MatrixH<char> MatrixH<char>::ConvG(MatrixH<char> &other);

//
//  AddGPU_Extension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 22/02/25.
//

#import <Foundation/Foundation.h>
#import "AlgebroHeap.hpp"
#import <Metal/Metal.h>

template <typename Type>
MatrixH<Type> MatrixH<Type>::AddGPU(MatrixH<Type> &kernel) {
    MatrixH<Type> result = MatrixH<Type>();
    result.values = new Type[total_size];
    result.total_size = total_size;
    result.shape = shape;
    
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    
    id<MTLBuffer> buffer1 = [metalDevice newBufferWithBytesNoCopy:values length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    id<MTLBuffer> buffer2 = [metalDevice newBufferWithBytesNoCopy:kernel.values length:kernel.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    id<MTLBuffer> buffer3 = [metalDevice newBufferWithBytesNoCopy:result.values length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
    }];
    
    id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
    id<MTLFunction> func1;
    
    if constexpr (std::is_integral<Type>::value) {
        func1 = [lib newFunctionWithName:@"AddGPU_I"];
    } else if constexpr (std::is_floating_point<Type>::value) {
        func1 = [lib newFunctionWithName:@"AddGPU_F"];
    } else {
        func1 = [lib newFunctionWithName:@"AddGPU_C"];
    }
    
    
    NSError *error = nil;
    id<MTLComputePipelineState> computeState = [metalDevice newComputePipelineStateWithFunction:func1 error:&error];
    
    if (error) {
        NSLog(@"Adder: %@", error.localizedDescription);
    }
    
    id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    
    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
    
    [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
    [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
    [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
    [commandEncoder setComputePipelineState:computeState];
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];
    
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return result;
}



template MatrixH<float> MatrixH<float>::AddGPU(MatrixH<float> &other);
template MatrixH<int> MatrixH<int>::AddGPU(MatrixH<int> &other);
template MatrixH<uint8_t> MatrixH<uint8_t>::AddGPU(MatrixH<uint8_t> &other);
template MatrixH<char> MatrixH<char>::AddGPU(MatrixH<char> &other);

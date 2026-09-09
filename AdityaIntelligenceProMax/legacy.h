////
////   FaceLandmarkWrapper.h
////  WorldOf3D
////
////  Created by Aditya Dudeja on 12/12/25.
////
//
//#ifndef _FaceLandmarkWrapper_h
//#define _FaceLandmarkWrapper_h
//
//// FaceLandmarkWrapper.h
//#import <Foundation/Foundation.h>
//#import <CoreVideo/CoreVideo.h>
//#include <opencv2/opencv.hpp>
////#include "mediapipe/framework/calculator_framework.h"
////#include "mediapipe/framework/formats/image_frame.h"
////#include "mediapipe/framework/formats/landmark.pb.h"
////#include "mediapipe/framework/formats/image_frame_opencv.h"
////#include "mediapipe/framework/packet.h"
////#include "mediapipe/framework/port/parse_text_proto.h"  // ADD THIS
////#include "mediapipe/framework/port/file_helpers.h"
////#include "mediapipe/framework/formats/video_stream_header.h"
//#include "absl/flags/flag.h"
//#include "absl/flags/parse.h"
//#include "absl/log/absl_log.h"
//#include "mediapipe/framework/calculator_framework.h"
//#include "mediapipe/framework/formats/image_frame.h"
//#include "mediapipe/framework/formats/image_frame_opencv.h"
//#include "mediapipe/framework/port/file_helpers.h"
//#include "mediapipe/framework/port/opencv_highgui_inc.h"
//#include "mediapipe/framework/port/opencv_imgproc_inc.h"
//#include "mediapipe/framework/port/opencv_video_inc.h"
//#include "mediapipe/framework/port/parse_text_proto.h"
//#include "mediapipe/framework/port/status.h"
//#include "mediapipe/util/resource_util.h"
//
//#ifdef NO
//#undef NO
//#endif
//
//#ifdef YES
//#undef YES
//#endif
//NS_ASSUME_NONNULL_BEGIN
//
//@interface FaceLandmarkResult : NSObject
//@property (nonatomic, strong) NSArray<NSValue *> *landmarks; // Array of CGPoint
//@property (nonatomic, assign) BOOL detected;
//@property (nonatomic, assign) float confidence;
//@end
//
//@interface BitchWrapper : NSObject {
//    std::unique_ptr<mediapipe::CalculatorGraph> _graph;
//    std::unique_ptr<mediapipe::OutputStreamPoller> _poller;
//    BOOL _initialized;
//    int64_t _frame_timestamp;
//}
//
//- (instancetype)init;
////- (FaceLandmarkResult *)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer;
////- (FaceLandmarkResult *)detectInImage:(NSImage *)image;
//- (FaceLandmarkResult *)detectInMat:(cv::Mat)frame width:(size_t)width height:(size_t)height;
//
//@end
//NS_ASSUME_NONNULL_END
//
//
//
//#endif /* _FaceLandmarkWrapper_h */


//static void copyCPUinplace( MatrixH<dims, Type>& outMat, const MatrixH<dims, Type>& inMat, int offset, bool commit = true) {
//#ifdef SAFE_MODE
//    if (inMat.total_size > outMat.total_size) {
//        std::cerr << "MatrixH: CopyInplace operation requires both mats to be of same size." << "\n";
//        throw;
//    }
//#endif
//    
//    if (!(inMat.flags & NON_CONTIGUOUS_FLAG) && !(outMat.flags & NON_CONTIGUOUS_FLAG)) {
//        memcpy(outMat.buffer, inMat.buffer, inMat.total_size * sizeof(Type));
//        return;
//    }
//            
//    auto res = collapse_dims(inMat.shape, outMat.strides, inMat.strides, dims, INT32_MAX);
//    auto cdims = res.out_dims;
//    // us stands for unsafe and fast subscripting so it doesnt suppor negative indices and is super fast.
//    if (cdims == 1) {
//        for (uint32_t i = 0; i < inMat.total_size; i++) {
//            outMat.buffer[i * res.stridesA[0]] = inMat.buffer[i * res.stridesB[0]];
//        }
//    } else if (cdims == 2) {
//        for (uint32_t i = 0; i < res.shape[0]; i++) {
//            for (uint32_t j = 0; j < res.shape[1]; j++) {
//                outMat.buffer[i * res.stridesA[0] + j * res.stridesA[1]] = inMat.buffer[i * res.stridesB[0] + j * res.stridesB[1]];
////                    outMat.us(i, j) = inMat.us(i, j);
//            }
//        }
//    } else if (cdims == 3) {
//        for (uint32_t i = 0; i < res.shape[0]; i++) {
//            for (uint32_t j = 0; j < res.shape[1]; j++) {
//                for (uint32_t k = 0; k < res.shape[2]; k++) {
//                    outMat.buffer[i * res.stridesA[0] + j * res.stridesA[1] + k * res.stridesA[2]] = inMat.buffer[i * res.stridesB[0] + j * res.stridesB[1] + k * res.stridesB[2]];
////                        outMat.us(i, j, k) = inMat.us(i, j, k);
//                }
//            }
//        }
//    
//    } else {
//        uint32_t outer_iterations = 1;
//        for (uint32_t o = 0; o <= cdims - 4; o++) {
//            outer_iterations *= res.shape[o];
//        }
//        for (uint32_t o = 0; o < outer_iterations; o++) {
//            uint32_t inMatIndex = 0;
//            uint32_t outMatIndex = 0;
//            uint32_t rem = o;
//            for (int i = dims-4; i >=0; i--) {
//                inMatIndex  += (rem % res.shape[i]) * res.stridesB[i];
//                outMatIndex += (rem % res.shape[i]) * res.stridesA[i];
//                rem /= res.shape[i];
//            }
//            
//            for (uint32_t i = 0; i < res.shape[cdims-3]; i++) {
//                for (uint32_t j = 0; j < res.shape[cdims-2]; j++) {
//                    for (uint32_t k = 0; k < res.shape[cdims-1]; k++) {
//                        outMat.buffer[outMatIndex + i * res.stridesA[dims-3] + j * res.stridesA[dims-2] + k * res.stridesA[dims-1]] = inMat.buffer[inMatIndex + i * res.stridesB[cdims-3] + j * res.strides[cdims-2] + k * inMat.strides[cdims-1]];
//                    }
//                }
//            }
//        }
//        
////            for (uint32_t o = 0; o < inMat.accumul(0, dims-3); o++) {
////                uint32_t inMatIndex = 0;
////                uint32_t outMatIndex = 0;
////                uint32_t rem = o;
////                for (int i = dims-4; i >=0; i--) {
////                    inMatIndex  += (rem % inMat.shape[i])  * inMat.strides[i];
////                    outMatIndex += (rem % outMat.shape[i]) * outMat.strides[i];
////                    rem /= inMat.shape[i];
////                }
////
////                for (uint32_t i = 0; i < inMat.shape[dims-3]; i++) {
////                    for (uint32_t j = 0; j < inMat.shape[dims-2]; j++) {
////                        for (uint32_t k = 0; k < inMat.shape[dims-1]; k++) {
////                            outMat.buffer[outMatIndex + i * outMat.strides[dims-3] + j * outMat.strides[dims-2] + k * outMat.strides[dims-1]] = inMat.buffer[inMatIndex + i * inMat.strides[dims-3] + j * inMat.strides[dims-2] + k * inMat.strides[dims-1]];
////                        }
////                    }
////                }
////            }
//    }
//}

//static void copyGPUinplace( MatrixH<dims, Type>& outMat, const MatrixH<dims, Type>& inMat, int offset, bool commit = true) {
//#ifdef SAFE_MODE
//    if (inMat.total_size > outMat.total_size) {
//        std::cerr << "MatrixH: CopyInplace operation requires both mats to be of same size." << "\n";
//        throw;
//    }
//#endif
//    uint8_t typeCode = get_dtype_code<Type>();
//    int ndims = dims;
//
//    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
//    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
//    
//    auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
//    auto _dispatchExecutionSize =  MTLSizeMake(inMat.total_size, 1, 1);
//    [commandEncoder setBuffer:outMat.metalBuffer offset:0 atIndex:0];
//    [commandEncoder setBuffer:inMat.metalBuffer offset:0 atIndex:1];
//    [commandEncoder setBytes:outMat.strides length:dims * sizeof(size_m) atIndex:2];
//    [commandEncoder setBytes:inMat.strides  length:dims * sizeof(size_m) atIndex:3];
//    [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
//    if (dims == 1) {
//        if (!GlobalGPUManager.CopyInplace[typeCode][0]) {
//            GlobalGPUManager.initCopyInplace(typeCode, 0);
//        }
//        [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][0]];
//    } else if (dims == 2) {
//        if (!GlobalGPUManager.CopyInplace[typeCode][1]) {
//            GlobalGPUManager.initCopyInplace(typeCode, 1);
//        }
//        _dispatchExecutionSize =  MTLSizeMake(inMat.shape[1], inMat.shape[0], 1);
//        [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][1]];
//    } else if (dims == 3) {
//        if (!GlobalGPUManager.CopyInplace[typeCode][2]) {
//            GlobalGPUManager.initCopyInplace(typeCode, 2);
//        }
//        _dispatchExecutionSize =  MTLSizeMake(inMat.shape[2], inMat.shape[1], inMat.shape[0]);
//        [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][2]];
//    
//    } else {
//        if (!GlobalGPUManager.CopyInplace[typeCode][3]) {
//            GlobalGPUManager.initCopyInplace(typeCode, 3);
//        }
//        _dispatchExecutionSize =  MTLSizeMake(inMat.shape[dims-1], inMat.shape[dims-2], inMat.accumul(0, dims-2));
//        [commandEncoder setBytes:inMat.shape length:dims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&ndims length:sizeof(int) atIndex:6];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][3]];
//    }
//    
//    
//    [commandEncoder dispatchThreads:_dispatchExecutionSize
//              threadsPerThreadgroup:_threadsPerThreadgroup];
//    if (commit) {
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//        GlobalGPUManager.gCommandBuffer = nil;
//        GlobalGPUManager.gCommandEncoder=nil;
//    }
//}


//MatrixH(const MatrixH<dims, Type>& other) : MatrixBase(dims, dtype_from_type<Type>()) {
//#ifdef CopyLog
//    std::cout << "Copied" << "\n";
//#endif
//    // copy constructor doesnt need to delete its buffer as  its called only on uninitlised matricies
////        if () {
//        buffer = new Type[other.total_size];
//        total_size = other.total_size;
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
////        gradFunc = other.gradFunc;
////        parentNodes = other.parentNodes;
////        } else if (total_size != other.total_size) {
////            // copy constructor doesnt need to delete it
//////            if (buffer) {
//////                delete [] buffer;
//////            }
////            buffer = new Type[other.total_size];
////            total_size = other.total_size;
////
////        }
//    flags = other.flags; // FIX: We are allocating new buffer, so we own it. Reset the ownership flags.
//    flags &= ~NON_OWNERSHIP_FLAG;
//    memcpy(buffer, other.buffer, sizeof(Type) * total_size);
//    memcpy(shape, other.shape, sizeof(size_m) * dims);
//    memcpy(strides, other.strides, dims * sizeof(size_m));
//    tape = other.tape;
//}

//static MatrixH<3, uint8_t> fromImage() {
//    #if !TARGET_OS_IPHONE
//    CFStringRef path = CFStringCreateWithCString(NULL, "/Users/adityadude/Documents/TUSHU.HEIC", kCFStringEncodingUTF8);
//    CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
//    CGImageSourceRef source;
//    CGImageRef cgImage;
//    for (int i = 0; i < 3; i++) {
//        source = CGImageSourceCreateWithURL(url, NULL);
//        cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
//        if (cgImage) {break;}
//    }
//    CFRelease(url);
//    CFRelease(path);
//    #endif
//
//    #if TARGET_OS_IPHONE
//    UIImage *image = [UIImage imageNamed:@"IMG_1278"];
//    CGImageRef cgImage = image.CGImage;
//    #endif
//
//    if (!cgImage) {
//        std::cerr << "Failed to create CGImage" << std::endl;
////            return;
//    }
////        CGImageRef depthImage = NULL;
////        if (includeDepth) {
////            NSDictionary *auxDataInfo =
////                (__bridge_transfer NSDictionary *)
////                CGImageSourceCopyAuxiliaryDataInfoAtIndex(source,
////                                                          0,
////                                                          kCGImageAuxiliaryDataTypeDisparity);
////            auto cfProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil);
////            NSError* err = nil;
////            auto depthData = [AVDepthData depthDataFromDictionaryRepresentation:auxDataInfo error:&err];
////            CVPixelBufferRef pb = depthData.depthDataMap;
////            size_t DepthWidth = CVPixelBufferGetWidth(pb);
////            size_t DepthHeight = CVPixelBufferGetHeight(pb);
////            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
////
////        }
//    size_t Imgwidth = CGImageGetWidth(cgImage);
//    size_t Imgheight = CGImageGetHeight(cgImage);
//    std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight << "Loaded \n";
//    size_t bytesPerRow = 4 * Imgwidth;
//    void *data = malloc(bytesPerRow * Imgheight);
//    CGContextRef context = CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
//                                                 CGImageGetColorSpace(cgImage),
//                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
//    CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
//    CGContextRelease(context);
//    CGImageRelease(cgImage);
//    
//    uint8_t* pixelData = static_cast<uint8_t*>(data);
//    
//    MatrixH<3, uint8_t> result;
//    result.buffer = pixelData;
//    result.shape[0] = Imgheight;
//    result.shape[1] = Imgwidth;
//    result.shape[2] = 4;
//    result.calcStrides();
//    result.total_size = Imgwidth * Imgheight * 4;
//    result.buildMetalBuffer();
//    return result;
//}


//    void copyFrom(MatrixH<dims, Type>& input) {
//        size_t eff_total_size = effectiveBufferSize();
//        if (!buffer) {
//            buffer = new Type[eff_total_size];
//            total_size = input.total_size;
//        }
//
//        memcpy(buffer, input.buffer, sizeof(Type) * eff_total_size);
//        memcpy(shape,input.shape, sizeof(size_m) * dims);
//        buildMetalBuffer();
//    }

//static MatrixH<dims, Type> constant(const std::vector<size_t>& shapeI ,Type value) {
//    MatrixH<dims, Type> result;
//    for (int i = 0; i< dims; i++) {
//        result.shape[i] = shapeI[i];
//    }
//    result.total_size = accumul(shapeI);
//    result.buffer = new Type[result.total_size]; // no need for effective total buffer size as the buffer is allocated by us;
//    std::fill(result.buffer, result.buffer + result.total_size, value);
//    result.buildMetalBuffer();
//    result.calcStrides();
//    return result;
//}
//
//template <int dimsI>
//static MatrixH<dims, Type> repeating(const std::vector<size_t>& shapeI, const MatrixH<dimsI, Type>& pattern) {
//    MatrixH<dims, Type> result;
//    if (shapeI.size() + dimsI != dims) {
//        std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//        throw ;
//    }
//    
//    for (int i = 0; i < shapeI.size(); i++) {
//        result.shape[i] = shapeI[i];
//    }
//    for (int i = 0; i < dimsI; i++) {
//        result.shape[shapeI.size() + i] = pattern.shape[i];
//    }
//    result.calcStrides();
//    result.total_size = result.accumul(0, dims);
//    result.buffer = new Type[result.total_size];
//    
////        for (int i = 0; i < result.accumul(0, shapeI.size()); ++i) {
////            memcpy(result.buffer + i * pattern.total_size, pattern.buffer, pattern.total_size * sizeof(Type));
////        }
//    PatternFill(result.buffer, pattern.buffer, pattern.total_size * sizeof(Type), result.accumul(0, shapeI.size()));
//    result.buildMetalBuffer();
//    return result;
//}

//static MatrixH<dims, Type> repeating(const std::vector<size_t>& shapeI, const Type& pattern) {
//    MatrixH<dims, Type> result;
//    if (shapeI.size() + 0 != dims) {
//        std::cerr << "Dimensions Dont Add up, Pattern: " << 0 << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//        throw ;
//    }
//    
//    for (int i = 0; i < shapeI.size(); i++) {
//        result.shape[i] = shapeI[i];
//    }
//
//    result.calcStrides();
//    result.total_size = result.accumul(0, dims);
//    result.buffer = new Type[result.total_size];
//    
////        for (int i = 0; i < result.accumul(0, shapeI.size()); ++i) {
////            memcpy(result.buffer + i * pattern.total_size, pattern.buffer, pattern.total_size * sizeof(Type));
////        }
//    PatternFill(result.buffer, &pattern, sizeof(Type), result.accumul(0, shapeI.size()));
//    result.buildMetalBuffer();
//    return result;
//}

//~MatrixH() {
//    #ifdef DestructionLog
//    std::cout << "Matrix Destroyed" << "\n";
//    #endif
//
//    // HUGE ERROR IN PREVIOUS CODE flags & 0 was always false and buffer wasnt getting deleted
////        if ((flags & 0)) {
////            delete [] buffer;
////            std::cout << "deleted" << "\n";
////        }
//    if (!(flags & NON_OWNERSHIP_FLAG)) {
//        delete [] buffer;
//        buffer = nullptr;
//        #ifdef DestructionLog
//        std::cout << "deleted" << "\n";
//        #endif
//    }
//}
//
//MatrixH(const MatrixH<dims, Type>& other) : MatrixBase(dims, dtype_from_type<Type>()) {
//#ifdef CopyLog
//    std::cout << "Copied" << "\n";
//#endif
//    // copy constructor doesnt need to delete its buffer as  its called only on uninitlised matricies
////        if () {
//        buffer = new Type[other.total_size];
//        total_size = other.total_size;
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
////        gradFunc = other.gradFunc;
////        parentNodes = other.parentNodes;
////        } else if (total_size != other.total_size) {
////            // copy constructor doesnt need to delete it
//////            if (buffer) {
//////                delete [] buffer;
//////            }
////            buffer = new Type[other.total_size];
////            total_size = other.total_size;
////
////        }
//    flags = other.flags; // FIX: We are allocating new buffer, so we own it. Reset the ownership flags.
//    flags &= ~NON_OWNERSHIP_FLAG;
//    memcpy(buffer, other.buffer, sizeof(Type) * total_size);
//    memcpy(shape, other.shape, sizeof(size_m) * dims);
//    memcpy(strides, other.strides, dims * sizeof(size_m));
//    tape = other.tape;
//}
//
//MatrixH( MatrixH<dims, Type>&& other) : MatrixBase(dims, dtype_from_type<Type>()) {
//#ifdef MoveLog
//    std::cout << "Moved" << "\n";
//#endif
//    
//    if (flags & NON_OWNERSHIP_FLAG) {
//        *this = (const MatrixH<dims, Type>&) other; // calls copy assignment
//        return;
//    }
//    if (buffer) {
//        delete [] buffer;
//    }
//    buffer = other.buffer;
//    flags = other.flags;
//    other.buffer = nullptr;
//    memcpy(shape, other.shape, dims * sizeof(size_m));
//    memcpy(strides, other.strides, dims * sizeof(size_m));
//    metalBuffer = other.metalBuffer;
//    total_size = other.total_size;
////        gradFunc = std::move(other.gradFunc);
////        parentNodes = std::move(other.parentNodes);
//    other.~MatrixH();
//}
//
//// const fill
//MatrixH<dims, Type>& operator=(Type value) {
//    if (flags & (1u << 1)) {
//        fill_nd_iterative(buffer, shape, strides, dims, value);
//    } else {
//        std::fill(buffer, buffer+total_size, value);
////            memset(buffer, 0, total_size * sizeof(Type));
//    }
//    
//    return *this;
//}
//
//MatrixH<2, Type>& operator=(const MatrixH<1, Type>& other) requires (dims == 2) {
//    
//    if (total_size != other.total_size) {
//        std::cerr << "Error: MatrixH size mismatch — total_size = " << total_size
//                  << ", other.total_size = " << other.total_size << std::endl;
//        throw std::runtime_error("Tensor size mismatch in operation");
//    }
//    if (shape[1] != other.shape[0]) {
//        std::cerr << "Error: MatrixH shape mismatch — shape[last] = " << shape[1]
//                  << ", other.shape[last] = " << other.shape[0] << std::endl;
//        throw std::runtime_error("MatrixH shape mismatch in operation");
//        throw;
//    }
//    
//    if ((flags & NON_CONTIGUOUS_FLAG) || (other.flags & NON_CONTIGUOUS_FLAG)) {
//        for (int i = 0; i < shape[0]; i++) {
//            for (int j = 0; j < shape[1]; j++) {
//                buffer[strides[0] * i + strides[1] * j] = other.buffer[j * other.strides[0]];
//            }
//        }
//        return *this;
//    } else {
//
//#ifdef CopyLog
//        std::cout << "Copy Assignment" << "\n";
//#endif
//        memcpy(buffer, other.buffer, total_size * sizeof(Type));
//        memcpy(shape, other.shape, dims * sizeof(size_m));
//        memcpy(strides, other.strides, dims * sizeof(size_m));
////            gradFunc = other.gradFunc;
////            parentNodes = other.parentNodes;
//        
//        
//    }
//    
//    return *this;
//}
//
//
//// copy assignment
//MatrixH<dims, Type>& operator=(const MatrixH<dims, Type>& other) {
//    
//    if (&other == this) { }
//    // 2. Data Buffer Check (Same Underlying Data)
//    // If both point to the same memory buffer, copying is redundant.
//    else if (this->buffer == other.buffer) {
//        return *this;
//    }
//    else if (total_size == other.total_size) {
//        if ((flags & NON_CONTIGUOUS_FLAG) || (other.flags & NON_CONTIGUOUS_FLAG)) {
//            
//            size_m indexA[dims];
//            size_m indexB[dims];
//            for (int gid = 0; gid < total_size; gid++) {
//                int remA = gid;
//                int remB = gid;
//                for (int i =dims-1; i >= 0; i--) {
//                    indexA[i] = remA % shape[i];
//                    indexB[i] = remB % other.shape[i];
//                    remA /= shape[i];
//                    remB /= other.shape[i];
//                }
//                
//                int offsetA = dotArray(indexA, strides, dims);
//                int offsetB = dotArray(indexB, other.strides, dims);
//                buffer[offsetA] = other.buffer[offsetB];
//            }
//            return *this;
//        }
//#ifdef CopyLog
//        std::cout << "Copy Assignment" << "\n";
//#endif
//        memcpy(buffer, other.buffer, total_size * sizeof(Type));
//        memcpy(shape, other.shape, dims * sizeof(size_m));
//        memcpy(strides, other.strides, dims * sizeof(size_m));
////            gradFunc = other.gradFunc;
////            parentNodes = other.parentNodes;
//        
//        
//    } else {
//#ifdef CopyLog
//        std::cout << "Copy Create Assignment" << "\n";
//#endif
//        if (buffer && !(flags & NON_OWNERSHIP_FLAG)) {
//            delete [] buffer;
//        }
//        flags = other.flags; // FIX: We are allocating new buffer, so we own it. Reset the ownership flags.
//        flags &= ~NON_OWNERSHIP_FLAG;
//        total_size = other.total_size;
//        buffer = new Type[total_size];
//        buildMetalBuffer();
//        memcpy(buffer, other.buffer, total_size * sizeof(Type));
//        memcpy(shape, other.shape, dims * sizeof(size_m));
//        memcpy(strides, other.strides, dims * sizeof(size_m));
////            gradFunc = other.gradFunc;
////            parentNodes = other.parentNodes;
//    }
//    
//    return *this;
//}
//
//
//MatrixH<dims, Type>& operator=(MatrixH<dims, Type>&& other) {
//    if (&other == this) { return *this; }
//    if (flags & NON_OWNERSHIP_FLAG) {
//#ifdef CopyLog
//        std::cout << "DONT OWN THE DATA COPYInG INSTEAD \n";
//#endif
//        *this = (const MatrixH<dims, Type>&) other;
//        return *this;
//    }
////        else if (total_size == other.total_size) {
////            std::cout << "Copy Assignment" << "\n";
////            memcpy(buffer, other.buffer, total_size * sizeof(Type));
////            memcpy(shape, other.shape, dims * sizeof(size_m));
////        } else {
//#ifdef MoveLog
//    std::cout << "Move Assignment" << "\n";
//#endif
//    // WRONGGGGGG
////        if (buffer && (flags & 0)) {
////            delete [] buffer;
////        }
//    if (buffer && !(flags & NON_OWNERSHIP_FLAG)) {
//        delete [] buffer;
//    }
//    buffer = other.buffer;
//    metalBuffer = other.metalBuffer;
//    flags = other.flags;
//    other.buffer = nullptr;
//    memcpy(shape, other.shape, dims * sizeof(size_m));
//    memcpy(strides, other.strides, dims * sizeof(size_m));
//    total_size = other.total_size;
////        gradFunc = std::move(other.gradFunc);
////        parentNodes = std::move(other.parentNodes);
////        parentNodes = other.parentNodes; // its a pointer to a vector
//    other.~MatrixH();
//    return *this;
//}
//
////    template <int d>
////    MatrixH<d, Type>& operator=(MatrixH<d, Type>&& other) {
////
////        this->~MatrixH();
////        return *other;
////    }
//MatrixH<dims, Type>& operator=(const simd_float3 other) {
//    
//    if ((float*)&other == this->buffer) { }
//    else if (total_size == 3) {
//        if (flags & NON_CONTIGUOUS_FLAG) {
//            size_m indexA[dims];
//            for (int gid = 0; gid < total_size; gid++) {
//                int remA = gid;
//                for (int i =dims-1; i >= 0; i--) {
//                    indexA[i] = remA % shape[i];
//                    remA /= shape[i];
//                }
//                
//                int offsetA = dotArray(indexA, strides, dims);
//                buffer[offsetA] = other[gid];
//            }
//            return *this;
//        }
//#ifdef CopyLog
//        std::cout << "Copy Assignment" << "\n";
//#endif
//        memcpy(buffer, &other, total_size * sizeof(Type));
//        
//        
//    } else {
//        std::cerr << "Cant Paste SIMD_FLOAT3 with total size of " << total_size << "\n";
//        throw;
//    }
//    
//    return *this;
//}

//#import "Matrix.mm"
//
//
//

//class Primitive {
//    mat operand1;
//    mat operand2;
//
//    std::string MSLCode;
//};

//class mat {
//public:
//    int dims;
//    size_t* shape;
//    size_t* strides;
//    size_t total_size;
//    std::vector<std::string> prevCode;
//    std::string MSLCode;
//    mat(const std::string& name): MSLCode(name) {
//        
//    }
//    mat() : dims(0), shape(nullptr), strides(nullptr), total_size(0), MSLCode("") {}
//
//    
//    mat operator +( mat operand2) {
//        mat result;
//        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
//        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
//        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
//        result.MSLCode = MSLCode + " + " + operand2.MSLCode;
//        return result;
//    }
//    
//    mat operator -(mat operand2) {
//        mat result;
//        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
//        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
//        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
//        result.MSLCode = MSLCode + " - " + operand2.MSLCode;
//        return result;
//    }
//    
//    mat operator *(mat operand2) {
//        mat result;
//        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
//        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
//        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
//        result.MSLCode = MSLCode + " * " + operand2.MSLCode;
//        return result;
//    }
//    
//    mat operator /(mat operand2) {
//        mat result;
//        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
//        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
//        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
//        result.MSLCode = MSLCode + " + " + operand2.MSLCode;
//        return result;
//    }
//};

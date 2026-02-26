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

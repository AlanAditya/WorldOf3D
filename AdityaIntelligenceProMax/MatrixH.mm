//
//  MatrixH.m
//  AdityaIntelligenceProMax
//
//  Created by Manoj Kumar on 09/03/25.
//
#include <opencv2/opencv.hpp>
#ifdef YES
#undef YES
#endif
#ifdef NO
#undef NO
#endif
#include <dlfcn.h>
#include <stdio.h>
#include <random>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <unistd.h>
// Include MediaPipe headers here
//#include "Bitch.h"

// Redefine for Objective-C compatibility
//#ifndef YES
//#define YES ((BOOL)1)
//#endif
//#ifndef NO
//#define NO  ((BOOL)0)
//#endif

//#include <absl/status/status.h>

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>
#import <AVFoundation/AVFoundation.h>
#import <Vision/Vision.h>
#import <ModelIO/ModelIO.h>
#import <MLCompute/MLCTensor.h>
#include <cstddef>
#include <typeinfo>
#if !TARGET_OS_IPHONE
#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#endif

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#include <stdint.h>
typedef uint16_t uint16;
#endif
#import <CoreText/CoreText.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mlx/mlx.h>
#include <span>
#import "Matrix.mm"
//#include "TestTracer.h"
#include <random>
#include <map>
#include <utility>

#import <CoreText/CoreText.h>
#import <CoreGraphics/CoreGraphics.h>
#include <vector>
#include <map>
#include <Accelerate/Accelerate.h>
@import GPUManager;
@import GeometryNode;
@import Utils;
@import Camera3D;
#import <SwiftUI/SwiftUI.h>

//#include "FaceLandmarkWrapper.h"
//#include <mediapipe/mediapipe/framework/calculator_graph.h>

@import SwiftUI;
#define FAST_MAX(a, b) ((a) > (b) ? (a) : (b))
#define R(start, end) Range(start, end)
#pragma once // <--- 1. Prevents recursive inclusion
#define SAFE_MODE

//#define CreationLog
//#define DestructionLog
//#define CopyLog
//#define MoveLog


int mediapipe_xcode_dylib_dummy_symbol();
constexpr auto null = std::nullopt;
using size_m = uint32_t;
simd_float3 ORIGIN = simd_make_float3(0, 0, 0);



std::ostream& operator<<(std::ostream& os, const simd::float4& matrix) {
    std::cout << "{ " << matrix.x << ", " << matrix.y << ", " << matrix.z << ", " << matrix.w <<" }";
    return os;
}

std::ostream& operator<<(std::ostream& os, const simd::float4x4& matrix) {
    std::cout << "{ " << matrix.columns[0] << ", " << matrix.columns[1] << ", " << matrix.columns[2] << ", " << matrix.columns[3] <<" }";
    return os;
}

std::ostream& operator<<(std::ostream& os, const simd::float3& matrix) {
    std::cout << "{ " << matrix.x << ", " << matrix.y << ", " << matrix.z << " }";
    return os;
}
std::ostream& operator<<(std::ostream& os, const simd::float2& matrix) {
    std::cout << "{ " << matrix.x << ", " << matrix.y  << " }";
    return os;
}
template <typename T>
void printArray(T* pointer, uint32_t size) {
    std::cout << "{ ";
    for (uint32_t i = 0; i < size; i++) {
        std::cout << pointer[i] << ", ";
    }
    std::cout << "} \n";
}

template <typename T>
T dotArray(const T* buff1, const T* buff2 ,const uint32_t size) {
    T acc = 0;
    for (uint32_t i = 0; i < size; i++) {
        acc += buff1[i] * buff2[i];
    }
    return acc;
}

typedef char simd_packed_char3 __attribute__((ext_vector_type(3)));



float CosOfVec(simd_float2 a, simd_float2 b, bool& orientation) {
    if (simd_cross(a, b).z >= 0) {
        orientation = 0;
    } else {
        orientation = 1;
    }
    return simd_dot(a, b) / (simd_length(a) * simd_length(b));
}

float CosOfVec(simd_float3 a, simd_float3 b, bool& orientation) {
    if (simd_cross(a, b).z > 0) {
        orientation = 0;
    } else {
        orientation = 1;
    }
    return simd_dot(a, b) / (simd_length(a) * simd_length(b));
}

int ring(int index, int size) {
    if (index >= 0) {
        return index % size;
    } else {
        return size + (index % size);
    }
}

template <typename T>
void reverseBuffer(const T* src, T* des, size_t len) {
    for (int i = 0; i < len / 2; i++) {
        des[i] = src[len-1-i];
        des[len-1-i] = src[i];
    }
}

void PatternFill(void* destination, const void* pattern, size_t patternSize, uint32_t n) {
    uint32_t exp = 0;
    uint32_t pO2 = 1;
    while ((1u << (exp + 1)) <= n) {
        ++exp;
    }
    
    memcpy(destination, pattern, patternSize);
    for (int i = 1; i < exp+1; i++) {
        memcpy((char*)destination + patternSize * pO2, destination, patternSize*pO2);
        // ByteShift To Multiply By 2
        pO2 <<= 1;
    }
    
    memcpy((char*)destination + patternSize * ((int)pO2), destination, (n - pO2) * patternSize);
}

class Timer {
public:
     Timer() {
        m_startTime = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        Stop();
    }
    
    void Stop() {
        auto endPointTime = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_startTime).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endPointTime).time_since_epoch().count();
        auto duration = end - start;
        double us = duration * 0.001;
        std::cout << "It took " << duration << " ns " << us << " us " <<  "\n";
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

CVPixelBufferRef createPixelBuffer(uint8_t *frameData, size_t width, size_t height, size_t channels) {
    NSDictionary *pixelAttributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
    
    CVPixelBufferRef pixelBuffer;
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)pixelAttributes, &pixelBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    void *bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);

    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
     
     // Assuming channels is the number of bytes per pixel (should be 4 for BGRA)
     size_t copyBytesPerRow = width * channels;
     
     // Copy row by row to respect the pixel buffer's stride.
     for (size_t row = 0; row < height; row++) {
         memcpy((uint8_t *)bufferAddress + row * bytesPerRow,
                frameData + row * copyBytesPerRow,
                copyBytesPerRow);
     }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    return pixelBuffer;
}

CMSampleBufferRef createSampleBuffer(CVPixelBufferRef pixelBuffer, CMTime pts) {
    CMVideoFormatDescriptionRef videoInfo;
    CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, &videoInfo);

    CMSampleTimingInfo timing = { .duration = CMTimeMake(1, 30), .presentationTimeStamp = pts, .decodeTimeStamp = kCMTimeInvalid };
    CMSampleBufferRef sampleBuffer;
    CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixelBuffer, videoInfo, &timing, &sampleBuffer);

    CFRelease(videoInfo);
    return sampleBuffer;
}

struct TypeInfo {
    int typeCode;
    int values;
    int baseType;
};


template<typename T> int get_dtype_code();

template<> int get_dtype_code<float>()           { return 0; }
template<> int get_dtype_code<float16_t>()    { return 1; }
template<> int get_dtype_code<uint8_t>()         { return 2; }
template<> int get_dtype_code<int>()             { return 3; }

template<> int get_dtype_code<int16_t>()           { return 4; }
template<> int get_dtype_code<uint16_t>()        { return 5; }
template<> int get_dtype_code<uint32_t>()        { return 6; }
template<> int get_dtype_code<simd_float2>()    { return 7; }
template<> int get_dtype_code<simd_float3>()    { return 8; }
template<> int get_dtype_code<simd_float4>()    { return 9; }


template<int code> struct dtype_from_code;  // primary template (undefined)

// Specializations
template<> struct dtype_from_code<0> { using type = float; };
template<> struct dtype_from_code<1> { using type = float16_t; };
template<> struct dtype_from_code<2> { using type = uint8_t; };
template<> struct dtype_from_code<3> { using type = int; };
template<> struct dtype_from_code<4> { using type = int16_t; };
template<> struct dtype_from_code<5> { using type = uint16_t; };
template<> struct dtype_from_code<6> { using type = uint32_t; };
template<> struct dtype_from_code<7> { using type = simd_float2; };
template<> struct dtype_from_code<8> { using type = simd_float3; };
template<> struct dtype_from_code<9> { using type = simd_float4; };

// Convenient alias:
template<int code>
using dtype_from_code_t = typename dtype_from_code<code>::type;

template<typename T> TypeInfo get_dtype_info();

template<> TypeInfo get_dtype_info<float>()           { return {0, 1, 0}; }
template<> TypeInfo get_dtype_info<float16_t>()           { return {1, 1, 0}; }
template<> TypeInfo get_dtype_info<uint8_t>()         { return {2, 1, 1}; }
template<> TypeInfo get_dtype_info<int>()             { return {3, 1, 2}; }
template<> TypeInfo get_dtype_info<int16_t>()         { return {4, 1, 3}; }
template<> TypeInfo get_dtype_info<uint16_t>()        { return {5, 1, 4}; }
template<> TypeInfo get_dtype_info<uint32_t>()        { return {6, 1, 5}; }
template<> TypeInfo get_dtype_info<simd_float2>()    { return {7, 2, 0}; }
template<> TypeInfo get_dtype_info<simd_float3>()    { return {8, 3, 0}; }
template<> TypeInfo get_dtype_info<simd_float4>()    { return {9, 4, 0}; }

int valueLimit = 5;


//enum class DType {
//    Float = 0,
//    Half = 1,
//    Int = 2,
//    UInt = 3,
//    Char = 4,
//    UChar = 5,
//    Short = 6,
//    UShort = 7,
//    // Add more as needed
//};

//const char* DTypeName(DType type) {
//    switch (type) {
//        case DType::Float: return "float";
//        case DType::Half: return "half";
//        case DType::Int: return "int";
//        case DType::UInt: return "uint";
//        case DType::Char: return "char"; // Metal uses char/uchar for 8-bit
//        case DType::UChar: return "uchar";
//        case DType::Short: return "short";
//        case DType::UShort: return "ushort";
//        default: return "void"; // Should not happen
//    }
//}

enum class DType : int {
    Float        = 0,
    Float16      = 1,
    UInt8        = 2,
    Int32        = 3,
    Int16        = 4,
    UInt16       = 5,
    UInt32       = 6,
    Float2       = 7,
    Float3       = 8,
    Float4       = 9,

    Unknown      = -1
};

template<typename T> constexpr DType dtype_from_type();

template<> constexpr DType dtype_from_type<float>()        { return DType::Float; }
template<> constexpr DType dtype_from_type<float16_t>()  { return DType::Float16; }
template<> constexpr DType dtype_from_type<uint8_t>()      { return DType::UInt8; }
template<> constexpr DType dtype_from_type<int>()          { return DType::Int32; }
template<> constexpr DType dtype_from_type<int16_t>()      { return DType::Int16; }
template<> constexpr DType dtype_from_type<uint16_t>()     { return DType::UInt16; }
template<> constexpr DType dtype_from_type<uint32_t>()     { return DType::UInt32; }
template<> constexpr DType dtype_from_type<simd_float2>()  { return DType::Float2; }
template<> constexpr DType dtype_from_type<simd_float3>()  { return DType::Float3; }
template<> constexpr DType dtype_from_type<Point3D>()  { return DType::Float3; }
template<> constexpr DType dtype_from_type<simd_float4>()  { return DType::Float4; }


template<DType code> struct type_dtype;

template<> struct type_dtype<DType::Float>   { using type = float; };
template<> struct type_dtype<DType::Float16>  { using type = float16_t; };
template<> struct type_dtype<DType::UInt8>   { using type = uint8_t; };
template<> struct type_dtype<DType::Int32>   { using type = int; };
template<> struct type_dtype<DType::Int16>   { using type = int16_t; };
template<> struct type_dtype<DType::UInt16>  { using type = uint16_t; };
template<> struct type_dtype<DType::UInt32>  { using type = uint32_t; };
template<> struct type_dtype<DType::Float2>  { using type = simd_float2; };
template<> struct type_dtype<DType::Float3>  { using type = simd_float3; };
template<> struct type_dtype<DType::Float4>  { using type = simd_float4; };


template<DType code>
using type_from_dtype = typename type_dtype<code>::type;




class MatrixBase {
public:
    int mDims;
    DType dtype;
    MatrixBase(int dims, DType dtype): mDims(dims), dtype(dtype) {
    }
};

template <int dims, typename Type>
class MatrixH;



template <typename T>
void fill_nd_iterative(
    T* base,
    const size_m* shape,
    const size_m* strides,
    const int dims,
    const T& value)
{
    const size_t ndim = dims;
    if (ndim == 0) return;
    if (ndim == 1) {
        // Single dimension — contiguous or strided
        for (size_t i = 0; i < shape[0]; i++) {
            base[i * strides[0]] = value;
        }
        return;
    }

    // Logical indices for each dimension
    std::vector<size_t> idx(ndim, 0);
    T* ptr = base;

    while (true) {
        // Fill the innermost dimension in a tight loop
        for (size_t i = 0; i < shape[ndim - 1]; i++) {
            ptr[i * strides[ndim - 1]] = value;
        }

        // Increment multi-dimensional index (except last dim)
        size_t dim = ndim - 2; // start from second-last dimension
        while (true) {
            idx[dim]++;
            ptr += strides[dim]; // move pointer by stride[dim] elements

            if (idx[dim] < shape[dim]) {
                // We can break and do the next row
                break;
            } else {
                // Reset this dim and move up one
                idx[dim] = 0;
                ptr = base;
                for (size_t d = 0; d < dim; d++) {
                    ptr += idx[d] * strides[d];
                }
                if (dim == 0) {
                    return; // All done
                }
                dim--;
            }
        }
    }
}

enum class ConvMode : uint8_t {
    Full = 0,
    Valid = 1,
    Same  = 2  // optional common mode
};

enum class ConvModeFull : uint8_t {
    Normal = 0,
    Extend = 1,
    Mirror  = 2  // optional common mode
};

template <typename T, int dims>
struct nested_initializer_list;

template <typename T>
struct nested_initializer_list<T, 1> {
    using type = std::initializer_list<T>;
};

template <typename T, int dims>
struct nested_initializer_list {
    using type = std::initializer_list<typename nested_initializer_list<T, dims - 1>::type>;
};
template <typename T>
struct nested_initializer_list<T, 0> {
    using type = T;
};

enum Flags : unsigned int {
    NON_OWNERSHIP_FLAG = 1u << 0,  // Bit 0
    NON_CONTIGUOUS_FLAG = 1u << 1,  // Bit 1
    COMPUTE_GRAPH = 1u << 2 // Bit 2
};


class OppNode {
public:
    std::shared_ptr<MatrixBase> A;
    std::shared_ptr<MatrixBase> B;
    
    std::string name;
    virtual std::string generateMSL() = 0;
    OppNode(const OppNode&) = default;
    OppNode(OppNode&&) noexcept = default;
    OppNode& operator=(const OppNode&) = default;
    OppNode& operator=(OppNode&&) noexcept = default;
    OppNode(std::shared_ptr<MatrixBase> A, std::shared_ptr<MatrixBase> B): A(A), B(B) {
        
    }
};

struct PairHash {
    std::size_t operator()(const std::pair<int, DType>& key) const noexcept {
        return std::hash<int>{}(key.first) ^ (std::hash<int>{}(static_cast<int>(key.second)) << 1);
    }
};

struct PairEqual {
    bool operator()(const std::pair<int, DType>& a, const std::pair<int, DType>& b) const noexcept {
        return a.first == b.first && a.second == b.second;
    }
};

class AddNode: public OppNode {
public:
    AddNode(std::shared_ptr<MatrixBase> A, std::shared_ptr<MatrixBase> B): OppNode(A, B) {
         
    }
    AddNode(const AddNode&) = default;
    AddNode(AddNode&&) noexcept = default;
    AddNode& operator=(const AddNode&) = default;
    AddNode& operator=(AddNode&&) noexcept = default;

    
    std::string generateMSL() override {
        std::string LHS;
        std::string RHS;
        
        using HandlerFn = std::string(*)(std::shared_ptr<MatrixBase>);
        
        std::unordered_map<std::pair<int, DType>, HandlerFn, PairHash, PairEqual> table = {
            {{1, DType::Float},  handleMatrix<1, float>},
            {{2, DType::Float},  handleMatrix<2, float>},
            {{3, DType::Float},  handleMatrix<3, float>},
        };
        
        LHS = table.at({A->mDims, A->dtype})(A);
        RHS = table.at({B->mDims, B->dtype})(B);
        return LHS + "+" +  RHS;
    }
    
    template<int dims, typename Type>
    static std::string handleMatrix(std::shared_ptr<MatrixBase> base_ptr);
};

// Range type for slicing
struct Range {
    int start, end;
    constexpr Range(int s, int e) : start(s), end(e) {}
};

// Proxy object to enable colon syntax
struct Slice {
    int value;
    
    // Implicit conversion from int - this is the key!
    constexpr Slice(int v) : value(v) {}
    
    // The magic: overload operator| to mimic colon
    constexpr Range operator|(const Slice& other) const {
        return Range(value, other.value);
    }
};


struct TensorRef {
    void* raw;           // points to MatrixH<dims,Type>
    const std::type_info* type;
};
struct TapeNode {
    // Parents of this node
    std::vector<TensorRef> parents;

    // Backward function: input is this node's gradient
    std::function<void(const TensorRef& grad)> backward;

    // Gradient accumulated during backward pass
    TensorRef grad;

    // For memory management (optional)
    int refcount = 0;
};

template<int dims, typename T>
TensorRef wrap(MatrixH<dims, T>* t) {
    return TensorRef{ t, &typeid(MatrixH<dims, T>) };
}

template<int dims, typename T>
void attachTape(MatrixH<dims, T>& out, TapeNode* node) {
//    out.tape = node;
}

template<int dimsB, typename Type>
inline void setBufferOrBytes(id<MTLComputeCommandEncoder> commandEncoder,
                              const MatrixH<dimsB, Type>& tensor,
                              NSUInteger index)  {
    if (tensor.metalBuffer) {
        [commandEncoder setBuffer:tensor.metalBuffer offset:0 atIndex:index];
    } else {
        size_t byteLength;
        if (tensor.flags & NON_CONTIGUOUS_FLAG) {
            // For non-contiguous: calculate actual buffer span
            // From first element to last element including stride gaps
            byteLength = (tensor.strides[0] * tensor.shape[0]) * sizeof(Type);
        } else {
            // For contiguous: use total size
            byteLength = tensor.total_size * sizeof(Type);
        }
        
        [commandEncoder setBytes:tensor.buffer length:byteLength atIndex:index];
    }
}

//template<int dims, typename T>
//void accumulate_grad(MatrixH<dims, T>& parent, const TensorRef& grad_out) {
//    if (!parent.tape->grad.raw) {
//        parent.tape->grad = grad_out;   // first time
//    } else {
//        // parent.tape->grad += grad_out
//        add_into(parent.tape->grad, grad_out);
//    }
//}
enum class OpType { NONE, ADD, MATMUL };
struct Primitive {
    OpType op;
    Primitive(OpType o = OpType::NONE) : op(o) {}
    virtual ~Primitive() = default;
    virtual void evaluate_inputs(MatrixBase* output) = 0;
};

template <int dims, typename Type>
struct SimmilarPrimitive : public Primitive {
    std::shared_ptr<MatrixH<dims, Type>> Mat1;
    std::shared_ptr<MatrixH<dims, Type>> Mat2;

    SimmilarPrimitive(OpType o, std::shared_ptr<MatrixH<dims, Type>> m1, std::shared_ptr<MatrixH<dims, Type>> m2)
        : Primitive(o), Mat1(m1), Mat2(m2) {}

    // NEW: Implement the bridge to traverse further down
    void evaluate_inputs(MatrixBase* output) override {
        Mat1->evaluate();
        Mat2->evaluate();
        if (op == OpType::ADD) {
            Mat1->BroadcastedAdd(*static_cast<MatrixH<dims, Type>*>(output), *Mat2);
        }
    }
};

template <int InDims, int OutDims, typename Type>
struct UnsqeezePrimitive : public Primitive {
    std::shared_ptr<MatrixH<InDims, Type>> input;
    int axis;

    UnsqeezePrimitive(std::shared_ptr<MatrixH<InDims, Type>> in, int axis) : input(in) {}

    void evaluate_inputs(MatrixBase* output) override {
        input->evaluate();
        auto out_casted = static_cast<MatrixH<OutDims, Type>*>(output);
        memcpy(out_casted->buffer, input->buffer, sizeof(Type) * out_casted->total_size);
    }
};

template <int dims, typename Type>
class MatrixH : public MatrixBase {
public:
    Type* buffer = nullptr;
    size_m shape[dims];
    size_m strides[dims];
    size_t total_size;
    id<MTLBuffer> metalBuffer = nil;
    uint8_t flags = 0;
    Primitive* tape = nullptr;
    std::atomic<uint32_t>* refCount = nil;
//    kes
//    using ParentList = std::vector<std::shared_ptr<MatrixH<dims, Type>>>;
//    __attribute__((annotate("lldb_hidden"))) ParentList* parentNodes = nullptr;
    
//    std::function<MatrixH<dims, Type>(MatrixH<dims, Type>&)>* gradFunc = nullptr;
//    std::conditional_t<
//        !std::is_same<Type, Point3D>::value,
//        Type,
//        std::nullptr_t
//    > grad;
    
    
    
    using initializer_type = typename nested_initializer_list<Type, dims>::type;
    MatrixH(initializer_type nestedList) requires (dims != 0) : MatrixBase(dims, dtype_from_type<Type>())  {
        total_size = 1;
        computeShape(nestedList, 0);
        buffer = new Type[total_size];
        int k = 0;
        writeInBuffer(nestedList, k);
        if (total_size > 10) {
            buildMetalBuffer();
        }
        #ifdef CreationLog
        std::cout << "Created" << "\n";
        #endif
        calcStrides();
    }
    
    MatrixH()
      : MatrixBase(dims, dtype_from_type<Type>()), buffer(nullptr), total_size(0), metalBuffer(nullptr) {
        #ifdef CreationLog
        std::cout << "Created" << "\n";
        #endif

      }
    
    template <typename T>
    struct is_initializer_list : std::false_type {};

    template <typename T>
    struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
    
    template <typename T>
    void computeShape(const T& nestedList, int d) {
        if constexpr (is_initializer_list<T>::value) {
            if (d == dims - 1) {
                shape[d] = nestedList.size();
                total_size *= nestedList.size();
            } else {
                shape[d] = nestedList.size();
                total_size *= nestedList.size();
                computeShape(*nestedList.begin(), d+1);
            }
        }
    }
    
    template <typename T>
    void writeInBuffer(const T& nestedList, int& currentIndex) {
        if constexpr (std::is_same<T, std::initializer_list<Type>>::value) {
            memcpy(buffer + currentIndex, nestedList.begin(), nestedList.size() * sizeof(Type));
            currentIndex += nestedList.size();
        } else {
            for (auto i: nestedList) {
                writeInBuffer(i, currentIndex);
            }
        }
        
    };
    
    MatrixH(Type value, bool buildGPUBuffer = true) requires (dims == 0) : MatrixBase(0, dtype_from_type<Type>()) {
        #ifdef CreationLog
        std::cout << "Created" << "\n";
        #endif
        buffer = new Type[1];
        *buffer = value;
        total_size = 1;
        if (buildGPUBuffer) {
            buildMetalBuffer();
        }
    }
    
    
    explicit MatrixH(size_t reserveCapacity) requires (dims != 0) : MatrixBase(dims, dtype_from_type<Type>()) {
        #ifdef CreationLog
        std::cout << "Created" << "\n";
        #endif
        buffer = new Type[reserveCapacity];
        shape[0] = reserveCapacity;
        std::fill(shape+1, shape+dims, 1);
        
        total_size = reserveCapacity;
        if (total_size > 10) {
            buildMetalBuffer();
        }
    }
    
    explicit MatrixH(std::initializer_list<Type> list) {
        total_size = list.size();
        shape[0] = list.size();
        buffer = new Type[total_size];
        memcpy(buffer, list.begin(), sizeof(Type) * list.size());
        strides[0] = 1;
        
        if (total_size > 10) {
            buildMetalBuffer();
        }
    }

    MatrixH<dims, Type> addWithTape(MatrixH<dims, Type>& A, MatrixH<dims, Type>& B) {
        MatrixH<dims, Type> out;
        // ... compute out = A + B on CPU or Metal ...

        // Create tape node
        TapeNode* node = new TapeNode;
        node->parents = { wrap(&A), wrap(&B) };

        // Define backward function
        node->backward = [&](const TensorRef& grad_out) {
            // dA = grad_out
//            accumulate_grad(A, grad_out);
            // dB = grad_out
//            accumulate_grad(B, grad_out);
        };

        attachTape(out, node);
        return out;
    }
    
    static MatrixH<dims, Type> solid() {
        MatrixH<dims, Type> outputM;
    }
    
    MatrixH<dims, Type> cloneData() {
        MatrixH<dims, Type> result;
        result.total_size = total_size;
        
        size_t eff_total_size = effectiveBufferSize();
        result.buffer = new Type[eff_total_size];
        memcpy(result.buffer, buffer, eff_total_size * sizeof(Type));
        
    }
    
    void evaluate() {
        if (tape) {
            // 1. Recursively go down the graph to ensure inputs are ready
            tape->evaluate_inputs(this);
                
            // 2. Compute THIS node's data (The actual math kernels go here)
            std::cout << "Executing OpType: " << static_cast<int>(tape->op) << " for a " << dims << "D matrix.\n";
//            if (tape->op == OpType::ADD) {
//                auto casted_tape = static_cast<SimmilarPrimitive<dims, Type>*>(tape);
//                casted_tape->Mat1->BroadcastedAdd(*this, *casted_tape->Mat2);
//            }
        }
//        is_computed = true; // Mark as done so we don't calculate it twice
        }
    
    template <int dimO>
    void cloneData(MatrixH<dimO, Type>& mat) {
        mat.total_size = total_size;
        
        size_t eff_total_size = effectiveBufferSize();
        mat.buffer = new Type[eff_total_size];
        memcpy(mat.buffer, buffer, eff_total_size * sizeof(Type));
    }
    
    MatrixH<dims, Type> copy() {
        MatrixH<dims, Type> result = cloneData();
        result.total_size = total_size;
        result.flags = flags;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        result.buildMetalBuffer();
        return result;
    }
    
    
//    BUG FIX: Metal Buffer Cache Coherency Issue for Non-Contiguous Buffers
//
//    PROBLEM:
//    When creating Metal buffers for non-contiguous data using newBufferWithBytesNoCopy,
//    the buffer length was incorrectly set to (total_size * sizeof(Type)), which only
//    accounts for the number of valid elements, not the actual memory span.
//
//    For non-contiguous buffers, elements may be scattered across a larger memory region
//    due to strides. When Metal is told the buffer is smaller than the actual accessed
//    memory range, it may not properly cache, synchronize, or maintain coherency for
//    memory beyond the declared boundary, leading to intermittent GPU read failures
//    (~10% failure rate observed) where later elements appear as zeros.
//
//    ROOT CAUSE:
//    Metal's memory management system uses the declared buffer length to determine:
//    - Cache line prefetching boundaries
//    - CPU→GPU memory synchronization scope
//    - Memory barrier coverage regions
//
//    Even though the underlying pointer is valid and CPU-side access works correctly,
//    GPU kernel access to indices beyond the declared buffer size may fail because
//    Metal's caching layer doesn't guarantee those memory regions are synchronized
//    or available in GPU cache.
//
//    FIX:
//    For non-contiguous buffers, use the actual memory span (shape[0] * strides[0] * sizeof(Type))
//    instead of the element count (total_size * sizeof(Type)). This ensures Metal properly
//    manages the entire accessed memory region.
//
//    BEFORE:
//    total_size * sizeof(Type)  // ❌ Only counts valid elements

//
//    AFTER:
//    length:shape[0] * strides[0] * sizeof(Type)  // ✅ Full memory span
//    IMPORTANT:
//    buildMetalBuffer() must be called AFTER setting the NON_CONTIGUOUS_FLAG, as the
//    flag determines which buffer size calculation to use:
//
//    CORRECT ORDER:
//    InstanceTransformView.flags |= NON_CONTIGUOUS_FLAG;  // Set flag first
//    InstanceTransformView.buildMetalBuffer();            // Then build buffer
//
//    INCORRECT ORDER:
//    InstanceTransformView.buildMetalBuffer();            // ❌ Wrong: uses wrong size
//    InstanceTransformView.flags |= NON_CONTIGUOUS_FLAG;  // ❌ Too late
//
//    IMPACT:
//    - Fixes intermittent GPU kernel failures where non-contiguous buffer elements
//      read as zero
//    - Ensures proper Metal cache coherency for strided/sparse tensor operations
//    - Eliminates race conditions in GPU compute operations on broadcasted/reshaped tensors
    void buildMetalBuffer() {
        if (total_size == 0 && !buffer) {std::cout << "failed to build MTL buffer with size 0"; return;}
//        if (total_size < 10) { return; }

       metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:effectiveBufferSize() * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
       }];
        
    }
    
    static size_t accumul(const std::vector<size_t>& shape) {
        size_t acc = 1;
        for (int i = 0; i < shape.size(); i ++) {
            acc *= shape[i];
        }
        return acc;
    }
    
    size_t accumul(int start, int end) const {
        size_t acc = 1;
        for (int i = start; i < end; i ++) {
            acc *= shape[i];
        }
        return acc;
    }
    
    void calcStrides() {
        size_t acc = 1;
        for (int i = dims-1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
    }
    
    bool compareShapes(size_m* Othershape) {
        bool res = true;
        for (int i = 0; i < dims; i ++) {
            if (shape[i] != Othershape[i]) {
                res = false;
            }
        }
        return res;
    }
    
    bool compareShapes(size_m* Othershape, int end) {
        if (end < 0) {
            end = dims - end;
        }
        bool res = true;
        for (int i = 0; i < end; i ++) {
            if (shape[i] != Othershape[i]) {
                res = false;
            }
        }
        return res;
    }
    
    size_t effectiveBufferSize() {
        if (flags & NON_CONTIGUOUS_FLAG) {
            // CORRECT: Calculate the maximum memory offset generated by the strides
            size_t max_offset = 0;
            for (int d = 0; d < dims; ++d) {
                if (shape[d] > 0) {
                    // The max index for this dimension is (count - 1) * stride
                    max_offset += (shape[d] - 1) * strides[d];
                }
            }
            // Buffer must be large enough to hold the last element
            return (max_offset + 1);
        } else {
            return total_size;
        }
    }
    
    static MatrixH<dims, Type> constant(std::initializer_list<size_m> shapeI ,Type value) {
        #ifdef SAFE_MODE
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape (" << shapeI.size() << ") should not exceed dim of matrix (" << dims << ")";
            throw std::invalid_argument("MatrixH: Shape dimensions mismatch."); // FIXED
        }
        #endif
        MatrixH<dims, Type> result;
        memcpy(result.shape, shapeI.begin(), shapeI.size() * sizeof(size_m));
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size]; // no need for effective total buffer size as the buffer is allocated by us;
        std::fill(result.buffer, result.buffer + result.total_size, value);
        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }
    
    template <int dimsI>
    static MatrixH<dims, Type> repeating(std::initializer_list<size_m> shapeI, const MatrixH<dimsI, Type>& pattern) {
        #ifdef SAFE_MODE
        if (shapeI.size() + dimsI != dims) {
            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
            throw std::invalid_argument("MatrixH: Repeating shape dimensions mismatch.");
        }
        #endif
        MatrixH<dims, Type> result;
        memcpy(result.shape, shapeI.begin(), shapeI.size() * sizeof(size_m));
        memcpy(result.shape + shapeI.size(), pattern.shape, dimsI * sizeof(size_m));
        result.calcStrides();
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        PatternFill(result.buffer, pattern.buffer, pattern.total_size * sizeof(Type), result.accumul(0, shapeI.size()));
        result.buildMetalBuffer();
        return result;
    }
    
    template <int dimsI>
    static MatrixH<dims, Type> repeatingGPU(std::initializer_list<size_m> shapeI, const MatrixH<dimsI, Type>& pattern) {
        #ifdef SAFE_MODE
        if (shapeI.size() + dimsI != dims) {
            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
            throw std::invalid_argument("Repeating shape dimensions mismatch."); // FIXED
        }
        #endif
        MatrixH<dims, Type> result;
        MatrixH<dims, Type> patternView;
        memcpy(patternView.shape, shapeI.begin(), shapeI.size() * sizeof(size_m));
        memcpy(patternView.shape + shapeI.size(), pattern.shape, dimsI * sizeof(size_m));
        memset(patternView.strides, 0, shapeI.size() * sizeof(size_m));
        memcpy(patternView.strides + shapeI.size(), pattern.strides, dimsI * sizeof(size_m));
        patternView.total_size = patternView.accumul(0, dims);
        patternView.buffer = pattern.buffer;
        patternView.metalBuffer = pattern.metalBuffer;
        patternView.flags |= NON_OWNERSHIP_FLAG;
        patternView.flags |= NON_CONTIGUOUS_FLAG;

        memcpy(result.shape, patternView.shape, dims * sizeof(size_m));
        result.calcStrides();
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        result.buildMetalBuffer();
        
        copyGPUinplace(result, patternView, 0);
        return result;
    }
    
    static MatrixH<dims, Type> repeating(std::initializer_list<size_m> shapeI, const Type& pattern) {
        #ifdef SAFE_MODE
        if (shapeI.size() != dims) {
            std::cerr << "Dimensions Dont Add up, Pattern: " << 0 << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
            throw std::invalid_argument("MatrixH: Repeating shape dimensions mismatch.");
        }
        #endif
        MatrixH<dims, Type> result;
        memcpy(result.shape, shapeI.begin(), shapeI.size() * sizeof(size_m));
        result.calcStrides();
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        PatternFill(result.buffer, &pattern, sizeof(Type), result.accumul(0, shapeI.size()));
        result.buildMetalBuffer();
        return result;
    }
    
    static MatrixH<dims, Type> Range(Type start, std::initializer_list<size_m> shapeI) {
        if (shapeI.size() != dims) {std::cerr << "MatrixH: Shape should not excede dim of matrix"; throw;}
        MatrixH<dims, Type> result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        
        for (int i = 0; i < result.total_size; i++) {
            result.buffer[i] = i+start;
        }
        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }
    
    static MatrixH<dims, Type> linspace(Type start, Type end, std::initializer_list<size_m> shapeI) {
        #ifdef SAFE_MODE
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape should not exceed dim of matrix";
            throw std::invalid_argument("MatrixH: Shape dimensions mismatch.");
        }
        #endif
        MatrixH<dims, Type> result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        
        Type step = (result.total_size > 1) ? (end - start) / (result.total_size - 1) : 0;
        
        for (int i = 0; i < result.total_size; i++) {
            result.buffer[i] = start + i * step;
        }
        
        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }
    
    // Random uniform distribution uses std::mt19937 generator which values perfection of distribution over speed.
    // std::mt19937 is ~2.5KB of state and comparitively slow.
    static MatrixH<dims, float> rand_uniform(std::initializer_list<size_m> shapeI, Type lower, Type upper) {
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape should not exceed dim of matrix";
            throw std::invalid_argument("Invalid shape dimensions");
        }
        
        MatrixH<dims, Type> result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        size_t grain_size = 4096;
        if (result.total_size > grain_size) {
            size_t iterations = (result.total_size + grain_size - 1) / grain_size;


            dispatch_apply(iterations, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t chunk_idx)  {
                size_t start = chunk_idx * grain_size;
                size_t end = std::min(start + grain_size, result.total_size);
                // Thread-local generator for parallel safety
                static thread_local std::mt19937 generator(std::random_device{}());
                // Create distribution locally (it's lightweight)
                std::uniform_real_distribution<float> distribution(lower, upper);
                for (size_m i = start; i < end; i++) {
                    result.buffer[i] = distribution(generator);
                }
            });
        } else {
            // Fast random generation using thread-local RNG
            static thread_local std::mt19937 generator(std::random_device{}());
            std::uniform_real_distribution<float> distribution(lower, upper);
            
            // Vectorized generation
            for (size_m i = 0; i < result.total_size; i++) {
                result.buffer[i] = distribution(generator);
            }
        }
        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }

    // Random uniform distribution uses std::minstd_rand generator which values speed over perfection of distribution.
    static MatrixH<dims, float> fast_rand_uniform(std::initializer_list<size_m> shapeI, Type lower, Type upper) {
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape should not exceed dim of matrix";
            throw std::invalid_argument("Invalid shape dimensions");
        }
        
        MatrixH<dims, Type> result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        size_t grain_size = 4096;
        if (result.total_size > grain_size) {
            size_t iterations = (result.total_size + grain_size - 1) / grain_size;
            // C++17 Parallel Policy
            dispatch_apply(iterations, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t chunk_idx)  {
                size_t start = chunk_idx * grain_size;
                size_t end = std::min(start + grain_size, result.total_size);
                // Thread-local generator for parallel safety
                static thread_local std::minstd_rand generator(std::random_device{}());
                // Create distribution locally (it's lightweight)
                std::uniform_real_distribution<float> distribution(lower, upper);
                for (size_m i = start; i < end; i++) {
                    result.buffer[i] = distribution(generator);
                }
            });
        } else {
            // Fast random generation using thread-local RNG
            static thread_local std::minstd_rand generator(std::random_device{}());
            std::uniform_real_distribution<float> distribution(lower, upper);
            
            // Vectorized generation
            for (size_m i = 0; i < result.total_size; i++) {
                result.buffer[i] = distribution(generator);
            }
        }
        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }
    
    
    static MatrixH<dims, float> noise_texture(
        std::initializer_list<size_m> shapeI,
        float scale = 5.0f,
        float detail = 2.0f,
        float roughness = 0.5f,
        float lacunarity = 2.0f,
        float offset = 0.0f,
        float gain = 1.0f,
        float distortion = 0.0f
    ) {
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape should not exceed dim of matrix";
            throw std::invalid_argument("Invalid shape dimensions");
        }
        static size_t NoiseC =0;
        std::cout << "Noise called: " << NoiseC << "\n";
        NoiseC+=1;
        MatrixH<dims, float> result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new float[result.total_size];
        
        // Permutation table for Perlin noise (standard)
        static const int p[512] = {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
            8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
            134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
            55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
            18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
            250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
            189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
            172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
            228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
            107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
            // Second half (same as first half)
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
            8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
            134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
            55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
            18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
            250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
            189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
            172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
            228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
            107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };
        
        // Helper functions for Perlin noise
        auto fade = [](float t) -> float {
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        };
        
        auto lerp = [](float t, float a, float b) -> float {
            return a + t * (b - a);
        };
        
        auto grad = [](int hash, float x, float y, float z) -> float {
            int h = hash & 15;
            float u = h < 8 ? x : y;
            float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        };
        
        // Perlin noise function
        auto perlin = [&](float x, float y, float z) -> float {
            int X = (int)std::floor(x) & 255;
            int Y = (int)std::floor(y) & 255;
            int Z = (int)std::floor(z) & 255;
            
            x -= std::floor(x);
            y -= std::floor(y);
            z -= std::floor(z);
            
            float u = fade(x);
            float v = fade(y);
            float w = fade(z);
            
            int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
            int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
            
            return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                          grad(p[BA], x - 1, y, z)),
                                  lerp(u, grad(p[AB], x, y - 1, z),
                                          grad(p[BB], x - 1, y - 1, z))),
                          lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                                          grad(p[BA + 1], x - 1, y, z - 1)),
                                  lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                                          grad(p[BB + 1], x - 1, y - 1, z - 1))));
        };
        
        // Fractal noise with octaves
        auto fractal_noise = [&](float x, float y, float z) -> float {
            float value = 0.0f;
            float amplitude = 1.0f;
            float frequency = 1.0f;
            float max_value = 0.0f;
            
            int octaves = (int)detail;
            float blend = detail - octaves;
            
            for (int i = 0; i < octaves; ++i) {
                float n = perlin(x * frequency, y * frequency, z * frequency);
                value += n * amplitude;
                max_value += amplitude;
                
                amplitude *= roughness;
                frequency *= lacunarity;
            }
            
            // Fractional octave blending
            if (blend > 0.0f && octaves < 16) {
                float n = perlin(x * frequency, y * frequency, z * frequency);
                value += n * amplitude * blend;
                max_value += amplitude * blend;
            }
            
            return value / max_value;
        };
        
        // Generate noise texture
        size_m indices[dims];
        for (size_m i = 0; i < dims; ++i) indices[i] = 0;
        
        for (size_m i = 0; i < result.total_size; ++i) {
            // Calculate normalized coordinates
            float coords[3] = {0.0f, 0.0f, 0.0f};
            
            for (size_m d = 0; d < dims && d < 3; ++d) {
                coords[d] = (float)indices[d] / (float)result.shape[d];
            }
            
            // Apply scale
            float x = coords[0] * scale;
            float y = coords[1] * scale;
            float z = coords[2] * scale;
            
            // Apply distortion if needed
            if (distortion > 0.0f) {
                float dx = perlin(x + 13.5f, y + 17.3f, z + 11.7f) * distortion;
                float dy = perlin(x + 23.1f, y + 31.9f, z + 29.3f) * distortion;
                float dz = perlin(x + 41.7f, y + 43.2f, z + 47.9f) * distortion;
                x += dx;
                y += dy;
                z += dz;
            }
            
            // Calculate fractal noise with offset and gain
            float noise_val = fractal_noise(x, y, z);
            noise_val = (noise_val + offset) * gain;
            
            // Clamp to [0, 1] range
            result.buffer[i] = std::max(0.0f, std::min(1.0f, noise_val * 0.5f + 0.5f));
            
            // Update indices
            for (int d = dims - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < result.shape[d]) break;
                indices[d] = 0;
            }
        }

        result.buildMetalBuffer();
        result.calcStrides();
        return result;
    }
    
    static MatrixH gaussian(std::initializer_list<size_m> shapeI, Type stddev = 1.0, bool normalize = true) {
        if (shapeI.size() != dims) {
            std::cerr << "MatrixH: Shape should not exceed dim of matrix";
            throw std::invalid_argument("Invalid shape dimensions");
        }
        
        MatrixH result;
        std::copy(shapeI.begin(), shapeI.end(), result.shape);
        result.total_size = result.accumul(0, dims);
        result.buffer = new Type[result.total_size];
        result.calcStrides();
        // Get shape dimensions
        size_m* s = result.shape;
        
        if constexpr (dims == 1) {
            // 1D Gaussian
            Type center = s[0] / 2.0;
            for (size_m i = 0; i < s[0]; i++) {
                Type x = (i - center) / stddev;
                result.buffer[i * result.strides[0] ] = std::exp(-0.5 * x * x);
            }
        } else if constexpr (dims == 2) {
            // 2D Gaussian
            Type center_y = (s[0]-1) / 2.0;
            Type center_x = (s[1]-1) / 2.0;
            for (size_m i = 0; i < s[0]; i++) {
                for (size_m j = 0; j < s[1]; j++) {
                    Type y = (i - center_y) / stddev;
                    Type x = (j - center_x) / stddev;
                    result.buffer[i * result.strides[0]+ j * result.strides[1]] = std::exp(-0.5 * (x * x + y * y));
                }
            }
        } else if constexpr (dims == 3) {
            // 3D Gaussian
            Type center_z = s[0] / 2.0;
            Type center_y = s[1] / 2.0;
            Type center_x = s[2] / 2.0;
            for (size_m i = 0; i < s[0]; i++) {
                for (size_m j = 0; j < s[1]; j++) {
                    for (size_m k = 0; k < s[2]; k++) {
                        Type z = (i - center_z) / stddev;
                        Type y = (j - center_y) / stddev;
                        Type x = (k - center_x) / stddev;
                        result.buffer[i * result.strides[0] + j * result.strides[1] + k * result.strides[2]] = std::exp(-0.5 * (x * x + y * y + z * z));
                    }
                }
            }
        }
        // Normalize if requested
        if (normalize) {
            Type sum = 0;
            for (size_m i = 0; i < result.total_size; i++) {
                sum += result.buffer[i];
            }
            if (sum > 0) {
                for (size_m i = 0; i < result.total_size; i++) {
                    result.buffer[i] /= sum;
                }
            }
        }
        result.buildMetalBuffer();
        
        return result;
    }
    
    void printShape() const {
        std::cout << "Shape is { ";
        for (int i=0; i<dims; i++) {
            std::cout << shape[i] << ", ";
        }
        std::cout << "}\n";
    }
    
    void printShape(bool verbose) const {
        if (verbose == true) { std::cout << "Shape is { "; }
        
        for (int i=0; i<dims; i++) {
            std::cout << shape[i] << ", ";
        }
        if (verbose == true) { std::cout << "}\n"; }
        
    }
    
    void printStrides(bool verbose = true) const {
        if (verbose == true) { std::cout << "Strides are { "; }
        
        for (int i=0; i<dims; i++) {
            std::cout << strides[i] << ", ";
        }
        if (verbose == true) { std::cout << "}\n"; }
        
    }
    
    CGColorRef createCGColor(float r, float g, float b, float a) {
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGFloat components[] = {r, g, b, a};
        CGColorRef color = CGColorCreate(colorSpace, components);
        CGColorSpaceRelease(colorSpace);
        return color;  // Remember to CFRelease when done using it
    }

    CGColorRef createCGColorFromMatrixH(const MatrixH<1, int> &colorMat) {
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGFloat components[] = {colorMat.buffer[0] / 255.0, colorMat.buffer[1] / 255.0, colorMat.buffer[2] / 255.0, colorMat.buffer[3] / 255.0};
        CGColorRef color = CGColorCreate(colorSpace, components);
        CGColorSpaceRelease(colorSpace);
        return color;  // Remember to CFRelease when done using it
    }


    void drawText(char* text, MatrixH<1, int> point, const MatrixH<1, int>& colorMat, float fontSize) {
        if (colorMat.total_size != 4) {
            std::cerr << "Error: 4 arguments are required for colour" << "\n";
            return;
        }
        
        if (point.total_size != 2) {
            std::cerr << "Error: 2 arguments are required for position" << "\n";
            return;
        }
        CGColorRef color = createCGColorFromMatrixH(colorMat);
        
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        size_t widthsize = total_size / shape[0];
        CGContextRef context = CGBitmapContextCreate(
                                                     buffer, shape[1], shape[0], 8 * sizeof(Type), sizeof(Type) * widthsize, colorSpace, kCGImageAlphaPremultipliedLast
        );
        
        if (!context) {
            fprintf(stderr, "Failed to create bitmap context!\n");
            CGColorSpaceRelease(colorSpace);
        }

        CFStringRef stringRef = CFStringCreateWithCString(NULL, text, kCFStringEncodingUTF8);
        CTFontRef font = CTFontCreateWithName(CFSTR("Helvetica"), fontSize, NULL);
        
        NSDictionary *attributes = @{ (__bridge id)kCTFontAttributeName: (__bridge id)font, (__bridge id)kCTForegroundColorAttributeName: (__bridge id)color };
        NSAttributedString *attributedString = [[NSAttributedString alloc] initWithString:(__bridge NSString *)stringRef attributes:attributes];
        
        
        CTLineRef line = CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)attributedString);
        
        CGContextSetTextPosition(context, point.buffer[0], point.buffer[1]);
        CGContextSetTextDrawingMode(context, kCGTextFillClip);
        
        // Draw text
        CTLineDraw(line, context);
    }

    
    static MatrixH<dims, Type> blend(MatrixH<dims, Type>& mat1, MatrixH<dims, Type>& mat2) {
        MatrixH<dims, Type> output;
        for (int i = 0; i < dims; i++) {
            if (mat1.shape[i] != mat2.shape[i]) {
                std::cerr << "shape error \n";
                return output;
            }
            output.shape[i] = mat1.shape[i];
        }
        output.total_size = mat1.total_size;
        output.buffer = new Type[output.total_size];
        output.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        if (!GlobalGPUManager.blendInit) {
            GlobalGPUManager.initBlend();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(mat1.shape[0] * mat1.shape[1],1, 1);
        
        
        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:mat1.buffer length:mat1.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:mat2.buffer length:mat2.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        
        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        
        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
        [commandEncoder setComputePipelineState:GlobalGPUManager.blendCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return output;
    }
    
    void invertImg(bool evenAlpha) {

        
        if (!GlobalGPUManager.invertInitImg) {
            GlobalGPUManager.initInvert();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
        
        
        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];

        
        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
        [commandEncoder setBytes:&evenAlpha length:sizeof(bool) atIndex:1];
        [commandEncoder setComputePipelineState:GlobalGPUManager.invertImgCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    void chromaKeyImg(simd_packed_char3 key) {

        
        if (!GlobalGPUManager.chromaKeyInit) {
            GlobalGPUManager.initchromaKey();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
        
        
        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:this->buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];

        
        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
        [commandEncoder setBytes:&key length:sizeof(simd_packed_char3) atIndex:1];
        [commandEncoder setComputePipelineState:GlobalGPUManager.chromaKeyCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    MatrixH<dims, Type> addImg(MatrixH<dims, Type> &other, bool evenAlpha) {
        MatrixH<dims, Type> result;
        result.buffer = new Type[total_size];
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        
        if (!GlobalGPUManager.AddImgInit) {
            GlobalGPUManager.initAddImg();
        }
        
        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1], 1, 1);
        
        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
        [commandEncoder setComputePipelineState:GlobalGPUManager.AddImgCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
    
    template <typename... Args> requires (std::convertible_to<Args, size_m> && ...)
    auto reshape(Args... newShape) && {
        constexpr size_t newDims = sizeof...(newShape);
        MatrixH<newDims, Type> output;
        memcpy(output.shape, (size_m[]){static_cast<size_m>(newShape)...}, sizeof(size_m) * newDims);
        output.total_size = output.accumul(0, newDims);

        if (output.total_size != total_size) {
            std::cerr << "MatrixH: Reshape total size mismatch. Original size: " << total_size << ", New size: " << output.total_size << "\n";
            throw;
        }
        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        output.calcStrides();
        return output;
    }
    
    template<int i>
    MatrixH<dims - i, Type> flatten() && {
        if (flags & NON_CONTIGUOUS_FLAG) {
            for (int k = 0; k < i; ++k) {
                        if (strides[k] != shape[k+1] * strides[k+1] && shape[k+1] != 1) {
                            throw std::invalid_argument(
                                "Cannot zero-copy flatten: Axes in the requested range are not contiguous with respect to each other."
                            );
                        }
            }
        }
        MatrixH<dims - i, Type> output;
        memcpy(output.shape+1, shape+1 + i, (dims - i) * sizeof(size_t));
        output.shape[0] = accumul(0, i+1);

        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        output.total_size = total_size;
        
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        
        if (output.flags & NON_CONTIGUOUS_FLAG) {
            size_m flat_stride = strides[i];
            for (int k = i; k >= 0; --k) {
                if (shape[k] > 1) {
                    flat_stride = strides[k];
                    break;
                }
            }
            output.strides[0] = flat_stride;
            if (dims - i - 1 > 0) {
                memcpy(output.strides + 1, strides + i + 1, (dims - i - 1) * sizeof(size_m));
            }
        } else {
            output.calcStrides();
        }
        return output;
    }

    template <int i>
    MatrixH<dims+i, Type> unsqueeze() & {
        MatrixH<dims+i, Type> output;
        cloneData(output);
        std::fill(output.shape, output.shape+i, 1);
        std::fill(output.strides, output.strides+i, strides[0]);
        memcpy(output.shape+i, shape, dims * sizeof(size_m));
        output.flags = flags;
        memcpy(output.strides+i, strides, dims * sizeof(size_m));
        if (metalBuffer) {
            output.buildMetalBuffer();
        }
        return output;
    }
    
    template <int i>
    MatrixH<dims+i, Type> unsqueeze() && {
        MatrixH<dims+i, Type> output;
        std::fill(output.shape, output.shape+i, 1);
        std::fill(output.strides, output.strides+i, strides[0]);
        memcpy(output.shape+i, shape, dims * sizeof(size_m));
        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        output.total_size = total_size;
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        
        return output;
    }
    
    MatrixH<dims+1, Type> unsqueeze(int axis) && {
        if (axis < 0) {
            axis = dims+1 + axis;
        }
        #ifdef SAFE_MODE
        if (axis < 0 || axis >= dims) {
            throw std::invalid_argument("Squeeze axis out of bounds");
        }

        #endif SAFE_MODE

        MatrixH<dims+1, Type> output;
        memcpy(output.shape, shape, axis * sizeof(size_m));
        memcpy(output.shape + axis + 1, shape + axis, (dims-axis) * sizeof(size_m));
        output.shape[axis] = 1;
        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        output.total_size = total_size;
        memcpy(output.strides, strides, axis * sizeof(size_m));
        memcpy(output.strides + axis + 1, strides + axis, (dims-axis) * sizeof(size_m));
        output.strides[axis] = (axis == dims) ? 1 : strides[axis];
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        
        return output;
    }
    
    MatrixH<dims-1, Type> squeeze(int axis) && {
        if (axis < 0) {
            axis = dims + axis;
        }
        
        #ifdef SAFE_MODE
        if (axis < 0 || axis >= dims) {
            throw std::invalid_argument("Squeeze axis out of bounds");
        }
        if (shape[axis] != 1) {
            throw std::invalid_argument("Cannot squeeze an axis that does not have a size of 1");
        }

        #endif SAFE_MODE

        MatrixH<dims-1, Type> output;
        memcpy(output.shape, shape, axis * sizeof(size_m));
        memcpy(output.shape + axis, shape + axis + 1, (dims-axis-1) * sizeof(size_m));
        memcpy(output.strides, strides, axis * sizeof(size_m));
        memcpy(output.strides + axis, strides + axis + 1, (dims-axis-1) * sizeof(size_m));
        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        output.total_size = total_size;
        
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        
        return output;
    }
    
    template <int i>
    MatrixH<dims-i, Type> squeeze() & {
        MatrixH<dims-i, Type> output;
#ifdef SAFE_MODE
        for (int j = 0; j < i; j++) {
            if (shape[j] != 1) {
                std::cerr << "MatrixH: squeeze func has dim: " << j+1 << "with shape: " << shape[j] << " != 1 \n";
                throw;
            }
        }
#endif
        cloneData(output);
        memcpy(output.shape, shape+i, (dims-i) * sizeof(size_m));
        memcpy(output.strides, strides+i, (dims-i) * sizeof(size_m));
        output.flags = flags;
        if (metalBuffer) {
            output.buildMetalBuffer();
        }
        return output;
    }
    
    template <int i>
    MatrixH<dims-i, Type> squeeze() && {
        MatrixH<dims-i, Type> output;
#ifdef SAFE_MODE
        for (int j = 0; j < i; j++) {
            if (shape[j] != 1) {
                std::cerr << "MatrixH: squeeze func has dim: " << j+1 << "with shape: " << shape[j] << " != 1 \n";
                throw;
            }
        }
#endif
        memcpy(output.shape, shape+i, (dims-i) * sizeof(size_m));
        memcpy(output.strides, strides+i, (dims-i) * sizeof(size_m));
        output.buffer = buffer;
        output.metalBuffer = metalBuffer;
        output.flags = flags;
        buffer = nullptr;
        metalBuffer = nil;
        total_size = 0;
        return output;
    }
    
    void AddSameCPU(MatrixH<dims, Type>& result, const MatrixH<dims, Type>& other) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = other.buffer[i] + buffer[i];
        }

    }
    
    void MulSameCPU(MatrixH<dims, Type>& result, const MatrixH<dims, Type>& other) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = other.buffer[i] * buffer[i];
        }

    }
    
    void SubSameCPU(MatrixH<dims, Type>& result, const MatrixH<dims, Type>& other) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = buffer[i] - other.buffer[i];
        }

    }
    
    void MulConstCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = constant * buffer[i];
        }
        
    }
    
    void DivSameCPU(MatrixH<dims, Type>& result, const MatrixH<dims, Type>& other) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] =  buffer[i] / other.buffer[i];
        }

    }
    
    void AddConstCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = constant * buffer[i];
        }

    }
    
    void SubConstCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = buffer[i] - constant;
        }
 
    }
    
    void SubConstRevCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] = constant - buffer[i];
        }
 
    }
    
    void DivConstCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] =  buffer[i] / constant;
        }

    }
    
    void DivConstRevCPU(MatrixH<dims, Type>& result, Type constant) const {
        if (!result.buffer) {
            result.buffer = new Type[total_size];
        } else if (result.total_size != total_size) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
        }
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
        for (size_t i = 0; i < total_size; i++) {
            result.buffer[i] =  constant / buffer[i];
        }

    }
    
    MatrixH<dims, Type> MulConst(Type constant) {
        MatrixH<dims, Type> result;
        result.buffer = new Type[total_size];
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.gradFunc = [constant](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            return p1 * constant;
//        };
        
        if (!GlobalGPUManager.MulAllInit) {
            GlobalGPUManager.initMulAll();
        }
        
        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:&constant length:sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
        }];
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
        
        int type = 0;
        size_t stride = 1;
        
        if constexpr (std::is_integral<Type>::value) {
            type = 0;
        } else if constexpr (std::is_floating_point<Type>::value) {
            type = 1;
        } else {
            type = 2;
        }
        
        [commandEncoder setBytes:&type length:sizeof(int) atIndex:3];
        [commandEncoder setBytes:&stride length:sizeof(size_m) atIndex:4];
        [commandEncoder setComputePipelineState:GlobalGPUManager.MulAllCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
    
    template <int dims2>
    MatrixH<dims, Type> MulMat(const MatrixH<dims2, Type>& other) {
        size_t stride = 1;
        size_t strideI = 1;
        for (int i = 0; i < dims; i++) {
            for (int j = 0; j < dims2; j++) {
                if (shape[i] == other.shape[j]) {
                    stride = shape[i];
                    strideI = accumul(i+1, dims);
                }
            }
        }
        MatrixH<dims, Type> result;
        result.buffer = new Type[total_size];
        result.total_size = total_size;
        memcpy(result.shape, shape, sizeof(size_m) * dims);
        memcpy(result.strides, strides, sizeof(size_m) * dims);
        result.buildMetalBuffer();
        
        
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(other));
//        result.gradFunc = [](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return (p1) * *(selfs.parentNodes[1])+
//                       p2 * *(selfs.parentNodes[0]);
//        };
        
        if (!GlobalGPUManager.MulAllInit) {
            GlobalGPUManager.initMulAll();
        }
        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:1];
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:2];
        
        int type = 0;
        
        
        if constexpr (std::is_integral<Type>::value) {
            if (std::is_unsigned<Type>::value) {
                type = 2;
            } else {
                type = 0;
            }
            
        } else if constexpr (std::is_floating_point<Type>::value) {
            type = 1;
        } else {
            type = 3;
        }
        
        [commandEncoder setBytes:&type length:sizeof(int) atIndex:3];
        [commandEncoder setBytes:&stride length:sizeof(size_t) atIndex:4];
        [commandEncoder setBytes:&strideI length:sizeof(size_t) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.MulAllCompute];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
    
    static void copyGPUinplace( MatrixH<dims, Type>& outMat, const MatrixH<dims, Type>& inMat, int offset, bool commit = true) {
#ifdef SAFE_MODE
        if (inMat.total_size > outMat.total_size) {
            std::cerr << "MatrixH: CopyInplace operation requires both mats to be of same size." << "\n";
            throw;
        }
#endif
        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
//        Blit fast path for the GPU. A compute shader (even a 1D one) requires the GPU's ALU execution units to run the copy loop. A Blit command skips the ALUs entirely and uses the GPU's direct memory access (DMA) engines to blast the bytes across VRAM. It is drastically faster.
        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) && !(outMat.flags & NON_CONTIGUOUS_FLAG)) {
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer:inMat.metalBuffer
                           sourceOffset:offset * sizeof(Type)
                               toBuffer:outMat.metalBuffer
                      destinationOffset:offset * sizeof(Type)
                                   size:inMat.total_size * sizeof(Type)];
            [blitEncoder endEncoding];
            if (commit) {
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
            return;
        }
        uint8_t typeCode = get_dtype_code<Type>();
        
        auto res = collapse_dims(inMat.shape, outMat.strides, inMat.strides, dims, INT32_MAX);
        uint32_t cdims = res.out_dims;
        
        
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(inMat.total_size, 1, 1);
        [commandEncoder setBuffer:outMat.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:inMat.metalBuffer offset:0 atIndex:1];
        [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:res.stridesB  length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1) {
            if (!GlobalGPUManager.CopyInplace[typeCode][0]) {
                GlobalGPUManager.initCopyInplace(typeCode, 0);
            }
            [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][0]];
        } else if (cdims == 2) {
            if (!GlobalGPUManager.CopyInplace[typeCode][1]) {
                GlobalGPUManager.initCopyInplace(typeCode, 1);
            }
            _dispatchExecutionSize =  MTLSizeMake(res.shape[1], res.shape[0], 1);
            [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][1]];
        } else if (cdims == 3) {
            if (!GlobalGPUManager.CopyInplace[typeCode][2]) {
                GlobalGPUManager.initCopyInplace(typeCode, 2);
            }
            _dispatchExecutionSize =  MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
            [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][2]];
        
        } else {
            if (!GlobalGPUManager.CopyInplace[typeCode][3]) {
                GlobalGPUManager.initCopyInplace(typeCode, 3);
            }
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) {acc *= res.shape[i]; }
            _dispatchExecutionSize =  MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
            [commandEncoder setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][3]];
        }
        
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        if (commit) {
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.gCommandBuffer = nil;
            GlobalGPUManager.gCommandEncoder=nil;
        }
    }
    
    static void copyCPUinplace( MatrixH<dims, Type>& outMat, const MatrixH<dims, Type>& inMat, int offset, bool commit = true) {
#ifdef SAFE_MODE
        if (inMat.total_size > outMat.total_size) {
            std::cerr << "MatrixH: CopyInplace operation requires both mats to be of same size." << "\n";
            throw;
        }
#endif
        
        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) && !(outMat.flags & NON_CONTIGUOUS_FLAG)) {
            memcpy(outMat.buffer, inMat.buffer, inMat.total_size * sizeof(Type));
            return;
        }
                
        auto res = collapse_dims(inMat.shape, outMat.strides, inMat.strides, dims, INT32_MAX);
        auto cdims = res.out_dims;
        // us stands for unsafe and fast subscripting so it doesnt suppor negative indices and is super fast.
        if (cdims == 1) {
            for (uint32_t i = 0; i < inMat.total_size; i++) {
                outMat.buffer[i * res.stridesA[0]] = inMat.buffer[i * res.stridesB[0]];
            }
        } else if (cdims == 2) {
            if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] == 1) {
                for (uint32_t i = 0; i < res.shape[0]; i++) { memcpy(outMat.buffer + res.stridesA[0] * i, outMat.buffer + res.stridesB[0] * i, res.shape[1] * sizeof(Type)); }
                return;
            }
            for (uint32_t i = 0; i < res.shape[0]; i++) {
                for (uint32_t j = 0; j < res.shape[1]; j++) {
                    outMat.buffer[i * res.stridesA[0] + j * res.stridesA[1]] = inMat.buffer[i * res.stridesB[0] + j * res.stridesB[1]];                }
            }
        } else if (cdims == 3) {
            if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] == 1) {
                for (uint32_t i = 0; i < res.shape[0]; i++) {
                    for (uint32_t j = 0; j < res.shape[1]; j++) { memcpy(outMat.buffer + res.stridesA[0] * i + res.stridesA[1] * j, outMat.buffer + res.stridesB[0] * i + res.stridesB[1] * j, res.shape[2] * sizeof(Type)); }
                }
                return;
            }
            for (uint32_t i = 0; i < res.shape[0]; i++) {
                for (uint32_t j = 0; j < res.shape[1]; j++) {
                    for (uint32_t k = 0; k < res.shape[2]; k++) {
                        outMat.buffer[i * res.stridesA[0] + j * res.stridesA[1] + k * res.stridesA[2]] = inMat.buffer[i * res.stridesB[0] + j * res.stridesB[1] + k * res.stridesB[2]];
                    }
                }
            }
        
        } else {
            uint32_t outer_iterations = 1;
            for (uint32_t o = 0; o <= cdims - 4; o++) {
                outer_iterations *= res.shape[o];
            }
            for (uint32_t o = 0; o < outer_iterations; o++) {
                uint32_t inMatIndex = 0;
                uint32_t outMatIndex = 0;
                uint32_t rem = o;
                for (int i = dims-4; i >=0; i--) {
                    inMatIndex  += (rem % res.shape[i]) * res.stridesB[i];
                    outMatIndex += (rem % res.shape[i]) * res.stridesA[i];
                    rem /= res.shape[i];
                }
                if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] == 1) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) { memcpy(outMat.buffer + res.stridesA[cdims-3] * i + res.stridesA[cdims-2] * j, outMat.buffer + res.stridesB[cdims-3] * i + res.stridesB[cdims-2] * j, res.shape[dims-1] * sizeof(Type)); }
                    }
                    break;
                }
                for (uint32_t i = 0; i < res.shape[cdims-3]; i++) {
                    for (uint32_t j = 0; j < res.shape[cdims-2]; j++) {
                        for (uint32_t k = 0; k < res.shape[cdims-1]; k++) {
                            outMat.buffer[outMatIndex + i * res.stridesA[cdims-3] + j * res.stridesA[cdims-2] + k * res.stridesA[cdims-1]] = inMat.buffer[inMatIndex + i * res.stridesB[cdims-3] + j * res.strides[cdims-2] + k * inMat.strides[cdims-1]];
                        }
                    }
                }
            }
        }
    }
    
    static std::pair<MatrixH<2 * dims, Type>, MatrixH<2 * dims, Type>> meshGrid(const MatrixH<dims, Type> X, const MatrixH<dims, Type> Y) {
        
        MatrixH<2 * dims, Type> gridX;
        memcpy(gridX.shape, X.shape, dims * sizeof(size_m));
        memcpy(gridX.shape + dims, Y.shape, dims * sizeof(size_m));
        gridX.initWithCurrentShape();
        
        MatrixH<2 * dims, Type> gridY;
        memcpy(gridY.shape, X.shape, dims * sizeof(size_m));
        memcpy(gridY.shape + dims, Y.shape, dims * sizeof(size_m));
        gridY.initWithCurrentShape();

        MatrixH<2 * dims, Type> X_buffer_view;
        X_buffer_view.buffer = X.buffer;
        memcpy(X_buffer_view.shape, gridX.shape, 2 * dims * sizeof(size_m));
        memcpy(X_buffer_view.strides, X.strides, dims * sizeof(size_m));
        memset(X_buffer_view.strides + dims, 0, dims * sizeof(size_m));
        
        X_buffer_view.total_size = X_buffer_view.accumul(0, 2 * dims);
        X_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
        X_buffer_view.flags |= NON_OWNERSHIP_FLAG;
        X_buffer_view.metalBuffer = X.metalBuffer;

        
        
        MatrixH<2 * dims, Type> Y_buffer_view;
        Y_buffer_view.buffer = Y.buffer;
        
        memcpy(Y_buffer_view.shape, gridY.shape, 2 * dims * sizeof(size_m));
        memcpy(Y_buffer_view.strides + dims, Y.strides, dims * sizeof(size_m));
        memset(Y_buffer_view.strides, 0, dims * sizeof(size_m));
        
        Y_buffer_view.total_size = Y_buffer_view.accumul(0, 2 * dims);
        Y_buffer_view.metalBuffer = Y.metalBuffer;
        Y_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
        Y_buffer_view.flags |= NON_OWNERSHIP_FLAG;

        MatrixH<2 * dims, Type>::copyGPUinplace(gridY, Y_buffer_view, 0, false);
        MatrixH<2 * dims, Type>::copyGPUinplace(gridX, X_buffer_view, 0, true);
        
        
        
        return std::make_pair(std::move(gridX), std::move(gridY));
    }
    
    static std::tuple<MatrixH<3 * dims, Type>, MatrixH<3 * dims, Type>, MatrixH<3 * dims, Type>> meshGrid(const MatrixH<dims, Type> X, const MatrixH<dims, Type> Y,  const MatrixH<dims, Type> Z) {
        
        MatrixH<3 * dims, Type> gridX;
        memcpy(gridX.shape, X.shape, dims * sizeof(size_m));
        memcpy(gridX.shape + dims, Y.shape, dims * sizeof(size_m));
        memcpy(gridX.shape + 2 * dims, Z.shape, dims * sizeof(size_m));
        gridX.initWithCurrentShape();
        
        MatrixH<3 * dims, Type> gridY;
        memcpy(gridY.shape, gridX.shape, 3 * dims * sizeof(size_m));
        gridY.initWithCurrentShape();
        
        MatrixH<3 * dims, Type> gridZ;
        memcpy(gridZ.shape, gridX.shape, 3 * dims * sizeof(size_m));
        gridZ.initWithCurrentShape();

        MatrixH<3 * dims, Type> X_buffer_view;
        X_buffer_view.buffer = X.buffer;
        memcpy(X_buffer_view.shape, gridX.shape, 3 * dims * sizeof(size_m));
        memcpy(X_buffer_view.strides, X.strides, dims * sizeof(size_m));
        memset(X_buffer_view.strides + dims, 0, 2 * dims * sizeof(size_m)); // sets Y and Z strides to 0
        
        X_buffer_view.total_size = X_buffer_view.accumul(0, 3 * dims);
        X_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
        X_buffer_view.flags |= NON_OWNERSHIP_FLAG;
        X_buffer_view.metalBuffer = X.metalBuffer;

        MatrixH<3 * dims, Type> Y_buffer_view;
        Y_buffer_view.buffer = Y.buffer;
        
        memcpy(Y_buffer_view.shape, gridY.shape, 3 * dims * sizeof(size_m));
        memcpy(Y_buffer_view.strides + dims, Y.strides, dims * sizeof(size_m));
        memset(Y_buffer_view.strides, 0, dims * sizeof(size_m)); // sets X strides to 0
        memset(Y_buffer_view.strides + 2 * dims, 0, dims * sizeof(size_m)); // sets Z strides to 0
        
        Y_buffer_view.total_size = Y_buffer_view.accumul(0, 3 * dims);
        Y_buffer_view.metalBuffer = Y.metalBuffer;
        Y_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
        Y_buffer_view.flags |= NON_OWNERSHIP_FLAG;
        
        MatrixH<3 * dims, Type> Z_buffer_view;
        Z_buffer_view.buffer = Z.buffer;
        
        memcpy(Z_buffer_view.shape, gridZ.shape, 3 * dims * sizeof(size_m));
        memcpy(Z_buffer_view.strides + 2 * dims, Z.strides, dims * sizeof(size_m));
        memset(Z_buffer_view.strides, 0, dims * sizeof(size_m)); // sets X and Y strides to 0
        
        Z_buffer_view.total_size = Z_buffer_view.accumul(0, 3 * dims);
        Z_buffer_view.metalBuffer = Z.metalBuffer;
        Z_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
        Z_buffer_view.flags |= NON_OWNERSHIP_FLAG;

        MatrixH<3 * dims, Type>::copyGPUinplace(gridY, Y_buffer_view, 0, false);
        MatrixH<3 * dims, Type>::copyGPUinplace(gridX, X_buffer_view, 0, false);
        MatrixH<3 * dims, Type>::copyGPUinplace(gridZ, Z_buffer_view, 0, true);
        
        
        return std::make_tuple(std::move(gridX), std::move(gridY), std::move(gridZ));
    }
    
    static MatrixH<dims, Type> concat(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, int axis) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        MatrixH<dims, Type> output;
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
            } else {
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        
        size_t strideMat2 = Mat2.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
        }
        
        return output;
    }
    
    static MatrixH<dims, Type> concatGPU(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, int axis, size_m batch_size = 1) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        uint8_t typeCode = get_dtype_code<Type>();
        if (!GlobalGPUManager.Concat_2M[typeCode]) {
            GlobalGPUManager.initConcat_2M_GPU(typeCode);
        }
        
        MatrixH<dims, Type> output;
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
            } else {
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        size_t noOfOpp = output.accumul(0, axis);
        size_m stride = (size_m)output.accumul(axis, dims);
        
        size_m strideMat1 =(size_m) Mat1.accumul(axis, dims);
        
        size_m strideMat2 = (size_m)Mat2.accumul(axis, dims);
        

        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(noOfOpp, 1, 1);
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, Mat1, 1);
        setBufferOrBytes(commandEncoder, Mat2, 2);
        [commandEncoder setBytes:&stride length:sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&strideMat1 length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:&strideMat2 length:sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&batch_size length:sizeof(size_m) atIndex:6];
        
        [commandEncoder setComputePipelineState:GlobalGPUManager.Concat_2M_ComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                   threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        
        return output;
    }
    
    static MatrixH<dims+1, Type> concatGPU_ID(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, const MatrixH<dims, Type>& Mat3, int axis) {
#ifdef SAFE_MODE
        if (Mat1.total_size != Mat2.total_size || Mat2.total_size != Mat3.total_size) {
            std::cerr << "MatrixH: Concat operation requires same size mats: " << "\n";
            throw;
        }
#endif
        uint8_t typeCode = get_dtype_code<Type>();
        if (!GlobalGPUManager.Concat_2M[typeCode]) {
            GlobalGPUManager.initConcat_2M_GPU(typeCode);
        }
        
        MatrixH<dims+1, Type> output;
        memcpy(output.shape, Mat1.shape, axis * sizeof(size_m));
        memcpy(output.shape + (axis + 1), Mat1.shape + (axis), (dims - axis) * sizeof(size_m));
        output.shape[axis] = 3;
        output.total_size = Mat1.total_size * 3;
        output.calcStrides();
        // strides are the property of buffer and should be calculated with its total size in mind;
        // Fix for garbage point clouds: newBufferWithBytesNoCopy requires page-aligned memory.
        size_t sizeBytes = Mat1.total_size * 3 * sizeof(Type);
        size_t pageSize = sysconf(_SC_PAGESIZE);
        // Round up to full page size for safety (though just alignment is strictly required for address)
        size_t alignedSize = (sizeBytes + pageSize - 1) & ~(pageSize - 1);
        
        void* raw_mem = nullptr;
        if (posix_memalign(&raw_mem, pageSize, alignedSize) != 0) {
            std::cerr << "MatrixH: Failed to allocate aligned memory" << "\n";
            throw std::bad_alloc();
        }
        
        output.buffer = (Type*)raw_mem;
        output.total_size = Mat1.total_size * 3;
        output.flags |= NON_OWNERSHIP_FLAG; // Prevent ~MatrixH from deleting buffer
        
        output.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:raw_mem
                                                                             length:alignedSize
                                                                            options:MTLResourceStorageModeShared
                                                                        deallocator:^(void * _Nonnull pointer, NSUInteger length) {
            free(pointer);
        }];
        
        output.flags |= NON_CONTIGUOUS_FLAG;
        output.shape[axis] = 1;
        output.total_size = Mat1.total_size;
        
        MatrixH<dims+1, Type> View;
        memcpy(View.shape, output.shape, (dims+1) * sizeof(size_m));
        View.shape[axis] = 1;
        View.calcStrides();
        View.total_size = Mat1.total_size;
        View.flags |= NON_OWNERSHIP_FLAG;
        
        size_m offset = output.strides[axis];
        
        View.buffer = Mat1.buffer;
        View.metalBuffer = Mat1.metalBuffer;
        MatrixH<dims+1, Type>::copyGPUinplace(output, View, 0, false);
        
        View.buffer = Mat2.buffer;
        View.metalBuffer = Mat2.metalBuffer;
        MatrixH<dims+1, Type>::copyGPUinplace(output, View, offset, false);
        
        View.buffer = Mat3.buffer;
        View.metalBuffer = Mat3.metalBuffer;
        MatrixH<dims+1, Type>::copyGPUinplace(output, View, 2*offset, true);
        
        output.shape[axis] = 3;
        output.total_size = Mat1.total_size * 3;
        output.flags &= ~NON_CONTIGUOUS_FLAG;
        return output;
    }
    
    
    
    static MatrixH<dims, Type> concat(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, const MatrixH<dims, Type>& Mat3, int axis) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG || Mat3.flags & NON_CONTIGUOUS_FLAG ) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
            
        MatrixH<dims, Type> output;
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i] + Mat3.shape[i];
            } else {
#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requiresn shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i] << "\n";
                    throw;
                }
#endif
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        size_t strideMat2 = Mat2.accumul(axis, dims);
        size_t strideMat3 = Mat3.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + i * stride, Mat3.buffer + i * strideMat3, strideMat3 * sizeof(Type));
        }
        
        return output;
    }
    
    static MatrixH<dims, Type> concat(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, const MatrixH<dims, Type>& Mat3, const MatrixH<dims, Type>& Mat4, int axis) {
        MatrixH<dims, Type> output;
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG || Mat3.flags & NON_CONTIGUOUS_FLAG ) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i] + Mat3.shape[i] + Mat4.shape[i];
            } else {
#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i] || Mat1.shape[i] != Mat4.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i]<<" != " <<  Mat4.shape[i] << "\n";
                    throw;
                }
    
#endif
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        size_t strideMat2 = Mat2.accumul(axis, dims);
        size_t strideMat3 = Mat3.accumul(axis, dims);
        size_t strideMat4 = Mat4.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + i * stride, Mat3.buffer + i * strideMat3, strideMat3 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + strideMat3 + i * stride, Mat4.buffer + i * strideMat4, strideMat4 * sizeof(Type));
        }
        
        return output;
    }
    
    static void concat_(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, MatrixH<dims, Type>& output, int axis) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG ) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
            
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
            } else {
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        if (output.total_size != output.accumul(0, dims)) {
            delete [] output.buffer;
            output.total_size = output.accumul(0, dims);
            output.buffer = new Type[output.total_size];
            output.buildMetalBuffer();
        }

        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        
        size_t strideMat2 = Mat2.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
        }
    }
    
    static void concatID(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, MatrixH<dims+1, Type>& output) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG ){
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        int axis = dims+1;
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
            } else {
#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i] << "\n";
                    throw;
                }
#endif
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        if (output.total_size != output.accumul(0, dims)) {
            delete [] output.buffer;
            output.total_size = output.accumul(0, dims);
            output.buffer = new Type[output.total_size];
            output.buildMetalBuffer();
        }

        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        
        size_t strideMat2 = Mat2.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
        }
        
        return output;
    }
    
    static MatrixH<dims+1, Type> concatID(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG ) {
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        MatrixH<dims+1, Type> output;
        int axis = dims+1;
        for (int i = 0; i < dims; i++) {
            if (i == axis) {
                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
            } else {
#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i] << "\n";
                    throw;
                }
#endif
                output.shape[i] = Mat1.shape[i];
            }
        }
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();

        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        
        size_t strideMat2 = Mat2.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
        }
        
        return output;
    }
    
    static MatrixH<dims+1, Type> concatID(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, const MatrixH<dims, Type>& Mat3, const MatrixH<dims, Type>& Mat4, int axis) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG || Mat3.flags & NON_CONTIGUOUS_FLAG || Mat4.flags & NON_CONTIGUOUS_FLAG ){
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        MatrixH<dims+1, Type> output;
//        int axis = dims+1;
        for (int i = 0; i < axis; i++) {

#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i] || Mat1.shape[i] != Mat4.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i]<<" != " <<  Mat4.shape[i] << "\n";
                    throw;
                }
#endif
            output.shape[i] = Mat1.shape[i];
        }
        
        output.shape[axis] = 4;
        for (int i = axis+1; i < dims+1; i++) {

#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i] || Mat1.shape[i] != Mat4.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i]<<" != " <<  Mat4.shape[i] << "\n";
                    throw;
                }
#endif
                output.shape[i] = Mat1.shape[i-1];
            
        }
        
        output.calcStrides();
        output.total_size = output.accumul(0, dims+1);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();

        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims+1);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        size_t strideMat2 = Mat2.accumul(axis, dims);
        size_t strideMat3 = Mat3.accumul(axis, dims);
        size_t strideMat4 = Mat4.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {

            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + i * stride, Mat3.buffer + i * strideMat3, strideMat3 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + strideMat3 + i * stride, Mat4.buffer + i * strideMat4, strideMat4 * sizeof(Type));
            
        }
        
        return output;
    }
    
    static MatrixH<dims+1, Type> concatID(const MatrixH<dims, Type>& Mat1, const MatrixH<dims, Type>& Mat2, const MatrixH<dims, Type>& Mat3, int axis) {
#ifdef SAFE_MODE
        if (Mat1.flags & NON_CONTIGUOUS_FLAG || Mat2.flags & NON_CONTIGUOUS_FLAG || Mat3.flags & NON_CONTIGUOUS_FLAG  ){
            std::cerr << "MatrixH: Concat operation requires contigious buffer mats: " << "\n";
            throw;
        }
#endif
        MatrixH<dims+1, Type> output;
//        int axis = dims+1;
        for (int i = 0; i < axis; i++) {

#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i]) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i] << "\n";
                    throw;
                }
#endif
            output.shape[i] = Mat1.shape[i];
        }
        
        output.shape[axis] = 3;
        for (int i = axis+1; i < dims+1; i++) {

#ifdef SAFE_MODE
                if (Mat1.shape[i] != Mat2.shape[i] || Mat1.shape[i] != Mat3.shape[i] ) {
                    std::cerr << "MatrixH: Concat operation requires shapes be equal for all dims except the axis and at dim: " << i << Mat1.shape[i]<<" != " << Mat2.shape[i]<<" != " <<  Mat3.shape[i] << "\n";
                    throw;
                }
#endif
                output.shape[i] = Mat1.shape[i-1];
            
        }
        
        output.calcStrides();
        output.total_size = output.accumul(0, dims+1);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();

        size_t noOfOpp = output.accumul(0, axis);
        size_t stride = output.accumul(axis, dims+1);
        
        size_t strideMat1 = Mat1.accumul(axis, dims);
        size_t strideMat2 = Mat2.accumul(axis, dims);
        size_t strideMat3 = Mat3.accumul(axis, dims);
        
        for (int i = 0; i < noOfOpp; i++) {

            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
            memcpy(output.buffer + strideMat1 + strideMat2 + i * stride, Mat3.buffer + i * strideMat3, strideMat3 * sizeof(Type));
            
        }
        
        return output;
    }
    
    static MatrixH<dims, Type> zeros(std::initializer_list<size_m> shapeI) {
        if (shapeI.size() != dims) {std::cerr << "MatrixH: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
        MatrixH<dims, Type> output;
        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        memset(output.buffer, 0, output.total_size * sizeof(Type));
        return output;
    }
    
    static MatrixH<dims, Type> ones(std::initializer_list<size_m> shapeI) {
        if (shapeI.size() != dims) {std::cerr << "MatrixH: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
        MatrixH<dims, Type> output;
        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        std::fill(output.buffer, output.buffer + output.total_size, 1);
        return output;
    }
    
    MatrixH<dims, Type> ones() const {
        MatrixH<dims, Type> output;
        memcpy(output.shape, shape, dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        std::fill(output.buffer, output.buffer + output.total_size, 1);
        return output;
    }
    
    MatrixH<dims, Type> zeros() const {
        MatrixH<dims, Type> output;
        memcpy(output.shape, shape, dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        memset(output.buffer, 0, output.total_size * sizeof(Type));
        return output;
    }
    
    static MatrixH<dims, Type> withShape(std::initializer_list<size_m> shapeI) {
        if (shapeI.size() != dims) {std::cerr << "MatrixH: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
        MatrixH<dims, Type> output;
        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, dims);
        output.buffer = new Type[output.total_size];
        output.buildMetalBuffer();
        return output;
    }
    
    void initWithCurrentShape() {
        if (buffer && !(flags & NON_OWNERSHIP_FLAG)) {
            delete [] buffer;
        }
        calcStrides();
        total_size = accumul(0, dims);
        buffer = new Type[total_size];
        buildMetalBuffer();
    }
    
//    template <typename = std::enable_if_t<(dims == 2)>>
    static MatrixH<2, Type> eye(uint m, uint n, int k) {
        MatrixH<2, Type> output = MatrixH<2, Type>::zeros({m, n});
        uint iteration = MIN(m, n-abs(k));
        if (0 <= k) {
            for (int i = 0; i < iteration; i++) {
                // [i, j+k]
                output.buffer[i * output.shape[1] + i + k] = 1;
            }

        } else {
            for (int i = 0; i < iteration; i++) {
                // [i, i-abs(k)] => same as shifting it down => [i+abs(k), i]
                // Since k is -ve => [i-k, i]
                // Moving the Diagnol Left is Same as moving it above as y = (x + k) ==> (y - k) = x
                output.buffer[(i - k) * output.shape[1] + i] = 1;
            }
        }
        return output;
    }
    
    static MatrixH<2, Type> eye(uint m) {
        return eye(m, m, 0);
    }
    
    void drawRect(simd_int4 rect, MatrixH<dims-2, Type> element) {
        int X = rect[0];
        int Y = rect[1];
        int width = rect[2];
        int height = rect[3];
        for (int i = 0; i < dims - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        
        
        if (X + width > shape[1] || Y + height > shape[0]) {
            std::cerr << "Error Dimensions excedeError Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        
        Type* rowBuffer = new Type[width * element.total_size];
        
        for (size_t i = 0; i < width; i++) {
            memcpy(buffer + Y * widthsize + (X + i) * elementSize, element.buffer, element.total_size * sizeof(Type));
        }
        
        for (int j = Y+1; j < Y + height; j++) {
            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y * widthsize + X * elementSize, element.total_size * width * sizeof(Type));
        }
    }
    
    simd_float2 NormaliseShapeToScreen(simd_int2 deviceCoord) {
        simd_float2 size = simd_make_float2(shape[1], shape[0]);
        
        return simd_make_float2((float)deviceCoord.x, (float)deviceCoord.y) / size;
    }
    
    void drawRect(simd_int2 p1, simd_int2 p2, MatrixH<dims, Type> element) {
        for (int i = 0; i < shape.size() - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        
        auto xDiff = p1.x-p2.x;
        auto yDiff = p1.y-p2.y;
        
        int width = abs(xDiff);
        int height = abs(yDiff);
        
        int X;
        int Y;
        
        if (xDiff > 0) {
            X = p2.x;
        } else {
            X = p1.x;
        }
        
        if (yDiff > 0) {
            Y = p2.y;
        } else {
            Y = p1.y;
        }
        
        if (X + width > shape[1] || Y + height > shape[0]) {
            std::cerr << "Error Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        
        Type* rowBuffer = new Type[width * element.total_size];
        
        for (size_t i = 0; i < width; i++) {
            memcpy(buffer + Y * widthsize + (X + i) * elementSize, element.buffer, element.total_size * sizeof(Type));
        }
        
        for (int j = Y+1; j < Y + height; j++) {
            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y * widthsize + X * elementSize, element.total_size * width * sizeof(Type));
        }
    }
    void drawElipse(const simd_int4& rect, const MatrixH<dims-2, Type>& element) {
        int X = rect[0];
        int Y = rect[1];
        int width = rect[2];
        int height = rect[3];
        for (int i = 0; i < dims - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Elipse: Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        if (X + width > shape[1] || Y + height > shape[0]) {
            std::cerr << "Error Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        auto centre = simd_make_float2(X + (width / 2.0), Y + (height / 2.0));
        auto rad = simd_make_float2(width, height) / 2;
        
        for (int i = X; i < X + width; i ++) {
            for (int j = Y; j < Y + height; j ++) {
                auto coord = simd_make_float2(i, j);
                float S1 = simd_dot(((coord - centre) / rad), ((coord - centre) / rad)) - 1.0;
                
                if (S1 < 0.0) {
                    memcpy(buffer + j * widthsize + i * elementSize , element.buffer, element.total_size * sizeof(Type));
                }
            }
        }
    }
    
    simd_float4 toSimdFloat4() {
        return simd_make_float4(buffer[0], buffer[1], buffer[2], buffer[3]);
    }
    simd_float3 toSimdFloat3() {
        return simd_make_float3(buffer[0], buffer[1], buffer[2]);
    }
    simd_float3 toSimdFloat3(int i) {
#ifdef SAFE_MODE
        if (i >= shape[0]) {
            std::cerr << "MatrixH: Index " << i << " Out of Bounds for shape of dim 1: " << shape[0];
            throw;
        }
        if (3 != shape[1] || dims != 2) {
            std::cerr << "MatrixH: Cant Call to SimdFloat3 with one index on dims greater than 2 or inner shape of " << shape[1];
            throw;
        }
#endif
        return simd_make_float3(buffer[i * strides[0] + 0 * strides[1]], buffer[i * strides[0] + 1 * strides[1]], buffer[i * strides[0] + 2 * strides[1]]);
    }
    
    simd_float3 toSimdFloat3(int i, int j) {
#ifdef SAFE_MODE
        if (i >= shape[0]) {
            std::cerr << "MatrixH: Index " << i << " Out of Bounds for shape of dim 1: " << shape[0];
            throw;
        }
        if (j >= shape[1]) {
            std::cerr << "MatrixH: Index " << j << " Out of Bounds for shape of dim 1: " << shape[1];
            throw;
        }
        if (3 != shape[2] || dims != 3) {
            std::cerr << "MatrixH: Cant Call to SimdFloat3 with one index on dims greater than 3 or inner shape of " << shape[2];
            throw;
        }
#endif
        return simd_make_float3(buffer[i * strides[0] + j * strides[1] + 0 * strides[2]], buffer[i * strides[0] + j * strides[1] + 1 * strides[2]], buffer[i * strides[0] + j * strides[1] + 2 * strides[2]]);
    }
    simd_float4 toSimdFloat4(size_t i, size_t j) {
        size_t offset = i * (shape[1] * 4) + j * (4);
        return simd_make_float4(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3]);
    }
    CVPixelBufferRef createPixelBufferFromMat() const {
        
        size_t width = shape[1];
        size_t height = shape[0];
        size_t channels = shape[2];
        NSDictionary *pixelAttributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
        
        CVPixelBufferRef pixelBuffer;
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)pixelAttributes, &pixelBuffer);

        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        void *bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);

        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
         
         // Assuming channels is the number of bytes per pixel (should be 4 for BGRA)
         size_t copyBytesPerRow = width * channels;
         
         // Copy row by row to respect the pixel buffer's stride.
         for (size_t row = 0; row < height; row++) {
             memcpy((uint8_t *)bufferAddress + row * bytesPerRow,
                    buffer + row * copyBytesPerRow,
                    copyBytesPerRow);
         }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        return pixelBuffer;
    }
    
    static MatrixH<3, uint8_t> fromCVmat(cv::Mat image) {
        cv::Mat m = image;
        if (!m.isContinuous()) m = m.clone();

        MatrixH<3, uint8_t> result;
        
        result.shape[0] = m.rows;
        result.shape[1] = m.cols;
        result.shape[2] = m.channels();
        result.calcStrides();
        result.total_size = result.accumul(0, 3);
        result.buffer = new uint8_t[result.total_size];
        memcpy(result.buffer, m.data, result.total_size);
        result.buildMetalBuffer();
        return result;
    }
    
    static MatrixH<3, uint8_t> fromImage(std::string path_str = std::string("/Users/adityadude/Documents/TUSHU.HEIC")) {
        #if !TARGET_OS_IPHONE
        CFStringRef path = CFStringCreateWithCString(NULL, path_str.c_str(), kCFStringEncodingUTF8);
        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
        CGImageSourceRef source;
        CGImageRef cgImage;
        for (int i = 0; i < 3; i++) {
            source = CGImageSourceCreateWithURL(url, NULL);
            cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
            if (cgImage) {break;}
        }
        CFRelease(url);
        CFRelease(path);
        #endif

        #if TARGET_OS_IPHONE
        UIImage *image = [UIImage imageNamed:@"IMG_1278"];
        CGImageRef cgImage = image.CGImage;
        #endif

        if (!cgImage) {
            std::cerr << "Failed to create CGImage" << std::endl;
        }
        size_t Imgwidth = CGImageGetWidth(cgImage);
        size_t Imgheight = CGImageGetHeight(cgImage);
        std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight << "Loaded \n";
        size_t bytesPerRow = 4 * Imgwidth;
        void *data = malloc(bytesPerRow * Imgheight);
        CGContextRef context = CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
                                                     CGImageGetColorSpace(cgImage),
                                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
        CGContextRelease(context);
        CGImageRelease(cgImage);
        
        uint8_t* pixelData = static_cast<uint8_t*>(data);
        
        MatrixH<3, uint8_t> result;
        result.buffer = pixelData;
        result.shape[0] = Imgheight;
        result.shape[1] = Imgwidth;
        result.shape[2] = 4;
        result.calcStrides();
        result.total_size = Imgwidth * Imgheight * 4;
        result.buildMetalBuffer();
        return result;
    }
    
    static std::pair<MatrixH<3, uint8_t>, MatrixH<2, float16_t>> fromImageWithDepth(std::string path_str = std::string("/Users/adityadude/Documents/TUSHU.HEIC") , bool nonReduced = true) {
        #if !TARGET_OS_IPHONE
        CFStringRef path = CFStringCreateWithCString(NULL, path_str.c_str(), kCFStringEncodingUTF8);
        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
        CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(url);
        CFRelease(path);
        
        #endif

        #if TARGET_OS_IPHONE
        CGImageSourceRef source;
        CGImageRef cgImage;
        // 1. Ask the Bundle where the file is on the iPhone's disk
        NSURL *fileURL = [[NSBundle mainBundle] URLForResource:@"TUSHU" withExtension:@"HEIC"];
        
        if (fileURL) {
            // 2. Convert the URL to a CFString path (if your engine needs a path string)
            // Note: It's better to use the URL directly if possible.
            CFStringRef path = (__bridge CFStringRef)[fileURL path];
            
            // OR just pass the URL to CGImageSource directly (Better way):
            source = CGImageSourceCreateWithURL((__bridge CFURLRef)fileURL, NULL);
            cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
            // ... do your processing
        } else {
            NSLog(@"Error: Could not find TUSHU.HEIC in the app bundle!");
        }
        
        #endif

        if (!cgImage) {
            std::cerr << "Failed to create CGImage" << std::endl;
        }
        
        size_t Imgwidth = CGImageGetWidth(cgImage);
        size_t Imgheight = CGImageGetHeight(cgImage);
        std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight << "Loaded \n";
        
        CGImageRef depthImage = NULL;

        NSDictionary *auxDataInfo =
            (__bridge_transfer NSDictionary *)
            CGImageSourceCopyAuxiliaryDataInfoAtIndex(source,
                                                      0,
                                                      kCGImageAuxiliaryDataTypeDisparity);
        auto cfProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil);
        NSError* err = nil;
        auto depthData = [AVDepthData depthDataFromDictionaryRepresentation:auxDataInfo error:&err];
        CVPixelBufferRef pb = depthData.depthDataMap;
        size_t DepthWidth = CVPixelBufferGetWidth(pb);
        size_t DepthHeight = CVPixelBufferGetHeight(pb);
        
        CGSize rgbSize = CGSizeMake(CGImageGetWidth(cgImage), CGImageGetHeight(cgImage));
        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        
        MatrixH<2, float16_t> depthBuffer;

        if (nonReduced) {
            depthBuffer.buffer = new float16_t[Imgwidth * Imgheight];
            depthBuffer.shape[0] = Imgheight;
            depthBuffer.shape[1] = Imgwidth;
            depthBuffer.calcStrides();
            depthBuffer.total_size = Imgwidth * Imgheight;
            depthBuffer.buildMetalBuffer();
            
            vImage_Buffer srcBuf = {
                .data     = CVPixelBufferGetBaseAddress(pb),
                .height   = CVPixelBufferGetHeight(pb),
                .width    = CVPixelBufferGetWidth(pb),
                .rowBytes = CVPixelBufferGetBytesPerRow(pb)
            };
            
            vImage_Buffer dstBuf = {
                .data     = depthBuffer.buffer,
                .height   = Imgheight,
                .width    = Imgwidth,
                .rowBytes = Imgwidth * sizeof(float16_t)
            };
            
            vImageScale_Planar16F(&srcBuf, &dstBuf, NULL, kvImageHighQualityResampling);
        } else {
            depthBuffer.buffer = new float16_t[DepthWidth * DepthHeight];
            depthBuffer.shape[0] = DepthHeight;
            depthBuffer.shape[1] = DepthWidth;
            depthBuffer.calcStrides();
            depthBuffer.total_size = DepthWidth * DepthHeight;
            depthBuffer.buildMetalBuffer();
            void *rawPtr = CVPixelBufferGetBaseAddress(pb);
            if (CVPixelBufferGetBytesPerRow(pb) == depthBuffer.shape[1] * sizeof(float16_t)) {
                memcpy(depthBuffer.buffer, rawPtr, depthBuffer.total_size * sizeof(float16_t));
            } else {
                for (int i = 0; i < DepthHeight; i++) {
                    memcpy(depthBuffer.buffer + i * DepthWidth, (float16_t*)rawPtr + i * CVPixelBufferGetBytesPerRow(pb), DepthWidth * sizeof(float16_t));
                }
            }
        }


        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        
        

        size_t bytesPerRow = 4 * Imgwidth;
        void *data = malloc(bytesPerRow * Imgheight);
        CGContextRef context = CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
                                                     CGImageGetColorSpace(cgImage),
                                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
        CGContextRelease(context);
        CGImageRelease(cgImage);
        
        uint8_t* pixelData = static_cast<uint8_t*>(data);
        
        MatrixH<3, uint8_t> result;
        result.buffer = pixelData;
        result.shape[0] = Imgheight;
        result.shape[1] = Imgwidth;
        result.shape[2] = 4;
        result.calcStrides();
        result.total_size = Imgwidth * Imgheight * 4;
        result.buildMetalBuffer();
        
        
        return {result, depthBuffer};
    }
    static MatrixH<4, uint8_t> fromVideo(const char* vidPath) {
    //    const char* vidPath = "/Users/adityadude/Downloads/WhatsApp Video 2025-01-01 at 14.49.11.mp4";
        NSString *filePath = [NSString stringWithUTF8String:vidPath];
        
        NSURL* url = [NSURL fileURLWithPath:filePath];
        AVURLAsset* asset = [[AVURLAsset alloc] initWithURL:url options:nil];
        NSLog(@"%@", asset);
        if (!asset) {
            
            std::cerr << "Asset Invalid \n";
        }
        __block AVAssetTrack* videoTrack;
        dispatch_semaphore_t trackSem = dispatch_semaphore_create(0);
        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> * videoArray, NSError * _Nullable) {
            videoTrack = videoArray.firstObject;
            dispatch_semaphore_signal(trackSem);
        }];
        dispatch_semaphore_wait(trackSem, DISPATCH_TIME_FOREVER);
        
        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:nil];
        NSDictionary* outputSettings = @{
            (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferMetalCompatibilityKey : @YES,
            (id)kCVPixelBufferIOSurfacePropertiesKey : @{}
        };

        for (AVAssetTrack *track in asset.tracks) {
            NSLog(@"Track media type: %@", track.mediaType);
        }
//        @try {
//            AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
//            // Proceed with using trackOutput
//        }
//        @catch (NSException *exception) {
//            NSLog(@"Exception occurred: %@, %@", exception.name, exception.reason);
//            // Handle the exception appropriately
//        }
        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
        [reader addOutput:trackOutput];
        
        if ([NSThread isMainThread]) {
            NSLog(@"Running on the main thread");
        } else {
            NSLog(@"Not running on the main thread");
        }
        
        // Blocking the Thread till we load the duration async
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        __block CMTime duration;

        [asset loadValuesAsynchronouslyForKeys:@[@"duration"] completionHandler:^{
            NSError *error = nil;
            AVKeyValueStatus status = [asset statusOfValueForKey:@"duration" error:&error];
            if (status == AVKeyValueStatusLoaded) {
                duration = asset.duration;
            } else {
                // Handle error
            }
            dispatch_semaphore_signal(semaphore);
        }];

        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        
        [reader startReading];
        
        size_t frameCount = (size_t)(CMTimeGetSeconds(duration) * videoTrack.nominalFrameRate) + 10;
        size_t width = (size_t)videoTrack.naturalSize.width;
        size_t height = (size_t)videoTrack.naturalSize.height;
        size_t channels = 4; // BGRA format has 4 channels
        std::cout << "data " << frameCount <<" "<< height <<" " <<width<< " "<<channels;
        uint8_t* values = new uint8_t[width*height*frameCount*channels];
        NSLog(@"CMTime %f", CMTimeGetSeconds(duration));
        
        size_t frameIndex = 0;
        while ([reader status] == AVAssetReaderStatusReading) {
            CMSampleBufferRef sampleBuffer = [trackOutput copyNextSampleBuffer];
            if (!sampleBuffer) {std::cout << "error "; break; }
            CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
            CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
            uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
            for (size_t y = 0; y < height; y++) {
                memcpy(values + (frameIndex * height * width * channels) + (y * width * channels),
                       baseAddress + (y * bytesPerRow),
                       width * channels * sizeof(Type));
            }
            frameIndex++;
            CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
            CVPixelBufferRelease(imageBuffer);
        }
        
        MatrixH<4, uint8_t> result = MatrixH();
        result.shape[0] = frameIndex;
        result.shape[1] = height;
        result.shape[2] = width;
        result.shape[3] = channels;
        result.total_size = width*height*frameCount*channels;
        result.buffer = values;
        result.calcStrides();
        result.buildMetalBuffer();
        result.flags |= (1u << 2);
        return result;
    }
    
    dispatch_group_t saveVideo(const char* outputPath, bool wait=false) {
        size_t frames = shape[0];
        size_t height = shape[1];
        size_t width = shape[2];
        size_t channels = shape[3];
        
        dispatch_group_t group = dispatch_group_create();
        @autoreleasepool {
            NSLog(@"Saving Video...");

            NSString *filePath = [NSString stringWithUTF8String:outputPath];
            NSURL *outputURL = [NSURL fileURLWithPath:filePath];
            
            NSFileManager *fileManager = [NSFileManager defaultManager];
            if ([fileManager fileExistsAtPath:filePath]) {
                NSError *removeError = nil;
                [fileManager removeItemAtPath:filePath error:&removeError];
                if (removeError) {
                    NSLog(@"Failed to remove existing file: %@", removeError);
                    return group;
                }
            }
            
            // Setup AVAssetWriter
            NSError *error = nil;
            AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:outputURL fileType:AVFileTypeQuickTimeMovie error:&error];

            if (error) {
                NSLog(@"Error creating writer: %@", error);
                return group;
            }

            NSDictionary *videoSettings = @{
                AVVideoCodecKey: AVVideoCodecTypeH264,
                AVVideoWidthKey: @(width),
                AVVideoHeightKey: @(height),
                AVVideoScalingModeKey: AVVideoScalingModeResize
            };

            AVAssetWriterInput *writerInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:videoSettings];
            writerInput.expectsMediaDataInRealTime = YES;
            writerInput.performsMultiPassEncodingIfSupported = NO;

            NSDictionary *bufferAttributes = @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferWidthKey: @(width),
                (id)kCVPixelBufferHeightKey: @(height),
            };

            AVAssetWriterInputPixelBufferAdaptor *adaptor = [[AVAssetWriterInputPixelBufferAdaptor alloc] initWithAssetWriterInput:writerInput sourcePixelBufferAttributes:bufferAttributes];

            if ([writer canAddInput:writerInput]) {
                [writer addInput:writerInput];
            } else {
                NSLog(@"Failed to add input");
                return group;
            }

            [writer startWriting];
            [writer startSessionAtSourceTime:kCMTimeZero];

            dispatch_queue_t queue = dispatch_queue_create("video_encoding_queue", NULL);
            
            
            dispatch_group_enter(group);
            
            [writerInput requestMediaDataWhenReadyOnQueue:queue usingBlock:^{
                int frameIndex = 0;
                while (frameIndex < frames) {
                    if ([writerInput isReadyForMoreMediaData]) {
//                        uint8_t *frameData = values + (frameIndex * height * width * channels);
                        auto frame = this->operator[](frameIndex);
                        CVPixelBufferRef pixelBuffer = frame.createPixelBufferFromMat();

                        if (pixelBuffer) {
                            CMTime pts = CMTimeMake(frameIndex, 30); // 30 FPS
                            [adaptor appendPixelBuffer:pixelBuffer withPresentationTime:pts];
                            CVPixelBufferRelease(pixelBuffer);
                        }

                        frameIndex++;
                    }
                }

                [writerInput markAsFinished];
                [writer finishWritingWithCompletionHandler:^{
                    dispatch_group_leave(group);
                    NSLog(@"Video saved to %@", filePath);
                }];
            }];
            if (wait) {
                dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
            }
        }
        return group;
    }
    
    void CopyToTexture(id<MTLTexture> texture) {
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)shape[1], (NSUInteger)shape[0]);
        NSUInteger bytesPerRow = shape[1] * 4;  // 4 bytes per pixel for BGRA8
        
        id<MTLCommandBuffer> commandBuffer = [GlobalGPUManager.gCommandQueue commandBuffer];

        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        [blitEncoder copyFromBuffer:metalBuffer
                       sourceOffset:0
                  sourceBytesPerRow:bytesPerRow
                sourceBytesPerImage:bytesPerRow * shape[0]
                         sourceSize:region.size
                          toTexture:texture
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:region.origin];

        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

    }
    
    id<MTLTexture> ToMTLTexture() {
        id<MTLTexture> resultTexture;
        MTLTextureDescriptor* drawableDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                                                                                width:(NSUInteger)shape[1]
                                                                                               height:(NSUInteger)shape[0]
                                                                                            mipmapped:NO];
        drawableDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        // Use shared storage so the CPU can read the texture data.
        drawableDesc.storageMode = MTLStorageModeShared;
    
    
        resultTexture = [GlobalGPUManager.metalDevice newTextureWithDescriptor:drawableDesc];
        
        CopyToTexture(resultTexture);
        return resultTexture;
    }
    
    void printNonCont() const {
        if (!buffer) {return; }
        else if (dims == 2) {
            std::cout << "{ \n";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << buffer[strides[0] * i + strides[1] * j] << " ";
                }
                std::cout << "} \n";
            }
            std::cout << "} \n";
        } else if (dims == 3) {
            std::cout << "{ \n";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << " { ";
                    for (size_t k = 0; k < shape[2]; k++) {
                        std::cout << buffer[strides[0] * i + strides[1] * j + strides[2] * k] << ", ";
                    }
                    std::cout << "}, ";
                }
                std::cout << "}, \n";
            }
            std::cout << "}, \n";
        } else if (dims == 4) {
            for (size_t l = 0; l < shape[0]; l++) {
                for (size_t i = 0; i < shape[1]; i++) {
                    for (size_t j = 0; j < shape[2]; j++) {
                        std::cout << "{ ";
                        for (size_t k = 0; k < shape[3]; k++) {
                            std::cout << buffer[strides[0] * l + strides[1] * i + strides[2] * j + strides[3] *k] << " ";
                        }
                        std::cout << "} ";
                    }
                    std::cout << std::endl;
                }
                std::cout <<"\n";
            }
        }
        
        else {
            std::cerr << "Printing only supported for 2D matrices." << std::endl;
            return;
        }
    }
    
    void print() const {
        if (!buffer) {return; }
//        if (flags & (1u << 1)) { printNonCont(); return;}
        if (flags & NON_CONTIGUOUS_FLAG) { printNonCont(); return;}
        if (dims == 0) {
            std::cout << "{ ";
                std::cout << buffer[0] << " ,";
            std::cout << "} \n";
        }
        else if (dims == 1) {
            std::cout << "{ ";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << buffer[i] << " ,";
            }
            std::cout << "} \n";
        }
        else if (dims == 2) {
            std::cout << "{ \n";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << buffer[shape[1] * i + j] << " ";
                }
                std::cout << "} \n";
            }
            std::cout << "} \n";
        } else if (dims == 3) {
            std::cout << "{ \n";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << " { ";
                    for (size_t k = 0; k < shape[2]; k++) {
                        std::cout << buffer[shape[2]*(shape[1] * i + j) + k] << ", ";
                    }
                    std::cout << "}, ";
                }
                std::cout << "}, \n";
            }
            std::cout << "}, \n";
        } else if (dims == 4) {
            for (size_t l = 0; l < shape[0]; l++) {
                for (size_t i = 0; i < shape[1]; i++) {
                    for (size_t j = 0; j < shape[2]; j++) {
                        std::cout << "{ ";
                        for (size_t k = 0; k < shape[3]; k++) {
                            std::cout << buffer[shape[3]*(shape[2]*(shape[1] * l + i) + j)  + k] << " ";
                        }
                        std::cout << "} ";
                    }
                    std::cout << std::endl;
                }
                std::cout <<"\n";
            }
        }
        
        else {
            std::cerr << "Printing only supported till 4D matrices." << std::endl;
            return;
        }

    }
    
    
    Type* operator()(size_t i) const {
        Type* value = buffer + i * total_size / shape[0];
        return value;
    }
    
    Type* operator()(size_t i, size_t j) {
        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1]));
        return value;
    }
    
    Type* operator()(size_t i, size_t j, size_t k) {
        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])) + k * (total_size / (shape[0] * shape[1] * shape[2]));
        return value;
    }
    
    template <int Newdims, typename OutType>
    void To(MatrixH<Newdims, OutType>& output, int type) const {
        int valuesIn = 1;
        int valuesOut = 1;
        int currentTypeCode = get_dtype_code<Type>();
        int OutTypeCode = get_dtype_code<OutType>();
        
        if (currentTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<Type>();
            valuesIn = typeInfo.values;
            currentTypeCode = typeInfo.baseType;
        }
        
        if (OutTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<OutType>();
            valuesOut = typeInfo.values;
            OutTypeCode = typeInfo.baseType;
        }
        
        if (output.total_size * valuesOut != total_size * valuesIn) {
            std::cerr << "MatrixH: Invalid Dims, cannot convert from (" << total_size <<", " << valuesIn << "(imp)) To (" << output.total_size <<", " << valuesOut << ") \n";
        }
        
        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn, 1, 1);
        
        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(OutType) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
        
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:&type length:sizeof(int) atIndex:2];
        [commandEncoder setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    template <typename OutType>
    void To(MatrixH<dims, OutType>& output, int type) const {
        int valuesIn = 1;
        int valuesOut = 1;
        int currentTypeCode = get_dtype_code<Type>();
        int OutTypeCode = get_dtype_code<OutType>();
        
        if (currentTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<Type>();
            valuesIn = typeInfo.values;
            currentTypeCode = typeInfo.baseType;
        }
        
        if (OutTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<OutType>();
            valuesOut = typeInfo.values;
            OutTypeCode = typeInfo.baseType;
        }
        
        if (output.total_size * valuesOut != total_size * valuesIn) {
            std::cerr << "MatrixH: Invalid Dims, cannot convert from (" << total_size <<", " << valuesIn << "(imp)) To (" << output.total_size <<", " << valuesOut << ") \n";
        }
        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn ,1, 1);
        
        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(OutType) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        [commandEncoder setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
//    explicit operator MatrixH<dims, float>() const {
//        MatrixH<dims, float> result;
//        result.total_size = total_size;
//        result.buffer = new float[total_size];
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
//        
//        this->To<float>(result, 0);
//
//        return result;
//    }
    
    template <typename OutType>
    explicit operator MatrixH<dims, OutType>() const {
        int valuesIn = 1;
        int valuesOut = 1;
        int currentTypeCode = get_dtype_code<Type>();
        int OutTypeCode = get_dtype_code<OutType>();
        
        if (valuesIn != valuesOut) {
            std::cerr << "MatrixH: increase dims" << "\n";
        }
        
        if (currentTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<Type>();
            valuesIn = typeInfo.values;
            currentTypeCode = typeInfo.baseType;
        }
        
        if (OutTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<OutType>();
            valuesOut = typeInfo.values;
            OutTypeCode = typeInfo.baseType;
        }
        
        MatrixH<dims, OutType> result;
        result.total_size = total_size ;
        result.buffer = new OutType[total_size];
        result.buildMetalBuffer();
        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
        std::memcpy(result.strides, strides, sizeof(size_m) * dims);
        this->To<OutType>(result, 0);
        return result;
    }
    
    template <typename OutType, int DimsNew>
    explicit operator MatrixH<DimsNew, OutType>() const {
        int valuesIn = 1;
        int valuesOut = 1;
        int currentTypeCode = get_dtype_code<Type>();
        int OutTypeCode = get_dtype_code<OutType>();
        uint8_t dimBias = 0;
        
        
        
        if (currentTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<Type>();
            valuesIn = typeInfo.values;
            currentTypeCode = typeInfo.baseType;
        }
        
        if (OutTypeCode > valueLimit) {
            auto typeInfo = get_dtype_info<OutType>();
            valuesOut = typeInfo.values;
            OutTypeCode = typeInfo.baseType;
        }
        
        MatrixH<DimsNew, OutType> result;
        result.total_size = (total_size * valuesIn) / valuesOut;
        result.buffer = new OutType[result.total_size];
        result.buildMetalBuffer();
        
        if (DimsNew - dims == 1){
            result.shape[DimsNew-1] = valuesIn;
            std::memcpy(result.shape, shape, sizeof(size_m) * dims);
        } else if (dims - DimsNew == 1) {
            if (shape[dims-1] != valuesOut) {
                std::cerr << "MatrixH: For conversion last dim should be " << valuesOut << "\n";
            }
            std::memcpy(result.shape, shape, sizeof(size_m) * DimsNew);
        } else {
            std::cerr << "MatrixH: not supported as of yet" << "\n";
        }
        
        
        this->To(result, 0);
        result.calcStrides();
        return result;
    }
    
    
    
    
    template <int dimsNew>
    explicit operator MatrixH<dimsNew, Type>() {
        return this->unsqueeze<dimsNew-dims>();
    }
    
    
    
//    template<int DimsNew>
//    explicit operator MatrixH<DimsNew, float>() const {
//        MatrixH<DimsNew, float> result;
//        uint8_t dimBias = 0;
//        int code = 00;
//        if (std::is_same<Type, simd_float2>::value) {
//            result.total_size = total_size * 2;
//            result.shape[DimsNew-1] = 2;
//            dimBias = 1;
//        } else if (std::is_same<Type, simd_float3>::value) {
//            result.total_size = total_size * 3;
//            result.shape[DimsNew-1] = 3;
//            dimBias = 1;
//        } else if (std::is_same<Type, simd_float4>::value) {
//            result.total_size = total_size * 4;
//            result.shape[DimsNew-1] = 4;
//            dimBias = 1;
//        } else if (std::is_same<Type, simd_float2x2>::value) {
//            result.total_size = total_size * 4;
//            result.shape[DimsNew-1] = 2;
//            result.shape[DimsNew-2] = 2;
//            dimBias = 2;
//        } else if (std::is_same<Type, simd_float3x3>::value) {
//            result.total_size = total_size * 9;
//            result.shape[DimsNew-1] = 3;
//            result.shape[DimsNew-2] = 3;
//            dimBias = 2;
//        } else if (std::is_same<Type, simd_float4x4>::value) {
//            result.total_size = total_size * 16;
//            result.shape[DimsNew-1] = 4;
//            result.shape[DimsNew-2] = 4;
//            dimBias = 2;
//        }
//        else {
//            result.total_size = total_size;
//        }
//        
//        result.buffer = new float[total_size];
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        std::memcpy(result.shape + (DimsNew - dims) - dimBias, shape, sizeof(size_m) * dims);
//        std::fill(result.shape, result.shape + (DimsNew - dims) - dimBias, 1);
//        
//        this->To<DimsNew, float>(result, 1);
//        
//
//        return result;
//    }
//    
//    template<int DimsNew>
//    explicit operator MatrixH<DimsNew, uint8_t>() const {
//        MatrixH<DimsNew, uint8_t> result;
//        result.total_size = total_size;
//        result.buffer = new uint8_t[total_size];
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
//        std::fill(result.shape + dims, result.shape + DimsNew, 1);
//        
//        if (std::is_same<Type, uint8_t>::value) {
//            this->To<DimsNew, uint8_t>(result, 1);
//        }
//        else if (std::is_same<Type, int16_t>::value) {
//            this->To<DimsNew, uint8_t>(result, 3);
//        }
//        else {
//            std::cerr << "MatrixH: Type Not Suported Yet" << "\n";
//        }
//
//        return result;
//    }
    
    
    
    
    
    
    MatrixH<dims, Type> Transpose(std::initializer_list<size_m> axis) {
        if (axis.size() != dims) {
            std::cerr << "MatrixH: Axis size should be equal to Dims of " << dims << "\n";
        }
        
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.TransposeInit[typeCode]) {
            GlobalGPUManager.initTransposeAll(typeCode);
        }
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        

        
//        size_m inputStrides[dims];
//        size_t acc = 1;
//        for (int i = dims-1; i >= 0; i--) {
//            inputStrides[i] = acc;
//            acc *= shape[i];
//        }
        
        size_m acc = 1;
        size_m outputStrides[dims];
        for (int i = dims-1; i >= 0; i--) {
            outputStrides[*(axis.begin() + i)] = acc;
            output.strides[i] = acc;
            if (*(axis.begin() + i) >= dims) {std::cerr << "MatrixH: Rearranged axis should not increase dims" << "\n"; }
            acc *= shape[*(axis.begin() + i)];
            output.shape[i] = shape[*(axis.begin() + i)];
        }
        
        int dimensions = dims;
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }
        [commandEncoder setBytes:strides length:dims * sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:outputStrides length:dims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&dimensions length: sizeof(dims) atIndex:4];
        [commandEncoder setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    
    
    MatrixH<dims, Type> sin() const {
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.SinInit[typeCode]) {
            GlobalGPUManager.initSin(typeCode);
        }
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, shape, sizeof(size_m) * dims);
        memcpy(output.strides, strides, sizeof(size_m) * dims);
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    static MatrixH<dims, Type> sqrt(const MatrixH<dims, Type>& mat) {
        
        
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.SqrtInit[typeCode]) {
            GlobalGPUManager.initSqrt_All(typeCode);
        }
        
        MatrixH<dims, Type> output;
        output.total_size = mat.total_size;
        output.buffer = new Type[mat.total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, mat.shape, sizeof(size_m) * dims);
        memcpy(output.strides, mat.strides, sizeof(size_m) * dims);
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(mat.total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (mat.metalBuffer) {
            [commandEncoder setBuffer:mat.metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:mat.buffer length:mat.total_size*sizeof(Type) atIndex:1];
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    // sqrts itself
    MatrixH<dims, Type> sqrt() const {
        uint8_t typeCode = get_dtype_code<Type>();
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, shape, sizeof(size_m) * dims);
        memcpy(output.strides, strides, sizeof(size_m) * dims);
        
        if (!GlobalGPUManager.SqrtInit[typeCode]) {
            GlobalGPUManager.initSqrt_All(typeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    MatrixH<dims, Type> exp() const {
        uint8_t typeCode = get_dtype_code<Type>();
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, shape, sizeof(size_m) * dims);
        memcpy(output.strides, strides, sizeof(size_m) * dims);
        
        if (!GlobalGPUManager.ExpInit[typeCode]) {
            GlobalGPUManager.initExp(typeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    MatrixH<dims, Type> cos() const {
        uint8_t typeCode = get_dtype_code<Type>();
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, shape, sizeof(size_m) * dims);
        memcpy(output.strides, strides, sizeof(size_m) * dims);
        
        if (!GlobalGPUManager.CosInit[typeCode]) {
            GlobalGPUManager.initCos(typeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];

        [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    MatrixH<dims, Type> tan() const {
        uint8_t typeCode = get_dtype_code<Type>();
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        memcpy(output.shape, shape, sizeof(size_m) * dims);
        memcpy(output.strides, strides, sizeof(size_m) * dims);
        
        if (!GlobalGPUManager.TanInit[typeCode]) {
            GlobalGPUManager.initTan(typeCode);
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return output;
    }
    
    MatrixH<dims-1, Type> Sum(int axis) const {
        if (axis < 0){
            axis += dims;
        }
        if ((dims-1 < axis)) {
            std::cerr << "MatrixH: Axis should not excede Dims of " << dims << "\n";
            throw;
        }
        
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.SumInit[typeCode]) {
            GlobalGPUManager.initSum_All(typeCode);
        }
        
        MatrixH<dims-1, Type> output;
        memcpy(output.shape, shape, axis * sizeof(size_m));
        memcpy(output.shape + axis, shape + axis + 1, (dims-axis) * sizeof(size_m));
        
        output.total_size = output.accumul(0, dims-1);
        output.buffer = new Type[output.total_size];
        memset(output.buffer, 0, output.total_size * sizeof(Type));
        output.buildMetalBuffer();
        output.calcStrides();
        std::cout << output.total_size << "\n";
        
        size_m ElStride = accumul(axis+1, dims);
        
        size_m noOfOpp = shape[axis];
        size_m axisStride;
        if (axis != dims-1) {
            axisStride = 1;
        }
        else {
            axisStride = shape[axis];
        }
        
        size_m inputStrides[dims-1];
        size_t acc = 1;
        for (int i = dims-2; i >= 0; i--) {
            inputStrides[i] = acc;
            acc *= output.shape[i];
        }
        
        size_m maskedStrides[dims-1];
        memcpy(maskedStrides, inputStrides, sizeof(size_m) * (dims-1));
        
        acc = 1;
        for (int i = 0; i < axis; i++) {
            maskedStrides[i] *= shape[axis];
        }
//        std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride << " noOfOpp: " << noOfOpp << "\n";
//        printArray(inputStrides, dims-1);
//        printArray(maskedStrides, dims-1);
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);
         
        size_t outputDims = dims -1;
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        setBufferOrBytes(commandEncoder, output, 0);
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }
        [commandEncoder setBytes:&axisStride length: sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:&ElStride length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:4];
        
        if (dims > 1) {
            [commandEncoder setBytes:&inputStrides length: (dims-1) * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:&maskedStrides length: (dims-1)* sizeof(size_m) atIndex:6];
        } else {
            size_m inpandmaskedStrides = 1;
            [commandEncoder setBytes:&inpandmaskedStrides length: sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:&inpandmaskedStrides length:sizeof(size_m) atIndex:6];
        }
        [commandEncoder setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        

        
        return output;
    }
    
    MatrixH<dims, Type> SumNoRed(int axis) {
        if (axis < 0){
            axis += dims;
        }
        if ((dims-1 < axis)) {
            std::cerr << "MatrixH: Axis should not excede Dims of " << dims << "\n";
            throw;
        }
        
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.SumInit[typeCode]) {
            GlobalGPUManager.initSum_All(typeCode);
        }
        
        MatrixH<dims, Type> output;
        memcpy(output.shape, shape, dims * sizeof(size_m));
        output.shape[axis] = 1;
        output.total_size = output.accumul(0, dims-1);
        output.buffer = new Type[output.total_size];
        memset(output.buffer, 0, output.total_size * sizeof(Type));
        output.buildMetalBuffer();
        output.calcStrides();
        std::cout << output.total_size << "\n";
        
        size_m ElStride = accumul(axis+1, dims);
        
        size_m noOfOpp = shape[axis];
        size_m axisStride;
        if (axis != dims-1) {
            axisStride = 1;
        }
        else {
            axisStride = shape[axis];
        }
        
        size_m inputStrides[dims-1];
        size_t acc = 1;
        for (int i = dims-2; i >= 0; i--) {
            inputStrides[i] = acc;
            acc *= output.shape[i];
        }
        
        size_m maskedStrides[dims-1];
        memcpy(maskedStrides, inputStrides, sizeof(size_m) * (dims-1));
        
        acc = 1;
        for (int i = 0; i < axis; i++) {
            maskedStrides[i] *= shape[axis];
        }
//        std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride << " noOfOpp: " << noOfOpp << "\n";
//        printArray(inputStrides, dims-1);
//        printArray(maskedStrides, dims-1);
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);
         
        size_t outputDims = dims -1;
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size * sizeof(Type) atIndex:1];
        }
        [commandEncoder setBytes:&axisStride length: sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:&ElStride length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:4];
        if (dims > 1) {
            [commandEncoder setBytes:&inputStrides length: (dims-1) * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:&maskedStrides length: (dims-1)* sizeof(size_m) atIndex:6];
        } else {
            size_m inpandmaskedStrides = 1;
            [commandEncoder setBytes:&inpandmaskedStrides length: sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:&inpandmaskedStrides length:sizeof(size_m) atIndex:6];
        }
        [commandEncoder setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        

        
        return output;
    }
    
    MatrixH<dims, Type> T() const {
        size_t axis[dims];
        
        
        uint8_t typeCode = get_dtype_code<Type>();

        if (!GlobalGPUManager.TransposeInit[typeCode]) {
            GlobalGPUManager.initTransposeAll(typeCode);
        }
        
        MatrixH<dims, Type> output;
        output.total_size = total_size;
        output.buffer = new Type[total_size];
        output.buildMetalBuffer();
        
        
        
//        size_m inputStrides[dims];
        size_m acc = 1;
        for (int i = dims-1; i >= 0; i--) {
//            inputStrides[i] = acc;
//            acc *= shape[i];
//            
//            // reverse the axis
            axis[dims-1-i]=i;
        }
        
        
        acc = 1;
        size_m outputStrides[dims];
        for (int i = dims-1; i >= 0; i--) {
            outputStrides[axis[i]] = acc;
            if (axis[i] >= dims) {std::cerr << "MatrixH: Rearranged axis should not increase dims" << "\n"; }
            output.strides[i] = acc;
            acc *= shape[axis[i]];
            output.shape[i] = shape[axis[i]];
        }
        int dimensions = dims;
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size*sizeof(Type) atIndex:1];
        }
        [commandEncoder setBytes:strides length:dims * sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:outputStrides length:dims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&dimensions length: sizeof(dimensions) atIndex:4];
        [commandEncoder setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        

        
        return output;
    }
    
    
    template<int kDims>
    void conv(MatrixH<dims, Type>& result, const MatrixH<kDims, Type> &kernel, int sepAxis = kDims, ConvMode mode = ConvMode::Same, ConvModeFull Fullmode = ConvModeFull::Normal) {
        
        uint8_t typeCode = get_dtype_code<Type>();
        

        
        size_m elementStride = accumul(sepAxis, dims); //
//        size_m elementStride = strides[sepAxis-1];
        size_m noOfOpp;
        size_m newShape[dims];
        switch (mode) {
            case ConvMode::Same:
                if (!GlobalGPUManager.ConvolveInit[typeCode]) {
                    GlobalGPUManager.initConvolve_All(typeCode);
                }
                noOfOpp = accumul(0, sepAxis);
                memcpy(newShape, shape, dims * sizeof(size_m));
                break;
            case ConvMode::Valid:
                noOfOpp = 1;
                for (size_m i = 0; i < sepAxis; ++i) {
                    newShape[i] = shape[i] - (kernel.shape[i] - 1);
                    noOfOpp *= newShape[i];
                }
                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) * sizeof(size_m));
                break;
            case ConvMode::Full:
                if (!GlobalGPUManager.ConvolveFullInit[typeCode]) {
                    GlobalGPUManager.initConvolve_FULL(typeCode);
                }
                noOfOpp = 1;
                for (size_m i = 0; i < sepAxis; ++i) {
                    newShape[i] = shape[i] + (kernel.shape[i] - 1);
                    noOfOpp *= newShape[i];
                }
                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) * sizeof(size_m));
                break;
        }
        size_m stridesNew[kDims];
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        MTLLogStateDescriptor *logStateDesc = [MTLLogStateDescriptor new];
//        logStateDesc.bufferSize = 1024*1024;
//        logStateDesc.level = MTLLogLevelInfo;
//        NSError* error = nil;
//        id<MTLLogState> logState = [GlobalGPUManager.metalDevice newLogStateWithDescriptor:logStateDesc error:&error];
//        [logState addLogHandler:^(NSString *substring, NSString *category,
//                                  MTLLogLevel level, NSString *message)
//        {
//        }];
//        
//        
//        MTLCommandBufferDescriptor *cbufDesc = [MTLCommandBufferDescriptor new];
//        cbufDesc.logState = logState;
//        
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBufferWithDescriptor:cbufDesc];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(noOfOpp, 1, 1);
        
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        if (kernel.metalBuffer) {
            [commandEncoder setBuffer:kernel.metalBuffer offset:0 atIndex:2];
        } else {
            [commandEncoder setBytes:kernel.buffer length:kernel.total_size * sizeof(Type) atIndex:2];
        }
        [commandEncoder setBytes:&kernel.total_size length: sizeof(size_t) atIndex:3];
        [commandEncoder setBytes:shape length: kDims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:kernel.shape length:kDims * sizeof(size_m) atIndex:5];
//        if (sepAxis == dims) {
            [commandEncoder setBytes:strides length:kDims * sizeof(size_m) atIndex:6];
//        } else {
//            
//            size_m acc = 1;
//            for (int i = kDims-1; i >= 0; i--) {
//                stridesNew[i] = acc;
//                acc *= shape[i];
//            }
//            [commandEncoder setBytes:stridesNew length:kDims * sizeof(size_m) atIndex:6];
//        }
        [commandEncoder setBytes:&elementStride length: sizeof(size_m) atIndex:7];
        [commandEncoder setBytes:&sepAxis length: sizeof(int) atIndex:8];
        if (mode == ConvMode::Full) {
            [commandEncoder setBytes:&Fullmode length: sizeof(uint8_t) atIndex:9];
            [commandEncoder setComputePipelineState:GlobalGPUManager.ConvolveFullComputeState[typeCode]];
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ConvolveComputeState[typeCode]];
        }
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    void Dot(MatrixH<dims, Type>& result, const MatrixH<dims, Type> &other, bool TransposeB) {
        
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (!GlobalGPUManager.GEMMAInit[typeCode]) {
            GlobalGPUManager.initGEMMA_All(typeCode);
        }
        
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
        
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        if (metalBuffer) {
            [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        } else {
            [commandEncoder setBytes:buffer length:total_size * sizeof(Type) atIndex:1];
        }
        
        
        // BUG FIX: B_transposed must be declared in the same scope as [commandBuffer commit]; to prevent premature deallocation.
        // Previously, B_transposed was declared inside the if block and [commandBuffer commit]; outside the if block, causing its metalBuffer to be
        // deallocated before the GPU command completed, resulting in undefined behavior and incorrect results.
        // Separated TransposeB branches to handle object lifetimes correctly.
        // When TransposeB=false, B_transposed must stay alive until GPU execution completes,
        // so we wait until after waitUntilCompleted() before it goes out of scope.
        // When TransposeB=true, reverseShapeBuffer can be safely deleted immediately after
        // waitUntilCompleted() since it's just a shape array, and we avoid unnecessary B_transposed allocation.
        
        if (!TransposeB) {
            MatrixH<dims, Type> B_transposed;
            B_transposed = other.T();
            if (other.metalBuffer) {
                [commandEncoder setBuffer:B_transposed.metalBuffer offset:0 atIndex:2];
            } else {
                [commandEncoder setBytes:B_transposed.buffer length:B_transposed.total_size * sizeof(Type) atIndex:2];
            }

            [commandEncoder setBytes:other.shape length:dims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:shape length:dims * sizeof(size_m) atIndex:3];
            
            [commandEncoder setComputePipelineState:GlobalGPUManager.GEMMAComputeState[typeCode]];
            [commandEncoder dispatchThreads:_dispatchExecutionSize
                      threadsPerThreadgroup:_threadsPerThreadgroup];
            
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
                    
        } else {
            size_m* reverseShapeBuffer = new size_m[dims];
            reverseBuffer(other.shape, reverseShapeBuffer, dims);
            if (other.metalBuffer) {
                [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
            } else {
                [commandEncoder setBytes:other.buffer length:other.total_size * sizeof(Type) atIndex:2];
            }
            [commandEncoder setBytes:reverseShapeBuffer length:dims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:shape length:dims * sizeof(size_m) atIndex:3];
            
            [commandEncoder setComputePipelineState:GlobalGPUManager.GEMMAComputeState[typeCode]];
            [commandEncoder dispatchThreads:_dispatchExecutionSize
                      threadsPerThreadgroup:_threadsPerThreadgroup];
            
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            delete [] reverseShapeBuffer;
        }
    }
    
    MatrixH<dims, Type> Dot(const MatrixH<dims, Type> &other) {
#ifdef SAFE_MODE
        if (shape[dims - 1] != other.shape[0]) {
            std::cerr << "ValueError: shapes (" ;
            printShape(false);
            std::cerr << ") and (";
            other.printShape(false);
            std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1 <<") != "<< other.shape[0] <<" (dim 0) \n";
            throw;
        }
#endif
        MatrixH<dims, Type> result = MatrixH<dims, Type>::withShape({shape[0], other.shape[1]});
        Dot(result, other, false);
        return result;
    }
    
    MatrixH<dims, Type> cross(const MatrixH<dims, Type> &other) {
#ifdef SAFE_MODE
        if (3 != shape[dims-1]) {
            std::cerr << "ValueError: last dim should be 3 (" ;
            printShape(false);
            std::cerr << ") \n";
            throw;
        }
        if (3 != other.shape[dims-1]) {
            std::cerr << "ValueError: last dim should be 3 (" ;
            other.printShape(false);
            std::cerr << ") \n";
            throw;
        }
#endif
        
        MatrixH<dims, Type> result = other.zeros();
        for (int i = 0; i < accumul(0, dims-1); i++) {
            *(simd_float3*)(result.buffer + 3*i) = simd_cross(*(simd_float3*)(buffer + 3*i), *(simd_float3*)(other.buffer + 3*i));
        }
        return result;
    }
    
    MatrixH<dims-1, Type> length(const MatrixH<dims, Type> &other) {
#ifdef SAFE_MODE
        if (3 != shape[dims-1]) {
            std::cerr << "ValueError: last dim should be 3 (" ;
            printShape(false);
            std::cerr << ") \n";
            throw;
        }
        if (3 != other.shape[dims-1]) {
            std::cerr << "ValueError: last dim should be 3 (" ;
            other.printShape(false);
            std::cerr << ") \n";
            throw;
        }
#endif
        
        MatrixH<dims-1, Type> result(accumul(0, dims-1));
        for (int i = 0; i < accumul(0, dims-1); i++) {
            *(simd_float3*)(result.buffer + 3*i) = simd_length(*(simd_float3*)(buffer + 3*i));
        }
        return result;
    }
    
    
    MatrixH<dims, Type> Dot(const MatrixH<dims, Type> &other, bool TransposeB) {

        if (TransposeB) {
#ifdef SAFE_MODE
            if (shape[dims - 1] != other.shape[dims-1]) {
                std::cerr << "ValueError: shapes (" ;
                printShape(false);
                std::cerr << ") and (";
                other.printShape(false);
                std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1 <<") != "<< other.shape[dims-1] <<" (dim dims-1) \n";
                throw;
            }
#endif
            MatrixH<dims, Type> result = MatrixH<dims, Type>::withShape({shape[0], other.shape[0]});
            Dot(result, other, TransposeB);
            return result;
        } else {
#ifdef SAFE_MODE
            if (shape[dims - 1] != other.shape[0]) {
                std::cerr << "ValueError: shapes (" ;
                printShape(false);
                std::cerr << ") and (";
                other.printShape(false);
                std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1 <<") != "<< other.shape[0] <<" (dim 0) \n";
                throw;
            }
#endif
            MatrixH<dims, Type> result = MatrixH<dims, Type>::withShape({shape[0], other.shape[1]});
            Dot(result, other, TransposeB);
            return result;
        }
        
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedAddCPU(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {

        uint8_t typeCode = get_dtype_code<Type>();
        
        if (resultDims != fmax(dims, dimsB)) {
            std::invalid_argument("Incompatible dims of the result mat");
            throw;
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    std::invalid_argument("Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            if (result.total_size > 10) {
                result.buildMetalBuffer();
            }
        }
        
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < resultDims; i++) {
                indexR = rem / result.strides[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= result.strides[i];
            }
            
            result.buffer[gid] = buffer[GindexA] + other.buffer[GindexB];
        }
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedSubCPU(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {

        uint8_t typeCode = get_dtype_code<Type>();
        
        if (resultDims != fmax(dims, dimsB)) {
            std::invalid_argument("Incompatible dims of the result mat");
            throw;
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    std::invalid_argument("Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            if (result.total_size > 10) {
                result.buildMetalBuffer();
            }
        }
        
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < resultDims; i++) {
                indexR = rem / result.strides[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= result.strides[i];
            }
            
            result.buffer[gid] = buffer[GindexA] - other.buffer[GindexB];
        }
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedMulCPU(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {

        uint8_t typeCode = get_dtype_code<Type>();
        
        if (resultDims != fmax(dims, dimsB)) {
            std::invalid_argument("Incompatible dims of the result mat");
            throw;
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    std::invalid_argument("Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            if (result.total_size > 10) {
                result.buildMetalBuffer();
            }
            
        }
        
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < resultDims; i++) {
                indexR = rem / result.strides[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= result.strides[i];
            }
            
            result.buffer[gid] = buffer[GindexA] * other.buffer[GindexB];
        }
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedDivCPU(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {

        uint8_t typeCode = get_dtype_code<Type>();
        
        if (resultDims != fmax(dims, dimsB)) {
            std::invalid_argument("Incompatible dims of the result mat");
            throw;
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    std::invalid_argument("Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            if (result.total_size > 10) {
                result.buildMetalBuffer();
            }
        }
        
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < resultDims; i++) {
                indexR = rem / result.strides[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= result.strides[i];
            }
            
            result.buffer[gid] = buffer[GindexA] / other.buffer[GindexB];
        }
    }
    

    
    template <int dimsB, int resultDims>
    void BroadcastedAdd(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<MatrixH<dimsB, Type>>(other));
//        result.gradFunc = [](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return p1 + p2;
//        };
//        auto add = AddNode(std::make_shared<MatrixBase>(*this), std::make_shared<MatrixBase>(other));
//        std::vector<AddNode> vec3 = {add};
//        result.operationNodes.push_back(add);
//
        uint8_t typeCode = get_dtype_code<Type>();
        
        
        if (resultDims != fmax(dims, dimsB)) {
            throw std::invalid_argument("MatrixH: Incompatible dims of the result mat");
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    throw std::invalid_argument("MatrixH: Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            result.buildMetalBuffer();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
        int rDims = resultDims;
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);
        
        
        if constexpr (resultDims == 1) {
            // 1 dim specialisation
            if (!GlobalGPUManager.BrodcastedAddInit[typeCode][0]) {
                GlobalGPUManager.initBrodcastedAddInit(typeCode, 0);
            }
            [commandEncoder setBytes:&result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&strideA length: sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&strideB length: sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][0]];
            _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        } else if constexpr (resultDims == 2) {
            // 2 dim specialisation
            if (!GlobalGPUManager.BrodcastedAddInit[typeCode][1]) {
                GlobalGPUManager.initBrodcastedAddInit(typeCode, 1);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][1]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[1], result.shape[0], 1);
        } else if constexpr (resultDims == 3) {
            // 3 dim specialisation
            if (!GlobalGPUManager.BrodcastedAddInit[typeCode][2]) {
                GlobalGPUManager.initBrodcastedAddInit(typeCode, 2);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][2]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[2], result.shape[1], result.shape[0]);
        } else {
            // N dim specialisation
            if (!GlobalGPUManager.BrodcastedAddInit[typeCode][3]) {
                GlobalGPUManager.initBrodcastedAddInit(typeCode, 3);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result.shape length:rDims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:7];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][3]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[resultDims-1], result.shape[resultDims-2], result.accumul(0, resultDims-2));
        }
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedSub(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<MatrixH<dimsB, Type>>(other));
//        result.gradFunc = [](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return p1 + p2;
//        };
//
        uint8_t typeCode = get_dtype_code<Type>();
    
        
        if (resultDims != fmax(dims, dimsB)) {
            throw std::invalid_argument("MatrixH: Incompatible dims of the result mat");
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    throw std::invalid_argument("MatrixH: Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            result.buildMetalBuffer();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
        int rDims = resultDims;
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);
        
        
        if constexpr (resultDims == 1) {
            // 1 dim specialisation
            if (!GlobalGPUManager.BrodcastedSubInit[typeCode][0]) {
                GlobalGPUManager.initBrodcastedSubInit(typeCode, 0);
            }
            [commandEncoder setBytes:&result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&strideA length: sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&strideB length: sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][0]];
            _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        } else if constexpr (resultDims == 2) {
            // 2 dim specialisation
            if (!GlobalGPUManager.BrodcastedSubInit[typeCode][1]) {
                GlobalGPUManager.initBrodcastedSubInit(typeCode, 1);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][1]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[1], result.shape[0], 1);
        } else if constexpr (resultDims == 3) {
            // 3 dim specialisation
            if (!GlobalGPUManager.BrodcastedSubInit[typeCode][2]) {
                GlobalGPUManager.initBrodcastedSubInit(typeCode, 2);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][2]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[2], result.shape[1], result.shape[0]);
        } else {
            // N dim specialisation
            if (!GlobalGPUManager.BrodcastedSubInit[typeCode][3]) {
                GlobalGPUManager.initBrodcastedSubInit(typeCode, 3);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result.shape length:rDims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:7];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][3]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[resultDims-1], result.shape[resultDims-2], result.accumul(0, resultDims-2));
        }
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedMul(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<MatrixH<dimsB, Type>>(other));
//        result.gradFunc = [](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return p1 + p2;
//        };
//
        uint8_t typeCode = get_dtype_code<Type>();
        
        if (resultDims != fmax(dims, dimsB)) {
            throw std::invalid_argument("MatrixH: Incompatible dims of the result mat");
        }
        
//        size_m* strideA = new size_m[resultDims];
//        size_m* strideB = new size_m[resultDims];
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    throw std::invalid_argument("MatrixH: Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }

        result.calcStrides();
        
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            result.buildMetalBuffer();
        }
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
        int rDims = resultDims;
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);
        
        if constexpr (resultDims == 1) {
            // 1 dim specialisation
            if (!GlobalGPUManager.BrodcastedMulInit[typeCode][0]) {
                GlobalGPUManager.initBrodcastedMulInit(typeCode, 0);
            }
            [commandEncoder setBytes:&result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&strideA length: sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&strideB length: sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][0]];
            _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        } else if constexpr (resultDims == 2) {
            // 2 dim specialisation
            if (!GlobalGPUManager.BrodcastedMulInit[typeCode][1]) {
                GlobalGPUManager.initBrodcastedMulInit(typeCode, 1);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][1]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[1], result.shape[0], 1);
        } else if constexpr (resultDims == 3) {
            // 3 dim specialisation
            if (!GlobalGPUManager.BrodcastedMulInit[typeCode][2]) {
                GlobalGPUManager.initBrodcastedMulInit(typeCode, 2);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][2]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[2], result.shape[1], result.shape[0]);
        } else {
            // N dim specialisation
            if (!GlobalGPUManager.BrodcastedMulInit[typeCode][3]) {
                GlobalGPUManager.initBrodcastedMulInit(typeCode, 3);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result.shape length:rDims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:7];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][3]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[resultDims-1], result.shape[resultDims-2], result.accumul(0, resultDims-2));
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    template <int dimsB, int resultDims>
    void BrodcastedDiv(MatrixH<resultDims, Type>& result, const MatrixH<dimsB, Type> &other) const {
//        result.parentNodes.push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<MatrixH<dimsB, Type>>(other));
//        result.gradFunc = [](MatrixH<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return p1 + p2;
//        };
//
        uint8_t typeCode = get_dtype_code<Type>();
        
        
        if (resultDims != fmax(dims, dimsB)) {
            throw std::invalid_argument("MatrixH: Incompatible dims of the result mat");
        }
        
        size_m strideA[resultDims];
        size_m strideB[resultDims];

        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
        
        for (int i = 0; i < resultDims; i++) {
            // dims - i-1 < 0
            if (dims < i+1) {
                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                strideA[resultDims-i-1] =0;
            } else if (dimsB < i+1) {
                result.shape[resultDims-i-1] = shape[dims-i-1];
                strideB[resultDims-i-1] =0;
            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
                if (shape[dims-i-1] == 1) {
                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
                    strideA[resultDims-i-1] =0;
                } else if (other.shape[dimsB-i-1] == 1) {
                    result.shape[resultDims-i-1] = shape[dims-i-1];
                    strideB[resultDims-i-1] =0;
                } else {
                    throw std::invalid_argument("MatrixH: Incompatible shapes for broadcasting");
                }
            } else {
                result.shape[resultDims-i-1] = shape[dims-i-1];
            }
        }
        
        result.calcStrides();
        if (result.total_size != result.accumul(0, resultDims)) {
            result.total_size = result.accumul(0, resultDims);
            if (result.buffer) {delete [] result.buffer; }
            result.buffer = new Type[result.total_size];
            result.buildMetalBuffer();
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
        int rDims = resultDims;
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);

        
        if constexpr (resultDims == 1) {
            // 1 dim specialisation
            if (!GlobalGPUManager.BrodcastedDivInit[typeCode][0]) {
                GlobalGPUManager.initBrodcastedDivInit(typeCode, 0);
            }
            [commandEncoder setBytes:&result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&strideA length: sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&strideB length: sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][0]];
            _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        } else if constexpr (resultDims == 2) {
            // 2 dim specialisation
            if (!GlobalGPUManager.BrodcastedDivInit[typeCode][1]) {
                GlobalGPUManager.initBrodcastedDivInit(typeCode, 1);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][1]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[1], result.shape[0], 1);
        } else if constexpr (resultDims == 3) {
            // 3 dim specialisation
            if (!GlobalGPUManager.BrodcastedDivInit[typeCode][2]) {
                GlobalGPUManager.initBrodcastedDivInit(typeCode, 2);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][2]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[2], result.shape[1], result.shape[0]);
        } else {
            // N dim specialisation
            if (!GlobalGPUManager.BrodcastedDivInit[typeCode][3]) {
                GlobalGPUManager.initBrodcastedDivInit(typeCode, 3);
            }
            [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result.shape length:rDims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:7];
            [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][3]];
            _dispatchExecutionSize = MTLSizeMake(result.shape[resultDims-1], result.shape[resultDims-2], result.accumul(0, resultDims-2));
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    
    void Add(MatrixH<dims, Type>& result, const MatrixH<dims, Type> &other) {
//        if (!result.parentNodes) {
//            result.parentNodes = new std::vector<std::shared_ptr<MatrixH<dims, Type>>>(2);
//        }
//        result.parentNodes->push_back(std::make_shared<MatrixH<dims, Type>>(*this));
//        result.parentNodes->push_back(std::make_shared<MatrixH<dims, Type>>(other));
//        result.gradFunc = new std::function<MatrixH<dims, Type>(MatrixH<dims, Type>&)>(
//            [](MatrixH<dims, Type>& selfs) {
//                auto p1 = (*(*selfs.parentNodes)[0]->gradFunc)
//                            ? (*(*selfs.parentNodes)[0]->gradFunc)(*(*selfs.parentNodes)[0])
//                            : (*selfs.parentNodes)[0]->ones();
//
//                auto p2 = (*(*selfs.parentNodes)[1]->gradFunc)
//                            ? (*(*selfs.parentNodes)[1]->gradFunc)(*(*selfs.parentNodes)[1])
//                            : (*selfs.parentNodes)[1]->ones();
//
//                return p1 + p2;
//            }
//        );

        
        id<MTLComputePipelineState> computeState;
        if constexpr (std::is_integral<Type>::value) {
            if (!GlobalGPUManager.AddIntInit) {
                GlobalGPUManager.initAddInt();
            }
            computeState = GlobalGPUManager.AddIntCompute;
        } else if constexpr (std::is_floating_point<Type>::value) {
            if (!GlobalGPUManager.AddFloatInit) {
                GlobalGPUManager.initAddFloat();
            }
            computeState = GlobalGPUManager.AddFloatCompute;
        } else {
            std::cerr << "MatrixH: Type not supported" << "\n";
        }
        

        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
        
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
        
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:1];
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:2];
        [commandEncoder setComputePipelineState:computeState];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    template<int dimsB>
    MatrixH<dims, Type> operator+(const MatrixH<dimsB, Type> &other) requires (dims > dimsB) {
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedAddCPU(result, other); } else { BroadcastedAdd(result, other); }
        return result;
    }
    
    template<int dimsB>
    MatrixH<dimsB, Type> operator+(const MatrixH<dimsB, Type> &other) requires (dims < dimsB) {
        MatrixH<dimsB, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedAddCPU(result, other); } else { BroadcastedAdd(result, other); }
        return result;
    }
    
    MatrixH<dims, Type> operator+(const Type value) const {
        MatrixH<dims, Type> result;
        
        if (total_size < 10) {
            auto other = MatrixH<0, Type>(value, false);
            BrodcastedAddCPU(result, other);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            BroadcastedAdd(result, other); }
        return result;
    }
    
    friend MatrixH<dims, Type> operator+(Type value, const MatrixH<dims, Type>& mat) {
        MatrixH<dims, Type> result;
        
        
        if (mat.total_size  < 10) {
            auto other = MatrixH<0, Type>(value, false);
            mat.BroadcastedAdd(result, other);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            mat.BroadcastedAdd(result, other);
        }
        return result;
    }
    
    MatrixH<dims, Type> operator+(const MatrixH<dims, Type> &other) {
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) {
            if (total_size == other.total_size) {
                AddSameCPU(result, other);
            } else {
                BrodcastedAddCPU(result, other);
            }
        } else if (total_size == other.total_size && !(flags & NON_CONTIGUOUS_FLAG)) {
            result.buffer = new Type[effectiveBufferSize()];
            result.total_size = total_size;
            result.buildMetalBuffer();
            memcpy(result.shape, shape, sizeof(size_m) * dims);
            memcpy(result.strides, strides, sizeof(size_m) * dims);
            Add(result, other);
        } else {
            BroadcastedAdd(result, other);
        }
        
//        id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
//        
//        id<MTLBuffer> buffer1 = [metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        
//        id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
//        id<MTLFunction> func1;
//        
//        if constexpr (std::is_integral<Type>::value) {
//            func1 = [lib newFunctionWithName:@"AddGPU_I"];
//        } else if constexpr (std::is_floating_point<Type>::value) {
//            func1 = [lib newFunctionWithName:@"AddGPU_F"];
//        } else {
//            func1 = [lib newFunctionWithName:@"AddGPU_C"];
//        }
//        
//        
//        NSError *error = nil;
//        id<MTLComputePipelineState> computeState = [metalDevice newComputePipelineStateWithFunction:func1 error:&error];
//        
//        if (error) {
//            NSLog(@"Adder: %@", error.localizedDescription);
//        }
//        
//        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//        
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//        [commandEncoder setComputePipelineState:computeState];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
        
        return result;
    }
    
    template<int dimsB>
    MatrixH<dims, Type> operator*(const MatrixH<dimsB, Type> &other) requires (dims > dimsB) {
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedMulCPU(result, other); } else { BrodcastedMul(result, other); }
        return result;
    }
    
    template<int dimsB>
    MatrixH<dimsB, Type> operator*(const MatrixH<dimsB, Type> &other) requires (dims < dimsB) {
        MatrixH<dimsB, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedMulCPU(result, other); } else { BrodcastedMul(result, other); }
        return result;
    }
    
    MatrixH<dims, Type> operator*(const Type value) const {
        MatrixH<dims, Type> result;
        
        if (total_size < 10) {
            MulConstCPU(result, value);
        }
        else {
            auto other = MatrixH<0, Type>(value, true);
            BrodcastedMul(result, other);
        }
        return result;
    }
    friend MatrixH<dims, Type> operator*(Type value, const MatrixH<dims, Type>& mat) {
        MatrixH<dims, Type> result;
        if (mat.total_size < 10) {
            mat.MulConstCPU(result, value); }
        else {
            auto other = MatrixH<0, Type>(value, true);
            mat.BrodcastedMul(result, other);
        }
        return result;
    }
    

    MatrixH<dims, Type> operator*(const MatrixH<dims, Type>& other) const {
        if (FAST_MAX(total_size, other.total_size) < 10) {
            MatrixH<dims, Type> result;
            if (total_size == other.total_size) {
                MulSameCPU(result, other);
            } else {
                BrodcastedMulCPU(result, other);
            }
            return result;
        }
        else if (total_size == other.total_size) {
//            return MulMat(other);
            MatrixH<dims, Type> result;
            BrodcastedMul(result, other);
            return result;
        } else {
            MatrixH<dims, Type> result;
            BrodcastedMul(result, other);
            return result;
        }
    }
    // Division
    
    template<int dimsB>
    MatrixH<dims, Type> operator/(const MatrixH<dimsB, Type> &other) requires (dims > dimsB) {
        
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedDivCPU(result, other); } else { BrodcastedDiv(result, other);}
        return result;
    }
    
    template<int dimsB>
    MatrixH<dimsB, Type> operator/(const MatrixH<dimsB, Type> &other) requires (dims < dimsB) {
        MatrixH<dimsB, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) { BrodcastedDivCPU(result, other); } else { BrodcastedDiv(result, other);}
        return result;
    }
    
    template<int dimsB>
    MatrixH<dimsB, Type> operator/(const MatrixH<dimsB, Type> &other) requires (dims == dimsB) {
        MatrixH<dimsB, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10) {
            if (total_size == other.total_size) {
                DivSameCPU(result, other);
            } else {
                BrodcastedDivCPU(result, other);
            }
        } else { BrodcastedDiv(result, other);
        }
        return result;
    }
    
    MatrixH<dims, Type> operator/(const Type value) {
        MatrixH<dims, Type> result;
        
        if (total_size < 10) {
            DivConstCPU(result, value);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            BrodcastedDiv(result, other);
        }
        return result;
    }
    
    friend MatrixH<dims, Type> operator/(Type value, const MatrixH<dims, Type>& mat) {
        MatrixH<dims, Type> result;

        if (mat.total_size < 10) {
            mat.DivConstRevCPU(result, value);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            other.BrodcastedDiv(result, mat);
        }
        
        return result;
    }
    
    // Subtraction
    
    template<int dimsB>
    MatrixH<dims, Type> operator-(const MatrixH<dimsB, Type> &other) requires (dims > dimsB) {
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10)  { BrodcastedSubCPU(result, other); } else { BrodcastedSub(result, other);}
        return result;
    }
    
    template<int dimsB>
    MatrixH<dimsB, Type> operator-(const MatrixH<dimsB, Type> &other) requires (dims < dimsB) {
        MatrixH<dimsB, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10)  { BrodcastedSubCPU(result, other); } else { BrodcastedSub(result, other);}
        return result;
    }
    
    template<int dimsB>
    MatrixH<dims, Type> operator-(const MatrixH<dimsB, Type> &other) requires (dims == dimsB) {
        MatrixH<dims, Type> result;
        if (FAST_MAX(total_size, other.total_size) < 10)  {
            if (total_size == other.total_size) {
                SubSameCPU(result, other);
            } else {
                BrodcastedSubCPU(result, other);
            }
        
        } else {
            BrodcastedSub(result, other);
        }
        return result;
    }
    
    MatrixH<dims, Type> operator-(const Type value) {
        MatrixH<dims, Type> result;
        
        if (total_size < 10) {
            SubConstCPU(result, value);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            BrodcastedSub(result, other);
        }
        return result;
    }
    friend MatrixH<dims, Type> operator-(Type value, const MatrixH<dims, Type>& mat) {
        MatrixH<dims, Type> result;
        
        if (mat.total_size < 10) {
            mat.SubConstRevCPU(result, value);
        } else {
            auto other = MatrixH<0, Type>(value, true);
            other.BrodcastedSub(result, mat);
        }
        return result;
    }

    
    template <size_t D = dims, typename = std::enable_if_t<(D == 1)>>
    Type& operator[] (int i) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        return buffer[i * strides[0]];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 2)>>
    Type& operator[] (int i, int j) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        return buffer[strides[0] * i + strides[1] * j];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 3)>>
    Type& operator[] (int i, int j, int k) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (k < 0) {
            k = shape[2] + k;
        }
        if (k >= shape[2]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        return buffer[strides[0] * i + strides[1] * j + strides[2] * k];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 4)>>
    Type& operator[] (int i, int j, int k, int l) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (k < 0) {
            k = shape[2] + k;
        }
        if (k >= shape[2]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (l < 0) {
            l = shape[3] + l;
        }
        if (l >= shape[3]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        return buffer[strides[0] * i + strides[1] * j + strides[2] * k + strides[3] * l];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 5)>>
    Type& operator[] (int i, int j, int k, int l, int w) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (k < 0) {
            k = shape[2] + k;
        }
        if (k >= shape[2]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (l < 0) {
            l = shape[3] + l;
        }
        if (l >= shape[3]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (w < 0) {
            w = shape[4] + w;
        }
        if (w >= shape[4]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        return buffer[strides[0] * i + strides[1] * j + strides[2] * k + strides[3] * l + strides[4] * w];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 1)>>
    inline Type& us(int i) const {
        return buffer[i * strides[0]];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 2)>>
    inline Type& us(int i, int j) const {
        return buffer[strides[0] * i + strides[1] * j];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 3)>>
    inline Type& us(int i, int j, int k) const {
        return buffer[strides[0] * i + strides[1] * j + strides[2] * k];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D == 4)>>
    inline Type& us (int i, int j, int k, int l) const {
        return buffer[strides[0] * i + strides[1] * j + strides[2] * k + strides[3] * l];
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D > 1)>>
    MatrixH<dims-1, Type> operator[] (int i) const {
#ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
#endif
//        MatrixH<dims-1, Type> result;
//        result.total_size = accumul(1, dims);
//        std::memcpy(result.shape, shape + 1, sizeof(size_m) * (dims-1));
//        result.buffer = buffer + result.total_size * i;
//        flags |= OWNERSHIP_FLAG;
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
        
        MatrixH<dims-1, Type> slicedMat;
        slicedMat.buffer = buffer + strides[0] * i;
        memcpy(slicedMat.strides, strides + 1, (dims-1) * sizeof(size_m));
        memcpy(slicedMat.shape, shape + 1, (dims-1) * sizeof(size_m));
        slicedMat.total_size = accumul(1, dims);
        if (slicedMat.total_size > 10) {
            slicedMat.buildMetalBuffer();
        }
        
        slicedMat.flags |= NON_OWNERSHIP_FLAG;
        
        return slicedMat;
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D > 2)>>
    MatrixH<dims-2, Type> operator[] (int i, int j) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        MatrixH<dims-2, Type> slicedMat;
        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j;
        memcpy(slicedMat.strides, strides + 2, (dims-2) *sizeof(size_m));
        memcpy(slicedMat.shape, shape + 2, (dims-2) *sizeof(size_m));
        slicedMat.total_size = accumul(2, dims);
        if (slicedMat.total_size > 10) {
            slicedMat.buildMetalBuffer();
        }
        slicedMat.flags |= NON_OWNERSHIP_FLAG;
        return slicedMat;
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D > 3)>>
    MatrixH<dims-3, Type> operator[] (int i, int j, int k) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (k < 0) {
            k = shape[2] + k;
        }
        if (k >= shape[2]) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif
        MatrixH<dims-3, Type> slicedMat;
        
        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j + strides[2] * k;
        memcpy(slicedMat.strides, strides + 3, (dims-3) *sizeof(size_m));
        memcpy(slicedMat.shape, shape + 3, (dims-3) *sizeof(size_m));
        slicedMat.total_size = accumul(3, dims);
        if (slicedMat.total_size > 10) {
            slicedMat.buildMetalBuffer();
        }
        slicedMat.flags |= NON_OWNERSHIP_FLAG;
        return slicedMat;
    }
    
    template <size_t D = dims, typename = std::enable_if_t<(D > 4)>>
    MatrixH<dims-4, Type> operator[] (int i, int j, int k, int l) const {
    #ifdef SAFE_MODE
        if (i < 0) {
            i = shape[0] + i;
        }
        if (i >= shape[0] || i < 0) { // Added < 0 catch in case abs(negative index) > shape
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (j < 0) {
            j = shape[1] + j;
        }
        if (j >= shape[1] || j < 0) {
            throw std::invalid_argument( "Index Out Of range" );
        }
        
        if (k < 0) {
            k = shape[2] + k;
        }
        if (k >= shape[2] || k < 0) {
            throw std::invalid_argument( "Index Out Of range" );
        }

        if (l < 0) {
            l = shape[3] + l;
        }
        if (l >= shape[3] || l < 0) {
            throw std::invalid_argument( "Index Out Of range" );
        }
    #endif // SAFE_MODE

        MatrixH<dims-4, Type> slicedMat;
        
        // Offset the buffer pointer by 4 strides
        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j + strides[2] * k + strides[3] * l;
        
        // Copy the remaining strides and shapes, shifting the pointer by 4
        memcpy(slicedMat.strides, strides + 4, (dims-4) * sizeof(size_m));
        memcpy(slicedMat.shape, shape + 4, (dims-4) * sizeof(size_m));
        
        // Accumulate from the 4th index onward
        slicedMat.total_size = accumul(4, dims);
        
        if (slicedMat.total_size > 10) {
            slicedMat.buildMetalBuffer();
        }
        
        slicedMat.flags |= NON_OWNERSHIP_FLAG;
        
        return slicedMat;
    }
    
    MatrixH<dims, Type> operator[](struct Range r1) {

        
        return Slice({ {{r1.start, r1.end}} });
    }
    
    MatrixH<dims, Type> operator[](struct Range r1, struct Range r2) {

        
        return Slice({ {{r1.start, r1.end}}, {{ r2.start, r2.end }} });
    }
    
    
    ~MatrixH() {
        #ifdef DestructionLog
        std::cout << "Matrix Destroyed" << "\n";
        #endif

        // HUGE ERROR IN PREVIOUS CODE flags & 0 was always false and buffer wasnt getting deleted
//        if ((flags & 0)) {
//            delete [] buffer;
//            std::cout << "deleted" << "\n";
//        }
        if (!(flags & NON_OWNERSHIP_FLAG)) {
            delete [] buffer;
            buffer = nullptr;
            #ifdef DestructionLog
            std::cout << "deleted" << "\n";
            #endif
        }
    }
    
    MatrixH(const MatrixH<dims, Type>& other) : MatrixBase(dims, dtype_from_type<Type>()) {
#ifdef CopyLog
        std::cout << "Copied" << "\n";
#endif
        // copy constructor doesnt need to delete its buffer as  its called only on uninitlised matricies
//        if () {
            buffer = new Type[other.total_size];
            total_size = other.total_size;
            metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
            }];
//        gradFunc = other.gradFunc;
//        parentNodes = other.parentNodes;
//        } else if (total_size != other.total_size) {
//            // copy constructor doesnt need to delete it
////            if (buffer) {
////                delete [] buffer;
////            }
//            buffer = new Type[other.total_size];
//            total_size = other.total_size;
//
//        }
        flags = other.flags; // FIX: We are allocating new buffer, so we own it. Reset the ownership flags.
        flags &= ~NON_OWNERSHIP_FLAG;
        memcpy(buffer, other.buffer, sizeof(Type) * total_size);
        memcpy(shape, other.shape, sizeof(size_m) * dims);
        memcpy(strides, other.strides, dims * sizeof(size_m));
        tape = other.tape;
    }
    
    MatrixH( MatrixH<dims, Type>&& other) : MatrixBase(dims, dtype_from_type<Type>()) {
#ifdef MoveLog
        std::cout << "Moved" << "\n";
#endif
        
        if (flags & NON_OWNERSHIP_FLAG) {
            *this = (const MatrixH<dims, Type>&) other; // calls copy assignment
            return;
        }
        if (buffer) {
            delete [] buffer;
        }
        buffer = other.buffer;
        flags = other.flags;
        other.buffer = nullptr;
        memcpy(shape, other.shape, dims * sizeof(size_m));
        memcpy(strides, other.strides, dims * sizeof(size_m));
        metalBuffer = other.metalBuffer;
        total_size = other.total_size;
//        gradFunc = std::move(other.gradFunc);
//        parentNodes = std::move(other.parentNodes);
        other.~MatrixH();
    }
    
    // const fill
    MatrixH<dims, Type>& operator=(Type value) {
        if (flags & (1u << 1)) {
            fill_nd_iterative(buffer, shape, strides, dims, value);
        } else {
            std::fill(buffer, buffer+total_size, value);
//            memset(buffer, 0, total_size * sizeof(Type));
        }
        
        return *this;
    }
    
    MatrixH<2, Type>& operator=(const MatrixH<1, Type>& other) requires (dims == 2) {
        
        if (total_size != other.total_size) {
            std::cerr << "Error: MatrixH size mismatch — total_size = " << total_size
                      << ", other.total_size = " << other.total_size << std::endl;
            throw std::runtime_error("Tensor size mismatch in operation");
        }
        if (shape[1] != other.shape[0]) {
            std::cerr << "Error: MatrixH shape mismatch — shape[last] = " << shape[1]
                      << ", other.shape[last] = " << other.shape[0] << std::endl;
            throw std::runtime_error("MatrixH shape mismatch in operation");
            throw;
        }
        
        if ((flags & NON_CONTIGUOUS_FLAG) || (other.flags & NON_CONTIGUOUS_FLAG)) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    buffer[strides[0] * i + strides[1] * j] = other.buffer[j * other.strides[0]];
                }
            }
            return *this;
        } else {

#ifdef CopyLog
            std::cout << "Copy Assignment" << "\n";
#endif
            memcpy(buffer, other.buffer, total_size * sizeof(Type));
            memcpy(shape, other.shape, dims * sizeof(size_m));
            memcpy(strides, other.strides, dims * sizeof(size_m));
//            gradFunc = other.gradFunc;
//            parentNodes = other.parentNodes;
            
            
        }
        
        return *this;
    }
    
    
    // copy assignment
    MatrixH<dims, Type>& operator=(const MatrixH<dims, Type>& other) {
        
        if (&other == this) { }
        // 2. Data Buffer Check (Same Underlying Data)
        // If both point to the same memory buffer, copying is redundant.
        else if (this->buffer == other.buffer) {
            return *this;
        }
        else if (total_size == other.total_size) {
            if ((flags & NON_CONTIGUOUS_FLAG) || (other.flags & NON_CONTIGUOUS_FLAG)) {
                
                size_m indexA[dims];
                size_m indexB[dims];
                for (int gid = 0; gid < total_size; gid++) {
                    int remA = gid;
                    int remB = gid;
                    for (int i =dims-1; i >= 0; i--) {
                        indexA[i] = remA % shape[i];
                        indexB[i] = remB % other.shape[i];
                        remA /= shape[i];
                        remB /= other.shape[i];
                    }
                    
                    int offsetA = dotArray(indexA, strides, dims);
                    int offsetB = dotArray(indexB, other.strides, dims);
                    buffer[offsetA] = other.buffer[offsetB];
                }
                return *this;
            }
#ifdef CopyLog
            std::cout << "Copy Assignment" << "\n";
#endif
            memcpy(buffer, other.buffer, total_size * sizeof(Type));
            memcpy(shape, other.shape, dims * sizeof(size_m));
            memcpy(strides, other.strides, dims * sizeof(size_m));
//            gradFunc = other.gradFunc;
//            parentNodes = other.parentNodes;
            
            
        } else {
#ifdef CopyLog
            std::cout << "Copy Create Assignment" << "\n";
#endif
            if (buffer && !(flags & NON_OWNERSHIP_FLAG)) {
                delete [] buffer;
            }
            flags = other.flags; // FIX: We are allocating new buffer, so we own it. Reset the ownership flags.
            flags &= ~NON_OWNERSHIP_FLAG;
            total_size = other.total_size;
            buffer = new Type[total_size];
            buildMetalBuffer();
            memcpy(buffer, other.buffer, total_size * sizeof(Type));
            memcpy(shape, other.shape, dims * sizeof(size_m));
            memcpy(strides, other.strides, dims * sizeof(size_m));
//            gradFunc = other.gradFunc;
//            parentNodes = other.parentNodes;
        }
        
        return *this;
    }

    
    MatrixH<dims, Type>& operator=(MatrixH<dims, Type>&& other) {
        if (&other == this) { return *this; }
        if (flags & NON_OWNERSHIP_FLAG) {
#ifdef CopyLog
            std::cout << "DONT OWN THE DATA COPYInG INSTEAD \n";
#endif
            *this = (const MatrixH<dims, Type>&) other;
            return *this;
        }
//        else if (total_size == other.total_size) {
//            std::cout << "Copy Assignment" << "\n";
//            memcpy(buffer, other.buffer, total_size * sizeof(Type));
//            memcpy(shape, other.shape, dims * sizeof(size_m));
//        } else {
#ifdef MoveLog
        std::cout << "Move Assignment" << "\n";
#endif
        // WRONGGGGGG
//        if (buffer && (flags & 0)) {
//            delete [] buffer;
//        }
        if (buffer && !(flags & NON_OWNERSHIP_FLAG)) {
            delete [] buffer;
        }
        buffer = other.buffer;
        metalBuffer = other.metalBuffer;
        flags = other.flags;
        other.buffer = nullptr;
        memcpy(shape, other.shape, dims * sizeof(size_m));
        memcpy(strides, other.strides, dims * sizeof(size_m));
        total_size = other.total_size;
//        gradFunc = std::move(other.gradFunc);
//        parentNodes = std::move(other.parentNodes);
//        parentNodes = other.parentNodes; // its a pointer to a vector
        other.~MatrixH();
        return *this;
    }
    
//    template <int d>
//    MatrixH<d, Type>& operator=(MatrixH<d, Type>&& other) {
//
//        this->~MatrixH();
//        return *other;
//    }
    MatrixH<dims, Type>& operator=(const simd_float3 other) {
        
        if ((float*)&other == this->buffer) { }
        else if (total_size == 3) {
            if (flags & NON_CONTIGUOUS_FLAG) {
                size_m indexA[dims];
                for (int gid = 0; gid < total_size; gid++) {
                    int remA = gid;
                    for (int i =dims-1; i >= 0; i--) {
                        indexA[i] = remA % shape[i];
                        remA /= shape[i];
                    }
                    
                    int offsetA = dotArray(indexA, strides, dims);
                    buffer[offsetA] = other[gid];
                }
                return *this;
            }
#ifdef CopyLog
            std::cout << "Copy Assignment" << "\n";
#endif
            memcpy(buffer, &other, total_size * sizeof(Type));
            
            
        } else {
            std::cerr << "Cant Paste SIMD_FLOAT3 with total size of " << total_size << "\n";
            throw;
        }
        
        return *this;
    }
    
    MatrixH<dims, Type> Derivative(MatrixH<dims, Type>& result, int axis, int loopBack) {
        size_t stride = accumul(axis+1, dims);
        size_t max = shape[axis] - 1;
        int lastResolve = loopBack;
        
        if (loopBack == 3) {
            memcpy(result.shape, shape, axis * sizeof(size_m));
            result.shape[axis] = shape[axis] - 1;
            memcpy(result.shape + axis + 1, shape + axis + 1, (dims-axis-1) * sizeof(size_m));
            result.total_size = result.accumul(0, dims);
        } else {
            memcpy(result.shape, shape, sizeof(size_m) * dims);
        }
        result.calcStrides();
        
        if (!result.buffer) {
            if (loopBack == 3) {
                result.buffer = new Type[result.total_size];
                result.buildMetalBuffer();
            } else {
                result.buffer = new Type[total_size];
                result.total_size = total_size;
                result.buildMetalBuffer();
            }
        }
        
        else if (result.total_size != total_size && loopBack != 3) {
            delete [] result.buffer;
            result.buffer = new Type[total_size];
            result.total_size = total_size;
            result.buildMetalBuffer();
        }
                

        
        if (!GlobalGPUManager.DerivativeAllInit) {
            GlobalGPUManager.initDerivativeAll();
        }
        
        int type = 0;
        
        // for treating simd_float2 as 2 floats
        int typeBias = 1;
        
        if constexpr (std::is_integral<Type>::value) {
            if (std::is_unsigned<Type>::value) {
                type = 2;
            } else {
                type = 1;
            }
            
        } else if constexpr (std::is_floating_point<Type>::value) {
            type = 0;
        } else if constexpr (std::is_same<Type, simd_float2>::value) {
            type = 0;
            stride *= 2;
            typeBias *= 2;
        }
        else {
            std::cerr << "MatrixH: Type Not supported" << "\n";
        }
        
        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(typeBias*total_size, 1, 1);
        
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
        [commandEncoder setBytes:&stride length:sizeof(size_t) atIndex:2];
        [commandEncoder setBytes:&max length:sizeof(size_t) atIndex:3];
        [commandEncoder setBytes:&lastResolve length:sizeof(int) atIndex:4];
        
        
        
        [commandEncoder setBytes:&type length:sizeof(int) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.DerivativeAll];
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
    
    void SliceBuffer(Type* outBuff, Type* inBuff, size_t stride, size_t size, uint rows) {
        for (int i = 0; i < rows; i++) {
            memcpy(outBuff + i * size, inBuff + i * stride, size * sizeof(Type));
        }
    }
    
    
    
    void SliceCopy(MatrixH<dims, Type>& slicedMat, std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
        uint index = 0;
        size_t offsets[dims];
        bool firstNonNullStrideFound = false;
        memset(offsets, 0, dims * sizeof(size_m));
        memcpy(slicedMat.shape, shape, dims * sizeof(size_m));
        Type* MemArena;
        
        for (auto i : slice) {
            if (i.has_value()) {
                offsets[index] = i->first;
                size_t stride = slicedMat.accumul(index, dims);
                slicedMat.shape[index] = i->second - i->first;
                size_t size = slicedMat.accumul(index, dims);
                size_t elementStride = slicedMat.accumul(index+1, dims);
                
                uint noOfOp = slicedMat.accumul(0, index);
                if (!firstNonNullStrideFound) {
                    MemArena = new Type[slicedMat.accumul(index, dims)];
                    SliceBuffer(MemArena, buffer + offsets[index] * elementStride, stride, size, noOfOp);
                    firstNonNullStrideFound = true;
                    index++;
                    continue;
                    
                }
                SliceBuffer(MemArena, MemArena + offsets[index] * elementStride, stride, size, noOfOp);
            }
            index++;
        }
        slicedMat.calcStrides();
//        slicedMat.total_size = slicedMat.accumul(0, dims);
        slicedMat.total_size = slicedMat.strides[0] * slicedMat.shape[0];
        memcpy(slicedMat.buffer, MemArena, sizeof(Type) * slicedMat.total_size);
    }
    
    MatrixH<dims, Type> SliceCopy(std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
        uint index = 0;
        size_t offsets[dims];
        bool firstNonNullStrideFound = false;
        
        size_t acc = 1;
        for (auto i : slice) {
            if (i.has_value()) {
                acc *= i->second - i->first;
            }
            else {
                acc *= shape[index];
            }
            index ++;
        }
        
        acc *= accumul(index, dims);
        MatrixH<dims, Type> slicedMat(acc);
        
        index = 0;
        memset(offsets, 0, dims * sizeof(size_m));
        memcpy(slicedMat.shape, shape, dims * sizeof(size_m));
        Type* MemArena;
        
        for (auto i : slice) {
            
            if (i.has_value()) {
                offsets[index] = i->first;
                size_t stride = slicedMat.accumul(index, dims);
                slicedMat.shape[index] = i->second - i->first;
                size_t size = slicedMat.accumul(index, dims);
                
                size_t elementStride = slicedMat.accumul(index+1, dims);
                
                uint noOfOp = slicedMat.accumul(0, index);
                std::cout << "Stride: " << stride << "size: " << size << "no of Opp: " << noOfOp << "\n";
                if (!firstNonNullStrideFound) {
//                    MemArena = new Type[slicedMat.accumul(index, dims)]; // not wrong wrong
                    MemArena = new Type[slicedMat.accumul(0, index+1) * accumul(index+1, dims)];
                    SliceBuffer(MemArena, buffer + offsets[index] * elementStride, stride, size, noOfOp);
                    firstNonNullStrideFound = true;
                    index++;
                    continue;
                    
                }
                SliceBuffer(MemArena, MemArena + offsets[index] * elementStride, stride, size, noOfOp);
            }
            index++;
        }
        slicedMat.calcStrides();
        slicedMat.total_size = acc;
        memcpy(slicedMat.buffer, MemArena, sizeof(Type) * slicedMat.total_size);
        return slicedMat;
    }
    
    MatrixH<dims, Type> Slice(std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
        uint index = 0;
        size_m offsets[dims];
        bool firstNonNullStrideFound = false;
        memset(offsets, 0, dims * sizeof(size_m));
        
#ifdef SAFE_MODE
            if (slice.size() > dims) { std::cerr << "MatrixH: slice has "<< slice.size() << " args for a " << dims << "D matrix "<< "\n"; }
#endif
        
        MatrixH<dims, Type> slicedMat;
        
        for (auto i : slice) {
            if (i.has_value()) {
#ifdef SAFE_MODE
                if (i->second > shape[index]) { std::cerr << "MatrixH: index "<< i->second << " excedes the shape " << shape[index] << "of axis "<< index << "\n"; }
                if (i->first > shape[index]) { std::cerr << "MatrixH: index "<< i->second << " excedes the shape " << shape[index] << "of axis "<< index << "\n"; }
#endif
                slicedMat.shape[index] = i->second - i->first;
                offsets[index] = i->first;
            }
            else {
                slicedMat.shape[index] = shape[index];
            }
            
            index ++;
        }
        memcpy(slicedMat.shape + slice.size(), shape + slice.size(),(dims-slice.size()) * sizeof(size_m));
        memcpy(slicedMat.strides, strides, dims * sizeof(size_m));
        slicedMat.total_size = slicedMat.accumul(0, dims);
        slicedMat.flags |= NON_OWNERSHIP_FLAG;
        slicedMat.flags |= NON_CONTIGUOUS_FLAG;
        slicedMat.buffer = buffer + dotArray(offsets, strides, dims);
        if (slicedMat.total_size > 10) {
            slicedMat.buildMetalBuffer();
        }
        
        return slicedMat;
    }
    
//    void operator=(const MatrixH<dims, Type> &other) {
//        if (buffer && total_size == other.total_size) {
//            memcpy(buffer, other.buffer, other.total_size * sizeof(Type));
//            memcpy(shape, other.shape, dims * sizeof(size_m));
////            delete [] other.buffer;
//            
//        } else {
//            if (buffer) {
//                delete [] buffer;
//            }
//            buffer = new Type[other.total_size];
//            total_size = other.total_size;
//            memcpy(buffer, other.buffer, other.total_size * sizeof(Type));
//            memcpy(shape, other.shape, dims * sizeof(size_m));
//        }
//    }
    
};

using MatF1 = MatrixH<1, float>;

template<int dims, typename Type>
std::string AddNode::handleMatrix(std::shared_ptr<MatrixBase> base_ptr) {
    MatrixH<dims, Type>* mat = reinterpret_cast<MatrixH<dims, Type>*>(base_ptr.get());
//    if (mat->operationNodes->size() > 1) {
//        return (*mat->operationNodes)[0].generateMSL();
//    }
    return "x";
}
template std::string AddNode::handleMatrix<1, float>(std::shared_ptr<MatrixBase>);
template std::string AddNode::handleMatrix<2, float>(std::shared_ptr<MatrixBase>);
//template std::string AddNode::handleMatrix<3, float>(std::shared_ptr<MatrixBase>);
template std::string AddNode::handleMatrix<4, float>(std::shared_ptr<MatrixBase>);
//class matrix {
//public:
//    size_m* shape;
//    char* buffer;
//    int dims = 0;
//    size_t totalsize = 0;
//    
//    template<typename U>
//    void construct(std::initializer_list<typename std::conditional<
//                   std::is_arithmetic_v<U>, U, std::initializer_list<U>>::type> list) {
////        dims += 1;
////        size_m* temp = new size_m[dims];
////        if (shape) { memcpy(temp, shape, (dims-1) * sizeof(size_m)); delete [] shape; }
////        shape = temp;
////        shape[dims-1] = list.size();
////        if constexpr (std::is_arithmetic_v<U>) {
////            totalsize += list.size();
////        } else {
////            for (auto& sub : list) {
////                construct(sub);
////            }
////        }
//    }
//    
//    void printShape() {
//        std::cout << "[ ";
//        for (int i =0; i< dims; i++) {
//            std::cout << shape[i] <<", ";
//        }
//        std::cout << "\n";
//    }
//};



template <int dims, typename Type>
MatrixH<dims, Type> sin(const MatrixH<dims, Type>& mat) {
    return mat.sin();
}

template <int dims, typename Type>
MatrixH<dims, Type> cos(const MatrixH<dims, Type>& mat) {
    return mat.cos();
}

template <int dims, typename Type>
MatrixH<dims, Type> tan(const MatrixH<dims, Type>& mat) {
    return mat.tan();
}

template <int dims, typename Type>
MatrixH<dims, Type> exp(const MatrixH<dims, Type>& mat) {
    return mat.exp();
}

template <int dims, typename Type>
MatrixH<dims, Type> sqrt(const MatrixH<dims, Type>& mat) {
    return mat.sqrt();
}

MatrixH<1, float> simd_matH(simd_float3 inp) {
    auto mat = MatrixH<1, float>(3);
    mat.buffer[0] = inp.x;
    mat.buffer[1] = inp.y;
    mat.buffer[2] = inp.z;
    mat.calcStrides();
    return mat;
}

// MARK: - HAND DETECTOR AI
static MatrixH<1, uint8_t> RED = {255, 0, 0, 255};
static MatrixH<1, uint8_t> GREEN = {0, 255, 255, 255};
static MatrixH<1, float> BLUE = MatrixH<1, float>({89.0, 149.0, 173.0, 255.0}) / 255.0;

class HandDetector {
public:
    void drawHands(MatrixH<3, uint8_t>& frame, MatrixH<3, int>& hands, bool lines) {
        
        for (int i= 0; i < hands.shape[0]; i++) {
            
            for (int j= 0; j < hands.shape[1]; j++) {
                if (i == 0) {
#if !TARGET_OS_IPHONE
                    frame.drawElipse({*hands(i, j, 0), *hands(i, j, 1), 50, 50}, RED);
#endif

#if TARGET_OS_IPHONE
                    frame.drawElipse({*hands(i, j, 0), *hands(i, j, 1), 10, 10}, RED);
#endif
                    
                } else {
#if !TARGET_OS_IPHONE
                    frame.drawElipse({*hands(i, j, 0), *hands(i, j, 1), 50, 50}, GREEN);
#endif

#if TARGET_OS_IPHONE
                    frame.drawElipse({*hands(i, j, 0), *hands(i, j, 1), 10, 10}, GREEN);
#endif
                }
                
            }
            
//            if (lines){
//                for (int j= 0; j < 5; j++) {
//                    for (int k= 0; k< 3; k++) {
//                        drawLine(simd_make_int2(*hands(i, k*5 +j , 0), *hands(i, k*5 +j, 1)), simd_make_int2(*hands(i, (k+1)*5 +j, 0), *hands(i, (k+1)*5 +j, 1)), 0, GREEN);
//                    }
//                }
//            }
            

        }
    }
    
    void detectHands(MatrixH<3, uint8_t>& frame, MatrixH<3, int>& detections, bool all_pts) {
        size_t obsPerFinger;
        size_t width = frame.shape[1];
        size_t height = frame.shape[0];
        
        CVPixelBufferRef pixelBuffer = frame.createPixelBufferFromMat();
        if (!pixelBuffer) {
            NSLog(@"Pixel buffer creation failed.");
            return;
        }
        
        if (all_pts) {obsPerFinger = 20;} else { obsPerFinger = 5;}
        VNDetectHumanHandPoseRequest *handReq = [[VNDetectHumanHandPoseRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
            if (error) {
                NSLog(@"Hand detection error: %@", error);
                return;
            }
            NSArray<VNHumanHandPoseObservation *> *observations = request.results;
            
            
            if (detections.total_size !=  observations.count * obsPerFinger * 2) {
                delete [] detections.buffer;
                detections.buffer = new int[observations.count * obsPerFinger * 2];
                detections.shape[0] = observations.count;
                detections.shape[1] = obsPerFinger;
                detections.shape[2] = 2;
                
                detections.total_size = observations.count * obsPerFinger * 2;
                
            }
            if (!detections.buffer) {
                detections.buffer = new int[observations.count * obsPerFinger * 2];
                detections.shape[0] = observations.count;
                detections.shape[1] = obsPerFinger;
                detections.shape[2] = 2;
                
                detections.total_size = observations.count * obsPerFinger * 2;
            }
            
            

            int index = 0;
            for (VNHumanHandPoseObservation *point in observations) {
                
                NSError *landmarkError = nil;
                VNRecognizedPoint *thumbTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbTip error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving thumb tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 0, 0) = (int)(thumbTip.location.x * width);
                *detections(index, 0, 1) = (int)( ((1.0-thumbTip.location.y)) * height);
                
                
                        // Similarly, you can retrieve other landmarks, e.g.,
                VNRecognizedPoint *indexTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexTip error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 1, 0) = (int)(indexTip.location.x * width);
                *detections(index, 1, 1) = (int)( ((1.0-indexTip.location.y)) * height);
                
                VNRecognizedPoint *middleTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleTip error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 2, 0) = (int)(middleTip.location.x * width);
                *detections(index, 2, 1) = (int)( ((1.0-middleTip.location.y)) * height);
                
                VNRecognizedPoint *ringTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingTip error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 3, 0) = (int)(ringTip.location.x * width);
                *detections(index, 3, 1) = (int)( ((1.0-ringTip.location.y)) * height);
                
                VNRecognizedPoint *pinkyTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleTip error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 4, 0) = (int)(pinkyTip.location.x * width);
                *detections(index, 4, 1) = (int)( ((1.0-pinkyTip.location.y)) * height);
                
                
                if (all_pts) {
                    VNRecognizedPoint *thumbDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 5, 0) = (int)(thumbDip.location.x * width);
                    *detections(index, 5, 1) = (int)( ((1.0-thumbDip.location.y)) * height);
                    
                    VNRecognizedPoint *indexDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexDIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 6, 0) = (int)(indexDip.location.x * width);
                    *detections(index, 6, 1) = (int)( ((1.0-indexDip.location.y)) * height);
                    
                    VNRecognizedPoint *middleDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleDIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 7, 0) = (int)(middleDip.location.x * width);
                    *detections(index, 7, 1) = (int)( ((1.0-middleDip.location.y)) * height);
                    
                    VNRecognizedPoint *ringDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingDIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 8, 0) = (int)(ringDip.location.x * width);
                    *detections(index, 8, 1) = (int)( ((1.0-ringDip.location.y)) * height);
                    
                    VNRecognizedPoint *pinkyDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleDIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 9, 0) = (int)(pinkyDip.location.x * width);
                    *detections(index, 9, 1) = (int)( ((1.0-pinkyDip.location.y)) * height);
                    
                    VNRecognizedPoint *thumbPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbMP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 10, 0) = (int)(thumbPip.location.x * width) ;
                    *detections(index, 10, 1) = (int)( ((1.0-thumbPip.location.y)) * height);
                    
                    VNRecognizedPoint *indexPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexPIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 11, 0) = (int)(indexPip.location.x * width);
                    *detections(index, 11, 1) = (int)( ((1.0-indexPip.location.y)) * height);
                    
                    VNRecognizedPoint *middlePip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddlePIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 12, 0) = (int)(middlePip.location.x * width);
                    *detections(index, 12, 1) = (int)( ((1.0-middlePip.location.y)) * height);
                    
                    VNRecognizedPoint *ringPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingPIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 13, 0) = (int)(ringPip.location.x * width);
                    *detections(index, 13, 1) = (int)( ((1.0-ringPip.location.y)) * height);
                    
                    VNRecognizedPoint *pinkyPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittlePIP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 14, 0) = (int)(pinkyPip.location.x * width);
                    *detections(index, 14, 1) = (int)( ((1.0-pinkyPip.location.y)) * height);
                    
                    VNRecognizedPoint *thumbMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbCMC error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 15, 0) = (int)(thumbMCP.location.x * width) ;
                    *detections(index, 15, 1) = (int)( ((1.0-thumbMCP.location.y)) * height) ;
                    
                    VNRecognizedPoint *indexMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexMCP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 16, 0) = (int)(indexMCP.location.x * width) ;
                    *detections(index, 16, 1) = (int)( ((1.0-indexMCP.location.y)) * height) ;
                    
                    VNRecognizedPoint *middleMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleMCP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 17, 0) = (int)(middleMCP.location.x * width);
                    *detections(index, 17, 1) = (int)( ((1.0-middleMCP.location.y)) * height);
                    
                    VNRecognizedPoint *ringMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingMCP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 18, 0) = (int)(ringMCP.location.x * width);
                    *detections(index, 18, 1) = (int)( ((1.0-ringMCP.location.y)) * height);
                    
                    VNRecognizedPoint *pinkyMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleMCP error:&landmarkError];
                            if (landmarkError) {
                                NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                                continue;
                            }
                    *detections(index, 19, 0) = (int)(pinkyMCP.location.x * width);
                    *detections(index, 19, 1) = (int)( ((1.0-pinkyMCP.location.y)) * height);
                }
                index++;
            }
        }];
        
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
        NSError *error = nil;
        [handler performRequests:@[handReq] error:&error];
        if (error) {
            NSLog(@"Error performing face detection: %@", error);
        }
        
        CVPixelBufferRelease(pixelBuffer);
    }
    
    float relativeDistance(MatrixH<3, int>& detections, int hand) {
        return simd_length(simd_make_float2(*detections(hand, 1, 0) - *detections(hand, 0, 0), *detections(hand, 1, 1) - *detections(hand, 0, 1)));
    }
};

// MARK: - Camera

#if !TARGET_OS_IPHONE
@interface CapReader : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>
#endif

#if TARGET_OS_IPHONE
@interface CapReader : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate,AVCaptureDepthDataOutputDelegate>
#endif

{
    @public
    int width;
    int height;
    CVPixelBufferRef mGrabbedPixels;
    CVImageBufferRef mCurrentimageBuffer;
    AVCaptureDevice* mCaptureDevice;
    AVCaptureSession* mCaptureSession;
    AVCaptureDeviceInput* mCaptureDeviceInput;
    AVCaptureVideoDataOutput* mCaptureVideoDataOutput;
    NSCondition* mHasNewFrame;

#if TARGET_OS_IPHONE
    //CVImageBufferRef == CVPixelBufferRef the samething 😱😱
    AVCaptureDepthDataOutput* mDepthDataOutput;
    CVPixelBufferRef depthBuffer;
    CVImageBufferRef mGrabbedDepthBuffer;
    NSCondition* mHasNewDepthFrame;
#endif
}
-(id) initWithCam:(int)CamNo;
-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection;


#if TARGET_OS_IPHONE
-(void) depthDataOutput:(AVCaptureDepthDataOutput *)output didOutputDepthData:(AVDepthData *)depthData timestamp:(CMTime)timestamp connection:(AVCaptureConnection *)connection;
- (void) getDepth:(MatrixH<2, float16_t >&) depthFrame;
#endif

-(void) read:(MatrixH<3, uint8_t>&) frame;
-(void) stopCap;
@end

@implementation CapReader

- (id)initWithCam:(int)CamNo {
    self = [super init];
    NSError* error = nil;
#if !TARGET_OS_IPHONE
    if (CamNo == 0) {
        mCaptureDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    } else {
        mCaptureDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeExternal mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    }
#endif

#if TARGET_OS_IPHONE
    mCaptureDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInLiDARDepthCamera
                                                        mediaType:AVMediaTypeVideo
                                                         position:AVCaptureDevicePositionBack];
    
#endif
    
    mCaptureDeviceInput =  [[AVCaptureDeviceInput alloc] initWithDevice:mCaptureDevice error:&error];
    mCaptureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    
    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", DISPATCH_QUEUE_SERIAL);
    [mCaptureVideoDataOutput setSampleBufferDelegate:self queue:queue];
    
    NSMutableDictionary *pixelBufferOptionsL = [mCaptureVideoDataOutput.videoSettings mutableCopy];
    
    width = 0;
    height = 0;
    while ( true ) {
        // auto matching
        pixelBufferOptionsL[(id)kCVPixelBufferWidthKey]  = @(1.0*width);
        pixelBufferOptionsL[(id)kCVPixelBufferHeightKey] = @(1.0*height);
        mCaptureVideoDataOutput.videoSettings = pixelBufferOptionsL;

        // compare matched size and my options
        CMFormatDescriptionRef format = mCaptureDevice.activeFormat.formatDescription;
        CMVideoDimensions deviceSize = CMVideoFormatDescriptionGetDimensions(format);
        if ( deviceSize.width == width && deviceSize.height == height ) {
            break;
        }

        // fit my options to matched size
        width = deviceSize.width;
        height = deviceSize.height;
    }
    
    std::cout << "Initilizing Cam with Width: " << width << " and Height: " << height << "\n";
    
    OSType pixelFormat;
    pixelFormat = kCVPixelFormatType_32BGRA;
    
    NSDictionary *pixelBufferOptions;
    if (width > 0 && height > 0) {
        pixelBufferOptions =
            @{
                (id)kCVPixelBufferWidthKey:  @(1.0*width),
                (id)kCVPixelBufferHeightKey: @(1.0*height),
                (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
            };
    } else {
        pixelBufferOptions =
            @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
            };
    }
    
    mCaptureVideoDataOutput.videoSettings = pixelBufferOptions;
    mCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = YES;
    
    mCaptureSession = [[AVCaptureSession alloc] init];
#if !TARGET_OS_IPHONE
    mCaptureSession.sessionPreset = AVCaptureSessionPresetMedium;
    [mCaptureSession addInput: mCaptureDeviceInput];
    [mCaptureSession addOutput: mCaptureVideoDataOutput];
#endif
#if TARGET_OS_IPHONE
    
    [mCaptureSession beginConfiguration];
    mCaptureSession.sessionPreset = AVCaptureSessionPresetInputPriority;

    // 3. Add Input FIRST (needed to configure the device)
    if ([mCaptureSession canAddInput:mCaptureDeviceInput]) {
        [mCaptureSession addInput:mCaptureDeviceInput];
    }

    // 4. Find a format that supports BOTH Video and Depth
    AVCaptureDeviceFormat *bestFormat = nil;
    AVCaptureDeviceFormat *bestDepthFormat = nil;

    for (AVCaptureDeviceFormat *format in mCaptureDevice.formats) {
        // We only want formats that support Depth
        NSArray *supportedDepthFormats = format.supportedDepthDataFormats;
        if (supportedDepthFormats.count > 0) {
            CMVideoDimensions dim = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
            
            // Target a specific resolution.
            // NOTE: High-res video (1080p) + Depth is expensive.
            // 640x480 (VGA) is the standard safe choice for smooth Depth+Video.
            if (dim.width == 1920) {
                bestFormat = format;
                
                // Pick the efficient depth format (usually kCVPixelFormatType_DepthFloat16 or 32)
                bestDepthFormat = supportedDepthFormats[0];
                break;
            }
        }
    }

    // 5. Apply the Format
    if (bestFormat) {
        if ([mCaptureDevice lockForConfiguration:nil]) {
            mCaptureDevice.activeFormat = bestFormat;
            mCaptureDevice.activeDepthDataFormat = bestDepthFormat; // You must set this explicitly!
            
            // Optional: Force Frame Rate (e.g., 30 FPS)
            mCaptureDevice.activeVideoMinFrameDuration = CMTimeMake(1, 30);
            mCaptureDevice.activeVideoMaxFrameDuration = CMTimeMake(1, 30);
            
            [mCaptureDevice unlockForConfiguration];
            NSLog(@"✅ Manual Format Configured: %@", bestFormat);
        }
    } else {
        NSLog(@"❌ ERROR: No suitable Depth+Video format found!");
    }

    // --- CRITICAL CHANGE END ---

    // 6. Setup Video Output (Same as your code)
    mCaptureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
//    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", DISPATCH_QUEUE_SERIAL);
    [mCaptureVideoDataOutput setSampleBufferDelegate:self queue:queue];
    // ... set videoSettings ...
    [mCaptureSession addOutput:mCaptureVideoDataOutput];

    // 7. Setup Depth Output
    mDepthDataOutput = [[AVCaptureDepthDataOutput alloc] init];
    [mDepthDataOutput setDelegate:self callbackQueue:queue];
    mDepthDataOutput.alwaysDiscardsLateDepthData = YES;
    mDepthDataOutput.filteringEnabled = YES; // Start with raw data for speed
    NSDictionary *videoSettings = @{
        (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
    };
    mCaptureVideoDataOutput.videoSettings = videoSettings;
    if ([mCaptureSession canAddOutput:mDepthDataOutput]) {
        [mCaptureSession addOutput:mDepthDataOutput];
        
        // Check connection status immediately
        AVCaptureConnection *depthConn = [mDepthDataOutput connectionWithMediaType:AVMediaTypeDepthData];
        depthConn.enabled = YES;
        NSLog(@"Depth Connection Active: %d", depthConn.active); // This should now be 1
    }

    [mCaptureSession commitConfiguration];
//    [mCaptureSession startRunning];
//    
//    mDepthDataOutput = [[AVCaptureDepthDataOutput alloc] init];
////    dispatch_queue_t depthQueue = dispatch_queue_create("depthQueue", DISPATCH_QUEUE_SERIAL);
//    [mDepthDataOutput setDelegate:self callbackQueue:queue];
//    // Optional: Configure depth data output settings, for example:
//    mDepthDataOutput.alwaysDiscardsLateDepthData = YES;
//
//    
//
//    if ([mCaptureSession canAddOutput:mDepthDataOutput]) {
//        [mCaptureSession addOutput:mDepthDataOutput];
//        
//        AVCaptureConnection *depthConnection = [mDepthDataOutput connectionWithMediaType:AVMediaTypeDepthData];
//        depthConnection.enabled = true;
////        depthConnection.depthDataDeliveryEnabled = YES;
//        if (depthConnection) {
//            NSLog(@"%@", depthConnection);
//        }
//    }
    mHasNewDepthFrame =  [[NSCondition alloc] init];
#endif
    [mCaptureSession startRunning];
    mHasNewFrame =  [[NSCondition alloc] init];
    return self;
}

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    
    (void)output;
    (void)sampleBuffer;
    (void)connection;
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVBufferRetain(imageBuffer);
    [mHasNewFrame lock];
    CVBufferRelease(mCurrentimageBuffer);
    mCurrentimageBuffer = imageBuffer;
    [mHasNewFrame broadcast];
    [mHasNewFrame unlock];
}

#if TARGET_OS_IPHONE

- (void) depthDataOutput:(AVCaptureDepthDataOutput *)output didOutputDepthData:(AVDepthData *)depthData timestamp:(CMTime)timestamp connection:(AVCaptureConnection *)connection {
    // Process depthData here.
    // For example, you can convert the depth data to a CVPixelBuffer:
    CVPixelBufferRef depthBufferLocal = [depthData depthDataMap];
//    std::cout << "Sptth:" << ([depthData depthDataType] == kCVPixelFormatType_DepthFloat32) << "\n";
    NSLog(@"%b", [depthData depthDataType] == kCVPixelFormatType_DepthFloat16);
    CVBufferRetain(depthBufferLocal);
    
    // Synchronize with your own processing (e.g., copying data)
    [mHasNewDepthFrame lock];
    CVBufferRelease(depthBuffer);
    depthBuffer = depthBufferLocal;
    // Process depthBuffer as needed...
    [mHasNewDepthFrame broadcast];
    [mHasNewDepthFrame unlock];
}

- (void)                getDepth:(MatrixH<2, float16_t>&) depthFrame {
    [mHasNewDepthFrame lock];
    if (mGrabbedDepthBuffer) {
        CVBufferRelease(mGrabbedDepthBuffer);
    }
    if ([mHasNewDepthFrame waitUntilDate:[NSDate dateWithTimeIntervalSinceNow:1]]) {
        mGrabbedDepthBuffer = CVPixelBufferRetain(depthBuffer);
    }
    [mHasNewDepthFrame unlock];
    
    if (! mGrabbedDepthBuffer ) {
        return;
    }
    CVPixelBufferLockBaseAddress(mGrabbedDepthBuffer, kCVPixelBufferLock_ReadOnly);
    if (!depthFrame.buffer) {
        depthFrame.buffer = new float16_t[width * height];
        depthFrame.shape[0] = height;
        depthFrame.shape[1] = width;
        depthFrame.calcStrides();
        depthFrame.total_size = width * height;
        depthFrame.buildMetalBuffer();
        
    } else if (depthFrame.shape[0] != height || depthFrame.shape[1] != width) {
        if (!(depthFrame.flags & NON_OWNERSHIP_FLAG)) {delete [] depthFrame.buffer; }
        depthFrame.buffer = new float16_t[width * height];
        depthFrame.shape[0] = height;
        depthFrame.shape[1] = width;
        depthFrame.calcStrides();
        depthFrame.total_size = width * height;
        depthFrame.buildMetalBuffer();
    }
    if (CVPixelBufferGetWidth(mGrabbedDepthBuffer) != width) {
        
        vImage_Buffer srcBuf = {
            .data     = CVPixelBufferGetBaseAddress(mGrabbedDepthBuffer),
            .height   = CVPixelBufferGetHeight(mGrabbedDepthBuffer),
            .width    = CVPixelBufferGetWidth(mGrabbedDepthBuffer),
            .rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedDepthBuffer)
        };
        
        vImage_Buffer dstBuf = {
            .data     = depthFrame.buffer,
            .height   = (unsigned long)height,
            .width    = (unsigned long)width,
            .rowBytes = width * sizeof(float16_t)
        };
        
        vImageScale_Planar16F(&srcBuf, &dstBuf, NULL, kvImageHighQualityResampling);
    } else {

        char* rawPtr = (char*)CVPixelBufferGetBaseAddress(mGrabbedDepthBuffer);
        // all the properties given by Core video methods are in bytes like strides and row bytes so adding them to a typed pointer will lead to issues as it adds + i * sizeof(Type)
        if (CVPixelBufferGetBytesPerRow(mGrabbedDepthBuffer) == depthFrame.shape[1] * sizeof(float16_t)) {
            memcpy(depthFrame.buffer, rawPtr, depthFrame.total_size * sizeof(float16_t));
        } else {
            for (int i = 0; i < height; i++) {
                memcpy(depthFrame.buffer + i * width, rawPtr + i * CVPixelBufferGetBytesPerRow(mGrabbedDepthBuffer), width * sizeof(float16_t));
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(mGrabbedDepthBuffer, kCVPixelBufferLock_ReadOnly);
    CVBufferRelease(mGrabbedDepthBuffer);
    
    mGrabbedDepthBuffer = NULL;
}
#endif

- (void) read:(MatrixH<3, uint8_t>&) frame {
    [mHasNewFrame lock];
    
    if (mGrabbedPixels) {
        CVBufferRelease(mGrabbedPixels);
    }
    if ([mHasNewFrame waitUntilDate:[NSDate dateWithTimeIntervalSinceNow:1]]) {
        mGrabbedPixels = CVBufferRetain(mCurrentimageBuffer);
    }
//    
    [mHasNewFrame unlock];
    

    if (! mGrabbedPixels ) {return;}
    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uint8_t *baseaddress = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);
    size_t height =  CVPixelBufferGetHeight(mGrabbedPixels);
    size_t width = CVPixelBufferGetWidth(mGrabbedPixels);
    
    bool res = false;
    if (rowBytes != 0 && (pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_422YpCbCr8)) {
        
        if (frame.buffer && frame.shape[0] == height && frame.shape[1] == width && frame.shape[2] == 4) {
            frame.total_size = width * height * 4;
            memcpy(frame.buffer, baseaddress, frame.total_size);
        } else {
            frame.shape[0] =  height; frame.shape[1] =  width; frame.shape[2] =  4;
            frame.calcStrides();
            frame.total_size = width * height * 4;
            frame.buffer = new uint8_t[frame.total_size];
            frame.buildMetalBuffer();
            memcpy(frame.buffer, baseaddress, frame.total_size);
        }
        res = true;
               
    } else {
        fprintf(stderr, "MatrixH: rowBytes == 0 or unknown pixel format 0x%08X\n", pixelFormat);
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    
    mGrabbedPixels = NULL;
}

-(void) stopCap {
    [mCaptureSession stopRunning];
}

@end

// Mic Capture

//#if !TARGET_OS_IPHONE
@interface MicReader : NSObject<AVCaptureAudioDataOutputSampleBufferDelegate>
//#endif
{
    @public
    /// The number of samples per frame — the height of the spectrogram.
    int sampleCount;
    
    /// The number of displayed buffers — the width of the spectrogram.
    int bufferCount;
    
    /// Determines the overlap between frames.
    int hopCount;
    
    AudioBufferList mBufferList;
    CMBlockBufferRef mBlockBuffer;
    AVCaptureDevice* mCaptureDevice;
    AVCaptureSession* mCaptureSession;
    AVCaptureDeviceInput* mCaptureDeviceInput;
    AVCaptureAudioDataOutput* mCaptureAudioDataOutput;
    NSCondition* mHasNewFrame;
    dispatch_queue_t SessionQueue;
    dispatch_queue_t CaptureQueue;
    
    // Ring buffer for audio data
    int16_t* audioRingBuffer;
    int ringBufferSize;
    int writeIndex;
    int readIndex;
    BOOL hasNewData;

#if TARGET_OS_IPHONE
    AVCaptureDepthDataOutput* mDepthDataOutput;
    CVPixelBufferRef depthBuffer;
#endif
}
-(id) initWithMic:(int)CamNo;
-(void) configureSession;
-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection;
-(void) read:(MatrixH<1, int16_t>&) frame;
-(void) stopMic;
-(void) dealloc;
@end

@implementation MicReader

- (id)initWithMic:(int)CamNo {
    self = [super init];
    if (!self) return nil;
    
    /// The number of samples per frame — the height of the spectrogram.
    sampleCount = 1024;
    
    /// The number of displayed buffers — the width of the spectrogram.
    bufferCount = 768;
    
    /// Determines the overlap between frames.
    hopCount = 512;
    
    // Initialize ring buffer
    ringBufferSize = sampleCount * 4; // 4x buffer size for safety
    audioRingBuffer = (int16_t*)calloc(ringBufferSize, sizeof(int16_t));
    writeIndex = 0;
    readIndex = 0;
    hasNewData = NO;
    
    // Initialize block buffer reference
    mBlockBuffer = NULL;
    
    // Initialize AudioBufferList
    memset(&mBufferList, 0, sizeof(AudioBufferList));
    
    // Create dispatch queues
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL,
        QOS_CLASS_USER_INITIATED,
        0
    );
    attr = dispatch_queue_attr_make_with_autorelease_frequency(
        attr,
        DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM
    );
    CaptureQueue = dispatch_queue_create("captureQueue", attr);
    
    attr = dispatch_queue_attr_make_with_autorelease_frequency(
        DISPATCH_QUEUE_SERIAL,
        DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM
    );
    SessionQueue = dispatch_queue_create("SessionQueue", attr);
    
    mHasNewFrame = [[NSCondition alloc] init];
    
    // Configure session first
    [self configureSession];
    
    return self;
}

-(void) configureSession {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    
    switch (status) {
        case AVAuthorizationStatusAuthorized: {
            [self setupCaptureSession];
            break;
        }
        case AVAuthorizationStatusNotDetermined: {
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
                if (!granted) {
                    std::cerr << "App requires microphone access.\n";
                } else {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self setupCaptureSession];
                    });
                }
            }];
            return;
        }
        default:
            std::cerr << "App requires microphone access.\n";
            break;
    }
}

-(void) setupCaptureSession {
    NSError* error = nil;
    
    // Get the default microphone
    mCaptureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
    if (!mCaptureDevice) {
        std::cerr << "No microphone device found.\n";
        return;
    }
    
    // Create device input
    mCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:mCaptureDevice error:&error];
    if (!mCaptureDeviceInput || error) {
        std::cerr << "Failed to create device input: " << error.localizedDescription.UTF8String << "\n";
        return;
    }
    
    // Create audio data output
    mCaptureAudioDataOutput = [[AVCaptureAudioDataOutput alloc] init];
#if !TARGET_OS_IPHONE
    // Configure audio settings - CRITICAL: Set sample rate
    NSDictionary* audioSettings = @{
        AVFormatIDKey: @(kAudioFormatLinearPCM),
        AVLinearPCMIsFloatKey: @(NO),
        AVLinearPCMBitDepthKey: @(16),
        AVNumberOfChannelsKey: @(1),
        AVSampleRateKey: @(44100.0) // IMPORTANT: Set sample rate
    };
    

    mCaptureAudioDataOutput.audioSettings = audioSettings;
#endif
    [mCaptureAudioDataOutput setSampleBufferDelegate:self queue:CaptureQueue];
    
    // Create and configure session
    mCaptureSession = [[AVCaptureSession alloc] init];
    [mCaptureSession beginConfiguration];
    
    if ([mCaptureSession canAddInput:mCaptureDeviceInput]) {
        [mCaptureSession addInput:mCaptureDeviceInput];
    } else {
        std::cerr << "Cannot add audio input to session.\n";
        [mCaptureSession commitConfiguration];
        return;
    }
    
    if ([mCaptureSession canAddOutput:mCaptureAudioDataOutput]) {
        [mCaptureSession addOutput:mCaptureAudioDataOutput];
    } else {
        std::cerr << "Cannot add audio output to session.\n";
        [mCaptureSession commitConfiguration];
        return;
    }
    
    [mCaptureSession commitConfiguration];
    [mCaptureSession startRunning];
    
    std::cout << "Initialized Mic with sample buffer: " << sampleCount << " and Buffer Count: " << bufferCount << "\n";
}

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    
    if (!sampleBuffer) return;
    
    [mHasNewFrame lock];
    
    // Clean up previous block buffer
    if (mBlockBuffer) {
        CFRelease(mBlockBuffer);
        mBlockBuffer = NULL;
    }
    
    // Get audio buffer list
    OSStatus status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        NULL,
        &mBufferList,
        sizeof(mBufferList),
        NULL,
        NULL,
        kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        &mBlockBuffer
    );
    
    if (status != noErr) {
        NSLog(@"AudioBufferList retrieval failed: %d", (int)status);
        [mHasNewFrame unlock];
        return;
    }
    
    // Copy data to ring buffer
    if (mBufferList.mNumberBuffers > 0) {
        AudioBuffer* audioBuffer = &mBufferList.mBuffers[0];
        int16_t* audioData = (int16_t*)audioBuffer->mData;
        int numSamples = audioBuffer->mDataByteSize / sizeof(int16_t);
        
        for (int i = 0; i < numSamples; i++) {
            audioRingBuffer[writeIndex] = audioData[i];
            writeIndex = (writeIndex + 1) % ringBufferSize;
        }
        
        hasNewData = YES;
    }
    
    [mHasNewFrame broadcast];
    [mHasNewFrame unlock];
}

- (void) read:(MatrixH<1, int16_t>&) frame {
    [mHasNewFrame lock];
    
    // Wait for new data with timeout
    if (!hasNewData) {
        [mHasNewFrame waitUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
    }
    
    if (hasNewData) {
        // Calculate available samples
        int availableSamples = (writeIndex - readIndex + ringBufferSize) % ringBufferSize;
        int samplesToRead = MIN(availableSamples, sampleCount);
        
        if (samplesToRead > 0) {
            // Resize frame buffer if needed
            if (!frame.buffer || frame.shape[0] != samplesToRead) {
                if (frame.buffer) {
                    delete[] frame.buffer;
                }
                frame.shape[0] = samplesToRead;
                frame.total_size = samplesToRead;
                frame.buffer = new int16_t[frame.total_size];
                frame.buildMetalBuffer();
            }
            
            // Copy data from ring buffer
            for (int i = 0; i < samplesToRead; i++) {
                frame.buffer[i] = audioRingBuffer[readIndex];
                readIndex = (readIndex + 1) % ringBufferSize;
            }
            
            // Update hasNewData flag
            hasNewData = ((writeIndex - readIndex + ringBufferSize) % ringBufferSize) > 0;
        }
    }
    
    [mHasNewFrame unlock];
}

-(void) stopMic {
    if (mCaptureSession && [mCaptureSession isRunning]) {
        [mCaptureSession stopRunning];
    }
}

-(void) dealloc {
    [self stopMic];
    
    if (mBlockBuffer) {
        CFRelease(mBlockBuffer);
        mBlockBuffer = NULL;
    }
    
    if (audioRingBuffer) {
        free(audioRingBuffer);
        audioRingBuffer = NULL;
    }
    
    if (CaptureQueue) {
        CaptureQueue = nil;
    }
    
    if (SessionQueue) {
        SessionQueue = nil;
    }
}
@end





std::ostream& operator<<(std::ostream& os, const Vertex3D& v) {
    os << "Position: (" << v.position.x << ", " << v.position.y << ", " << v.position.z << ")  "
       << "Color: (" << v.colour.x << ", " << v.colour.y << ", " << v.colour.z << ", " << v.colour.w << ")  "
       << "TexCoord: (" << v.textureCoordinates.x << ", " << v.textureCoordinates.y << ")  "
       << "Normal: (" << v.normal.x << ", " << v.normal.y << ", " << v.normal.z << ")";
    return os;
}


//class PointCloud {
//public:
//    MatrixH<1, Point3D> points;
//    id<MTLBuffer> pointsBuffer = nil;
//    
//    simd_float3 position= {0.0, 0.0, 0.0};
//    simd_float3 scale = {1.0, 1.0, 1.0};
//    simd_float3 rotation = {0.0, 0.0, 0.0};
//    
//    bool update = true;
//    
//    
//    static PointCloud fromGrid(size_t rows, size_t cols, simd_float4 colour, simd_float3 start, simd_float3 end) {
//        PointCloud output(rows * cols);
//        simd_float3 diff = end - start;
//        diff.x /= cols;
//        diff.y /= rows;
//        diff.z /= rows * cols;
//        for ( int i = 0; i < rows; i++) {
//            for (int j=0; j < cols; j++) {
////                output.points.buffer[i * cols + j] = {{start.x + j * diff.x, start.y + i * diff.y, start.z + (i * j) * diff.z}, colour};
//            }
//        }
//        return output;
//    }
//    
//    static PointCloud fromImage(simd_float3 start, simd_float3 end) {
//        MatrixH<3, uint8_t> img = MatrixH<3, uint8_t>::fromImage(true);
//        size_t rows =img.shape[0];
//        size_t cols = img.shape[1];
//        PointCloud output(img.shape[0] * img.shape[1]);
//        simd_float3 diff = end - start;
//        diff.x /= cols;
//        diff.y /= rows;
//        diff.z /= rows * cols;
//        for ( int i = 0; i < rows; i++) {
//            for (int j=0; j < cols; j++) {
////                output.points.buffer[i * cols + j] = {{start.x + j * diff.x, start.y + i * diff.y, start.z + (i * j) * diff.z}, img.toSimdFloat4(i, j) / 255};
//            }
//        }
//        return output;
//    }
//    
////    static PointCloud fromModel(simd_float3 start, simd_float3 end) {
////
////        return output;
////    }
//    
//    void buildBuffers(id<MTLDevice> metalDevice) {
//        if (!pointsBuffer || update) {
//            pointsBuffer = [metalDevice newBufferWithLength: points.total_size * sizeof(Point3D) options:MTLResourceStorageModeShared];
//            memcpy([pointsBuffer contents], points.buffer, points.total_size*sizeof(Point3D));
//            
//            std::cout << "buffer built" << "\n";
//        }
//        update = false;
//    }
//};

class ObjMesh {
public:
    MDLMesh* modelIOMesh = nil;
    MTKMesh* metalMesh = nil;
    id<MTLTexture> texture = nil;
    NSMutableArray<id<MTLTexture>> *textures = [NSMutableArray array];
    
    ObjMesh(id<MTLDevice> device, MTKMeshBufferAllocator* allocator, NSString* filename ) {
//        NSURL* meshURL = [[NSURL alloc] initWithString:filename];
        
        NSURL* meshURL = [[NSBundle mainBundle] URLForResource:@"11_7_2024"
                                          withExtension:@"usdz"];
        NSError* error = nil;
        
        MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init ];
        // Store position in attribute[0]
        mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
        mtlVertexDescriptor.attributes[0].offset = 0;
        mtlVertexDescriptor.attributes[0].bufferIndex = 0;

        // Store texture coordinates in attribute[1]
        mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
        mtlVertexDescriptor.attributes[1].offset = offsetof(Vertex3D, colour);
        mtlVertexDescriptor.attributes[1].bufferIndex = 0;

        
        mtlVertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
        mtlVertexDescriptor.attributes[2].offset = offsetof(Vertex3D, textureCoordinates);
        mtlVertexDescriptor.attributes[2].bufferIndex = 0;
        
        mtlVertexDescriptor.attributes[3].format = MTLVertexFormatFloat3;
        mtlVertexDescriptor.attributes[3].offset = offsetof(Vertex3D, normal);
        mtlVertexDescriptor.attributes[3].bufferIndex = 0;
        
        // Set stride to twice the bytes per float2.
        mtlVertexDescriptor.layouts[0].stride = sizeof(Vertex3D);
        mtlVertexDescriptor.layouts[0].stepRate = 1;
        mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
        
        
        
        MDLVertexDescriptor* MeshDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor);
        
        MeshDescriptor.attributes[0].name = MDLVertexAttributePosition;
        MeshDescriptor.attributes[1].name = @"primvars:displayColor";
        MeshDescriptor.attributes[2].name = MDLVertexAttributeTextureCoordinate;
        MeshDescriptor.attributes[3].name = MDLVertexAttributeNormal;
        
        auto asset = [[MDLAsset alloc] initWithURL:meshURL vertexDescriptor:MeshDescriptor bufferAllocator:allocator];
        modelIOMesh = (MDLMesh*)[asset childObjectsOfClass:MDLMesh.class].firstObject;
        [modelIOMesh addNormalsWithAttributeNamed:MDLVertexAttributeNormal creaseThreshold:0.0f];
        metalMesh = [[MTKMesh alloc] initWithMesh:modelIOMesh device:device error:&error];
        
//        NSArray<NSString*> *tryNames = @[@"displayColor", @"primvars:displayColor"];
//        MDLVertexAttributeData *colorData = nil;
//
//        for (NSString *name in tryNames) {
//            colorData = [modelIOMesh vertexAttributeDataForAttributeNamed:name];
//            if (colorData) break;
//        }
//
//        if (colorData) {
//            // colorData.dataStart is a pointer to float4 colors
//            vector_float4 *colors = (vector_float4 *)colorData.dataStart;
//            size_t count = modelIOMesh.vertexCount;
//            for (size_t i = 0; i < count; i++) {
//                // e.g. stash into your own Vertex struct:
////                myVertices[i].color = colors[i];
//                std::cout << colors[i].x << " "<< colors[i].y << " " << colors[i].z << "\n";
//            }
//        } else {
//            NSLog(@"⚠️ No displayColor primvar found on the mesh.");
//        }
        
        NSUInteger count = 0;
        MDLVertexAttributeData *colorData =
        [modelIOMesh vertexAttributeDataForAttributeNamed:@"displayColor"
        ];
        if (!colorData) {
          NSLog(@"⚠️ No displayColor primvar found");
        } else {
          // 2) Read out real 0…1 colour values
          vector_float3 *colors = (vector_float3 *)colorData.dataStart;
          for (NSUInteger i = 0; i < count; i++) {
            vector_float3 c = colors[i];
              std::cout << colors[i].x << " "<< colors[i].y << " " << colors[i].z << "\n";
            // c.x, c.y, c.z are now your R/G/B in [0,1]
          }
        }
        
        MTKTextureLoader* textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];
        [asset loadTextures];
        
        
        
        for (MDLObject *object in asset) {
                if (![object isKindOfClass:[MDLMesh class]]) continue;

                MDLMesh *mesh = (MDLMesh *)object;

                for (MDLSubmesh *submesh in mesh.submeshes) {
                    MDLMaterial *material = submesh.material;
                    if (!material) continue;

                    MDLMaterialProperty *diffuse = [material propertyWithSemantic:MDLMaterialSemanticBaseColor];
                    if (diffuse.type == MDLMaterialPropertyTypeString) {
                        // Get texture file path
                        NSString *textureFileName = diffuse.stringValue;
                        NSURL *textureURL = [meshURL URLByAppendingPathComponent:textureFileName];

                        // Load it into a Metal texture
                        MTKTextureLoader *textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];
                        NSError *error = nil;
                        NSDictionary *options = @{
                            MTKTextureLoaderOptionSRGB : @NO
                        };

                        texture = [textureLoader newTextureWithContentsOfURL:textureURL
                                                                                     options:options
                                                                                       error:&error];
                        if (error) {
                            NSLog(@"Error loading texture: %@", error.localizedDescription);
                        } else {
                            break;// Return first diffuse texture found
                        }
                    }
                }
            }
        NSDictionary *options = @{
            MTKTextureLoaderOptionSRGB : @NO
        };
        texture = [textureLoader newTextureWithContentsOfURL:[[NSBundle mainBundle] URLForResource:@"Texture_4_ffffff"
                                                                                     withExtension:@"png"]
                                                                     options:options
                                                                       error:&error];
        if (meshURL) {
            NSLog(@"NotNull %@", metalMesh);
        }
        
    }
    
};



template<typename T>
class Shape {
public:
    Vertex3D* Verticies = nil;
    int vertexCount;
    
    T* indices = nil;
    int indexCount;
    
    id<MTLBuffer> vertexBuffer = nil;
    id<MTLBuffer> indexBuffer = nil;
    id<MTLTexture> texture = nil;
    
    simd_float3 position= {0.0, 0.0, 0.0};
    simd_float3 scale = {1.0, 1.0, 1.0};
    simd_float3 rotation = {0.0, 0.0, 0.0};
    
    bool update = true;
    bool textured = false;
    bool dragable = false;
    
    MTLPrimitiveType drawType = MTLPrimitiveTypeTriangle;
    
    std::vector<std::function<void(simd_float3)>> transformChangeCallbacks;
    
    Shape(Vertex3D* Verticies, int vertexCount, T* indices, int indexCount){
        this->Verticies = Verticies;
        this->vertexCount = vertexCount;
        this->indices = indices;
        
        for (int i = 0; i < indexCount; i++) {
            std::cout << (int)indices[i] << " ";
        }
        this->indexCount = indexCount;
    }
    void buildBuffers(id<MTLDevice> metalDevice) {
        if (!vertexBuffer || !indexBuffer || update) {
            
//            vertexBuffer = [metalDevice newBufferWithLength:vertexCount * sizeof(Vertex3D) options:MTLResourceStorageModeShared];
//            memcpy([vertexBuffer contents], Verticies, vertexCount*sizeof(Vertex3D));
            vertexBuffer = [metalDevice newBufferWithBytesNoCopy:Verticies length:vertexCount * sizeof(Vertex3D)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
                delete [] pointer;
            }];
            
            indexBuffer = [metalDevice newBufferWithBytesNoCopy:indices length:indexCount * sizeof(T)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
                delete [] pointer;
            }];
            
//            indexBuffer = [metalDevice newBufferWithLength:indexCount * sizeof(T) options:MTLResourceStorageModeShared];
//            
//            memcpy([indexBuffer contents], indices, indexCount*sizeof(T));
            std::cout << "buffer built" << "\n";
        }
        update = false;
    }
    void printV() {
        for (int i = 0; i < vertexCount; i++) {
            std::cout << Verticies[i].position << " ";
        }
        std::cout << " \n";
    }
    void printI() {
        for (int i = 0; i < indexCount; i++) {
            std::cout << indices[i] << " ";
        }
        std::cout << " \n";
    }
    
    bool intersectRay(simd_float4 m, simd_float4 b, simd_float4x4& cam, simd_float4& intersectionPoint, bool InCamSpace) {
        simd_float4x4 transform = Transformer();
        if (drawType == MTLPrimitiveTypeTriangle) {
            // Method 1 Check for every triangle by passing a ray with b = {cursorX, cursorY, 0, 1}
            // loop through verticies make triangle and multiply them with cam to get their view splace coords then take only the x and y component and then see if b lies in the triangle
            for (int i = 0; i < indexCount; i += 3) {
                
                simd_float4 v1 = simd_mul(simd_mul(cam, transform), simd_make_float4(Verticies[ indices[i] ].position, 1.0));
                simd_float4 v2 = simd_mul(simd_mul(cam, transform), simd_make_float4(Verticies[ indices[i+1] ].position, 1.0));
                simd_float4 v3 = simd_mul(simd_mul(cam, transform), simd_make_float4(Verticies[ indices[i+2] ].position, 1.0));
                
                float w1 = v1.w;
                float w2 = v2.w;
                float w3 = v3.w;
                
                v1 = v1 / v1.w;
                v2 = v2 / v2.w;
                v3 = v3 / v3.w;
                
                //  shift b to origin
                simd_float2 t1 = ( v1 - b).xy;
                simd_float2 t2 = ( v2 - b).xy;
                simd_float2 t3 = ( v3 - b).xy;
                
                // check if origin lies in the triangle by taking crossproduct as crossproduct will have same sign as origin is always on same side for all three edges
                float e1 = simd_determinant(simd_matrix_from_rows(t2-t1, t1));
                float e2 = simd_determinant(simd_matrix_from_rows(t3-t2, t2));
                float e3 = simd_determinant(simd_matrix_from_rows(t1-t3, t3));
                
 
                
                if ((e1 > 0 && e2 > 0 && e3 > 0) || (e1 < 0 && e2 < 0 && e3 < 0)) {
                    simd_float2 lm = simd_mul(simd_inverse(simd_matrix((v2-v1).xy, (v3-v1).xy)), b.xy - v1.xy);
                    if (InCamSpace) {
                        v1 = v1 * w1;
                        v2 = v2 * w2;
                        v3 = v3 * w3;
                    }
                    intersectionPoint = (v1 + lm.x * (v2-v1) + lm.y * (v3-v1));
                    return true;
                }
            }
        }
        return false;
    }
    
    void triggerCallbacks() {
        for (auto& callback : transformChangeCallbacks) {
            callback(position);
        }
    }
    
    simd_float4x4 Transformer() {
        return Translation(position) * RotationX( rotation.x) * RotationY( rotation.y) * RotationZ( rotation.z) * Scale( scale );
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
    Triangle(float side, MatrixH<1, float> colour): Shape<uint16>(new Vertex3D[3], 3, new uint16[3]{0, 1, 2}, 3) {  // Allocate indices dynamically
        Verticies[0] = {{0, 0, 0}, colour.toSimdFloat4()};
        Verticies[1] = {{side, 0, 0}, {1, 1, 1, 1}};
        Verticies[2] = {{0, side, 0}, {1, 1, 1, 1}};
    }
};

class Quad: public Shape<uint16> {
public:
    Quad(float s): Shape<uint16>(new Vertex3D[4], 4, new uint16[6], 6) {
        Verticies[0]  = { { -s, -s, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        Verticies[1]  = { { +s, -s, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {1, 0}, { 0.f,  0.f,  1.f } };
        Verticies[2]  = { { +s, +s, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {1, 1}, { 0.f,  0.f,  1.f } };
        Verticies[3]  = { { -s, +s, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 1}, { 0.f,  0.f,  1.f } };

        
        uint16_t indicesC[] = {
             0,  1,  2,  2,  3,  0, /* front */
        };
        memcpy(indices, indicesC, sizeof(uint16_t) * 6);
    }
};

class Circle: public Shape<uint16> {
public:
    Circle(float r, int n, simd_float3 colour): Shape<uint16>(new Vertex3D[ n + 1], n + 1, new uint16[3 * n], 3 * n) {
        Verticies[0] = {{0, 0, 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / n;
        for (int i = 0; i < n; i++) {
            Verticies[1+i] = {{r * cos(theta), r * sin(theta), 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
        
        for (int i = 0; i < n-1; i++) {
            indices[3 * i + 0] = 0;
            indices[3 * i + 1] = 1+ i;
            indices[3 * i + 2] = 1+ i+1;
        }
        indices[3 * (n - 1) + 0] = 0;
        indices[3 * (n - 1) + 1] = 1+ n-1;
        indices[3 * (n - 1) + 2] = 1+ 0;
    }
};

class Pipe: public Shape<uint16> {
public:
    Pipe(simd_float3 dir, float r, int n, simd_float3 colour): Shape<uint16>(new Vertex3D[ 2 * (n + 1) ], 2 * (n + 1), new uint16[3 * n + 6 * n + 3 * n], 3 * n + 6 * n + 3 * n) {
        
        simd_float3 up = {0, 1, 0};
        simd_float3 right = simd_normalize(simd_cross(up, dir));
        up = -simd_normalize(simd_cross(right, dir));
        
        Verticies[0] = {{0, 0, 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        Verticies[n+1] = {dir, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / n;
        simd_float3 resultant;
        for (int i = 0; i < n; i++) {
            resultant = r * right * cos(theta) + r * up * sin(theta);
            
            Verticies[1+i] = { resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            Verticies[1+n+1+i] = { dir + resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
        
        uint16_t offset1 = 3 * n;
        uint16_t offset2 = 3 * n + 6 * n;
        for (int i = 0; i < n-1; i++) {
            indices[3 * i + 0] = 0;
            indices[3 * i + 1] = 1+ i;
            indices[3 * i + 2] = 1+ i+1;
            
            indices[offset1 + 6 * i + 0] = 1 + i;
            indices[offset1 + 6 * i + 1] = n + 1 + i;
            indices[offset1 + 6 * i + 2] = n + 1 + i + 1;
            indices[offset1 + 6 * i + 3] = n + 1 + i + 1;;
            indices[offset1 + 6 * i + 4] = 1 + i + 1;
            indices[offset1 + 6 * i + 5] = 1 + i;
            
            indices[offset2 + 3 * i + 0] = n + 0;
            indices[offset2 + 3 * i + 1] = n + 1 + i;
            indices[offset2 + 3 * i + 2] = n + 1 + i + 1;
        }
        indices[3 * (n - 1) + 0] = 0;
        indices[3 * (n - 1) + 1] = 1+ n-1;
        indices[3 * (n - 1) + 2] = 1+ 0;
        
        indices[offset1 + 6 * (n - 1) + 0] = 1 + (n - 1);
        indices[offset1 + 6 * (n - 1) + 1] = n + 1 + (n - 1);
        indices[offset1 + 6 * (n - 1) + 2] = n + 1;
        indices[offset1 + 6 * (n - 1) + 3] = n + 1;
        indices[offset1 + 6 * (n - 1) + 4] = 1;
        indices[offset1 + 6 * (n - 1) + 5] = 1 + (n - 1);
        
        indices[offset2 + 3 * (n - 1) + 0] = n + 0;
        indices[offset2 + 3 * (n - 1) + 1] = n + 1+ n-1;
        indices[offset2 + 3 * (n - 1) + 2] = n + 1+ 0;
    }
    
    static void updateBuffers(simd_float3 dir, float r, int n, simd_float3 colour, Shape<uint16>& pipe) {
        if (2 * (n + 1) != pipe.vertexCount || 3 * n + 6 * n + 3 * n != pipe.indexCount) {
            std::cout << "n value wrong, cannot expand buffer of already existing shape" << "\n";
            return;
        }
        simd_float3 up = {0, 1, 0};
        simd_float3 right = simd_normalize(simd_cross(up, dir));
        up = -simd_normalize(simd_cross(right, dir));
        
        pipe.Verticies[0] = {{0, 0, 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        pipe.Verticies[n+1] = {dir, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / n;
        simd_float3 resultant;
        for (int i = 0; i < n; i++) {
            resultant = r * right * cos(theta) + r * up * sin(theta);
            
            pipe.Verticies[1+i] = { resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            pipe.Verticies[1+n+1+i] = { dir + resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
    }
};

class Cone: public Shape<uint16> {
public:
    Cone(simd_float3 dir, float r, int n, simd_float3 colour): Shape<uint16>(new Vertex3D[ (n + 1) + 1], (n + 1) + 1, new uint16[3 * n + 3 * n], 3 * n + 3 * n) {
//        buildShape(dir, r, n, colour, Verticies, indices);
    }
    
    void buildShape(simd_float3 dir, float r, int n, simd_float3 colour, Vertex3D* VertexBuffer, uint16 indexBuffer) {
        simd_float3 up = {0, 1, 0};
        simd_float3 right = simd_normalize(simd_cross(up, dir));
        up = -simd_normalize(simd_cross(right, dir));
        
        VertexBuffer[0] = {{0, 0, 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        VertexBuffer[n+1] = {dir, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / n;
        simd_float3 resultant;
        for (int i = 0; i < n; i++) {
            resultant = r * right * cos(theta) + r * up * sin(theta);
            Verticies[1+i] = { resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
        uint16_t offset1 = 3 * n;
        uint16_t offset2 = 3 * n + 6 * n;
        for (int i = 0; i < n-1; i++) {
            indices[3 * i + 0] = 0;
            indices[3 * i + 1] = 1+ i;
            indices[3 * i + 2] = 1+ i+1;
            
            indices[offset1 + 3 * i + 0] = 1 + i;
            indices[offset1 + 3 * i + 1] = n + 1;
            indices[offset1 + 3 * i + 3] = 1 + i + 1;
        }
        indices[3 * (n - 1) + 0] = 0;
        indices[3 * (n - 1) + 1] = 1+ n-1;
        indices[3 * (n - 1) + 2] = 1+ 0;
        
        indices[offset1 + 3 * (n - 1) + 0] = 1 + (n - 1);
        indices[offset1 + 3 * (n - 1) + 1] = n + 1;
        indices[offset1 + 3 * (n - 1) + 2] = 1 + 1;
    }
    
    static void updateBuffers(simd_float3 dir, float r, int n, simd_float3 colour, Shape<uint16>& cone) {
        if ( (n + 1) + 1 != cone.vertexCount || 3 * n + 3 * n != cone.indexCount) {
            std::cout << "n value wrong, cannot expand buffer of already existing shape" << "\n";
            return;
        }
        simd_float3 up = {0, 1, 0};
        simd_float3 right = simd_normalize(simd_cross(up, dir));
        up = -simd_normalize(simd_cross(right, dir));
        
        cone.Verticies[0] = {{0, 0, 0}, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        cone.Verticies[n+1] = {dir, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / n;
        simd_float3 resultant;
        for (int i = 0; i < n; i++) {
            resultant = r * right * cos(theta) + r * up * sin(theta);
            cone.Verticies[1+i] = { resultant, simd_make_float4(colour, 1.0), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
    }
};



class Cube: public Shape<uint16> {
public:
    Cube(float s): Shape<uint16>(new Vertex3D[24], 24, new uint16[36], 36) {

        
        Verticies[0]  = { { -s, -s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        Verticies[1]  = { { +s, -s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        Verticies[2]  = { { +s, +s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        Verticies[3]  = { { -s, +s, +s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        Verticies[4]  = { { +s, -s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { 1.f,  0.f,  0.f } };
        Verticies[5]  = { { +s, -s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { 1.f,  0.f,  0.f } };
        Verticies[6]  = { { +s, +s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { 1.f,  0.f,  0.f } };
        Verticies[7]  = { { +s, +s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { 1.f,  0.f,  0.f } };
        Verticies[8]  = { { +s, -s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f, -1.f } };
        Verticies[9]  = { { -s, -s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f, -1.f } };
        Verticies[10] = { { -s, +s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f, -1.f } };
        Verticies[11] = { { +s, +s, -s },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f, -1.f } };
        Verticies[12] = { { -s, -s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { -1.f, 0.f,  0.f } };
        Verticies[13] = { { -s, -s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { -1.f, 0.f,  0.f } };
        Verticies[14] = { { -s, +s, +s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { -1.f, 0.f,  0.f } };
        Verticies[15] = { { -s, +s, -s },  {0.0f, 0.0f, 1.0f, 1.0}, {0, 0}, { -1.f, 0.f,  0.f } };
        Verticies[16] = { { -s, +s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  1.f,  0.f } };
        Verticies[17] = { { +s, +s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  1.f,  0.f } };
        Verticies[18] = { { +s, +s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  1.f,  0.f } };
        Verticies[19] = { { -s, +s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  1.f,  0.f } };
        Verticies[20] = { { -s, -s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f, -1.f,  0.f } };
        Verticies[21] = { { +s, -s, -s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f, -1.f,  0.f } };
        Verticies[22] = { { +s, -s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f, -1.f,  0.f } };
        Verticies[23] = { { -s, -s, +s },  {1.0f, 0.0f, 0.0f, 1.0}, {0, 0}, { 0.f, -1.f,  0.f } };
        
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




using GeometryNodeHalf = GeometryNode<uint16>;

class TriangleNode: public GeometryNode<uint16> {
public:
    Vertex3D verticesL[3] = {
        {{0,0,0}, {1,1,1,1}},
        {{1,0,0}, {1,1,1,1}},
        {{0,1,0}, {1,1,1, 1}}
    };
    uint16 indicesL[3] = {0, 1, 2};
    TriangleNode(float side, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(new Vertex3D[3], 3, new uint16[3]{0, 1, 2}, 3) {  // Allocate indices dynamically
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        verts[0] = {{0, 0, 0}, colour.toSimdFloat4()};
        verts[1] = {{side, 0, 0}, colour.toSimdFloat4()};
        verts[2] = {{0, side, 0}, colour.toSimdFloat4()};
    }
    
    TriangleNode(float base, float height,MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(new Vertex3D[3], 3, new uint16[3]{0, 1, 2}, 3) {  // Allocate indices dynamically
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        verts[0] = {{-base/2, 0, 0}, colour.toSimdFloat4()};
        verts[1] = {{base/2, 0, 0}, colour.toSimdFloat4()};
        verts[2] = {{0, height, 0}, colour.toSimdFloat4()};
    }
};

class QuadNode: public GeometryNode<uint16> {
public:
    QuadNode(float l, float h, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}, uint8_t centre = 0): GeometryNode<uint16>(new Vertex3D[4], 4, new uint16[6], 6) {
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        if (centre == 0) {
            verts[0]  = { { -l/2, -h/2, 0 },  colour.toSimdFloat4(), {0, 0}, { 0.f,  0.f,  1.f } };
            verts[1]  = { { +l/2, -h/2, 0 },  colour.toSimdFloat4(), {1, 0}, { 0.f,  0.f,  1.f } };
            verts[2]  = { { +l/2, +h/2, 0 },  colour.toSimdFloat4(), {1, 1}, { 0.f,  0.f,  1.f } };
            verts[3]  = { { -l/2, +h/2, 0 },  colour.toSimdFloat4(), {0, 1}, { 0.f,  0.f,  1.f } };
        } else if (centre == 1) {
            verts[0]  = { { 0, -h/2, 0 },  colour.toSimdFloat4(), {0, 0}, { 0.f,  0.f,  1.f } };
            verts[1]  = { { +l, -h/2, 0 },  colour.toSimdFloat4(), {1, 0}, { 0.f,  0.f,  1.f } };
            verts[2]  = { { +l, +h/2, 0 },  colour.toSimdFloat4(), {1, 1}, { 0.f,  0.f,  1.f } };
            verts[3]  = { { 0, +h/2, 0 },  colour.toSimdFloat4(), {0, 1}, { 0.f,  0.f,  1.f } };
        } else if (centre == 2) {
            verts[0]  = { { -l/2, 0, 0 },  colour.toSimdFloat4(), {0, 0}, { 0.f,  0.f,  1.f } };
            verts[1]  = { { +l/2, 0, 0 },  colour.toSimdFloat4(), {1, 0}, { 0.f,  0.f,  1.f } };
            verts[2]  = { { +l/2, +h, 0 },  colour.toSimdFloat4(), {1, 1}, { 0.f,  0.f,  1.f } };
            verts[3]  = { { -l/2, +h, 0 },  colour.toSimdFloat4(), {0, 1}, { 0.f,  0.f,  1.f } };
        }
        
        uint16_t indicesC[] = {
             0,  1,  2,  2,  3,  0, /* front */
        };
        memcpy(indices, indicesC, sizeof(uint16_t) * 6);
    }
    
};

class ArrowNode: public GeometryNode<uint16> {
public:
    MatrixH<1, float> start;
    MatrixH<1, float> vec;
    float mlength;
    float mwidth;
    
    ArrowNode(MatrixH<1, float> _vec, MatrixH<1, float> _start = {0.0, 0.0, 0.0}, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): vec(_vec), start(_start), GeometryNode<uint16>(nil, 0, nil, 0)  {
        mwidth = 0.1;
        auto TriangleObj = TriangleNode(2* mwidth * 1.415, 2*mwidth* (1/1.415), colour);
        TriangleObj.position.y += 1-2*mwidth* (1/1.415);
        
        TriangleObj.buildModelMatrix();
        
        auto QuadObj = QuadNode(mwidth, 1-2*mwidth* (1/1.415), colour, 2);
        
        MatrixH<2, float> transformMat = MatrixH<2, float>::eye(4);
        transformMat.Slice({ {{0,1}}, {{0,3}} }) = -1 * simd_matH(simd_cross(vec.toSimdFloat3(), simd_make_float3(0, 0, 1)));
        transformMat.Slice({ {{1,2}}, {{0,3}} }) = vec;
        transformMat.Slice({ {{2,3}}, {{0,3}} }) = simd_matH(simd_cross(vec.toSimdFloat3(), transformMat.Slice({ {{0,1}}, {{0,3}} }).toSimdFloat3(0)));
        transformMat.Slice({ {{3,4}}, {{0,3}} }) = start;
        
        
        AddNodes(std::move(TriangleObj), std::move(QuadObj));
        memcpy((float*)&modelMatrix, transformMat.buffer, transformMat.total_size * sizeof(float));
        BuildInstances();
    }
    
    void updateNode(MatrixH<1, float> _vec, MatrixH<1, float> _start = {0.0, 0.0, 0.0}, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}) {
        vec = _vec;
        start = _start;
        MatrixH<2, float> transformMat = MatrixH<2, float>::eye(4);
//        transformMat.Slice({ {{0,1}}, {{0,3}} }) = -1 * vec.cross({0.0, 0.0, 1.0});
//        transformMat.Slice({ {{1,2}}, {{0,3}} }) = _vec;
//        transformMat.Slice({ {{2,3}}, {{0,3}} }) = vec.cross(transformMat.Slice({ {{0,1}}, {{0,3}} }).squeeze<1>());
//        transformMat.Slice({ {{3,4}}, {{0,3}} }) = _start;
        transformMat.Slice({ {{0,1}}, {{0,3}} }) = -1 * simd_matH(simd_cross(vec.toSimdFloat3(), simd_make_float3(0, 0, 1)));
        transformMat.Slice({ {{1,2}}, {{0,3}} }) = vec;
        transformMat.Slice({ {{2,3}}, {{0,3}} }) = simd_matH(simd_cross(vec.toSimdFloat3(), transformMat.Slice({ {{0,1}}, {{0,3}} }).toSimdFloat3(0)));
        transformMat.Slice({ {{3,4}}, {{0,3}} }) = start;
        

        memcpy((float*)&modelMatrix, transformMat.buffer, transformMat.total_size * sizeof(float));
        BuildInstances();
    }
    
    ArrowNode(float length, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(nil, 0, nil, 0) {
        mlength = length;
        mwidth = 0.5;
        auto TriangleObj = TriangleNode(1.415, (1/1.415), colour);
        TriangleObj.position.y += mlength*0.5 ;
        
        TriangleObj.buildModelMatrix();
        auto QuadObj = QuadNode(mwidth, mlength);
        
        
        AddNodes(std::move(TriangleObj), std::move(QuadObj));
    }
    
    ArrowNode(float length, float width, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(nil, 0, nil, 0) {
        mlength = length;
        mwidth = width;
        
        auto TriangleObj = TriangleNode(mwidth*1.9*1.415, (2.5*mwidth/1.415), colour);
        TriangleObj.position.y += mlength*0.5 ;
        
        TriangleObj.buildModelMatrix();
        auto QuadObj = QuadNode(mwidth, mlength, colour);
        
        AddNodes(std::move(TriangleObj), std::move(QuadObj));
    }
    
    ArrowNode(simd_float2 vec, float width, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(nil, 0, nil, 0) {
        mlength = simd_length(vec);
        mwidth = width;
        
        auto TriangleObj = TriangleNode(mwidth*1.9*1.415, (2.5*mwidth/1.415), colour);
        TriangleObj.position.y += mlength*0.5 ;
        
        TriangleObj.buildModelMatrix();
        auto QuadObj = QuadNode(mwidth, mlength, colour);
        QuadObj.buildModelMatrix();
        
        AddNodes(std::move(TriangleObj), std::move(QuadObj));

        auto row1 = simd_make_float4( vec.y, vec.x, 0, 0);
        auto row2 = simd_make_float4(-vec.x, vec.y, 0, 0);
        auto row3 = simd_make_float4(0     ,     0, 1, 0);
        auto row4 = simd_make_float4(0     , 0    , 0, 1);
        
        auto mat1 = simd_matrix_from_rows(row1, row2, row3, row4);
        
        modelMatrix = mat1;
        BuildInstances();
        setTailAt(ORIGIN); // includes the
    }
    
    void setTailAt(simd_float3 tail) {
//        setPos(tail + simd_make_float3(0, mlength/2, 0));
        auto col1 = simd_make_float4(1, 0, 0, 0);
        auto col2 = simd_make_float4(0, 1, 0, 0);
        auto col3 = simd_make_float4(0, 0, 1, 0);
        auto col4 = simd_make_float4(tail + simd_make_float3(0, mlength/2, 0), 1);
        
        auto mat1 = simd_matrix(col1, col2, col3, col4);
        modelMatrix = simd_mul(modelMatrix, mat1);
        buildGlobalMatrix();
    }
};

class CircleNode: public GeometryNode<uint16> {
public:
    CircleNode(float r, int n, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(new Vertex3D[ n + 1], n + 1, new uint16[3 * n], 3 * n) {
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        verts[0] = {{0, 0, 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, 1.0}};
        std::cout << verts[0] << "\n";
        float theta = 0;
        float stride = 2 * M_PI / n;
        for (int i = 0; i < n; i++) {
            verts[1+i] = {{r * cos(theta), r * sin(theta), 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, 1.0}};
            theta += stride;
        }
        
        for (int i = 0; i < n-1; i++) {
            indices[3 * i + 0] = 0;
            indices[3 * i + 1] = 1+ i;
            indices[3 * i + 2] = 1+ i+1;
        }
        indices[3 * (n - 1) + 0] = 0;
        indices[3 * (n - 1) + 1] = 1+ n-1;
        indices[3 * (n - 1) + 2] = 1+ 0;
        drawType = MTLPrimitiveTypeTriangle;
    }
};

class SphereNode: public GeometryNode<uint16> {
public:
    SphereNode(float r, int rS, int lS, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(new Vertex3D[rS * lS + 2 ], rS * lS + 2, new uint16[rS * 3 + 6*rS*(lS-1) + rS * 3],rS * 3 + 6*rS*(lS-1) + rS * 3) {
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        // ls Linear subdivisions
        // rs radial subdivisions
        verts[0] = {{0, r, 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, 1.0}};
        verts[vertexCount - 1] = {{0, -r, 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, -1.0}};
        float theta = 0;
        float stride = 2 * M_PI / rS;
        float polarAngle = M_PI / (lS + 1);
//        simd_float3 yV = simd_make_float3(0, r - (2*r/(lS+1)), 0);
        for (int h = 0; h < lS; h++) {
//            float rMag = sqrt(r*r - simd_dot(yV, yV));
            float rMag = r * sin(polarAngle);
            for (int i = 0; i < rS; i++) {
                simd_float3 rV = rMag * simd_make_float3(cos(theta), 0, sin(theta));
                simd_float3 yV = simd_make_float3(0, r * cos(polarAngle), 0);
                verts[1+h*rS + i] = {yV + rV, colour.toSimdFloat4(), {1, 0}, (yV+rV)/r};
                theta += stride;
            }
//            yV -= simd_make_float3(0, (2*r/(lS+1)), 0);
            polarAngle += M_PI / (lS + 1);
        }
        
        MatrixH<3, uint16> indiciesStridedView;
        indiciesStridedView.buffer = indices + 3 * rS;
        indiciesStridedView.shape[0] = lS-1;
        indiciesStridedView.shape[1] = rS;
        indiciesStridedView.shape[2] = 6;
        indiciesStridedView.total_size = 6*rS*(lS-1);
        indiciesStridedView.flags |= NON_OWNERSHIP_FLAG;
        indiciesStridedView.calcStrides();
        
        MatrixH<2, uint16> vertexIndexStridedView = MatrixH<2, uint16>::Range(1, { (unsigned)lS, (unsigned)rS }); // for no. of verticies range starts like 1, 2,3 in a shape shape[0] is latitudes so ican get help in no. verticies during index buffer

        for ( int j=0; j < lS-1; j++ ) {
            

            
            for (int i = 0; i < rS-1; i++) {
                // Top Fan Cap
                indices[3 * i + 0] = 0;
                indices[3 * i + 1] = 1+ i;
                indices[3 * i + 2] = 1+ i+1;
                
                // Quads in the middle
                indiciesStridedView[j, i, 0] = vertexIndexStridedView[j, i];
                indiciesStridedView[j, i, 1] = vertexIndexStridedView[j+1, i];
                indiciesStridedView[j, i, 2] = vertexIndexStridedView[j+1, i+1];
                indiciesStridedView[j, i, 3] = vertexIndexStridedView[j, i];
                indiciesStridedView[j, i, 4] = vertexIndexStridedView[j, i+1];
                indiciesStridedView[j, i, 5] = vertexIndexStridedView[j+1, i+1];
                
                
                // Bottom Fan Cap
                indices[rS * 3 + 6*rS*(lS-1) + 3 * i + 0] = vertexCount-1;
                indices[rS * 3 + 6*rS*(lS-1) + 3 * i + 1] = rS * (lS-1) + 1 + i;
                indices[rS * 3 + 6*rS*(lS-1) + 3 * i + 2] = rS * (lS-1) + 1 + i+1;
            }
            indiciesStridedView[j, rS-1, 0] = vertexIndexStridedView[j   , rS-1];
            indiciesStridedView[j, rS-1, 1] = vertexIndexStridedView[j +1, rS-1];
            indiciesStridedView[j, rS-1, 2] = vertexIndexStridedView[j +1, 0];
            indiciesStridedView[j, rS-1, 3] = vertexIndexStridedView[j   , rS-1];
            indiciesStridedView[j, rS-1, 4] = vertexIndexStridedView[j   , 0];
            indiciesStridedView[j, rS-1, 5] = vertexIndexStridedView[j +1, 0];
            
            

        }
        

        indices[3 * (rS - 1) + 0] = 0;
        indices[3 * (rS - 1) + 1] = 1+ rS-1;
        indices[3 * (rS - 1) + 2] = 1+ 0;
        
        indices[rS * 3 + 6*rS*(lS-1) + 3 * (rS - 1) + 0] = vertexCount-1;
        indices[rS * 3 + 6*rS*(lS-1) + 3 * (rS - 1) + 1] = rS * (lS-1) + 1 + rS-1;
        indices[rS * 3 + 6*rS*(lS-1) + 3 * (rS - 1) + 2] = rS * (lS-1) + 1 + 0;
        
        drawType = MTLPrimitiveTypeTriangle;
    }
};

class IcosphereNode: public GeometryNode<uint16> {
public:
    IcosphereNode(float r, int subdivisions, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}):
        GeometryNode<uint16>(
            new Vertex3D[calculateVertexCount(subdivisions)],
            calculateVertexCount(subdivisions),
            new uint16[calculateIndexCount(subdivisions)],
            calculateIndexCount(subdivisions)
        ) {
        
        // Golden ratio for icosahedron construction
        const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
        const float scale = r / sqrtf(1.0f + t * t);
        
        // Create initial 12 vertices of icosahedron
        simd_float3 baseVertices[12] = {
            {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
            { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
            { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
        };
        
        // Normalize and scale base vertices
        for (int i = 0; i < 12; i++) {
            baseVertices[i] = simd_normalize(baseVertices[i]) * r;
        }
        
        // Create initial 20 triangular faces of icosahedron
        uint16 baseFaces[20][3] = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };
        
        // Store vertices and indices for subdivision
        std::vector<simd_float3> vertices;
        std::vector<uint16> faceIndices;
        
        for (int i = 0; i < 12; i++) {
            vertices.push_back(baseVertices[i]);
        }
        
        for (int i = 0; i < 20; i++) {
            faceIndices.push_back(baseFaces[i][0]);
            faceIndices.push_back(baseFaces[i][1]);
            faceIndices.push_back(baseFaces[i][2]);
        }
        
        // Subdivide triangles
        for (int sub = 0; sub < subdivisions; sub++) {
            std::vector<uint16> newFaces;
            std::map<std::pair<uint16, uint16>, uint16> midpointCache;
            
            for (size_t i = 0; i < faceIndices.size(); i += 3) {
                uint16 v1 = faceIndices[i];
                uint16 v2 = faceIndices[i + 1];
                uint16 v3 = faceIndices[i + 2];
                
                // Get or create midpoint vertices
                uint16 a = getMidpoint(v1, v2, vertices, midpointCache, r);
                uint16 b = getMidpoint(v2, v3, vertices, midpointCache, r);
                uint16 c = getMidpoint(v3, v1, vertices, midpointCache, r);
                
                // Create 4 new triangles
                newFaces.push_back(v1); newFaces.push_back(a); newFaces.push_back(c);
                newFaces.push_back(v2); newFaces.push_back(b); newFaces.push_back(a);
                newFaces.push_back(v3); newFaces.push_back(c); newFaces.push_back(b);
                newFaces.push_back(a);  newFaces.push_back(b); newFaces.push_back(c);
            }
            
            faceIndices = newFaces;
        }
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        // Copy vertices to geometry buffer
        for (size_t i = 0; i < vertices.size(); i++) {
            simd_float3 pos = vertices[i];
            simd_float3 normal = simd_normalize(pos);
            
            // Simple spherical UV mapping
            float u = 0.5f + atan2f(normal.z, normal.x) / (2.0f * M_PI);
            float v = 0.5f - asinf(normal.y) / M_PI;
            
            verts[i] = {
                pos,
                colour.toSimdFloat4(),
                {u, v},
                normal
            };
        }
        
        // Copy indices
        for (size_t i = 0; i < faceIndices.size(); i++) {
            indices[i] = faceIndices[i];
        }
        
        drawType = MTLPrimitiveTypeTriangle;
    }
    
private:
    static int calculateVertexCount(int subdivisions) {
        int faces = 20;
        for (int i = 0; i < subdivisions; i++) {
            faces *= 4;
        }
        // Using Euler's formula for closed mesh: V - E + F = 2
        // For triangulated sphere: E = 3F/2, so V = F/2 + 2
        return faces / 2 + 2;
    }
    
    static int calculateIndexCount(int subdivisions) {
        int faces = 20;
        for (int i = 0; i < subdivisions; i++) {
            faces *= 4;
        }
        return faces * 3;
    }
    
    uint16 getMidpoint(uint16 i1, uint16 i2, std::vector<simd_float3>& vertices,
                       std::map<std::pair<uint16, uint16>, uint16>& cache, float r) {
        // Order indices consistently for cache lookup
        std::pair<uint16, uint16> key = i1 < i2 ? std::make_pair(i1, i2) : std::make_pair(i2, i1);
        
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
        
        // Calculate midpoint and project to sphere surface
        simd_float3 mid = (vertices[i1] + vertices[i2]) * 0.5f;
        mid = simd_normalize(mid) * r;
        
        uint16 index = (uint16)vertices.size();
        vertices.push_back(mid);
        cache[key] = index;
        
        return index;
    }
};

class CylinderNode: public GeometryNode<uint16> {
public:
    CylinderNode(float r, float l, int rS, MatrixH<1, float> colour = {1.0, 1.0, 1.0, 1.0}): GeometryNode<uint16>(new Vertex3D[rS * 2 + 2 ], rS * 2 + 2, new uint16[rS * 3 + 6*rS + rS * 3], rS * 3 + 6*rS + rS * 3) {
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        verts[0] = {{0, 0, 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, -1.0}};
        verts[vertexCount - 1] = {{0, l, 0}, colour.toSimdFloat4(), {0, 0}, {0.0, 0.0, 1.0}};
        float theta = 0;
        float stride = 2 * M_PI / rS;
        simd_float3 axisVec = {0, l, 0};
        for (int i = 0; i < rS; i++) {
            simd_float3 rVhat = r * simd_make_float3(cos(theta), 0, sin(theta));
            verts[1 + i] = {rVhat * r, colour.toSimdFloat4(), {1, 0}, simd_normalize(rVhat + simd_make_float3(0, 1, 0))};
            verts[1 + rS + i] = {rVhat * r + axisVec, colour.toSimdFloat4(), {1, 0}, simd_normalize(rVhat + simd_make_float3(0, 1, 0))};
            theta += stride;
        }

        MatrixH<2, uint16> indiciesStridedView;
        indiciesStridedView.buffer = indices + 3 * rS;
        indiciesStridedView.shape[0] = rS;
        indiciesStridedView.shape[1] = 6;
        indiciesStridedView.total_size = 6*rS;
        indiciesStridedView.flags |= NON_OWNERSHIP_FLAG;
        indiciesStridedView.calcStrides();
        
        MatrixH<2, uint16> vertexIndexStridedView = MatrixH<2, uint16>::Range(1, { 2, (unsigned)rS });


        for (int i = 0; i < rS-1; i++) {
            indices[3 * i + 0] = 0;
            indices[3 * i + 1] = 1+ i;
            indices[3 * i + 2] = 1+ i+1;
            
            indiciesStridedView[i, 0] = vertexIndexStridedView[0, i];
            indiciesStridedView[i, 1] = vertexIndexStridedView[1, i];
            indiciesStridedView[i, 2] = vertexIndexStridedView[1, i+1];
            indiciesStridedView[i, 3] = vertexIndexStridedView[0, i];
            indiciesStridedView[i, 4] = vertexIndexStridedView[0, i+1];
            indiciesStridedView[i, 5] = vertexIndexStridedView[1, i+1];
            
            indices[rS * 3 + 6*rS + 3 * i + 0] = vertexCount-1;
            indices[rS * 3 + 6*rS + 3 * i + 1] = 1+rS + i;
            indices[rS * 3 + 6*rS + 3 * i + 2] = 1+rS + i+1;
        }
        
        // Cylinder Rec face loop back
        indiciesStridedView[rS-1, 0] = vertexIndexStridedView[0 , rS-1];
        indiciesStridedView[rS-1, 1] = vertexIndexStridedView[1, rS-1];
        indiciesStridedView[rS-1, 2] = vertexIndexStridedView[1, 0];
        indiciesStridedView[rS-1, 3] = vertexIndexStridedView[0, rS-1];
        indiciesStridedView[rS-1, 4] = vertexIndexStridedView[0, 0];
        indiciesStridedView[rS-1, 5] = vertexIndexStridedView[1, 0];
        
        // Top Fan Loop Triangle
        indices[3 * (rS - 1) + 0] = 0;
        indices[3 * (rS - 1) + 1] = 1+ rS-1;
        indices[3 * (rS - 1) + 2] = 1+ 0;
        
        // Bottom Fan Loop Back
        indices[rS * 3 + 6*rS + 3 * (rS - 1) + 0] = vertexCount-1;
        indices[rS * 3 + 6*rS + 3 * (rS - 1) + 1] = rS + 1 + rS-1;
        indices[rS * 3 + 6*rS + 3 * (rS - 1) + 2] = rS + 1 + 0;
        
        drawType = MTLPrimitiveTypeTriangle;
    }
};

class LineNode: public GeometryNode<uint16> {
public:
    LineNode(MatrixH<2, float> points): GeometryNode<uint16>(new Vertex3D[2 * points.shape[0]], 2 * points.shape[0], new uint16[(points.shape[0] - 1) * 6], (points.shape[0] - 1) * 6) {
        MatrixH<2, float> vecs;
        points.Derivative(vecs, 0, 3);
//        vecs.print();
        
        auto vecs_normalised = vecs / (vecs * vecs).SumNoRed(1).sqrt();
//        vecs_normalised.print();

        
        MatrixH<2, float> perp = {{0, 1}, {-1, 0}};
        MatrixH<2, float> normals;
        
        normals = perp.Dot(vecs_normalised, true).T();
//        normals.print();
        
        MatrixH<2, float> convNormals = MatrixH<2, float>::zeros({normals.shape[0] + 1, normals.shape[1]});
        normals.conv<1>(convNormals, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
//        convNormals.print();
        
//        (convNormals * convNormals).Sum(1).sqrt().print();
        
        auto convNormals_normalised = convNormals / (convNormals * convNormals).SumNoRed(1);
//        convNormals_normalised.print();
        
        auto verts = MatrixH<2, float>::concat(points, points + convNormals_normalised * 0.1, 1);
        verts.shape[0] = points.shape[0] * 2;
        verts.shape[1] = 2;
        
        verts = MatrixH<2, float>::concat(verts, MatrixH<2, float>::zeros({points.shape[0] * 2, 1}), 1);
//        verts.print();
        
        MatrixH<2, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = points.shape[0] * 2;
        VertexBufferView.shape[1] = 3;
        VertexBufferView.strides[0] = sizeof(Vertex3D) / sizeof(float);
        VertexBufferView.strides[1] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = verts;
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, colour) / sizeof(float);
        VertexBufferView.shape[1] = 4;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = MatrixH<2, float>::repeating({points.shape[0] * 2}, BLUE);
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, normal) / sizeof(float);
        VertexBufferView.shape[1] = 3;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = MatrixH<2, float>::repeating({points.shape[0] * 2}, MatrixH<1, float>{0.0, 0.0, 1.0});

        MatrixH<2, uint16> indexBufferRange = MatrixH<2, uint16>::Range(0, {points.shape[0], 2});
        MatrixH<2, uint16> indexBufferView;
        indexBufferView.buffer = (uint16*)indices;
        indexBufferView.shape[0] = points.shape[0] - 1;
        indexBufferView.shape[1] = 6;
        indexBufferView.calcStrides();
        indexBufferView.flags |= NON_OWNERSHIP_FLAG;
        
        for (int i = 0; i < points.shape[0] - 1; i++) {
            indexBufferView[i, 0] = indexBufferRange[i, 0];
            indexBufferView[i, 1] = indexBufferRange[i, 1];
            indexBufferView[i, 2] = indexBufferRange[i+1, 1];
            indexBufferView[i, 3] = indexBufferRange[i, 0];
            indexBufferView[i, 4] = indexBufferRange[i+1, 0];
            indexBufferView[i, 5] = indexBufferRange[i+1, 1];
        }
        drawType = MTLPrimitiveTypeTriangle;
    }
    
    void UpdatePoints(MatrixH<2, float> points) {
        if (points.shape[0] * 2 != vertexCount) {
            delete [] Verticies;
            delete [] indices;
            
            Verticies = new Vertex3D[points.shape[0] * 2];
            indices = new uint16[(points.shape[0] - 1) * 6];
            
            vertexCount =  2 * points.shape[0];
            indexCount = (points.shape[0] - 1) * 6;
            update = true;
            
            MatrixH<2, uint16> indexBufferRange = MatrixH<2, uint16>::Range(0, {points.shape[0], 2});
            MatrixH<2, uint16> indexBufferView;
            indexBufferView.buffer = (uint16*)indices;
            indexBufferView.shape[0] = points.shape[0] - 1;
            indexBufferView.shape[1] = 6;
            indexBufferView.calcStrides();
            indexBufferView.flags |= NON_OWNERSHIP_FLAG;
            
            for (int i = 0; i < points.shape[0] - 1; i++) {
                indexBufferView[i, 0] = indexBufferRange[i, 0];
                indexBufferView[i, 1] = indexBufferRange[i, 1];
                indexBufferView[i, 2] = indexBufferRange[i+1, 1];
                indexBufferView[i, 3] = indexBufferRange[i, 0];
                indexBufferView[i, 4] = indexBufferRange[i+1, 0];
                indexBufferView[i, 5] = indexBufferRange[i+1, 1];
            }
        }
        MatrixH<2, float> vecs;
        points.Derivative(vecs, 0, 3);
        auto vecs_normalised = vecs / (vecs * vecs).SumNoRed(1).sqrt();
        MatrixH<2, float> perp = {{0, 1}, {-1, 0}};
        auto normals = perp.Dot(vecs_normalised, true).T();

        MatrixH<2, float> convNormals = MatrixH<2, float>::zeros({normals.shape[0] + 1, normals.shape[1]});
        normals.conv<1>(convNormals, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
        auto convNormals_normalised = convNormals / (convNormals * convNormals).SumNoRed(1);
        auto verts = MatrixH<2, float>::concat(points, points + convNormals_normalised * 0.1, 1);
        verts.shape[0] = points.shape[0] * 2;
        verts.shape[1] = 2;
        
        verts = MatrixH<2, float>::concat(verts, MatrixH<2, float>::zeros({points.shape[0] * 2, 1}), 1);
        MatrixH<2, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = points.shape[0] * 2;
        VertexBufferView.shape[1] = 3;
        VertexBufferView.strides[0] = sizeof(Vertex3D) / sizeof(float);
        VertexBufferView.strides[1] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = verts;
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, colour) / sizeof(float);
        VertexBufferView.shape[1] = 4;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = MatrixH<2, float>::repeating({points.shape[0] * 2}, BLUE);
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, normal) / sizeof(float);
        VertexBufferView.shape[1] = 3;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        
        VertexBufferView = MatrixH<2, float>::repeating({points.shape[0] * 2}, MatrixH<1, float>{0.0, 0.0, 1.0});
    }
};

class LineNode3D: public GeometryNode<uint16> {
public:
    LineNode3D(MatrixH<2, float> path, MatrixH<2, float> points): GeometryNode<uint16>(new Vertex3D[points.shape[0] * path.shape[0]], points.shape[0] * path.shape[0], new uint16[points.shape[0] * (path.shape[0] - 1) * 6], points.shape[0] * (path.shape[0] - 1) * 6) {

        MatrixH<2, float> vecs;
        path.Derivative(vecs, 0, 3);
        

        auto vecs_normalised = vecs / (vecs * vecs).SumNoRed(1).sqrt();


        MatrixH<2, float> iHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
        MatrixH<2, float> jHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
        iHat[0] = simd_normalize( simd_cross(vecs_normalised.toSimdFloat3(0), simd_make_float3(0, 0, 1)) );

        for (int i = 1; i < path.shape[0]-1; i++) {
            float sign =  simd_dot(iHat.toSimdFloat3(i-1), (vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1))) > 0 ? 1: -1;
            MatrixH<1, float> diff = simd_matH(simd_reflect((vecs_normalised[i] - vecs_normalised[i-1]).toSimdFloat3(), iHat[i-1].toSimdFloat3()));
            iHat[i] = iHat[i-1] + 1 * sign * diff;
            //            iHat[i] =  iHat[i] / (iHat[i] * iHat[i]).SumNoRed(0).sqrt();
            iHat[i] = simd_matH(simd_normalize(iHat.toSimdFloat3(i)));
        }
        
        // FASTER USING SIMD about 40 us faster
//        for (int i = 1; i < path.shape[0]-1; i++) {
//            float sign =  simd_dot(iHat.toSimdFloat3(i-1), (vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1))) > 0 ? 1: -1;
//            simd_float3 diff = (simd_reflect((vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1)), iHat.toSimdFloat3(i-1)));
//            iHat[i] = simd_matH(iHat.toSimdFloat3(i-1) + 1 * sign * diff);
//            iHat[i] = simd_matH(simd_normalize(iHat.toSimdFloat3(i)));
//        }
    //    iHat = iHat / (iHat * iHat).SumNoRed(1).sqrt();
        
        for (int i = 0; i < path.shape[0] - 1; i++) {
            jHat[i] = simd_normalize(simd_cross(iHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
        }
        for (int i = 0; i < path.shape[0] - 1; i++) {
            iHat[i] = simd_normalize(simd_cross(jHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
        }

        MatrixH<2, float> conviHat = MatrixH<2, float>::zeros({iHat.shape[0] + 1, iHat.shape[1]});
        iHat.conv<1>(conviHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
        
        MatrixH<2, float> convjHat = MatrixH<2, float>::zeros({jHat.shape[0] + 1, jHat.shape[1]});
        jHat.conv<1>(convjHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
        
        conviHat = conviHat / (conviHat * conviHat).SumNoRed(1);
        convjHat = convjHat / (convjHat * convjHat).SumNoRed(1);
        
        
        MatrixH<3, float> transforms = MatrixH<3, float>::repeating({path.shape[0]}, MatrixH<2, float>::eye(3));
        transforms.Slice({null, {{0, 1}} }) = conviHat.unsqueeze<1>();
        transforms.Slice({null, {{1, 2}} }) = convjHat.unsqueeze<1>();
        transforms.Slice({null, {{2, 3}} }) = MatrixH<2, float>::concat(vecs_normalised, vecs_normalised[vecs_normalised.shape[0]-1].unsqueeze<1>(), 0).unsqueeze<1>();
        

        MatrixH<3, float> transformedpoints = MatrixH<3, float>::withShape({path.shape[0], points.shape[0], 3});

            for (int i = 0; i < path.shape[0]; i++) {
                for (int j = 0; j < points.shape[0]; j++) {
                    transformedpoints[i, j] = path[i] + conviHat[i] * points[j, 0] + convjHat[i] * points[j, 1];
                    //                transformedpoints[i, j] = path[i] + transforms[i, 0] * points[j, 0] + transforms[i, 1] * points[j, 1];
                    
                    
                    
                    //                transformedpoints[i, j] = path[i] + transforms[i].T().Dot(points[j].unsqueeze<1>(), true).T().squeeze<1>();
                }
            }
            
        
        // FASTER USING SIMD about 40 us faster
//        MatrixH<3, float> transformedpoints = MatrixH<3, float>::withShape({path.shape[0], points.shape[0], 3});
//        for (int i = 0; i < path.shape[0]; i++) {
//            for (int j = 0; j < points.shape[0]; j++) {
//                transformedpoints[i, j] = simd_matH(path.toSimdFloat3(i) + conviHat.toSimdFloat3(i) * points[j, 0] + convjHat.toSimdFloat3(i) * points[j, 1]);
//            }
//        }
        
        MatrixH<3, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = path.shape[0];
        VertexBufferView.shape[1] = points.shape[0];
        VertexBufferView.shape[2] = 3;
        
        VertexBufferView.strides[0] = points.shape[0] * ( sizeof(Vertex3D) / sizeof(float) );
        VertexBufferView.strides[1] = sizeof(Vertex3D) / sizeof(float);
        VertexBufferView.strides[2] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 3);
        
        VertexBufferView = transformedpoints;
        
        MatrixH<2, uint16> indexBufferRange = MatrixH<2, uint16>::Range(0, {path.shape[0] ,points.shape[0]});
        MatrixH<3, uint16> indexBufferView;
        indexBufferView.buffer = (uint16*)indices;
        indexBufferView.shape[0] = path.shape[0]-1;
        indexBufferView.shape[1] = points.shape[0];
        indexBufferView.shape[2] = 6;
        indexBufferView.calcStrides();
        indexBufferView.flags |= NON_OWNERSHIP_FLAG;
        
        for (int j = 0; j < path.shape[0] -1; j++) {
            for (int i = 0; i< points.shape[0] - 1; i++) {
                indexBufferView[j, i, 0] = indexBufferRange[j+0, i];
                indexBufferView[j, i, 1] = indexBufferRange[j+1, i];
                indexBufferView[j, i, 2] = indexBufferRange[j+1, i+1];
                indexBufferView[j, i, 3] = indexBufferRange[j+0, i];
                indexBufferView[j, i, 4] = indexBufferRange[j+0, i+1];
                indexBufferView[j, i, 5] = indexBufferRange[j+1, i+1];
            }
            indexBufferView[j, points.shape[0]-1, 0] = indexBufferRange[j+0 , points.shape[0]-1];
            indexBufferView[j, points.shape[0]-1, 1] = indexBufferRange[j+1, points.shape[0]-1];
            indexBufferView[j, points.shape[0]-1, 2] = indexBufferRange[j+1, 0];
            indexBufferView[j, points.shape[0]-1, 3] = indexBufferRange[j+0, points.shape[0]-1];
            indexBufferView[j, points.shape[0]-1, 4] = indexBufferRange[j+0, 0];
            indexBufferView[j, points.shape[0]-1, 5] = indexBufferRange[j+1, 0];
        }
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, colour) / sizeof(float);
        VertexBufferView.shape[2] = 4;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 3);
        
        VertexBufferView = MatrixH<3, float>::repeating({path.shape[0], points.shape[0] }, BLUE);
        
        VertexBufferView.buffer = (float*)Verticies + offsetof(Vertex3D, normal) / sizeof(float);
        VertexBufferView.shape[2] = 3;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 3);
        
        VertexBufferView = MatrixH<3, float>::repeating({ path.shape[0], points.shape[0] }, MatrixH<1, float>{0.0, 0.0, 1.0});
        
        drawType = MTLPrimitiveTypeTriangle;
    }
    
    void UpdatePoints(MatrixH<2, float> path, MatrixH<2, float> points) {
        if (points.shape[0] * path.shape[0] != vertexCount) {
            delete [] Verticies;
            delete [] indices;
            
            Verticies = new Vertex3D[points.shape[0] * path.shape[0]];
            indices = new uint16[points.shape[0] * (path.shape[0] - 1) * 6];
            
            vertexCount =  points.shape[0] * path.shape[0];
            indexCount = points.shape[0] * (path.shape[0] - 1) * 6;
            update = true;
            
            MatrixH<2, uint16> indexBufferRange = MatrixH<2, uint16>::Range(0, {path.shape[0] ,points.shape[0]});
            MatrixH<3, uint16> indexBufferView;
            indexBufferView.buffer = (uint16*)indices;
            indexBufferView.shape[0] = path.shape[0]-1;
            indexBufferView.shape[1] = points.shape[0];
            indexBufferView.shape[2] = 6;
            indexBufferView.calcStrides();
            indexBufferView.flags |= NON_OWNERSHIP_FLAG;
            
            for (int j = 0; j < path.shape[0] -1; j++) {
                for (int i = 0; i< points.shape[0] - 1; i++) {
                    indexBufferView[j, i, 0] = indexBufferRange[j+0, i];
                    indexBufferView[j, i, 1] = indexBufferRange[j+1, i];
                    indexBufferView[j, i, 2] = indexBufferRange[j+1, i+1];
                    indexBufferView[j, i, 3] = indexBufferRange[j+0, i];
                    indexBufferView[j, i, 4] = indexBufferRange[j+0, i+1];
                    indexBufferView[j, i, 5] = indexBufferRange[j+1, i+1];
                }
                indexBufferView[j, points.shape[0]-1, 0] = indexBufferRange[j+0 , points.shape[0]-1];
                indexBufferView[j, points.shape[0]-1, 1] = indexBufferRange[j+1, points.shape[0]-1];
                indexBufferView[j, points.shape[0]-1, 2] = indexBufferRange[j+1, 0];
                indexBufferView[j, points.shape[0]-1, 3] = indexBufferRange[j+0, points.shape[0]-1];
                indexBufferView[j, points.shape[0]-1, 4] = indexBufferRange[j+0, 0];
                indexBufferView[j, points.shape[0]-1, 5] = indexBufferRange[j+1, 0];
            }
        }
        MatrixH<2, float> vecs;
        path.Derivative(vecs, 0, 3);
        

        auto vecs_normalised = vecs / (vecs * vecs).SumNoRed(1).sqrt();


        MatrixH<2, float> iHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
        MatrixH<2, float> jHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
        iHat[0] = simd_normalize( simd_cross(vecs_normalised.toSimdFloat3(0), simd_make_float3(0, 0, 1)) );

        for (int i = 1; i < path.shape[0]-1; i++) {
            float sign =  simd_dot(iHat.toSimdFloat3(i-1), (vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1))) > 0 ? 1: -1;
            MatrixH<1, float> diff = simd_matH(simd_reflect((vecs_normalised[i] - vecs_normalised[i-1]).toSimdFloat3(), iHat[i-1].toSimdFloat3()));
            iHat[i] = iHat[i-1] + 1 * sign * diff;
            //            iHat[i] =  iHat[i] / (iHat[i] * iHat[i]).SumNoRed(0).sqrt();
            iHat[i] = simd_matH(simd_normalize(iHat.toSimdFloat3(i)));
        }
        
        // FASTER USING SIMD about 40 us faster
//        for (int i = 1; i < path.shape[0]-1; i++) {
//            float sign =  simd_dot(iHat.toSimdFloat3(i-1), (vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1))) > 0 ? 1: -1;
//            simd_float3 diff = (simd_reflect((vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1)), iHat.toSimdFloat3(i-1)));
//            iHat[i] = simd_matH(iHat.toSimdFloat3(i-1) + 1 * sign * diff);
//            iHat[i] = simd_matH(simd_normalize(iHat.toSimdFloat3(i)));
//        }
    //    iHat = iHat / (iHat * iHat).SumNoRed(1).sqrt();
        
        for (int i = 0; i < path.shape[0] - 1; i++) {
            jHat[i] = simd_normalize(simd_cross(iHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
        }
        for (int i = 0; i < path.shape[0] - 1; i++) {
            iHat[i] = simd_normalize(simd_cross(jHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
        }

        MatrixH<2, float> conviHat = MatrixH<2, float>::zeros({iHat.shape[0] + 1, iHat.shape[1]});
        iHat.conv<1>(conviHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
        
        MatrixH<2, float> convjHat = MatrixH<2, float>::zeros({jHat.shape[0] + 1, jHat.shape[1]});
        jHat.conv<1>(convjHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
        
        conviHat = conviHat / (conviHat * conviHat).SumNoRed(1);
        convjHat = convjHat / (convjHat * convjHat).SumNoRed(1);
        
        
        MatrixH<3, float> transforms = MatrixH<3, float>::repeating({path.shape[0]}, MatrixH<2, float>::eye(3));
        transforms.Slice({null, {{0, 1}} }) = conviHat.unsqueeze<1>();
        transforms.Slice({null, {{1, 2}} }) = convjHat.unsqueeze<1>();
        transforms.Slice({null, {{2, 3}} }) = MatrixH<2, float>::concat(vecs_normalised, vecs_normalised[vecs_normalised.shape[0]-1].unsqueeze<1>(), 0).unsqueeze<1>();
        

        MatrixH<3, float> transformedpoints = MatrixH<3, float>::withShape({path.shape[0], points.shape[0], 3});

            for (int i = 0; i < path.shape[0]; i++) {
                for (int j = 0; j < points.shape[0]; j++) {
                    transformedpoints[i, j] = path[i] + conviHat[i] * points[j, 0] + convjHat[i] * points[j, 1];
                    //                transformedpoints[i, j] = path[i] + transforms[i, 0] * points[j, 0] + transforms[i, 1] * points[j, 1];
                    
                    
                    
                    //                transformedpoints[i, j] = path[i] + transforms[i].T().Dot(points[j].unsqueeze<1>(), true).T().squeeze<1>();
                }
            }
            
        
        // FASTER USING SIMD about 40 us faster
//        MatrixH<3, float> transformedpoints = MatrixH<3, float>::withShape({path.shape[0], points.shape[0], 3});
//        for (int i = 0; i < path.shape[0]; i++) {
//            for (int j = 0; j < points.shape[0]; j++) {
//                transformedpoints[i, j] = simd_matH(path.toSimdFloat3(i) + conviHat.toSimdFloat3(i) * points[j, 0] + convjHat.toSimdFloat3(i) * points[j, 1]);
//            }
//        }
        
        MatrixH<3, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = path.shape[0];
        VertexBufferView.shape[1] = points.shape[0];
        VertexBufferView.shape[2] = 3;
        
        VertexBufferView.strides[0] = points.shape[0] * ( sizeof(Vertex3D) / sizeof(float) );
        VertexBufferView.strides[1] = sizeof(Vertex3D) / sizeof(float);
        VertexBufferView.strides[2] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 3);
        
        VertexBufferView = transformedpoints;
    }
};


class PointCloudNode: public GeometryNode<uint16> {
public:
    PointCloudNode(const MatrixH<2, float>& points_pos, const MatrixH<2, float>& points_color): GeometryNode<uint16>(new Point3D[points_pos.shape[0]], points_pos.shape[0], new uint16[points_pos.shape[0]], points_pos.shape[0]) {
        if (points_pos.shape[0] != points_color.shape[0]) {
            throw std::runtime_error("GeometryNode: points_pos and points_color must have the same number of points.");
        }
        if (points_pos.shape[1] != 3) {
            throw std::runtime_error("GeometryNode: points_pos must have shape Nx3.");
        }
        if (points_color.shape[1] != 4) {
            throw std::runtime_error("GeometryNode: points_color must have shape Nx4.");
        }
        renderPipelineType = RenderPipelineType::Predefined;
        RenderStateNo = static_cast<int>(PredefinedRenderPipelineState::PointCloud);
        drawType = MTLPrimitiveTypePoint;
        vertexType = VertexType::Point;
        
        if (points_pos.total_size == 0) { return; }
        buildBuffers(GlobalGPUManager.metalDevice);
        MatrixH<2, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = points_pos.shape[0];
        VertexBufferView.shape[1] = 3;
        VertexBufferView.strides[0] = sizeof(Point3D) / sizeof(float);
        VertexBufferView.strides[1] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        VertexBufferView.metalBuffer = vertexBuffer;
        
        MatrixH<2, float>::copyGPUinplace(VertexBufferView, points_pos, 0, false);
        VertexBufferView.shape[1] = 4; // as colour is simd_float4
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        MatrixH<2, float>::copyGPUinplace(VertexBufferView, points_color, offsetof(Point3D, colour) / sizeof(float), true);
        VertexBufferView.shape[1] = 8;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        

    }
    void updatePoints(const MatrixH<2, float>& points_pos, const MatrixH<2, float>& points_color) {
        if (points_pos.shape[0] != points_color.shape[0]) {
            throw std::runtime_error("GeometryNode: points_pos and points_color must have the same number of points.");
        }

        if (points_pos.shape[0] != vertexCount || points_color.shape[0] != vertexCount) {
            delete [] Verticies;
            delete [] indices;
            
            Verticies = new Point3D[points_pos.shape[0]];
            indices = new uint16[points_pos.shape[0]];
            
            vertexCount =  points_pos.shape[0];
            indexCount = points_pos.shape[0];
            update = true;
        }
        
        if (points_pos.shape[1] != 3) {
            throw std::runtime_error("GeometryNode: points_pos must have shape Nx3.");
        }
        if (points_color.shape[1] != 4) {
            throw std::runtime_error("GeometryNode: points_color must have shape Nx4.");
        }
        buildBuffers(GlobalGPUManager.metalDevice);
        MatrixH<2, float> VertexBufferView;
        VertexBufferView.buffer = (float*)Verticies;
        VertexBufferView.shape[0] = points_pos.shape[0];
        VertexBufferView.shape[1] = 3;
        VertexBufferView.strides[0] = sizeof(Point3D) / sizeof(float);
        VertexBufferView.strides[1] = 1;
        VertexBufferView.flags |= NON_OWNERSHIP_FLAG;
        VertexBufferView.flags |= NON_CONTIGUOUS_FLAG;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        VertexBufferView.metalBuffer = vertexBuffer;
        
        MatrixH<2, float>::copyGPUinplace(VertexBufferView, points_pos, 0, false);
        VertexBufferView.shape[1] = 4; // as colour is simd_float4
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
        MatrixH<2, float>::copyGPUinplace(VertexBufferView, points_color, offsetof(Point3D, colour) / sizeof(float), true);
        VertexBufferView.shape[1] = 8;
        VertexBufferView.total_size = VertexBufferView.accumul(0, 2);
    }
};

// Helper structure for 2D points during triangulation
struct Point2D {
    float x, y;
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

// Helper class for triangulating text paths
//class TextTriangulator {
//private:
//    std::vector<Point2D> contourPoints;
//    std::vector<std::vector<Point2D>> holes;
//    
//    // Simple ear clipping triangulation
//    bool isEar(const std::vector<Point2D>& polygon, int i) {
//        int n = polygon.size();
//        int prev = (i - 1 + n) % n;
//        int next = (i + 1) % n;
//        
//        // Check if triangle is counter-clockwise (convex)
//        float cross = (polygon[i].x - polygon[prev].x) * (polygon[next].y - polygon[i].y) -
//                     (polygon[i].y - polygon[prev].y) * (polygon[next].x - polygon[i].x);
//        
//        if (cross <= 0) return false; // Not convex
//        
//        // Check if any other point is inside this triangle
//        for (int j = 0; j < n; j++) {
//            if (j == prev || j == i || j == next) continue;
//            if (pointInTriangle(polygon[prev], polygon[i], polygon[next], polygon[j])) {
//                return false;
//            }
//        }
//        return true;
//    }
//    
//    bool pointInTriangle(const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& p) {
//        float denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
//        if (abs(denom) < 1e-10) return false;
//        
//        float alpha = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
//        float beta = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
//        float gamma = 1.0f - alpha - beta;
//        
//        return alpha >= 0 && beta >= 0 && gamma >= 0;
//    }
//    
//public:
//    std::vector<Point2D> triangulate(const std::vector<Point2D>& polygon) {
//        std::vector<Point2D> result;
//        std::vector<Point2D> working = polygon;
//        
//        while (working.size() > 3) {
//            bool foundEar = false;
//            for (int i = 0; i < working.size(); i++) {
//                if (isEar(working, i)) {
//                    int prev = (i - 1 + working.size()) % working.size();
//                    int next = (i + 1) % working.size();
//                    
//                    // Add triangle
//                    result.push_back(working[prev]);
//                    result.push_back(working[i]);
//                    result.push_back(working[next]);
//                    
//                    // Remove the ear
//                    working.erase(working.begin() + i);
//                    foundEar = true;
//                    break;
//                }
//            }
//            if (!foundEar) break; // Prevent infinite loop
//        }
//        
//        // Add final triangle
//        if (working.size() == 3) {
//            result.push_back(working[0]);
//            result.push_back(working[1]);
//            result.push_back(working[2]);
//        }
//        
//        return result;
//    }
//};
//
//class Text3D : public Shape<uint16> {
//private:
//    NSString* text;
//    NSString* fontName;
//    float fontSize;
//    float depth;
//    simd_float4 color;
//    
//    // Core Text path extraction callback
//    static void pathApplierCallback(void* info, const CGPathElement* element) {
//        std::vector<std::vector<Point2D>>* contours = static_cast<std::vector<std::vector<Point2D>>*>(info);
//        
//        switch (element->type) {
//            case kCGPathElementMoveToPoint: {
//                contours->push_back(std::vector<Point2D>());
//                CGPoint pt = element->points[0];
//                contours->back().push_back(Point2D(pt.x, pt.y));
//                break;
//            }
//            case kCGPathElementAddLineToPoint: {
//                if (!contours->empty()) {
//                    CGPoint pt = element->points[0];
//                    contours->back().push_back(Point2D(pt.x, pt.y));
//                }
//                break;
//            }
//            case kCGPathElementAddQuadCurveToPoint: {
//                if (!contours->empty()) {
//                    // Approximate quadratic curve with line segments
//                    CGPoint control = element->points[0];
//                    CGPoint end = element->points[1];
//                    Point2D lastPt = contours->back().back();
//                    
//                    // Simple approximation - add a few points along the curve
//                    for (int i = 1; i <= 4; i++) {
//                        float t = i / 4.0f;
//                        float x = (1-t)*(1-t)*lastPt.x + 2*(1-t)*t*control.x + t*t*end.x;
//                        float y = (1-t)*(1-t)*lastPt.y + 2*(1-t)*t*control.y + t*t*end.y;
//                        contours->back().push_back(Point2D(x, y));
//                    }
//                }
//                break;
//            }
//            case kCGPathElementAddCurveToPoint: {
//                if (!contours->empty()) {
//                    // Approximate cubic curve with line segments
//                    CGPoint control1 = element->points[0];
//                    CGPoint control2 = element->points[1];
//                    CGPoint end = element->points[2];
//                    Point2D lastPt = contours->back().back();
//                    
//                    // Simple approximation - add a few points along the curve
//                    for (int i = 1; i <= 6; i++) {
//                        float t = i / 6.0f;
//                        float x = (1-t)*(1-t)*(1-t)*lastPt.x + 3*(1-t)*(1-t)*t*control1.x +
//                                 3*(1-t)*t*t*control2.x + t*t*t*end.x;
//                        float y = (1-t)*(1-t)*(1-t)*lastPt.y + 3*(1-t)*(1-t)*t*control1.y +
//                                 3*(1-t)*t*t*control2.y + t*t*t*end.y;
//                        contours->back().push_back(Point2D(x, y));
//                    }
//                }
//                break;
//            }
//            case kCGPathElementCloseSubpath:
//                // Close the current contour
//                break;
//        }
//    }
//    
//    void generateTextGeometry() {
//        // Create attributed string
//        CFStringRef cfText = (__bridge CFStringRef)text;
//        CTFontRef font = CTFontCreateWithName((__bridge CFStringRef)fontName, fontSize, NULL);
//        
//        if (!font) {
//            font = CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, fontSize, NULL);
//        }
//        
//        // Create attributed string
//        CFStringRef keys[] = { kCTFontAttributeName };
//        CFTypeRef values[] = { font };
//        CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
//                                                       (const void**)&values, 1,
//                                                       &kCFTypeDictionaryKeyCallBacks,
//                                                       &kCFTypeDictionaryValueCallBacks);
//        
//        CFAttributedStringRef attributedString = CFAttributedStringCreate(kCFAllocatorDefault, cfText, attributes);
//        
//        // Create line
//        CTLineRef line = CTLineCreateWithAttributedString(attributedString);
//        
//        // Get glyph runs
//        CFArrayRef runs = CTLineGetGlyphRuns(line);
//        CFIndex runCount = CFArrayGetCount(runs);
//        
//        std::vector<std::vector<Point2D>> allContours;
//        
//        for (CFIndex runIndex = 0; runIndex < runCount; runIndex++) {
//            CTRunRef run = (CTRunRef)CFArrayGetValueAtIndex(runs, runIndex);
//            CFIndex glyphCount = CTRunGetGlyphCount(run);
//            
//            CGGlyph* glyphs = (CGGlyph*)malloc(sizeof(CGGlyph) * glyphCount);
//            CGPoint* positions = (CGPoint*)malloc(sizeof(CGPoint) * glyphCount);
//            
//            CTRunGetGlyphs(run, CFRangeMake(0, 0), glyphs);
//            CTRunGetPositions(run, CFRangeMake(0, 0), positions);
//            
//            CTFontRef runFont = (CTFontRef)CFDictionaryGetValue(CTRunGetAttributes(run), kCTFontAttributeName);
//            
//            for (CFIndex glyphIndex = 0; glyphIndex < glyphCount; glyphIndex++) {
//                CGPathRef glyphPath = CTFontCreatePathForGlyph(runFont, glyphs[glyphIndex], NULL);
//                
//                if (glyphPath) {
//                    // Transform path by glyph position
//                    CGAffineTransform transform = CGAffineTransformMakeTranslation(positions[glyphIndex].x,
//                                                                                  positions[glyphIndex].y);
//                    CGPathRef transformedPath = CGPathCreateCopyByTransformingPath(glyphPath, &transform);
//                    
//                    // Extract path data
//                    std::vector<std::vector<Point2D>> glyphContours;
//                    CGPathApply(transformedPath, &glyphContours, pathApplierCallback);
//                    
//                    // Add to all contours
//                    allContours.insert(allContours.end(), glyphContours.begin(), glyphContours.end());
//                    
//                    CGPathRelease(transformedPath);
//                    CGPathRelease(glyphPath);
//                }
//            }
//            
//            free(glyphs);
//            free(positions);
//        }
//        
//        // Clean up Core Text objects
//        CFRelease(line);
//        CFRelease(attributedString);
//        CFRelease(attributes);
//        CFRelease(font);
//        
//        // Triangulate all contours
//        TextTriangulator triangulator;
//        std::vector<Point2D> allTriangles;
//        
//        for (const auto& contour : allContours) {
//            if (contour.size() >= 3) {
//                std::vector<Point2D> triangles = triangulator.triangulate(contour);
//                allTriangles.insert(allTriangles.end(), triangles.begin(), triangles.end());
//            }
//        }
//        
//        // Generate 3D geometry
//        generateMeshFromTriangles(allTriangles);
//    }
//    
//    void generateMeshFromTriangles(const std::vector<Point2D>& triangles) {
//        std::vector<Vertex3D> vertices;
//        std::vector<uint16> indices;
//        
//        // Create front face vertices
//        for (const auto& point : triangles) {
//            Vertex3D vertex;
//            vertex.position = {point.x, point.y, 0.0f};
//            vertex.colour = color;
//            vertices.push_back(vertex);
//        }
//        
//        // Create back face vertices
//        for (const auto& point : triangles) {
//            Vertex3D vertex;
//            vertex.position = {point.x, point.y, -depth};
//            vertex.colour = color;
//            vertices.push_back(vertex);
//        }
//        
//        int frontVertexCount = triangles.size();
//        
//        // Front face indices
//        for (int i = 0; i < frontVertexCount; i++) {
//            indices.push_back(i);
//        }
//        
//        // Back face indices (reversed winding)
//        for (int i = frontVertexCount - 1; i >= 0; i--) {
//            indices.push_back(i + frontVertexCount);
//        }
//        
//        // Side faces (connecting front to back)
//        // This is simplified - in practice you'd want to connect the outline edges
//        std::map<std::pair<float, float>, std::vector<int>> pointToVertices;
//        
//        // Group vertices by their 2D position
//        for (int i = 0; i < frontVertexCount; i++) {
//            std::pair<float, float> key = {triangles[i].x, triangles[i].y};
//            pointToVertices[key].push_back(i);
//            pointToVertices[key].push_back(i + frontVertexCount);
//        }
//        
//        // Allocate final arrays
//        vertexCount = vertices.size();
//        indexCount = indices.size();
//        
//        Verticies = new Vertex3D[vertexCount];
//        this->indices = new uint16[indexCount];
//        
//        // Copy data
//        for (int i = 0; i < vertexCount; i++) {
//            Verticies[i] = vertices[i];
//        }
//        
//        for (int i = 0; i < indexCount; i++) {
//            this->indices[i] = indices[i];
//        }
//    }
//    
//public:
//    Text3D(NSString* text, NSString* fontName, float fontSize, float depth, simd_float4 color)
//        : Shape<uint16>(nullptr, 0, nullptr, 0), text(text), fontName(fontName),
//          fontSize(fontSize), depth(depth), color(color) {
//        
//        // Generate the text geometry
//        generateTextGeometry();
//    }
//    
//    // Convenience constructor with default font
//    Text3D(NSString* text, float fontSize, float depth, simd_float4 color)
//        : Text3D(text, @"Helvetica", fontSize, depth, color) {
//    }
//    
//    // Destructor
//    ~Text3D() {
//        if (Verticies) delete[] Verticies;
//        if (indices) delete[] indices;
//    }
//    
//    // Helper method to get text bounds (useful for positioning)
//    CGRect getTextBounds() {
//        CTFontRef font = CTFontCreateWithName((__bridge CFStringRef)fontName, fontSize, NULL);
//        if (!font) {
//            font = CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, fontSize, NULL);
//        }
//        
//        CFStringRef keys[] = { kCTFontAttributeName };
//        CFTypeRef values[] = { font };
//        CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys,
//                                                       (const void**)&values, 1,
//                                                       &kCFTypeDictionaryKeyCallBacks,
//                                                       &kCFTypeDictionaryValueCallBacks);
//        
//        CFAttributedStringRef attributedString = CFAttributedStringCreate(kCFAllocatorDefault,
//                                                                         (__bridge CFStringRef)text,
//                                                                         attributes);
//        
//        CTLineRef line = CTLineCreateWithAttributedString(attributedString);
//        CGRect bounds = CTLineGetBoundsWithOptions(line, 0);
//        
//        CFRelease(line);
//        CFRelease(attributedString);
//        CFRelease(attributes);
//        CFRelease(font);
//        
//        return bounds;
//    }
//};


class ConvexPolygon: public Shape<uint16> {
public:
    ConvexPolygon(MatrixH<1, simd_float2>& points): Shape<uint16>(new Vertex3D[points.total_size], points.total_size, new uint16[3 + 3*(points.total_size-3)], 3 + 3*(points.total_size-3)) {
        Vertex3D* verts = static_cast<Vertex3D*>(Verticies);
        for (int i = 0; i < points.total_size; i ++) {
            Verticies[i]  = { { points[i].x, points[i].y, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        }
        for (int i = 0; i < points.total_size - 2; i ++) {
            indices[3*i + 0] = 0;
            indices[3*i + 1] = i+1;
            indices[3*i + 2] = i+2;
        }
    }
};

class Line: public Shape<uint16> {
public:
    Line(MatrixH<2, float>& points): Shape<uint16>(new Vertex3D[2 * points.total_size], 2 * points.total_size, new uint16[6 * points.total_size], 6 * points.total_size) {
        points.print();
        MatrixH<2, float> derivativeTemp;
        points.Derivative(derivativeTemp, 0, false);
        MatrixH<2, float> derivative = derivativeTemp.SliceCopy({{{0, derivativeTemp.shape[0]-1}}});
        derivativeTemp.print();
        derivative.print();
        derivative.T().print();
        auto perpendicular = MatrixH<2, float>({{0, -1}, {1, 0}});
        auto normals = perpendicular.Dot(derivative.T()).T();
        normals.print();
        
        for (int i = 0; i < points.shape[0]; i++) {
            Verticies[2*i]   = { { points[i, 0], points[i, 1], 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
            Verticies[2*i+1] = { { points[i, 0], points[i, 1], 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        }
        for (int i = 0; i < points.total_size - 2; i ++) {
            indices[3*i + 0] = 0;
            indices[3*i + 1] = i+1;
            indices[3*i + 2] = i+2;
        }
    }
};

class Text3D: public Shape<uint16> {
public:
    Text3D(): Shape<uint16>(nullptr, 0, nullptr, 0) {
        #if !TARGET_OS_IPHONE
        MatrixH<1, simd_float2> pts = generatePtsFromText(@"Hello World", 10,  [NSFont systemFontOfSize:14.0]);
        #endif
        #if TARGET_OS_IPHONE
        MatrixH<1, simd_float2> pts = generatePtsFromText(@"Hello World", 10,  [UIFont systemFontOfSize:14.0]);
        #endif
        indexCount = 3 + 3*(pts.total_size-3);
        vertexCount = pts.total_size;
        indices = new uint16[indexCount];
        Verticies = new Vertex3D[vertexCount];
        updateShape(pts, *this);
    }
    
    #if !TARGET_OS_IPHONE
    using sys_font = NSFont;
    #endif
    #if TARGET_OS_IPHONE
    using sys_font = UIFont;
    #endif
    static MatrixH<1, simd_float2> generatePtsFromText(NSString* text, size_t size, sys_font* font) {
        MatrixH<1, simd_float2> points;
        CFStringRef cfText = (__bridge CFStringRef)text;
        CTFontRef CTfont = (__bridge CTFontRef)font;
        if (!CTfont) {
            CTfont = CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, size, NULL);
        }
        
        CFStringRef keys[] = { kCTFontAttributeName };
        CFTypeRef values[] = { CTfont };
        CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault, (const void**)&keys, (const void**)&values, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        
        CFAttributedStringRef attributedString = CFAttributedStringCreate(kCFAllocatorDefault, cfText, attributes);
        
        CTLineRef line = CTLineCreateWithAttributedString(attributedString);
        CFArrayRef runs = CTLineGetGlyphRuns(line);
        CFIndex runCount = CFArrayGetCount(runs);
        
        std::vector<std::vector<simd_float2>> allContours;
        for (CFIndex runIndex = 0; runIndex < runCount; runIndex++) {
            CTRunRef run = (CTRunRef)CFArrayGetValueAtIndex(runs, runIndex);
            CFIndex glyphCount = CTRunGetGlyphCount(run);
            
            CGGlyph* glyphs = (CGGlyph*)malloc(sizeof(CGGlyph) * glyphCount);
            CGPoint* positions = (CGPoint*)malloc(sizeof(CGPoint) * glyphCount);
        
            
            CTRunGetGlyphs(run, CFRangeMake(0, 0), glyphs);
            CTRunGetPositions(run, CFRangeMake(0, 0), positions);
            
            points.buffer = new simd_float2[glyphCount];
            points.total_size = glyphCount;
            points.shape[0] = glyphCount;
            points.buildMetalBuffer();
            for (int i = 0; i < glyphCount; i++) {
                *(points(i)) = simd_make_float2(positions[i].x, positions[i].y);
            }
            
            CTFontRef runFont = (CTFontRef)CFDictionaryGetValue(CTRunGetAttributes(run), kCTFontAttributeName);
            for (CFIndex glyphIndex = 0; glyphIndex < glyphCount; glyphIndex++) {
                CGPathRef glyphPath = CTFontCreatePathForGlyph(runFont, glyphs[glyphIndex], NULL);
                if (glyphPath) {
                    CGAffineTransform transform = CGAffineTransformMakeTranslation(positions[glyphIndex].x, positions[glyphIndex].y);
                    CGPathRef transformedPath = CGPathCreateCopyByTransformingPath(glyphPath, &transform);
                    std::vector<std::vector<simd_float2>> glyphContours;
                    CGPathApply(transformedPath, &glyphContours, pathApplierCallback);
                    allContours.insert(allContours.end(), glyphContours.begin(), glyphContours.end());
                }
            }
            
        }
        points.print();
        return points;
        
    }
    
    static void pathApplierCallback(void* info, const CGPathElement* element) {
        std::vector<std::vector<simd_float2>>* contours = static_cast<std::vector<std::vector<simd_float2>>*>(info);
        switch (element->type) {
            case kCGPathElementMoveToPoint: {
                contours->push_back(std::vector<simd_float2>());
                simd_float2 pt = simd_make_float2(element->points[0].x, element->points[0].y);
                contours->back().push_back(pt);
                break;
            }
            default:
                break;
        }
    }
    static void updateShape(const MatrixH<1, simd_float2>& points, Shape<uint16>& s) {
        if (s.indexCount != 3 + 3*(points.total_size-3) || s.vertexCount != points.total_size) {
            std::cerr << "index or vertex count excedes the buffer" << "\n";
        }
        
        for (int i = 0; i < points.total_size; i ++) {
            s.Verticies[i]  = { { points[i].x, points[i].y, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        }
        std::vector<uint16> index(points.total_size);
        for (int i = 0; i < points.total_size; i++) {
            index[i] = i;
        }
        bool orientation;
        int lastAppendCound = 0;
        while (index.size() > 2) {
            for (int i = 0; i < index.size(); i++) {
                int i0 = ring(i - 1, index.size());
                int i1 = ring(i, index.size());
                int i2 = ring(i + 1, index.size());
                
                simd_float2 e1 = points[index[i1]] - points[index[i0]];
                simd_float2 e2 = points[index[i2]] - points[index[i0]];
                CosOfVec(e1, e2, orientation);
                
                if (orientation == 0 || index.size() == 3) {
                    s.indices[lastAppendCound + 0] = index[i0];
                    s.indices[lastAppendCound + 1] = index[i1];
                    s.indices[lastAppendCound + 2] = index[i2];
                    lastAppendCound += 3;
                    index.erase(index.begin() + i1);
                    break;
                }
                
            }
        }
    }
};

//class ConcavePolygon: public Shape<uint16> {
//public:
//    ConcavePolygon(MatrixH<1, simd_float2>& points): Shape<uint16>(new Vertex3D[points.total_size], points.total_size, new uint16[3 + 3*(points.total_size-3)], 3 + 3*(points.total_size-3)) {
//        
//        for (int i = 0; i < points.total_size; i ++) {
//            Verticies[i]  = { { points[i].x, points[i].y, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
//        }
//        MatrixH<1, simd_float2> edgeVec;
//        points.Derivative(edgeVec, 0, true);
//        
//        std::vector<int> indexOfConcavity = {};
//        for (int i = 0; i < points.total_size; i++) {
//            float Angle = simd_dot(points[i], points[i+1]) / (simd_length(points[i]) * simd_length(points[i+1]));
//            if ( M_PI < Angle ) {
//                indexOfConcavity.push_back(i);
//            }
//        }
//        
//        std::vector<MatrixH<1, simd_float2>> Convexes(indexOfConcavity.size());
//        
//        float currentRec = 1000000000000.0;
//        int currentRecIndex = 0;
//        
//        std::vector<simd_int2> newEdges(indexOfConcavity.size());
//        
//        for (int i = 0; i < indexOfConcavity.size(); i++) {
//            for (int j = 0; j < points.total_size; j ++) {
//                if (j == points.total_size) {
//                    if (j == i || j == 0 || j == i-1) {
//                        if (simd_length(points[i] - points[j]) < currentRec) {
//                            currentRec = simd_length(points[i] - points[j]);
//                            currentRecIndex = j;
//                        }
//                    }
//                } else if (j > 0 ) {
//                    if (j == i || j == i+1 || j == i-1) {
//                        if (simd_length(points[i] - points[j]) < currentRec) {
//                            currentRec = simd_length(points[i] - points[j]);
//                            currentRecIndex = j;
//                        }
//                    }
//                } else if (j == 0) {
//                    if (j == i || j == i+1 || j == points.total_size) {
//                        if (simd_length(points[i] - points[j]) < currentRec) {
//                            currentRec = simd_length(points[i] - points[j]);
//                            currentRecIndex = j;
//                        }
//                    }
//                }
//            }
//            newEdges.push_back(simd_make_int2(i, currentRecIndex));
//            currentRec = 1000000000000.0;
//            
//            // there will be doubles in it i --> j , j --> i
//        }
//        
//        for (int i = 0; i < points.total_size - 2; i ++) {
//            indices[3*i + 0] = 0;
//            indices[3*i + 1] = i+1;
//            indices[3*i + 2] = i+2;
//        }
//    }
//};

class ConcavePolygon: public Shape<uint16> {
public:
    ConcavePolygon(MatrixH<1, simd_float2>& points): Shape<uint16>(new Vertex3D[points.total_size], points.total_size, new uint16[3 + 3*(points.total_size-3)], 3 + 3*(points.total_size-3)) {
        updateShape(points, *this);
    }
    
    static void updateShape(const MatrixH<1, simd_float2>& points, Shape<uint16>& s) {
        if (s.indexCount != 3 + 3*(points.total_size-3) || s.vertexCount != points.total_size) {
            std::cerr << "index or vertex count excedes the buffer" << "\n";
        }
        
        for (int i = 0; i < points.total_size; i ++) {
            s.Verticies[i]  = { { points[i].x, points[i].y, 0 },  {0.0f, 1.0f, 0.0f, 1.0}, {0, 0}, { 0.f,  0.f,  1.f } };
        }
        std::vector<uint16> index(points.total_size);
        for (int i = 0; i < points.total_size; i++) {
            index[i] = i;
        }
        bool orientation;
        int lastAppendCound = 0;
        while (index.size() > 2) {
            for (int i = 0; i < index.size(); i++) {
                int i0 = ring(i - 1, index.size());
                int i1 = ring(i, index.size());
                int i2 = ring(i + 1, index.size());
                
                simd_float2 e1 = points[index[i1]] - points[index[i0]];
                simd_float2 e2 = points[index[i2]] - points[index[i0]];
                CosOfVec(e1, e2, orientation);
                
                if (orientation == 0 || index.size() == 3) {
                    s.indices[lastAppendCound + 0] = index[i0];
                    s.indices[lastAppendCound + 1] = index[i1];
                    s.indices[lastAppendCound + 2] = index[i2];
                    lastAppendCound += 3;
                    index.erase(index.begin() + i1);
                    break;
                }
                
            }
        }
    }
};

struct ArrayShape {
    Shape<uint16> shape;
    simd_float4x4* transform;
    int instanceCount;
    bool update = true;
    id<MTLBuffer> transformBuffer = nil;
    
    ArrayShape(Shape<uint16>& shapeA, simd_float4x4* transformArr, int count): shape(shapeA), transform(transformArr), instanceCount(count), update(true), transformBuffer(nil) {
    }
    
    void buildBuffer(id<MTLDevice> metalDevice) {
        if (!transformBuffer || update) {
//            transformBuffer = [metalDevice newBufferWithLength:instanceCount * sizeof(simd_float4x4) options:MTLResourceStorageModeShared];
//            
//            memcpy([transformBuffer contents], transform, instanceCount * sizeof(simd_float4x4));
            transformBuffer = [metalDevice newBufferWithBytesNoCopy:transform length:instanceCount * sizeof(simd_float4x4) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
                delete [] pointer;
            }];
            std::cout << "buffer built transform" << "\n";
        }
        update = false;
    }
};




class ArrayModifier {
public:
    simd_float3 dPosition = {0, 0, 0};
    simd_float3 dRotation = {0, 0, 0};
    simd_float3 dScale = {0, 0, 0};
    
    template <typename Type>
    ArrayShape MakeModifierArray(Shape<Type>& shapeI, int count) {
        simd_float4x4* transform = new simd_float4x4[count];
        for (int i = 0; i < count; i++) {
            transform[i] = Translation(i * dPosition) * RotationX( i * dRotation.x) * RotationY( i * dRotation.y ) * RotationZ( i * dRotation.z ) *  Scale(1 + i * dScale);
        }
        return ArrayShape(shapeI, transform, count);
    }
    
    ArrayShape MakeModifierSuperArray(ArrayShape& ArrayI, int count) {
        simd_float4x4* transform = new simd_float4x4[count * ArrayI.instanceCount];
        for (int i = 0; i < count; i++) {
            memcpy(transform + i * ArrayI.instanceCount, ArrayI.transform, sizeof(simd_float4x4) * ArrayI.instanceCount);
            for (int j = 0; j < ArrayI.instanceCount; j++) {
                transform[i * ArrayI.instanceCount + j] = Translation(i * dPosition) * RotationX( i * dRotation.x) * RotationY( i * dRotation.y ) * RotationZ( i * dRotation.z ) *  Scale(1 + i * dScale) * transform[i * ArrayI.instanceCount + j];
            }
        }
        return ArrayShape(ArrayI.shape, transform, count * ArrayI.instanceCount);
    }
    
    template <typename Type>
    ArrayShape MakeModifierArrayRadial(Shape<Type>& shapeI, int count) {
        simd_float4x4* transform = new simd_float4x4[count];
        
        for (int i = 0; i < count; i++) {
            transform[i] = RotationX( i * dRotation.x) * RotationY( i * dRotation.y ) * RotationZ( i * dRotation.z ) * Translation(i * dPosition)  *  Scale(1 + i * dScale);
        }
        return ArrayShape(shapeI, transform, count);
    }
    
    ArrayShape MakeModifierSuperArrayRadial(ArrayShape& ArrayI, int count) {
        simd_float4x4* transform = new simd_float4x4[count * ArrayI.instanceCount];
        for (int i = 0; i < count; i++) {
            memcpy(transform + i * ArrayI.instanceCount, ArrayI.transform, sizeof(simd_float4x4) * ArrayI.instanceCount);
            for (int j = 0; j < ArrayI.instanceCount; j++) {
                transform[i * ArrayI.instanceCount + j] =  RotationX( i * dRotation.x) * RotationY( i * dRotation.y ) * RotationZ( i * dRotation.z ) * Translation(i * dPosition)  *  Scale(1 + i * dScale) * transform[i * ArrayI.instanceCount + j];
            }
        }
        return ArrayShape(ArrayI.shape, transform, count * ArrayI.instanceCount);
    }
};


template <typename Type>
ArrayShape buildLineArray(float start, float end, int count, Shape<Type> shapeI) {
    simd_float4x4* transform = new simd_float4x4[count];
    float d = (end - start) / (count-1);
    for (int i = 0; i < count; i++) {
        transform[i] = Translation(simd_make_float3(start + i * d, 0, 0));
    }
    return ArrayShape(shapeI, transform, count);
}

template <typename Type>
ArrayShape buildDottedLine(simd_float3 dir, int count, Shape<Type> shapeI) {
    simd_float4x4* transform = new simd_float4x4[count];
    simd_float3 d = (dir) / (count-1);
    for (int i = 0; i < count; i++) {
        transform[i] = Translation(d * i);
    }
    return ArrayShape(shapeI, transform, count);
}
void updateDottedLine(simd_float3 dir, int count, simd_float4x4* transform) {
    simd_float3 d = (dir) / (count-1);
    for (int i = 0; i < count; i++) {
        transform[i] = Translation(d * i);
    }
}


//class Camera3D {
//    simd_float3 oldPosition = {0.0, 0.0, -3.0};
//
//
//    simd_float3 up = {0, 1, 0};
//    simd_float3 right = {1, 0, 0};
//    simd_float3 forward = {0, 0, 1};
//    
//    float azimuthalAngle = 0;
//    float polarAngle = 0;
//    
//    float farP = 100;
//    float nearP = 0.1;
//    
//    float r = 1;
//    float d = 1;
//    float aspectRatio = 1;
//    
//    float minPolarAngle = -M_PI / 2;
//    float maxPolarAngle = M_PI / 2;
//    
//    bool AxisUpdated = false;
//    
//
//    float sensitivity = 1;
//public:
//    bool updated = false;
//    bool OrthoPara = 1;
//    simd_float3 position = {0.0, 0.0, -3.0};
//    simd_float3 target = {0, 0, 0};
//    simd_float4x4 viewMatrix;
//    Camera3D() {
//        simd_float3 vec = target - position;
//        
//        forward = simd::normalize(vec);
//        right = simd::normalize(simd::cross(forward, up));
//        up = simd::normalize(simd::cross(right, forward));
//        AxisUpdated = true;
//    }
//    
//    void updatePosition(simd_float2 drag, TransformationMode mode) {
//        if (mode == TransformationMode::Translate) {
//            if (!AxisUpdated) {
//                simd_float3 vec = target - position;
//                
//                forward = simd::normalize(vec);
//                right = simd::normalize(simd::cross(forward, up));
//                up = simd::normalize(simd::cross(right, forward));
//                AxisUpdated = true;
//            }
//
////            std::cout << "Camera Pos" << position << " Angles: "<< azimuthalAngle << ", " << polarAngle <<"\n";
//            
//            auto dist = simd::length(target - position);
//            position += (up * drag.y + right * drag.x) * sensitivity * dist;
//            
//            target = position + simd_normalize(forward) * dist;
//            updateOLD();
//        }
//        else if (mode == TransformationMode::Orbit) {
////            std::cout << "Camera Pos" << position << " Angles: "<< azimuthalAngle << ", " << polarAngle <<"\n";
//            float sensitivity = 10.0;
////            position = (RotationY(drag.x * 100) * RotationX(drag.y * 100) * simd_make_float4(position - target, 1) ).xyz + target;
////            updateOLD();
////            simd_float3 vec = target - position;
////            forward = simd::normalize(vec);
////            right = simd::normalize(simd::cross(forward, up));
////            up = simd::normalize(simd::cross(right, forward));
//            azimuthalAngle += drag.y * sensitivity;
//            polarAngle     += drag.x * sensitivity;
//
//            // Update up vector based on azimuthal angle
//            if (std::cos(azimuthalAngle) < 0.0f) {
//                up = simd::float3{0, -1, 0};
//            } else {
//                up = simd::float3{0, 1, 0};
//            }
//
//            // Calculate camera distance
//            float distance = simd::length(target - position);
//
//            // Optionally clamp polar angle
//            // polarAngle = std::fmax(0.01f, std::fmin(static_cast<float>(M_PI) - 0.01f, polarAngle));
//            std::cout << up << "\n";
//            // Calculate new camera position
//            float x = target.x + distance * std::sin(polarAngle) * std::cos(azimuthalAngle) ;
//            float y = target.y + distance * std::sin(azimuthalAngle) ;
//            float z = target.z + distance * -std::cos(polarAngle) * std::cos(azimuthalAngle);
//
//            // Update camera position
//            position = simd::float3{x, y, z};
//
//            // Call your camera update method here
//            updateOLD(); // Assuming it's defined
//            AxisUpdated = false;
//            
//        }
//        else if (mode == TransformationMode::Zoom) {
//            position = target + (oldPosition - target) * ( 1/drag.x);
//            std::cout << (( drag.x)) << "\n";
//        }
//        updated = false;
//    }
//    
//    void updateOLD() {
//        oldPosition = position;
//    }
//    
//    void setPosition(simd_float2 drag, TransformationMode mode) {
//        if (mode == TransformationMode::Translate) {
//            auto dist = simd::length(target - position);
//            position = oldPosition + up * drag.y + right * drag.x;
//            target = position + simd_normalize(forward) * dist;
//        }
//        else if (mode == TransformationMode::Orbit) {
//            
//            simd_float3 vec = position - target;
//
//            position = (RotationY(drag.x * 0.01) * RotationX(drag.y * 0.01) * simd_make_float4(vec, 1) ).xyz + target;
//            updateOLD();
////            vec = target - position;
////            forward = simd::normalize(vec);
////            right = simd::normalize(simd::cross(forward, up));
////            up = simd::normalize(simd::cross(right, forward));
//        }
//        else if (mode == TransformationMode::Zoom) {
//            position = target + (position - target) * drag.x;
//        }
//        updated = false;
//    }
//    
//    void updateMatrix() {
//        if (!updated) {
//            simd_float3 zAxis = simd::normalize( target - position );
//            simd_float3 xAxis = simd::normalize(simd::cross(up, zAxis));
//
//            simd_float3 yAxis = simd::normalize(simd::cross(zAxis, xAxis));
//            
//            // dot is used because first the rotation is undone then
//            simd_float4 row0 = {xAxis.x, xAxis.y, xAxis.z, -simd::dot(xAxis, position)};
//            simd_float4 row1 = {yAxis.x, yAxis.y, yAxis.z, -simd::dot(yAxis, position)};
//            simd_float4 row2 = {zAxis.x, zAxis.y, zAxis.z, -simd::dot(zAxis, position)};
//            simd_float4 row3 = {0,      0,      0,      1         };
//
//            if (OrthoPara) {
//                d = tan(0.5 * M_PI * 0.25);
//                
//                r = d * aspectRatio;
//                
//                float k = farP - nearP;
//                
//                simd_float4 row4 = {1/r,  0,        0,                           0};
//                simd_float4 row5 = {0,        1/d,  0,                           0};
//                simd_float4 row6 = {0,        0,        farP/k,  -nearP * farP * (1/k)};
//                simd_float4 row7 = {0.0f,     0.0f,     1.0f,                        0};
//                
//                simd_float4x4 lookat =  simd_matrix_from_rows(row0, row1, row2, row3);
//                simd_float4x4 clipMat = simd_matrix_from_rows(row4, row5, row6, row7);
//                
//                viewMatrix = simd_mul(clipMat, lookat);
//                
//                updated = true;
//            } else {
//                float k = farP - nearP;
//                simd_float4 row4 = {2/r,  0,    0,    0};
//                simd_float4 row5 = {0,    2/d,  0,    0};
//                simd_float4 row6 = {0,    0,    1/k,  -nearP * (1/k)};
//                simd_float4 row7 = {0.0f, 0.0f, 0.0f, static_cast<float>(0.33 * simd_length(target - position))};
//                simd_float4x4 lookat =  simd_matrix_from_rows(row0, row1, row2, row3);
//                simd_float4x4 clipMat = simd_matrix_from_rows(row4, row5, row6, row7);
//                viewMatrix = simd_mul(clipMat, lookat);
//                updated = true;
//            }
// 
//        }
//    }
//    
//    
//    void updateAspectRatio(float ratio) {
//        aspectRatio = ratio;
//        r = ratio * d;
//        updated = false;
//    }
//};




struct Assets {
    std::vector<std::shared_ptr<GeometryNode<uint16>>> _NodesQueuePtr;
};


@interface Renderer : NSObject<MTKViewDelegate>
{
    @public
        id<MTLDevice> metalDevice;
        id<MTLCommandQueue> CommandQueue;
        id<MTLRenderPipelineState> PlaneRenderPipelineState;
        id<MTLRenderPipelineState> InfiniteGridRenderPipelineState;
        id<MTLComputePipelineState> BasicComputePipelineState;
        id<MTLComputePipelineState> ClearPassPipelineState;
        id<MTLRenderPipelineState> BasicRenderPipelineState;
        id<MTLRenderPipelineState> LightingRenderPipelineState;
        id<MTLDepthStencilState> BasicDepthStencilState;
        id<MTLTexture> offscreenTexture;
        id<MTLRenderPipelineState> nodeRenderPipelineState;
        id<MTLRenderPipelineState> instanceRenderPipelineState;
        id<MTLRenderPipelineState> PointCloudRenderPipelineState;
        id<MTLRenderPipelineState> MeshPointCloudRenderPipelineState;
        id<MTLLibrary> library;
        NSMutableArray<id<MTLRenderPipelineState>>* customRenderPipelineStates;
        id<MTLRenderPipelineState> predefinedRenderPipelineState[3];
        size_t width;
        size_t height;
        uint8_t sampleCount;
        std::vector<Shape<uint16>> _objectQueue;
        std::vector<ArrayShape> _objectQueueInstanced;
//        std::vector<PointCloud> _pointCloudQueue;
        std::vector<ObjMesh> _meshpointCloudObjectQueue;
        std::vector<ObjMesh> _meshObjectQueue;
        std::vector<ObjMesh> _lightingObjectQueue;
        std::vector<Shape<uint16>> _ComposedObjectQueue;
        MTLClearColor worldColour;
        std::vector<GeometryNode<uint16_t>> _NodesQueue;
        std::vector<GeometryNode<uint32_t>> _NodesQueue_32;
        std::vector<std::shared_ptr<GeometryNode<uint16>>> _NodesQueuePtr;
        Assets* ass;
        Camera3D cam;
    
    }

//@property std::vector<Shape<uint16>> objectQueue;
- (instancetype)initWithDevice:(id<MTLDevice>)device :(size_t)widthV :(size_t)heightV :(uint8_t)sampleCount;
- (MTLVertexDescriptor*) createPlaneMetalVertexDescriptor;

- (void)updateBaseImage:(MatrixH<3, uint8_t>&) layer;
- (void)createRenderPiplineState:(NSString*)vertexFunc :(NSString*)fragmentFunc;

@end

@implementation Renderer


- (instancetype)initWithDevice:(id<MTLDevice>)device :(size_t)widthV :(size_t)heightV :(uint8_t)sampleCount{
    NSLog(@"Init Nothing");
    self = [super init];
    NSLog(@"%@", device);
    NSError* error = nil;
    metalDevice = device;
    CommandQueue = [metalDevice newCommandQueue];
    width = 1920;
    height = 1080;
    self->sampleCount = sampleCount;
    worldColour = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    customRenderPipelineStates = [[NSMutableArray alloc] init];
    library = [metalDevice newDefaultLibrary];
    
    id<MTLFunction> planeVertexFunc = [library newFunctionWithName:@"planeVertexShader"];
    id<MTLFunction> planeFragmentFunc = [library newFunctionWithName:@"planeFragmentShader"];
    
    MTLVertexDescriptor* VertexDesc = [self createPlaneMetalVertexDescriptor];
    
    MTLRenderPipelineDescriptor *desc =  [[MTLRenderPipelineDescriptor alloc] init];
    [[desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    MTLRenderPipelineColorAttachmentDescriptor *colorAttachment = [desc colorAttachments][0];
    colorAttachment.blendingEnabled = YES;
    colorAttachment.rgbBlendOperation = MTLBlendOperationAdd;
    colorAttachment.alphaBlendOperation = MTLBlendOperationAdd;
    colorAttachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    colorAttachment.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    colorAttachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    colorAttachment.blendingEnabled = YES;
    colorAttachment.rgbBlendOperation = MTLBlendOperationAdd;
    colorAttachment.alphaBlendOperation = MTLBlendOperationAdd;
    colorAttachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    
    [desc setVertexDescriptor:VertexDesc];
    [desc setVertexFunction:planeVertexFunc];
    [desc setFragmentFunction:planeFragmentFunc];
    [desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [desc setRasterSampleCount:sampleCount];
    PlaneRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:desc error:&error];
    
    
    id<MTLFunction> InfiniteGridVertexFunc = [library newFunctionWithName:@"InfiniteGridVertexShader"];
    id<MTLFunction> InfiniteGridFragmentFunc = [library newFunctionWithName:@"InfiniteGridFragmentShader"];
    MTLRenderPipelineDescriptor *InfiniteGridDesc =  [[MTLRenderPipelineDescriptor alloc] init];
    [[InfiniteGridDesc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    colorAttachment = [InfiniteGridDesc colorAttachments][0];
    colorAttachment.blendingEnabled = YES;
    colorAttachment.rgbBlendOperation = MTLBlendOperationAdd;
    colorAttachment.alphaBlendOperation = MTLBlendOperationAdd;
    colorAttachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    
    colorAttachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    colorAttachment.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    colorAttachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    [InfiniteGridDesc setVertexFunction:InfiniteGridVertexFunc];
    [InfiniteGridDesc setFragmentFunction:InfiniteGridFragmentFunc];
    [InfiniteGridDesc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [InfiniteGridDesc setRasterSampleCount:sampleCount];
    InfiniteGridRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:InfiniteGridDesc error:&error];
    
    id<MTLFunction> basicVertexFunc = [library newFunctionWithName:@"basicVertexShader"];
    id<MTLFunction> basicFragmentFunc = [library newFunctionWithName:@"basicFragmentShader"];
    
    MTLVertexDescriptor* basicVertexDesc = [self createBasicMetalVertexDescriptor];
    
    MTLRenderPipelineDescriptor *basic_desc =  [[MTLRenderPipelineDescriptor alloc] init];
    [[basic_desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [basic_desc setVertexDescriptor:basicVertexDesc];
    [basic_desc setVertexFunction:basicVertexFunc];
    [basic_desc setFragmentFunction:basicFragmentFunc];
    [basic_desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [basic_desc setRasterSampleCount:sampleCount];
    BasicRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:basic_desc error:&error];
    
    MTLDepthStencilDescriptor *depthStencilDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStencilDesc.depthWriteEnabled = YES;
    
    // Create the depth stencil state from the device.
    BasicDepthStencilState = [metalDevice newDepthStencilStateWithDescriptor:depthStencilDesc];
    
    id<MTLFunction> instanceVertexFunc = [library newFunctionWithName:@"instanceVertexShader"];
    id<MTLFunction> instanceFragmentFunc = [library newFunctionWithName:@"basicFragmentShader"]; // same fragment shader
    
    // Set up the render pipeline descriptor for instanced rendering.
    MTLRenderPipelineDescriptor *instancedDesc = [[MTLRenderPipelineDescriptor alloc] init];
    [[instancedDesc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [instancedDesc setVertexDescriptor:basicVertexDesc];
    [instancedDesc setVertexFunction:instanceVertexFunc];
    [instancedDesc setFragmentFunction:instanceFragmentFunc];
    [instancedDesc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [instancedDesc setRasterSampleCount:sampleCount];
    // Create the pipeline state object
    instanceRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:instancedDesc error:&error];
    
    
    id<MTLFunction> pointCloudVertexFunc = [library newFunctionWithName:@"pointCloudVertexShader"];
    id<MTLFunction> pointCloudFragmentFunc = [library newFunctionWithName:@"pointCloudFragmentShader"];
    
    // Create a vertex descriptor appropriate for instanced rendering.
    MTLVertexDescriptor* pointCloudVertexDesc = [self createPointCloudVertexDescriptor];
    // Set up the render pipeline descriptor for instanced rendering.
    MTLRenderPipelineDescriptor *pointCloudDesc = [[MTLRenderPipelineDescriptor alloc] init];
    [[pointCloudDesc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [pointCloudDesc setVertexDescriptor:pointCloudVertexDesc];
    [pointCloudDesc setVertexFunction:pointCloudVertexFunc];
    [pointCloudDesc setFragmentFunction:pointCloudFragmentFunc];
    [pointCloudDesc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [pointCloudDesc setRasterSampleCount:sampleCount];
    // Create the pipeline state object
    PointCloudRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:pointCloudDesc error:&error];
    
    id<MTLFunction> meshPointCloudVertexFunc = [library newFunctionWithName:@"MeshPointCloudVertexShader"];
    id<MTLFunction> meshPointCloudFragmentFunc = [library newFunctionWithName:@"meshPointCloudFragmentShader"];
    
    
    MTLRenderPipelineDescriptor *meshPointCloud_desc =  [[MTLRenderPipelineDescriptor alloc] init];
    [[meshPointCloud_desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [meshPointCloud_desc setVertexDescriptor:basicVertexDesc];
    [meshPointCloud_desc setVertexFunction:meshPointCloudVertexFunc];
    [meshPointCloud_desc setFragmentFunction:meshPointCloudFragmentFunc];
    [meshPointCloud_desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [meshPointCloud_desc setRasterSampleCount:sampleCount];
    MeshPointCloudRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:meshPointCloud_desc error:&error];
    
    id<MTLFunction> lightingVertexFunc = [library newFunctionWithName:@"lightingVertexShader"];
    id<MTLFunction> lightingFragmentFunc = [library newFunctionWithName:@"lightingFragmentShader"];
    
    MTLRenderPipelineDescriptor *lighting_desc =  [[MTLRenderPipelineDescriptor alloc] init];
    [[lighting_desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [lighting_desc setVertexDescriptor:basicVertexDesc];
    [lighting_desc setVertexFunction:lightingVertexFunc];
    [lighting_desc setFragmentFunction:lightingFragmentFunc];
    [lighting_desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [lighting_desc setRasterSampleCount:sampleCount];
    LightingRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:lighting_desc error:&error];
    
    id<MTLFunction> NodeVertexFunc = [library newFunctionWithName:@"NodeVertexShader"];
    
    // Set up the render pipeline descriptor for instanced rendering.
    MTLRenderPipelineDescriptor *node_Desc = [[MTLRenderPipelineDescriptor alloc] init];
    [[node_Desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [node_Desc setVertexDescriptor:basicVertexDesc];
    [node_Desc setVertexFunction:NodeVertexFunc];
    [node_Desc setFragmentFunction:lightingFragmentFunc];
    [node_Desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [node_Desc setRasterSampleCount:sampleCount];
    
    // Create the pipeline state object
    nodeRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:node_Desc error:&error];
    
    id<MTLFunction> pointCloudNodeVertexFunc = [library newFunctionWithName:@"pointCloudNodeVertexShader"];
    
    // Create a vertex descriptor appropriate for instanced rendering.

    MTLRenderPipelineDescriptor *pointCloudNodeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    [[pointCloudDesc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [pointCloudDesc setVertexDescriptor:pointCloudVertexDesc];
    [pointCloudDesc setVertexFunction:pointCloudNodeVertexFunc];
    [pointCloudDesc setFragmentFunction:pointCloudFragmentFunc];
    [pointCloudDesc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [pointCloudDesc setRasterSampleCount:sampleCount];
    // Create the pipeline state object
    id<MTLRenderPipelineState> PointCloudNodeRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:pointCloudDesc error:&error];
    
    id<MTLFunction> BillboardNodeVertexFunc = [library newFunctionWithName:@"BillboardNodeVertexShader"];
    
    // Set up the render pipeline descriptor for instanced rendering.
    MTLRenderPipelineDescriptor *node_billboard_Desc = [[MTLRenderPipelineDescriptor alloc] init];
    [[node_billboard_Desc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [node_billboard_Desc setVertexDescriptor:basicVertexDesc];
    [node_billboard_Desc setVertexFunction:BillboardNodeVertexFunc];
    [node_billboard_Desc setFragmentFunction:lightingFragmentFunc];
    [node_billboard_Desc setDepthAttachmentPixelFormat:MTLPixelFormatDepth16Unorm];
    [node_billboard_Desc setRasterSampleCount:sampleCount];
    id<MTLRenderPipelineState> BillboardNodeRenderPipelineState = [metalDevice newRenderPipelineStateWithDescriptor:node_billboard_Desc error:&error];

    predefinedRenderPipelineState[0] = nodeRenderPipelineState;
    predefinedRenderPipelineState[1] = PointCloudNodeRenderPipelineState;
    predefinedRenderPipelineState[2] = BillboardNodeRenderPipelineState;
    
    
    cam = Camera3D();
    auto A = MatrixH<1, uint8_t>({1, 0, 0, 0});
    MatrixH<3, uint8_t> clearImg = MatrixH<3, uint8_t>::repeating({(size_m)height, (size_m)width}, A);
//    auto c = (MatrixH<3, float>)clearImg;
//    c.Slice({ {{0, 10}}, {{0, 10}} }).printNonCont();
    
    offscreenTexture = clearImg.ToMTLTexture();
//    printArray((uint8_t*)offscreenTexture.buffer.contents, 100);
    
    return self;
}

- (void) createRenderPiplineState:(NSString*)vertexFunc :(NSString*)fragmentFunc {
    id<MTLFunction> CustomVertexFunc   = [library newFunctionWithName:vertexFunc];
    id<MTLFunction> CustomFragmentFunc = [library newFunctionWithName:fragmentFunc];

    MTLVertexDescriptor* basicVertexDesc = [self createBasicMetalVertexDescriptor];

    // Pipeline descriptor
    MTLRenderPipelineDescriptor *Custom_Desc = [[MTLRenderPipelineDescriptor alloc] init];
    Custom_Desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    Custom_Desc.vertexDescriptor                = basicVertexDesc;
    Custom_Desc.vertexFunction                  = CustomVertexFunc;
    Custom_Desc.fragmentFunction                = CustomFragmentFunc;
    Custom_Desc.depthAttachmentPixelFormat      = MTLPixelFormatDepth16Unorm;
    Custom_Desc.rasterSampleCount               = sampleCount;

    NSError *error = nil;

    auto customPipelineState =
        [metalDevice newRenderPipelineStateWithDescriptor:Custom_Desc error:&error];

    if (error) {
        NSLog(@"Pipeline creation failed: %@", error);
    }

    [customRenderPipelineStates addObject:customPipelineState];
}



- (void)updateBaseImage:(MatrixH<3, uint8_t>&) layer {
    if (layer.shape[2] != 4) {
        throw std::runtime_error("MatrixH: updateBaseImage: expected 4 channels (RGBA), got " + std::to_string(layer.shape[2]));
    }
    if (layer.total_size == 0 || !layer.buffer) {
        layer.buffer = new uint8_t[width * height * 4];
        layer.shape[0] = height;
        layer.shape[1] = width;
        layer.shape[2] = 4;
        layer.total_size = height * width * 4;
    }
    
    if (!offscreenTexture || layer.shape[0] != height || layer.shape[1] != width) {
        NSLog(@"Texture was Wrong; Setting the correct one");
        width = layer.shape[1];
        height = layer.shape[0];
        MTLTextureDescriptor* drawableDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                                                                                width:(NSUInteger)width
                                                                                               height:(NSUInteger)height
                                                                                            mipmapped:NO];
        drawableDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        // Use shared storage so the CPU can read the texture data.
        drawableDesc.storageMode = MTLStorageModeShared;
    
    
        offscreenTexture = [metalDevice newTextureWithDescriptor:drawableDesc];
    }

    MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
    NSUInteger bytesPerRow = width * 4;  // 4 bytes per pixel for BGRA8
    
    
//    id<MTLBuffer> buffer = [metalDevice newBufferWithBytesNoCopy:layer.buffer length:bytesPerRow * height options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        delete[] (uint8_t*)pointer;
//    }];

//    offscreenTexture = layer.ToMTLTexture();
    
    if (layer.metalBuffer) {
        id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder copyFromBuffer:layer.metalBuffer
                       sourceOffset:0
                  sourceBytesPerRow:bytesPerRow
                sourceBytesPerImage:bytesPerRow * height
                         sourceSize:region.size
                          toTexture:offscreenTexture
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:region.origin];
        
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        // BUG FIX: Previously missing waitUntilCompleted caused the layer to go out of scope  its buffer being freed before the gpu completed the execution . Small textures
        // (10x10x4) rendered before GPU finished uploading, showing black/corrupted pixels. Large
        // textures (100x100+) worked by accident because i think that large chucks of memory dond get rycyled so easily so maybe it still had data

    } else {
        // Direct CPU-to-texture upload. We don't use layer.metalBuffer here because
        // MatrixH deliberately avoids creating Metal buffers for small matrices as an
        // optimization. For small data, the overhead of Metal buffer allocation exceeds
        // the benefit, so replaceRegion:withBytes: is more efficient.
        [offscreenTexture replaceRegion:region
                            mipmapLevel:0
                              withBytes:layer.buffer
                            bytesPerRow:bytesPerRow];
    }
//    layer.print();
    

//    [offscreenTexture replaceRegion:region
//                        mipmapLevel:0
//                          withBytes:layer.buffer
//                        bytesPerRow:bytesPerRow];
}



- (void)drawInMTKView:(nonnull MTKView *)view {
    id<CAMetalDrawable> drawable = [view currentDrawable];
    id<MTLCommandBuffer> cmdBuffer = [CommandQueue commandBuffer];
    
    MTLRenderPassDescriptor *passDescriptor = [view currentRenderPassDescriptor];
    
//    passDescriptor.defaultRasterSampleCount = 4;
//    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];
//    [blitEncoder copyFromTexture:offscreenTexture
//                     sourceSlice:0
//                     sourceLevel:0
//                    sourceOrigin:MTLOriginMake(0, 0, 0)
//                      sourceSize:MTLSizeMake(offscreenTexture.width, offscreenTexture.height, 1)
//                       toTexture:drawable.texture
//                destinationSlice:0
//                destinationLevel:0
//               destinationOrigin:MTLOriginMake(0, 0, 0)];
//    [blitEncoder endEncoding];
    
//    passDescriptor.colorAttachments[0].texture = drawable.texture;
    passDescriptor.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    passDescriptor.colorAttachments[0].clearColor = worldColour;
//    passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    
    passDescriptor.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    
//    passDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
//    passDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
    passDescriptor.depthAttachment.clearDepth = 1.0;
   
    id<MTLRenderCommandEncoder> cmdEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:passDescriptor];

#if !TARGET_OS_IPHONE
    float vertexData[] = {-1, -1, 0, 1,
        1, -1, 1, 1,
        -1,  1, 0, 0,
        1,  1, 1, 0};
#endif

#if TARGET_OS_IPHONE
    float vertexData[] = {-1, -1, 1, 0,
        1, -1, 1, 1,
        -1,  1, 0, 0,
        1,  1, 0, 1};
#endif
//
    [cmdEncoder setDepthStencilState:BasicDepthStencilState];
    [cmdEncoder setRenderPipelineState:PlaneRenderPipelineState];
    [cmdEncoder setVertexBytes:vertexData length:  16 * sizeof(float) atIndex:0];
    [cmdEncoder setFragmentTexture:offscreenTexture atIndex:0];
    [cmdEncoder setRenderPipelineState:PlaneRenderPipelineState];
    [cmdEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [cmdEncoder setCullMode:MTLCullModeNone];
    [cmdEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
    if (!cmdEncoder) {
        NSLog(@"Drawable is nil! Nothing will be displayed.");
        return;
    }
    
    cam.updateMatrix();
    simd_float4x4 transform;
    [cmdEncoder setRenderPipelineState:InfiniteGridRenderPipelineState];
    [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:0];
    [cmdEncoder setVertexBytes:&cam.target length:sizeof(simd_float3) atIndex:1];
    [cmdEncoder setVertexBytes:&cam.inverseProjectionMatrix length:sizeof(simd_float4x4) atIndex:2];
    [cmdEncoder setFragmentBytes:&cam.target length:sizeof(simd_float3) atIndex:0];
    [cmdEncoder setFragmentBytes:&cam.inverseProjectionMatrix length:sizeof(simd_float4x4) atIndex:1];
    [cmdEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    
    if (0 < _objectQueue.size() || 0 < _meshObjectQueue.size()) {
        [cmdEncoder setRenderPipelineState:BasicRenderPipelineState];
        
        for (int i = 0; i < _objectQueue.size(); i++) {
            
            Shape<uint16> shape = _objectQueue[i];
            _objectQueue[i].buildBuffers(metalDevice);
            transform = _objectQueue[i].Transformer();
            [cmdEncoder setVertexBuffer:_objectQueue[i].vertexBuffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(simd_float4x4) atIndex:1];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            [cmdEncoder drawIndexedPrimitives:_objectQueue[i].drawType indexCount:shape.indexCount indexType:MTLIndexTypeUInt16 indexBuffer:_objectQueue[i].indexBuffer indexBufferOffset:0];
            
        }
        
        for (int i = 0; i < _meshObjectQueue.size(); i++) {
            transform = Identity();
            [cmdEncoder setVertexBuffer:_meshObjectQueue[i].metalMesh.vertexBuffers[0].buffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(simd_float4x4) atIndex:1];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            
            for (int j = 0; j < _meshObjectQueue[i].metalMesh.submeshes.count; j++) {
                
                [cmdEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:_meshObjectQueue[i].metalMesh.submeshes[j].indexCount indexType:_meshObjectQueue[i].metalMesh.submeshes[j].indexType indexBuffer:_meshObjectQueue[i].metalMesh.submeshes[j].indexBuffer.buffer indexBufferOffset:_meshObjectQueue[i].metalMesh.submeshes[j].indexBuffer.offset];
            }
            
        }
    }
    if (_objectQueueInstanced.size() > 0) {
        [cmdEncoder setRenderPipelineState:instanceRenderPipelineState];
        
        for (int i = 0; i < _objectQueueInstanced.size(); i++) {
            _objectQueueInstanced[i].buildBuffer(metalDevice);
            _objectQueueInstanced[i].shape.buildBuffers(metalDevice);
            transform = _objectQueueInstanced[i].shape.Transformer();
            
            [cmdEncoder setVertexBuffer:_objectQueueInstanced[i].shape.vertexBuffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(matrix_float4x4) atIndex:1];
            [cmdEncoder setVertexBuffer:_objectQueueInstanced[i].transformBuffer offset:0 atIndex:3];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            [cmdEncoder drawIndexedPrimitives:_objectQueueInstanced[i].shape.drawType
                                   indexCount:_objectQueueInstanced[i].shape.indexCount
                                    indexType:MTLIndexTypeUInt16
                                  indexBuffer:_objectQueueInstanced[i].shape.indexBuffer
                            indexBufferOffset:0
                                instanceCount:_objectQueueInstanced[i].instanceCount];
        }
    }
//    [cmdEncoder setRenderPipelineState:PointCloudRenderPipelineState];
//
//    for (int i = 0; i < _pointCloudQueue.size(); i++) {
//        _pointCloudQueue[i].buildBuffers(metalDevice);
//        transform = Identity();
//
//        [cmdEncoder setVertexBuffer:_pointCloudQueue[i].pointsBuffer offset:0 atIndex:0];
//        [cmdEncoder setVertexBytes:&transform length:sizeof(matrix_float4x4) atIndex:1];
//        [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
//        [cmdEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:_pointCloudQueue[i].points.total_size];
//    }
    if (_meshpointCloudObjectQueue.size() > 0) {
        [cmdEncoder setRenderPipelineState:MeshPointCloudRenderPipelineState];
        for (int i = 0; i < _meshpointCloudObjectQueue.size(); i++) {
            transform = Identity();
            [cmdEncoder setVertexBuffer:_meshpointCloudObjectQueue[i].metalMesh.vertexBuffers[0].buffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(simd_float4x4) atIndex:1];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            
            for (int j = 0; j < _meshpointCloudObjectQueue[i].metalMesh.submeshes.count; j++) {
                
                [cmdEncoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip indexCount:_meshpointCloudObjectQueue[i].metalMesh.submeshes[j].indexCount indexType:_meshpointCloudObjectQueue[i].metalMesh.submeshes[j].indexType indexBuffer:_meshpointCloudObjectQueue[i].metalMesh.submeshes[j].indexBuffer.buffer indexBufferOffset:_meshpointCloudObjectQueue[i].metalMesh.submeshes[j].indexBuffer.offset];
            }
            
        }
    }
    if (0 < _lightingObjectQueue.size() || 0 < _ComposedObjectQueue.size()) {
        [cmdEncoder setRenderPipelineState:LightingRenderPipelineState];
        for (int i = 0; i < _lightingObjectQueue.size(); i++) {
            transform = Identity();
            
            [cmdEncoder setVertexBuffer:_lightingObjectQueue[i].metalMesh.vertexBuffers[0].buffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(simd_float4x4) atIndex:1];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            
            [cmdEncoder setFragmentTexture:_lightingObjectQueue[i].texture atIndex:0];
            for (int j = 0; j < _lightingObjectQueue[i].metalMesh.submeshes.count; j++) {
                
                [cmdEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:_lightingObjectQueue[i].metalMesh.submeshes[j].indexCount indexType:_lightingObjectQueue[i].metalMesh.submeshes[j].indexType indexBuffer:_lightingObjectQueue[i].metalMesh.submeshes[j].indexBuffer.buffer indexBufferOffset:_lightingObjectQueue[i].metalMesh.submeshes[j].indexBuffer.offset];
            }
            
        }
        for (int i = 0; i < _ComposedObjectQueue.size(); i++) {
            
            Shape<uint16> shape = _ComposedObjectQueue[i];
            _ComposedObjectQueue[i].buildBuffers(metalDevice);
            transform = _ComposedObjectQueue[i].Transformer();
            [cmdEncoder setVertexBuffer:_ComposedObjectQueue[i].vertexBuffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&transform length:sizeof(simd_float4x4) atIndex:1];
            [cmdEncoder setFragmentBytes:&shape.textured length:sizeof(bool) atIndex:0];
            [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            [cmdEncoder setFragmentTexture:_ComposedObjectQueue[i].texture atIndex:0];
            [cmdEncoder drawIndexedPrimitives:_ComposedObjectQueue[i].drawType indexCount:shape.indexCount indexType:MTLIndexTypeUInt16 indexBuffer:_ComposedObjectQueue[i].indexBuffer indexBufferOffset:0];
            
        }
    }
//
//    [cmdEncoder setRenderPipelineState:nodeRenderPipelineState];
    [cmdEncoder setVertexBytes:&cam.viewMatrix length:sizeof(simd_float4x4) atIndex:2];
    
    // -1: undefined
    // 0-100: Predefined
    // 101-.. Custom
    int active_state = -1;
    for (int i=0; i < _NodesQueue.size(); i++) {
        
        _NodesQueue[i].draw(cmdEncoder, metalDevice, predefinedRenderPipelineState, customRenderPipelineStates, &cam, active_state);
    }
    
    for (int i=0; i < _NodesQueue_32.size(); i++) {
        
        _NodesQueue_32[i].draw(cmdEncoder, metalDevice, predefinedRenderPipelineState, customRenderPipelineStates, &cam, active_state);
    }

    for (int i=0; i < _NodesQueuePtr.size(); i++) {
        
        _NodesQueuePtr[i]->draw(cmdEncoder, metalDevice, predefinedRenderPipelineState, customRenderPipelineStates, &cam, active_state);
    }
    
    [cmdEncoder endEncoding];
    [cmdBuffer presentDrawable:drawable];
    [cmdBuffer commit];
//    [cmdBuffer waitUntilCompleted]; // BUGFIX: Delaying the cpu
    
}

- (MTLVertexDescriptor*) createPlaneMetalVertexDescriptor {
    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init ];
    // Store position in attribute[0]
    mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[0].offset = 0;
    mtlVertexDescriptor.attributes[0].bufferIndex = 0;

    // Store texture coordinates in attribute[1]
    mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[1].offset = sizeof(simd_float2);
    mtlVertexDescriptor.attributes[1].bufferIndex = 0;

    // Set stride to twice the bytes per float2.
    mtlVertexDescriptor.layouts[0].stride = 2 * sizeof(simd_float2);
    mtlVertexDescriptor.layouts[0].stepRate = 1;
    mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    
    return mtlVertexDescriptor;
}

- (MTLVertexDescriptor*) createPointCloudVertexDescriptor {
    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init ];
    // Store position in attribute[0]
    mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[0].offset = 0;
    mtlVertexDescriptor.attributes[0].bufferIndex = 0;

    // Store texture coordinates in attribute[1]
    mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[1].offset = offsetof(Point3D, colour);
    mtlVertexDescriptor.attributes[1].bufferIndex = 0;

    // Set stride to the bytes per float3 and float4.
    mtlVertexDescriptor.layouts[0].stride = sizeof(Point3D);
    mtlVertexDescriptor.layouts[0].stepRate = 1;
    mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    
    return mtlVertexDescriptor;
}

- (MTLVertexDescriptor*) createBasicMetalVertexDescriptor {
    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init ];
    // Store position in attribute[0]
    mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[0].offset = 0;
    mtlVertexDescriptor.attributes[0].bufferIndex = 0;

    // Store texture coordinates in attribute[1]
    mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[1].offset = offsetof(Vertex3D, colour);
    mtlVertexDescriptor.attributes[1].bufferIndex = 0;

    
    mtlVertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[2].offset = offsetof(Vertex3D, textureCoordinates);
    mtlVertexDescriptor.attributes[2].bufferIndex = 0;
    
    mtlVertexDescriptor.attributes[3].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[3].offset = offsetof(Vertex3D, normal);
    mtlVertexDescriptor.attributes[3].bufferIndex = 0;
    
    // Set stride to twice the bytes per float2.
    mtlVertexDescriptor.layouts[0].stride = sizeof(Vertex3D);
    mtlVertexDescriptor.layouts[0].stepRate = 1;
    mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    
    return mtlVertexDescriptor;
}


- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    cam.updateAspectRatio(size.width / size.height);
}



@end

@interface MyMetalView : MTKView {
    @public
        bool shiftPressed;
        bool isActiveView;
        simd_float4 intPointInWorldSpaceOld;
        bool updatedOldRay;
}



@property (nonatomic, strong) NSMutableSet<NSNumber *> *pressedKeys;
@property (nonatomic, strong) NSTimer *cameraTimer;



@end

@implementation MyMetalView


- (BOOL)acceptsFirstResponder {
    return YES;
}

- (instancetype)initWithFrame:(CGRect)frameRect device:(id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    
    if (self) {
        self.pressedKeys = [NSMutableSet set];
        #if !TARGET_OS_IPHONE
        self.cameraTimer = [NSTimer scheduledTimerWithTimeInterval:(1.0 / 120.0)
                                                            target:self
                                                          selector:@selector(updateCamera)
                                                          userInfo:nil
                                                           repeats:YES];
        #endif
        shiftPressed = false;
        isActiveView = true;
    }
    return self;
}

#if TARGET_OS_IPHONE
    - (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
        NSArray<UITouch *> *allTouches = [[event allTouches] allObjects];
        NSUInteger touchCount = [allTouches count];
        
        Renderer *renderer = (Renderer *)self.delegate;

        // --- CASE 1: Two-Finger Pan (Priority) ---
        // We handle this FIRST to prevent accidentally clicking objects while trying to pan
        printf("%lu fingers detected. \n", static_cast<unsigned long>(touchCount));
        if (touchCount == 2) {
            
            float totalDeltaX = 0;
            float totalDeltaY = 0;
            
            // Calculate the average movement of BOTH fingers
            // This fixes the issue where one stationary finger cancels out the movement
            for (UITouch *t in allTouches) {
                CGPoint cur = [t locationInView:self];
                CGPoint prev = [t previousLocationInView:self];
                totalDeltaX += (cur.x - prev.x);
                totalDeltaY += (cur.y - prev.y);
            }
            
            float avgDeltaX = totalDeltaX / 2.0f;
            float avgDeltaY = totalDeltaY / 2.0f;
            
            // Optional: Touch screens need a slight boost compared to mouse
            float touchSensitivity = 2.5f;
            
            // Send Translate Command immediately and return
            renderer->cam.handleMouseEvents(avgDeltaX * touchSensitivity,
                                            avgDeltaY * touchSensitivity,
                                            NO,
                                            NO, // isShift
                                            TransformationMode::Translate);
            return;
        }

        // --- CASE 2: Single Finger (Orbit or Object Drag) ---
        if (touchCount == 1) {
            UITouch *touch = [touches anyObject];
            CGPoint currentPos = [touch locationInView:self];
            CGPoint previousPos = [touch previousLocationInView:self];
            
            float deltaX = currentPos.x - previousPos.x;
            float deltaY = currentPos.y - previousPos.y;
            
            bool onShape = false;
            
            // --- Object Dragging Logic ---
            CGPoint mouseLocation = currentPos;
            simd_float2 currentMousePos = simd_make_float2((mouseLocation.x / self.frame.size.width), (mouseLocation.y / self.frame.size.height));
            simd_float4 currentMouseDrag = simd_make_float4((deltaX / (float)self.frame.size.width), (-deltaY / (float)self.frame.size.height), 0, 1);
            
            currentMouseDrag.xyz *= 2;
            currentMousePos = 2 * currentMousePos - 1.0f;
            simd_float4 rayInViewSpace = simd_make_float4(currentMousePos, 0, 1);
            simd_float4 intPoint;
            
            for (int i = 0; i < renderer->_objectQueue.size(); i++) {
                if (renderer->_objectQueue[i].dragable) {
                    // ... (Keep your existing Raycast logic exactly as is) ...
                    if (renderer->_objectQueue[i].intersectRay(simd_make_float4(0, 0, 1, 1), rayInViewSpace, renderer->cam.viewMatrix, intPoint, true)) {
                        onShape = true;
                        // ... (Your existing drag logic) ...
                        currentMouseDrag *= intPoint.w;
                        currentMouseDrag.z = intPoint.z;
                        currentMouseDrag.zw *= 0;
                        currentMouseDrag = simd_mul(renderer->cam.inverseProjectionMatrix, currentMouseDrag);
                        renderer->_objectQueue[i].position += simd_make_float3(currentMouseDrag.x, currentMouseDrag.y, 0);
                        renderer->_objectQueue[i].triggerCallbacks();
                        break;
                    }
                }
            }
            
            // --- Orbit Logic (Only if we didn't touch an object) ---
            if (!onShape) {
                renderer->cam.handleMouseEvents(deltaX, -deltaY, NO, NO, TransformationMode::Orbit);
            }
        }
    }
    
    - (void)didMoveToWindow {
        [super didMoveToWindow];
        // iOS doesn't strictly require makeFirstResponder for gestures, but good for menus
        [self becomeFirstResponder];

        // 1. Add the Pinch Gesture
        UIPinchGestureRecognizer *pinch = [[UIPinchGestureRecognizer alloc]
                                           initWithTarget:self
                                           action:@selector(handlePinch:)];
        [self addGestureRecognizer:pinch];
    }

    - (void)handlePinch:(UIPinchGestureRecognizer *)gesture {
        // 2. Get the scale (iOS starts at 1.0)
        float currentScale = gesture.scale;

        Renderer *renderer = (Renderer *)self.delegate;
        if (renderer) {
            // macOS: passed (1 + mag).
            // iOS: 'scale' IS (1 + mag), so pass it directly.
            renderer->cam.handleMouseEvents(currentScale, currentScale, NO, NO, TransformationMode::Zoom);
        }

        // 3. CRITICAL: Reset the scale to 1.0
        // On iOS, the scale accumulates. If you don't reset it,
        // the zoom will accelerate exponentially (1.1 * 1.2 * 1.3...).
        // By resetting to 1.0, we get the "delta" change for the next frame.
//        gesture.scale = 1.0;

        // 4. Handle End State
        if (gesture.state == UIGestureRecognizerStateEnded) {
            renderer->cam.updateOLD();
        }
    }
#endif
#if !TARGET_OS_IPHONE
- (void)mouseDragged:(NSEvent *)event {
    float deltaX = event.deltaX;
    float deltaY = event.deltaY;
    BOOL isShift = (event.modifierFlags & NSEventModifierFlagShift) != 0;
    bool onShape = false;
    Renderer *renderer = (Renderer *)self.delegate;
//    NSPoint mouseLocation = [self convertPoint:[event locationInWindow] fromView:nil];
//    simd_float2 currentMousePos = simd_make_float2((mouseLocation.x / self.frame.size.width), (mouseLocation.y / self.frame.size.height));
//    simd_float4 currentMouseDrag = simd_make_float4((deltaX / (float)self.frame.size.width), (-deltaY / (float)self.frame.size.height), 0, 1);
//    
//    currentMouseDrag.xyz *= 2;
//    currentMousePos = 2 * currentMousePos - 1.0f;
//    simd_float4 rayInViewSpace = simd_make_float4(currentMousePos, 0, 1);
//    simd_float4 intPoint;
//    for (int i = 0; i < renderer->_objectQueue.size(); i++) {
//        if (renderer->_objectQueue[i].dragable) {
//            if (renderer->_objectQueue[i].intersectRay(simd_make_float4(0, 0, 1, 1), rayInViewSpace, renderer->cam.viewMatrix, intPoint, true)) {
//                onShape = true;
//                currentMouseDrag *= intPoint.w;
//                currentMouseDrag.z = intPoint.z;
//                currentMouseDrag.zw *= 0;
//                std::cout << "current W " <<  intPoint.w << "\n";
//                std::cout << "intersection pont: "<< intPoint << " cursor in ViewSpace: " << rayInViewSpace << "current MouseDrag" << currentMouseDrag << "\n";
//                currentMouseDrag = simd_mul(renderer->cam.inverseProjectionMatrix, currentMouseDrag);
//                std::cout << currentMouseDrag << "\n";
//                renderer->_objectQueue[i].position += simd_make_float3(currentMouseDrag.x,currentMouseDrag.y,0);
//                renderer->_objectQueue[i].triggerCallbacks();
//                break;
//            }
//        }
//         
//    }

//    if (!onShape) {
        renderer->cam.handleMouseEvents(deltaX, -deltaY, NO, NO, isShift ? TransformationMode::Translate : TransformationMode::Orbit);
//    }
}
    
- (void)rightMouseDragged:(NSEvent *)event {
    float deltaX = event.deltaX;
    float deltaY = event.deltaY;
    Renderer *renderer = (Renderer *)self.delegate;
    renderer->cam.handleMouseEvents(deltaX, deltaY, NO, NO, TransformationMode::Translate);
}

- (void)scrollWheel:(NSEvent *)event {
    float deltaY = event.scrollingDeltaY;
    float deltaX = event.scrollingDeltaX;
    Renderer *renderer = (Renderer *)self.delegate;
    renderer->cam.handleMouseEvents(deltaX, deltaY, NO, NO, TransformationMode::Translate);
}

- (void) updateCamera {
    if (!isActiveView) {
        return;
    }
    float deltaY = 0;
    float deltaX = 0;
    float deltaZ = 0;
    float amount = 20;
    
    if ([self.pressedKeys containsObject:@123]) { // Left
        deltaX = -amount;
    }
    if ([self.pressedKeys containsObject:@124]) { // Right
        deltaX = amount;
    }
    if ([self.pressedKeys containsObject:@125]) { // Down
        deltaY = -amount;
    }
    if ([self.pressedKeys containsObject:@126]) { // Up
        deltaY = amount;
    }
    if ([self.pressedKeys containsObject:@13]) { // Up
        deltaZ = amount;
    }
    if ([self.pressedKeys containsObject:@1]) { // Up
        deltaZ = -amount;
    }

    if (deltaX != 0 || deltaY != 0 || deltaZ != 0) {
        Renderer *renderer = (Renderer *)self.delegate;
        if (shiftPressed) {
            renderer->cam.updateRawCamPosition(simd_make_float3(deltaX * 0.005, deltaY * 0.005, deltaZ * 0.005));
        } else {
            renderer->cam.handleMouseEvents(deltaX , deltaY , NO, NO, TransformationMode::Translate);
        }
        
        
    }
    if (shiftPressed) {
//        std::cout << shiftPressed << "\n";
    }
        
//    switch (event.keyCode) {
//        case 123:
//            deltaX = amount;
//            break;
//        case 124:
//            deltaX = -amount;
//            break;
//        case 125:
//            deltaY = +amount;
//            break;
//        case 126:
//            deltaY = -amount;
//            break;
//        case 13:
//            deltaZ = -0.1;
//            break;
//        case 1:
//            deltaZ = 0.1;
//            break;
//
//    }

}

- (void)keyDown:(NSEvent *)event {
    // There is no official code for shift
    [self.pressedKeys addObject:@(event.keyCode)];
    if (event.modifierFlags & NSEventModifierFlagShift) {
        shiftPressed = true;
    }
}

- (void)keyUp:(NSEvent *)event {
    [self.pressedKeys removeObject:@(event.keyCode)];
    if (event.modifierFlags & NSEventModifierFlagShift) {
        shiftPressed = false;
    }
    
}

//- (void)magnifyWithEvent:(NSEvent *)event {
//    float mag = event.magnification;
//    Renderer *renderer = (Renderer *)self.delegate;
//    std::cout << mag << "\n";
//    renderer->cam.handleMouseEvents(1+mag*0.1, 1+mag*0.1, NO, NO, TransformationMode::Zoom);
////    [renderer.cam handleMagnificationWithFactor:mag];
//}

- (void)viewDidMoveToWindow {
    [super viewDidMoveToWindow];
    [self.window makeFirstResponder:self];

    NSMagnificationGestureRecognizer *mag = [[NSMagnificationGestureRecognizer alloc]
        initWithTarget:self action:@selector(handleMagnify:)];
    [self addGestureRecognizer:mag];
}

- (void)handleMagnify:(NSMagnificationGestureRecognizer *)gesture {
    float magnification = gesture.magnification;  // e.g., +0.05 for slight zoom-in

    Renderer *renderer = (Renderer *)self.delegate;
    if (renderer) {
        renderer->cam.handleMouseEvents(1+magnification, 1+magnification, NO, NO, TransformationMode::Zoom);
    }
    if (gesture.state == NSGestureRecognizerStateEnded) {
        renderer->cam.updateOLD();
        gesture.magnification = 0.0;
    }
    // Reset or accumulate magnification based on your camera logic
//    gesture.magnification = 0.0;
}
#endif

//- (void)mouseDragged:(NSEvent *)event {
//    float deltaX = event.deltaX;
//    float deltaY = event.deltaY;
//    BOOL isShift = (event.modifierFlags & NSEventModifierFlagShift) != 0;
//    bool onShape = false;
//    
//    Renderer *renderer = (Renderer *)self.delegate;
//    NSPoint mouseLocation = [self convertPoint:[event locationInWindow] fromView:nil];
//    simd_float2 currentMousePos = simd_make_float2((mouseLocation.x / self.frame.size.width), (mouseLocation.y / self.frame.size.height));
//    currentMousePos = 2 * currentMousePos - 1.0f;
//    
//    simd_float4 rayInViewSpace = simd_make_float4(currentMousePos, 0, 1);
//    simd_float4 intPoint;
//    for (int i = 0; i < renderer->_objectQueue.size(); i++) {
//        if (renderer->_objectQueue[i].dragable) {
//            if (renderer->_objectQueue[i].intersectRay(simd_make_float4(0, 0, 1, 1), rayInViewSpace, renderer->cam.viewMatrix, intPoint, true)) {
//                onShape = true;
//                simd_float4 intPointInWorldSpace = simd_mul(renderer->cam.inverseProjectionMatrix, intPoint);
//                if (!updatedOldRay) {
//                    intPointInWorldSpaceOld = intPointInWorldSpace;
//                    updatedOldRay = true;
//                }
//                renderer->_objectQueue[i].position += (intPointInWorldSpace - intPointInWorldSpaceOld).xyz;
//                intPointInWorldSpaceOld = intPointInWorldSpace;
//                // so that if two faces of same obj behind each other they dont get double draged
//                break;
//            }
//        }
//         
//    }
//
//    if (!onShape) {
//        renderer->cam.handleMouseEvents(deltaX, -deltaY, NO, isShift, TransformationMode::Orbit);
//    }
//}




@end

class TextureBuilder {
public:
    
    static id<MTLTexture> buildGridTexture() {
        int width = 1000;
        int height = 1000;
        int nX = 10;
        int nY = 10;
        auto pixel = MatrixH<1, uint8_t>({0, 0, 0, 225});
        auto Mat = MatrixH<3, uint8_t>::repeating({1000, 1000}, pixel);
        int lineWidth = 10;
        int strideX = (width - lineWidth) / nX;
        int strideY = (height - lineWidth) / nY;
        
        
        int thinLineWidth = 2;
        int thin_nx =5;
        int thin_ny = 5;
        int thinStrideX = (strideX - lineWidth + thinLineWidth) / thin_nx;
        int thinStrideY = (strideY - lineWidth + thinLineWidth) / thin_ny;
        
        // Thick Lines
        for (int i= 0; i < 1; i++) {
            for (int j=0; j < lineWidth; j++) {
                for (int k = 0; k < width; k++) {
                    *Mat(i*strideY + j,k, 0) = 255;
                    *Mat(i*strideY + j,k, 1) = 255;
                    *Mat(i*strideY + j,k, 2) = 255;
                    *Mat(i*strideY + j,k, 3) = 255;
                }
                
            }
        }
        for (int i= 0; i < nX+1; i++) {
            for (int j=0; j < lineWidth; j++) {
                for (int k = 0; k < strideY; k++) {
                    *Mat(k, i*strideX + j, 0) = 255;
                    *Mat(k, i*strideX + j, 1) = 255;
                    *Mat(k, i*strideX + j, 2) = 255;
                    *Mat(k, i*strideX + j, 3) = 255;
                }
                
            }
        }
        
        // Thin Lines
        for (int i= 0; i < 1; i++) {
            for (int j=0; j < thinLineWidth; j++) {
                for (int k = 0; k < width; k++) {
                    *Mat(lineWidth + (thinStrideY - thinLineWidth) + i*thinStrideY + j,k, 0) = 255;
                    *Mat(lineWidth + (thinStrideY - thinLineWidth) + i*thinStrideY + j,k, 1) = 255;
                    *Mat(lineWidth + (thinStrideY - thinLineWidth) + i*thinStrideY + j,k, 2) = 255;
                    *Mat(lineWidth + (thinStrideY - thinLineWidth) + i*thinStrideY + j,k, 3) = 255;
                }
                
            }
        }
        
        for (int i= 0; i < nX; i++) {
            for (int l= 0; l < thin_nx-1; l++) {
                for (int j=0; j < thinLineWidth; j++) {
                    for (int k = 0; k < thinStrideY ; k++) {
                        *Mat(lineWidth + k, i*strideX + lineWidth + (thinStrideY - thinLineWidth) + l*thinStrideX + j, 0) = 255;
                        *Mat(lineWidth + k, i*strideX + lineWidth + (thinStrideY - thinLineWidth) + l*thinStrideX + j, 1) = 255;
                        *Mat(lineWidth + k, i*strideX + lineWidth + (thinStrideY - thinLineWidth) + l*thinStrideX + j, 2) = 255;
                        *Mat(lineWidth + k, i*strideX + lineWidth + (thinStrideY - thinLineWidth) + l*thinStrideX + j, 3) = 255;
                    }
                    
                }
            }
        }
        

//        PatternFill(Mat.buffer + width * 4 * lineWidth + width * 4 * thinStrideY, Mat.buffer + width * 4 * lineWidth, width * 4 * thinStrideY, thin_ny-1);
//        
        PatternFill(Mat.buffer + width * 4 * strideY, Mat.buffer, width * 4 * strideY, nY-1);
//        PatternFill(Mat.buffer + width * 4 * strideY * nY, Mat.buffer, width * 4, lineWidth);
        
        return Mat.ToMTLTexture();
    }
};


//// MARK: Tracer
//
//struct Expr {
//    virtual std::string to_metal() const = 0;
//    virtual ~Expr() = default;
//};
//
//struct Var : Expr {
//    std::string name;
//    Var(std::string n) : name(n) {}
//    std::string to_metal() const override { return name; }
//};
//
//struct Const : Expr {
//    float value;
//    Const(float v) : value(v) {}
//    std::string to_metal() const override { return std::to_string(value); }
//};
//
//struct Add : Expr {
//    std::unique_ptr<Expr> lhs, rhs;
//    Add(std::unique_ptr<Expr> l, std::unique_ptr<Expr> r) : lhs(std::move(l)), rhs(std::move(r)) {}
//    std::string to_metal() const override {
//        return "(" + lhs->to_metal() + " + " + rhs->to_metal() + ")";
//    }
//};
//
//struct Mul : Expr {
//    std::unique_ptr<Expr> lhs, rhs;
//    Mul(std::unique_ptr<Expr> l, std::unique_ptr<Expr> r) : lhs(std::move(l)), rhs(std::move(r)) {}
//    std::string to_metal() const override {
//        return "(" + lhs->to_metal() + " * " + rhs->to_metal() + ")";
//    }
//};
//
//struct Sin : Expr {
//    std::unique_ptr<Expr> arg;
//    Sin(std::unique_ptr<Expr> a) : arg(std::move(a)) {}
//    std::string to_metal() const override {
//        return "sin(" + arg->to_metal() + ")";
//    }
//};
//
//std::unique_ptr<Expr> operator+(std::unique_ptr<Expr>& a, std::unique_ptr<Expr>& b) {
//    return std::make_unique<Add>(a, b);
//}
//
//std::unique_ptr<Expr> operator*(std::unique_ptr<Expr>& a, std::unique_ptr<Expr>& b) {
//    return std::make_unique<Mul>(a, b);
//}
//
//std::unique_ptr<Expr> operator*(std::unique_ptr<Expr>&& a, std::unique_ptr<Expr>&& b) {
//    return std::make_unique<Mul>(a, b);
//}
//
//std::unique_ptr<Expr> operator*(float a, std::unique_ptr<Expr>& b) {
//    return std::make_unique<Mul>(std::make_unique<Const>(a), b);
//}
//
//std::unique_ptr<Expr> sin(std::unique_ptr<Expr>& a) {
//    return std::make_unique<Sin>(a);
//}
//
//std::string generate_metal_function(std::unique_ptr<Expr> expr, const std::string& func_name = "f") {
//    return "float " + func_name + "(float x) {\n    return " + expr->to_metal() + ";\n}";
//}
//
//std::unique_ptr<Expr> build_expr(std::unique_ptr<Expr> x) {
//    return 2.0 * x * x + sin(x); // This builds an expression tree
//}

@interface UIComponent : NSObject

    @property (nonatomic, strong) NSString *cid;
    @property (nonatomic, strong) NSString *title;
    @property (nonatomic, assign) NSInteger type;
    @property (nonatomic, assign) float value;
    @property (nonatomic, assign) float minValue;
    @property (nonatomic, assign) float maxValue;
    @property (nonatomic, copy) void (^valueChangedCallback)(float newValue);

- (instancetype)initWithTitle:(NSString *)title
                         type:(NSInteger)type
                        value:(float)value
                     minValue:(float)minValue
                     maxValue:(float)maxValue
                     callback:(std::function<void(float)>)callback;
@end

@implementation UIComponent
{
@public
    std::function<void(float)> callbackFloat;  // <--- holds your callback!

}

- (instancetype)initWithTitle:(NSString *)title
                         type:(NSInteger)type
                        value:(float)value
                     minValue:(float)minValue
                     maxValue:(float)maxValue
                     callback:(std::function<void(float)>)callback {
    self = [super init];
    if (self) {
        _cid = [[NSUUID UUID] UUIDString];  // Auto-generate
        _title = title;
        _type = type;
        _value = value;
        _minValue = minValue;
        _maxValue = maxValue;
        callbackFloat = callback;
    }
    return self;
}

@end

#import "SidePannel.h"

//Xcode will find and link old .dylib files sitting anywhere in your project directory, even if they're not in "Link Binary With Libraries" or any build settings. This causes runtime symbol errors because it uses the old version instead of the one you explicitly linked.
//Fix: Delete old dylib copies from your project directory. Xcode searches the project tree and picks up matching filenames automatically.
int main2(cv::Mat& camera_frame, cv::Mat& outMat);
int facial_landmarks(cv::Mat& camera_frame, cv::Mat& outMat, float* landmarks, int& num_landmarks);
int hand_tracking(cv::Mat& camera_frame, cv::Mat& outMat, float* landmarks, int& num_landmarks);

@interface Intelligence : NSObject
{
@public
    Renderer* pRender;
    Renderer* pRender2;
    CapReader* cap;
    dispatch_source_t cameraTimer;
    
}
@property (nonatomic) float Red;
@property (nonatomic) float Blue;
@property (nonatomic) float Green;

@property (nonatomic, strong) MTKView *view1;
@property (nonatomic, strong) MTKView *view2;
@property (nonatomic, strong) SidePannel* sidePanel;
@property (strong, nonatomic) AVPlayer *player;
@property (nonatomic, strong) NSMutableArray<UIComponent *> *components;
@property (nonatomic, strong) Renderer* pRender;

-(void) Logic;
-(void) concatLogic;
-(void) PointCloudLogic;
-(void) MotionDetection;
- (id)init:(CGRect)frameView1;
- (void) updateCam:(CGSize) drag :(TransformationMode)Mode :(int)viewNumber;
- (void) updateCamOLD:(int)viewNumber;
-(void) MemoryTestLogic;
-(void) VideoPlayer;
-(void) _3dEditor;
-(void) MicTesting123;
- (void) updateCamProjection:(bool) OrthoPara;
-(void) TeselatorTester;
-(void) TextTesselator;
-(void) MatTester;
-(void) MatTesterV2;
-(void) GeometryNodeTester;
-(void) GeometryNodeTesterV2;
-(void) RGBCam;
- (void) DepthFromImage;
- (void) convPerChannel;
-(void) VideoMerger;
-(void) NewEfficientApproachTester;
    -(void) iOS_Depth;
    -(void) computational_graph;
    -(void) vec_field;
@end

@implementation Intelligence

@synthesize pRender = pRender;
- (id)init:(CGRect)frameView1 {
    self = [super init];
    
    _components = [[NSMutableArray alloc] init];
    
    UIComponent *slider = [[UIComponent alloc] init];
    slider.title = @"Brightness";
    slider.type = 0;
    slider.value = 0.5;
    slider.minValue = 0.0;
    slider.maxValue = 1.0;
    slider.valueChangedCallback = ^(float newValue) {
        NSLog(@"Brightness changed to: %f", newValue);
        // Do something with the value
    };
    [_components addObject:slider];
    
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    std::cout << "Initaialised Metal Device \n";
    _view1 = [[MyMetalView alloc] initWithFrame:frameView1 device:metalDevice];
    [_view1 setColorPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
    [_view1 setClearColor:MTLClearColorMake(1.0, 0.0, 0.0, 1.0)];
    [_view1 setClearDepth:1.0];
    [_view1 setPreferredFramesPerSecond:120];
    [_view1 setFramebufferOnly: false];
    [_view1 setDepthStencilPixelFormat:MTLPixelFormatDepth16Unorm];
    [_view1 setSampleCount:4];
    
//    _view2 = [[MyMetalView alloc] initWithFrame:frameView1 device:metalDevice];
//    [_view2 setColorPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
//    [_view2 setClearColor:MTLClearColorMake(1.0, 0.0, 0.0, 1.0)];
//    [_view2 setClearDepth:1.0];
//    [_view2 setPreferredFramesPerSecond:30];
//    [_view2 setFramebufferOnly: false];
//    [_view2 setDepthStencilPixelFormat:MTLPixelFormatDepth16Unorm];
//    [_view2 setSampleCount:4];
    
    pRender = [[Renderer alloc] initWithDevice:metalDevice :frameView1.size.width :frameView1.size.height :4];
//    pRender2 = [[Renderer alloc] initWithDevice:metalDevice :frameView1.size.width :frameView1.size.height :4];se
    
    _sidePanel = [[SidePannel alloc] initWithVector:&pRender->_NodesQueuePtr];
    
    NSLog(@"hello");
    [_view1 setDelegate:pRender];
    [_view2 setDelegate:pRender2];
    
    _Red = 1;
    _Blue = 1;
    _Green = 1;
    
    return self;
}

- (void)addSliderWithTitle:(NSString*)title
                       min:(float)min
                       max:(float)max
                  callback:(std::function<void(float)>)callback
{
    UIComponent *slider = [[UIComponent alloc] init];
    slider.title = @"Brightness";
    slider.type = 0;
    slider.value = 0.5;
    slider.minValue = 0.0;
    slider.maxValue = 1.0;

    [_components addObject:slider];


}
    
-(void) computational_graph {
    auto x = std::make_shared<MatrixH<1, float>>(
        MatrixH<1, float>::Range(0, {10})
    );

    auto y = std::make_shared<MatrixH<1, float>>(
        MatrixH<1, float>::Range(0, {10})
    );
    
    auto addPrimitive1 = SimmilarPrimitive<1, float>(OpType::ADD, x, y);
    
    auto result = x->zeros();
    result.tape = &addPrimitive1;
    result.flags |= COMPUTE_GRAPH;
    
    auto addPrimitive2 = SimmilarPrimitive<1, float>(OpType::ADD, std::make_shared<MatrixH<1, float>>(result), y);

    auto result2 = x->zeros();
    result2.tape = &addPrimitive2;
    result2.flags |= COMPUTE_GRAPH;
    
    auto unsqeeze_primitive = UnsqeezePrimitive<1, 2, float>(std::make_shared<MatrixH<1, float>>(result2), 0);
    auto result3 = result2.unsqueeze<1>();
    result3.tape = &unsqeeze_primitive;
    result3.evaluate();
    result3.print();
    
}
    
- (void) vec_field {
    size_t no_of_points = 100;
    auto points_charges_location = MatrixH<2, float>::withShape({(uint32_t)no_of_points, 3});
    
    float Angle = 0;
    for (size_t i = 0; i < no_of_points; i++) {
        points_charges_location[i] = {cos(Angle), sin(Angle), 0};
        Angle += 2 * M_PI / (no_of_points);
    }
    auto transforms = MatrixH<3, float>::repeating({(size_m)no_of_points}, MatrixH<2, float>::eye(4) );
    transforms.shape[2] = 3;
    for (size_t i = 0; i < no_of_points; i++) {
        transforms[i, 3] = points_charges_location[i];
    }
    transforms.shape[2] = 4;
    transforms.flags |= NON_OWNERSHIP_FLAG;
    
    auto charges = std::make_shared<CircleNode>(0.01, 10, MatrixH<1, float>{1.0, 0.0, 0.0, 1.0});
    charges->buildInstanceFromBuffer((simd_float4x4*)transforms.buffer, no_of_points);
    pRender->_NodesQueuePtr.push_back(charges);
    
    
    uint32_t grid_x_size = 10;
    uint32_t grid_y_size = 10;
    uint32_t grid_z_size = 10;
    auto E_field = MatrixH<5, float>::repeating({grid_x_size, grid_y_size, grid_z_size}, MatrixH<2, float>::eye(4));
    auto x_axis = MatrixH<1, float>::linspace(-1, 1, {grid_x_size});
    auto y_axis = MatrixH<1, float>::linspace(-1, 1, {grid_y_size});
    auto z_axis = MatrixH<1, float>::linspace(-1, 1, {grid_z_size});
    
    // vectorised ops
    auto [X, Y, Z] = MatrixH<1, float>::meshGrid(x_axis, y_axis, z_axis);
    auto cubic_grid = MatrixH<3, float>::concatGPU_ID(X, Y, Z, 3);
    cubic_grid.print();
    auto cubic_grid_per_charge = MatrixH<5, float>::repeatingGPU({(uint32_t)no_of_points}, cubic_grid).Transpose({1, 2, 3, 0, 4});
    auto r_vec_cubic_grid_per_charge = cubic_grid_per_charge - points_charges_location;
    auto r_square = (r_vec_cubic_grid_per_charge * r_vec_cubic_grid_per_charge).SumNoRed(-1);
    auto field = (0.001 * r_vec_cubic_grid_per_charge / (r_square.sqrt() * r_square)).Sum(-2);
    auto transforms_for_arrow = MatrixH<5, float>::repeatingGPU({grid_x_size, grid_y_size, grid_z_size}, MatrixH<2, float>::eye(4));
    transforms_for_arrow.Slice({ {null} , {null}, {null} , {{3,4}}, {{0, 3}} }).squeeze(-2) = cubic_grid ;
    transforms_for_arrow.flags |= NON_OWNERSHIP_FLAG;
    transforms_for_arrow.Slice({ {null} , {null}, {null} , {{1,2}}, {{0, 3}} }).squeeze(-2) = field;
    transforms_for_arrow.Slice({ {null} , {null}, {null} , {{0,1}}, {{0, 3}} }).flatten<3>() = MatrixH<2, float>({ {0, -1, 0}, {1, 0, 0}, {0, 0, 1} }).Dot(std::move(field).flatten<2>().T()).T();
//    std::move(transforms_for_arrow).flatten<2>().print();
    // E_field.shape[4] = 3;
    // for (int i = 0; i < grid_x_size; i++) {
    //     for (int j = 0; j < grid_y_size; j++) {
    //         for (int w = 0; w < grid_z_size; w++) {
    //             E_field[i, j, w, 3] = {x_axis[i], y_axis[j], z_axis[w]};
    //             E_field[i, j, w, 1, 1] = 0.0f;
    //             for (int k = 0; k < no_of_points; k++) {
    //                 auto r_vec = MatrixH<1, float>{ x_axis[i], y_axis[j], z_axis[w] } - points_charges_location[k];
    //                 float dist_sq = simd_dot(r_vec.toSimdFloat3(), r_vec.toSimdFloat3());
                        
    //                     // 2. Prevent NaN singularities when grid points overlap charges
    //                 dist_sq += 1e-6f;
                        
    //                     // 3. To get 1/r^3 for the unnormalized vector, we need 1 / (dist_sq * sqrt(dist_sq))
    //                 float dist_cube_inv = 1.0f / (dist_sq * std::sqrt(dist_sq));
    //                 E_field[i, j, w, 1] = E_field[i, j, w, 1] +  0.001 * r_vec * dist_cube_inv;
                    
    //             }
    //             E_field[i, j,w, 0] = {-E_field[i, j, w, 1, 1], E_field[i, j,w, 1, 0], 0};
    //         }
    //     }
    // }
    // E_field.shape[4] = 4;
    // E_field.flags |= NON_OWNERSHIP_FLAG;
    auto arrow = std::make_shared<ArrowNode>(1, 0.1);
    arrow->buildInstanceFromBuffer((simd_float4x4*)transforms_for_arrow.buffer, grid_x_size * grid_y_size * grid_z_size);
    pRender->_NodesQueuePtr.push_back(arrow);
    
}
#if TARGET_OS_IPHONE
-(void) iOS_Depth {
    cap = [[CapReader alloc] initWithCam:0];
    __block MatrixH<3, uint8_t> frame;
    
    auto x = MatrixH<1, float>::linspace(1, -1, {(unsigned int)cap->width});
    auto y = MatrixH<1, float>::linspace(1, -1, {(unsigned int)cap->height});
    auto gridResult = MatrixH<1, float>::meshGrid(y, x);

    __block auto Y = gridResult.first;  // or gridResult.x
    __block auto X = gridResult.second; // or gridResult.y
    __block MatrixH<2, float16_t> halfZ;

    
    [cap read:frame];
    [cap getDepth:halfZ];
    MatrixH<2, float> Z = (MatrixH<2, float>)halfZ;
    
    auto floatCol = (((MatrixH<3, float>)frame) / 255.0f).flatten<1>();
    auto points = MatrixH<2, float>::concatID(X, Y, Z, 2).flatten<1>();
    __block auto pointCloud = std::make_shared<PointCloudNode>(points, floatCol);
    __block auto block = ^{
//        NSTimer* timer = [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
//                                                         repeats:YES
//                                                          block:^(NSTimer * _Nonnull timer) {
        @autoreleasepool
        {
            Timer time;
//            [cap read:frame];
//            [cap getDepth:halfZ];
//            MatrixH<2, float> Z = (MatrixH<2, float>)(10.0f / halfZ);
            
//            auto floatCol = (((MatrixH<3, float>)frame) / 255.0f).flatten<1>();
//            auto points = MatrixH<2, float>::concatGPU_ID(X, Y, Z, 2).flatten<1>();
//            pointCloud->updatePoints(points, floatCol);
        }
//        }];
//        NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
//        [runLoop addTimer:timer forMode:NSDefaultRunLoopMode];
//        [runLoop run];
    };
        
    pRender->_NodesQueuePtr.push_back(pointCloud);
//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), block);

    
//    dispatch_queue_t queue = dispatch_queue_create("com.engine.processing", DISPATCH_QUEUE_SERIAL);
//    cameraTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
//
//    // 3. Set the timing (1/30th of a second in nanoseconds)
//    uint64_t interval = (uint64_t)((1.0 / 30.0) * NSEC_PER_SEC);
//    uint64_t leeway = 1ull * NSEC_PER_MSEC; // Tell the OS 1ms of drift is okay for battery saving
//
//    dispatch_source_set_timer(cameraTimer, dispatch_walltime(NULL, 0), interval, leeway);
//    dispatch_source_set_event_handler(cameraTimer, block);
//    dispatch_resume(cameraTimer);
    return;
}
#endif
-(void) NewEfficientApproachTester {
    
//    __block MatrixH<4, uint8_t> vid = MatrixH<4, uint8_t>::fromVideo("/Users/adityadude/Documents/Screenshots/Screen Recording 2026-02-09 at 12.37.46 AM.mov");
//    
//    
//    __block int frame_index = 0;
//    __block bool breakIt = false;
//    [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
//                                    repeats:YES
//                                      block:^(NSTimer * _Nonnull timer) {
//        
//        auto frame = vid[frame_index];
//        frame.drawText("SARS-COV-2", MatrixH<1, int>{300, 100}, MatrixH<1, int>{0, 0, 0, 255}, 100);
//        vid[frame_index] = frame;
//        [pRender updateBaseImage:frame];
//        if (frame_index < vid.shape[0]-1) {
//            frame_index++;
//        } else {
//            if (breakIt != true) {
//                breakIt = true;
//                vid.saveVideo("/Users/adityadude/Documents/Screenshots/out.mp4", true);
//            }
//        }
//        
//    }];
//    
//    return;
//    
    

    
    __block auto Cube = std::make_shared<QuadNode>(1.0, 1.0);
    Cube->RenderStateNo = 2;
    pRender->_NodesQueuePtr.push_back(Cube);
    
//    auto Model = GeometryNode<uint32_t>::BuildGeoNodeFromModel("11_7_2024.usdz");
//    pRender->_NodesQueue_32.push_back(std::move(Model));
//
    __block MatrixH<2, float> RN = MatrixH<2, float>::noise_texture({1000, 3}, 5.9, 2.0, 1.0);
    __block MatrixH<2, float> RSS   = MatrixH<2, float>::concatGPU( MatrixH<2, float>::noise_texture({1000, 3}), MatrixH<2, float>::repeating({1000, 1}, 1.0f) , 1);
    __block auto RS = std::make_shared<PointCloudNode>(RN, RSS);
//    
////    __block MatrixH<3, float> depth =
//    auto noisy_img = MatrixH<3, float>::concatGPU( MatrixH<3, float>::noise_texture({100, 100, 3}), MatrixH<3, float>::repeating({100, 100, 1}, 1.0f) , 1);
    pRender->_NodesQueuePtr.push_back(RS);
////
    [_sidePanel updateAssetManager];
    auto [img, depthMap] = MatrixH<3, uint8_t>::fromImageWithDepth();
    MatrixH<2, float> depthMapFull = ((MatrixH<2, float>)depthMap);
    
    MatrixH<2, float> posEncodingX = MatrixH<2, float>::repeating({depthMapFull.shape[0]}, (MatrixH<1, float>::Range(0, {depthMapFull.shape[1]}) / depthMapFull.shape[1]));
    MatrixH<2, float> posEncodingY = MatrixH<2, float>::repeating({depthMapFull.shape[1]}, (MatrixH<1, float>::Range(0, {depthMapFull.shape[0]}) / depthMapFull.shape[0])).T();
    
    MatrixH<3, float> posEncodingMeshGrid = MatrixH<2, float>::concatGPU_ID(posEncodingX, posEncodingY, depthMapFull, 2);
    
    MatrixH<3, float> normalisedFloatColour = ((MatrixH<3, float>)img ) / 255.0f;
    
    RS->updatePoints(std::move(posEncodingMeshGrid).reshape(posEncodingMeshGrid.shape[0] * posEncodingMeshGrid.shape[1], posEncodingMeshGrid.shape[2]), std::move(normalisedFloatColour).reshape(normalisedFloatColour.shape[0] * normalisedFloatColour.shape[1], normalisedFloatColour.shape[2]));
    

    [pRender updateBaseImage:img];
//
//    std::vector<float> paramValues = {
//        5.0f,   // Scale
//        2.0f,   // Detail
//        0.5f,   // Roughness
//        2.0f,   // Lacunarity
//        0.0f,   // Offset
//        1.0f,   // Gain
//        0.0f    // Distortion
//    };
//    std::vector<std::string> paramNames = {
//        "Scale",
//        "Detail",
//        "Roughness",
//        "Lacunarity",
//        "Offset",
//        "Gain",
//        "Distortion"
//    };
    
//    MatrixH<1, float> vecX = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
//    vecX.buildMetalBuffer();
//    MatrixH<1, float> vecY = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
//    vecY.buildMetalBuffer();
//    
//    auto X = MatrixH<1, float>::linspace(-1, 1, {1000});
//    auto Y = MatrixH<1, float>::linspace(-1, 1, {1000});
//    auto Colour = MatrixH<2, float>::repeating({1000 * 1000}, MatrixH<1, float>{1.0, 0.0, 1.0, 1.0});
//    
//    auto [gridXx, gridYx] = MatrixH<1, float>::meshGrid(X, Y);
//    
//    auto gridZz = sqrt(1 - (gridXx * gridXx + gridYx * gridYx));
//    auto points_for_nish = MatrixH<3, float>::concat(std::move(gridXx).unsqueeze(-1), std::move(gridZz).unsqueeze(-1), std::move(gridYx).unsqueeze(-1), 2).flatten<1>();
//    
//    RS->updatePoints(points_for_nish, Colour);
    
//                        MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
//                        MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
//                        captureDescriptor.captureObject = GlobalGPUManager.metalDevice;
//    
//                        NSError *error = nil;
//                        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
//                            NSLog(@"Capture start failed: %@", error);
//                        }
    

//    ////
//    MatrixH<2, float> gridX = MatrixH<2, float>::withShape({vecX.shape[0], vecY.shape[0]});
//    MatrixH<2, float> gridY = MatrixH<2, float>::withShape({vecX.shape[0], vecY.shape[0]});
//    MatrixH<2, float> vecX_buffer_view;
//    vecX_buffer_view.buffer = vecX.buffer;
//    vecX_buffer_view.shape[0] = vecX.shape[0];
//    vecX_buffer_view.shape[1] = vecY.shape[0];
//    vecX_buffer_view.strides[0] = vecX.strides[0];
//    vecX_buffer_view.strides[1] = 0;
//    vecX_buffer_view.total_size = vecX_buffer_view.accumul(0, 2);
//    vecX_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
//    vecX_buffer_view.flags |= OWNERSHIP_FLAG;
//    vecX_buffer_view.metalBuffer = vecX.metalBuffer;
//
//    MatrixH<2, float>::copyGPUinplace(gridX, vecX_buffer_view, 0, false);
//    
//    MatrixH<2, float> vecY_buffer_view;
//    vecY_buffer_view.buffer = vecY.buffer;
//    vecY_buffer_view.shape[0] = vecX.shape[0];
//    vecY_buffer_view.shape[1] = vecY.shape[0];
//    vecY_buffer_view.strides[0] = 0;
//    vecY_buffer_view.strides[1] = vecY.strides[0];
//    vecY_buffer_view.total_size = vecY_buffer_view.accumul(0, 2);
//    vecY_buffer_view.metalBuffer = vecY.metalBuffer;
//    vecY_buffer_view.flags |= NON_CONTIGUOUS_FLAG;
//    vecY_buffer_view.flags |= OWNERSHIP_FLAG;
//    
//    
//    MatrixH<2, float>::copyGPUinplace(gridY, vecY_buffer_view, 0, true);
//    gridX.print();
//    gridY.print();
//    [captureManager stopCapture];
//    for (int i = 0; i < paramNames.size(); i++) {
//        NSString *paramName = [NSString stringWithUTF8String:paramNames[i].c_str()];
//        UIComponent *slider = [[UIComponent alloc] init];
//        slider.title = paramName;
//        slider.type = 0;
//        slider.value = paramValues[i]; // Default value from vector
//        slider.minValue = 0.0;
//        slider.maxValue = 10.0;
//        slider.valueChangedCallback = ^(float newValue) {
//            float scale = [_components[0] value];
//            float detail = [_components[1] value];
//            float roughness = [_components[2] value];
//            float lacunarity = [_components[3] value];
//            float offset = [_components[4] value];
//            float gain = [_components[5] value];
//            float distortion = [_components[6] value];
//            auto noisy_img = MatrixH<3, float>::concatGPU( MatrixH<3, float>::noise_texture({100, 100, 3}, scale, detail, roughness, lacunarity, offset, gain, distortion) * 1.2, MatrixH<3, float>::repeating({100, 100, 1}, 1.0f) , 2);
//            auto noisy_img_uint8 = (MatrixH<3, uint8_t>)(noisy_img * 255.0f);
//            
//            [self->pRender updateBaseImage:noisy_img_uint8];
//        };
//        [_components addObject:slider];
//    }
//    
    
//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
//        @autoreleasepool {
//            NSTimer *timer = [NSTimer timerWithTimeInterval:(1.0/30.0)
//                                                     repeats:YES
//                                                       block:^(NSTimer * _Nonnull timer) {
//                RN = MatrixH<2, float>::rand_uniform({1000, 3}, -10, 10);
//                RSS   = MatrixH<2, float>::concatGPU( MatrixH<2, float>::rand_uniform({1000, 3}, 0, 1), MatrixH<2, float>::repeating({1000, 1}, 1.0f) , 1);
//                RS->updatePoints(RN, RSS);
//            }];
//    
//            NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
//            [runLoop addTimer:timer forMode:NSDefaultRunLoopMode];
//            [runLoop run];
//        }
//    });

}
#if !TARGET_OS_IPHONE
-(void) concatLogic {
        __block MatrixH<3, uint8_t> img = MatrixH<3, uint8_t>::fromImage();
        __block MatrixH<3, uint8_t> img2 = MatrixH<3, uint8_t>::constant({img.shape[0], img.shape[1], img.shape[2]}, 0);

        img2.drawText("TUSHUUU", {10, 500}, {255, 255, 255, 255}, 300);
        
        img2.invertImg(true);
//        img2.drawRect(simd_make_int4(10, 10, 500, 500), {255, 255, 255, 255});
        
        __block MatrixH<3, uint8_t> concImg = MatrixH<3, uint8_t>::blend(img2, img);

        concImg.chromaKeyImg({-1, -1, -1});
        
        concImg = MatrixH<3, uint8_t>::blend(concImg, img);
        [self->pRender updateBaseImage:concImg];

}

-(void) PointCloudLogic {
//    __block MatrixH<3, uint8_t> img = MatrixH<3, uint8_t>::fromImage();
//    __block MatrixH<3, uint8_t> img2 = MatrixH<3, uint8_t>::constant({img.shape[0], img.shape[1], img.shape[2]}, 0);
//
//    img2.drawText("TUSHUUU", {10, 500}, {255, 255, 255, 255}, 300);
//    
//    img2.invertImg(true);
////        img2.drawRect(simd_make_int4(10, 10, 500, 500), {255, 255, 255, 255});
//    
//    __block MatrixH<3, uint8_t> concImg = MatrixH<3, uint8_t>::blend(img2, img);
//
//    concImg.chromaKeyImg({-1, -1, -1});
//    
//    concImg = MatrixH<3, uint8_t>::blend(concImg, img);
//    [self->pRender updateBaseImage:concImg];
     
//    auto tt2 = Cube(0.05);
//    
//        pRender->_objectQueue.push_back(tt2);
    
//    __block PointCloud ptCloud = PointCloud::fromImage({-1.5, -1.5, 0.1}, {1.5, 1.5, 0.9});
//    pRender->_pointCloudQueue.push_back(ptCloud);
    auto meshufferallocator = [[MTKMeshBufferAllocator alloc] initWithDevice:pRender->metalDevice];
    __block auto mesh = ObjMesh(pRender->metalDevice, meshufferallocator, [[NSBundle mainBundle] pathForResource:@"11_7_2024"
                                                                                                          ofType:@"g"]);
    pRender->_meshpointCloudObjectQueue.push_back(mesh);

}

-(void) MotionDetection {
    cap = [[CapReader alloc] initWithCam:0];
//
    
    __block MatrixH<2, float> constantArray1 = MatrixH<2, float>::Range(0, {3, 4});
    __block MatrixH<1, float> constantArray2 = {1, 2, 3};
    constantArray1.print();
    constantArray2.print();
//    constantArray1 = constantArray1.MulMat(constantArray2);
    constantArray1.print();
    
    __block MatrixH<3, uint8_t> camOut;
    __block MatrixH<3, uint8_t> cached;
    __block MatrixH<3, uint8_t> result;
    sleep(2);
//    __block MatrixH<3, uint8_t> rect = MatrixH<3, uint8_t>::constant({1080, 1920, 4}, 0);
//    
//    rect.drawRect(simd_make_int4(0, 0, 100, 100), {255, 255, 255, 255});
    [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
        
        [self->cap read:camOut];
        
        if (!cached.buffer) {
            cached = camOut;
            cached.invertImg(false);
        }
        
        result = cached.addImg(camOut, false);
        
        result.invertImg(false);
        [self->pRender updateBaseImage:result];
//        cached.copyFrom(camOut);
        
        
    }];
    
    
}

-(void) MemoryTestLogic {
    __block auto A = MatrixH<1, float>({1.0f, 2.0f});
    __block auto B = MatrixH<3, float>::repeating({3, 4}, A);
    __block MatrixH<3, float> cached;
    cap = [[CapReader alloc] initWithCam:0];
    __block MatrixH<3, uint8_t> img;
    __block MatrixH<3, uint8_t> C;
    sleep(2);
    [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
//        {
//            Timer time;
            [self->cap read:img];
            
            auto Floatimg = (MatrixH<3, float>)img;
            
            
            [self->pRender updateBaseImage:img];
            
            auto C_b =  Floatimg * (MatrixH<1, float>({self->_Blue, 0, 0, 1}));
            auto C_g =  Floatimg * (MatrixH<1, float>({0, self->_Green, 0, 1}));
            auto C_r =  Floatimg * (MatrixH<1, float>({0, 0, self->_Red, 1}));
            
            if (!cached.buffer) {
                cached = (C_b + C_g + C_r);
            } else {
                cached = 0;
                cached.Add(cached, C_b);
                cached.Add(cached, C_g);
                cached.Add(cached, C_r);
            }
            
            if (!C.buffer) {
                C = (MatrixH<3, uint8_t>)cached;
            } else {
                cached.To(C, 1);
            }
            
            
            
            [self->pRender2 updateBaseImage:C];
//        };

    }];
}

-(void) VideoMerger {
    __block MatrixH<4, uint8_t> vid = MatrixH<4, uint8_t>::fromVideo("/Users/adityadude/Downloads/IMG_0605.mov");
    __block int i = 0;
    [NSTimer scheduledTimerWithTimeInterval:(1.0/120)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
        auto frame = vid[i];
        
        [self->pRender updateBaseImage:frame];
        if (i != vid.shape[0]-1) {
            i++;
        }
        printf("frame: %d \n", i);
    }];
}

-(void) VideoPlayer {
    __block MatrixH<4, uint8_t> vid = MatrixH<4, uint8_t>::fromVideo("/Users/adityadude/Documents/SadiGaliFarewellClip2.mp4");
    __block int i = 0;
    NSString *filePath = [NSString stringWithUTF8String:"/Users/adityadude/Documents/SadiGaliFarewellClip2.mp4"];
    
    NSURL* url = [NSURL fileURLWithPath:filePath];
    AVAsset* asset = [AVAsset assetWithURL:url];

    // Create an AVPlayerItem
    AVPlayerItem *playerItem = [AVPlayerItem playerItemWithAsset:asset];

    // Initialize AVPlayer with the player item
     _player = [AVPlayer playerWithPlayerItem:playerItem];

    // Start playback
    [_player play];
    // Objective-C++ code within videoPlayer
//    dispatch_async(dispatch_get_main_queue(), ^{
        [NSTimer scheduledTimerWithTimeInterval:(1.0/120)
                                        repeats:YES
                                          block:^(NSTimer * _Nonnull timer) {
            auto frame = vid[i];
            
            [self->pRender updateBaseImage:frame];
            auto Floatimg = (MatrixH<3, float>)frame;
            auto C_b =  Floatimg * (MatrixH<1, float>({self->_Blue, 0, 0, 1}));
            auto C_g =  Floatimg * (MatrixH<1, float>({0, self->_Green, 0, 1}));
            auto C_r =  Floatimg * (MatrixH<1, float>({0, 0, self->_Red, 1}));
            
            auto C = (MatrixH<3, uint8_t>)((C_b + C_g + C_r));
            [self->pRender2 updateBaseImage:C];
//            
            if (i < vid.shape[0]) {
                i++;
            }
        }];
//    });



}

-(void) _3dEditor {
    auto tt = Quad(0.1);
    tt.texture = TextureBuilder::buildGridTexture();
    tt.position.z= 0.1;
    
    tt.textured = true;
    pRender->_ComposedObjectQueue.push_back(tt);
    
    auto cameraObj = Cube(0.1);
    cameraObj.position = pRender->cam.position;
    
    auto TargetObj = Circle(0.1, 10, simd_make_float3(1, 1, 1));
    TargetObj.position = pRender->cam.target;
//    pRender2->cam.OrthoPara = 0;
    pRender2->cam.updated = false;
    
    Pipe pipe = Pipe((TargetObj.position - cameraObj.position) / 10, 0.01, 10, simd_make_float3(1, 0, 0));
    ArrayShape DottedLine = buildDottedLine(TargetObj.position - cameraObj.position, 10, pipe);
    
    pRender2->_objectQueue.push_back(TargetObj);
    pRender2->_objectQueue.push_back(cameraObj);
//    pRender2->_objectQueue.push_back(pipe);
    pRender2->_objectQueueInstanced.push_back(DottedLine);
    
}


-(void) _3dVideoPlayer {
    auto tt = Quad(0.1);
    __block MatrixH<4, uint8_t> vid = MatrixH<4, uint8_t>::fromVideo("/Users/adityadude/Documents/SadiGaliFarewellClip2.mp4");
    __block MatrixH<3, uint8_t> frame = vid[10];
    tt.texture = frame.ToMTLTexture();
    tt.textured = true;
    pRender->_ComposedObjectQueue.push_back(tt);
    __block int i = 0;
    [NSTimer scheduledTimerWithTimeInterval:(1.0/120)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
        frame = vid[i];
        
        frame.CopyToTexture(self->pRender->_ComposedObjectQueue[0].texture);
//
        if (i < vid.shape[0]-1) {
            i++;
        }
    }];
}

-(void) MatTesterV2 {
    MatrixH<1, float> x = MatrixH<1, float>::Range(0.0, {5});
    mlx::core::array mlxArray = mlx::core::array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    mlxArray =mlx::core::arange(5000);
    auto Mat2 =  (x * x) + x;
    auto mat3 = sin(x);
//    mat3.print();
    MatrixH<1, simd_float2> pts = {
        simd_make_float2(0.0f, 0.0f),
        simd_make_float2(1.0f, 0.0f),
        simd_make_float2(1.0f, 1.0f),
    };
    MatrixH<3, float> y = MatrixH<3, float>::Range(1.0, {100, 100, 100});
    auto points = (MatrixH<2, float>)pts;
//    points.print();
    
//    x[2] = 6.0f;
//    x.print();
    
    

    auto L1 = Line(points);
//    
    auto Mat = MatrixH<3, float>::zeros({5, 5, 5});
    Mat = y.Slice({{{0, 5}}, {{0, 5}}, {{0, 5}}});
    
    Mat.printNonCont();
    
    auto z = MatrixH<3, float>::zeros({5, 5, 3});
    auto kernel = MatrixH<2, float>({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    auto convMat = MatrixH<3, float>::repeating({5, 3}, MatrixH<1, float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    convMat.print();
    auto newM = convMat.Transpose({0, 2, 1});
    newM.print();
    newM.conv(z, kernel, 2, ConvMode::Same);
    z.print();
    
    

//    std::cout << (Mat2.parentNodes[1]->parentNodes.size()) << "\n";
//    (Mat2.gradFunc(Mat2)).print();

//    std::cout <<  sizeof(mlx::core::) << "\n";
    
}

- (void) DepthFromImage {
    auto [img, depthMap] = MatrixH<3, uint8_t>::fromImageWithDepth();
    auto floatImg = (MatrixH<3, float>)img / 255.0f;
    auto icospNode = TriangleNode(0.01, 0.01);
    MatrixH<4, float> transforms = MatrixH<4, float>::repeating({depthMap.shape[0], depthMap.shape[1]}, MatrixH<2, float>::eye(4));
    transforms.flags |= NON_OWNERSHIP_FLAG;
    
    for (int y = 0; y < transforms.shape[0]; y++) {
        for (int x = 0; x < transforms.shape[1]; x++) {
            transforms[y, x, 3, 0] = 2 * (float)x / transforms.shape[1];
            transforms[y, x, 3, 1] = 2 * (float)y / transforms.shape[0];
            transforms[y, x, 3, 2] = depthMap[y, x];
        }
    }
    icospNode.buildInstanceFromBuffer((simd_float4x4*)transforms.buffer, (int)depthMap.total_size);
    [pRender createRenderPiplineState:@"CustomShader1" :@"lightingFragmentShader"];
    icospNode.RenderStateNo = 1;
    icospNode.renderStateBlock = ^(id<MTLRenderCommandEncoder> encoder) {
        [encoder setVertexBuffer:floatImg.metalBuffer offset:0 atIndex:4];
        // Do something with the value
    };
    pRender->_NodesQueue.push_back(std::move(icospNode));
    [pRender updateBaseImage:img];
    auto alphaMap = MatrixH<2, uint8_t>::repeating({depthMap.shape[0], depthMap.shape[1]},  MatrixH<0, uint8_t>(255));
    auto depthMapUint = (MatrixH<2, uint8_t>)(depthMap * 200);
    auto combined = MatrixH<2, uint8_t>::concatID(depthMapUint, depthMapUint, depthMapUint, alphaMap, 2);

    
    [pRender updateBaseImage:combined];
}

- (void) convPerChannel {
//    auto [img, depthMap] = MatrixH<3, uint8_t>::fromImageWithDepth();
    __block auto img =MatrixH<3, uint8_t>::fromImage();
    auto floatImg = (MatrixH<3, float>)img / 255.0f;
    auto result = floatImg.zeros();
    __block auto Cube = std::make_shared<QuadNode>(1.0, 1.0);
    pRender->_NodesQueuePtr.push_back(Cube);
//    [pRender updateBaseImage:img];
    [_sidePanel updateAssetManager];

    MatrixH<2, float> kernel = MatrixH<2, float>::gaussian({25, 25}, 12, true);
    MatrixH<2, float> kernel2 = MatrixH<2, float>::constant({100, 100}, 1.0) / 10000;
    MatrixH<3, float> sl = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 1});
    
    
////    auto RChannel = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 1});
//    auto RChannel = floatImg.Slice({null, null, {{0,1}} });
//    {
//        Timer time;
//        RChannel.conv(sl, kernel2);
//    }
//    result.Slice({ null, null, {{0, 1}}}) = sl;
//    result.Slice({ null, null, {{1, 4}}}) = floatImg.Slice({ null, null, {{1, 4}}});
//
//    {
//        Timer time;
//        floatImg.conv(result, kernel2);
//    }
//    auto uint8img = (MatrixH<3, uint8_t>) (result * 255.0f);
////

//    auto RChannelConved = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 1});
//    auto GChannelConved = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 1});
//    auto BChannelConved = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 1});
//    
//    auto RChannel = floatImg.SliceCopy({null, null, {{0,1}} });
//    auto GChannel = floatImg.SliceCopy({null, null, {{1,2}} });
//    auto BChannel = floatImg.SliceCopy({null, null, {{2,3}} });
//    
//    {
//        Timer time;
//        RChannel.conv(RChannelConved, kernel2);
//        GChannel.conv(GChannelConved, kernel2);
//        BChannel.conv(BChannelConved, kernel2);
//    }
//    auto blurImg = MatrixH<3, float>::zeros({floatImg.shape[0], floatImg.shape[1], 4});
//    blurImg.Slice({null, null, {{0, 1}} }) = BChannel;
//    blurImg.Slice({null, null, {{1, 2}} }) = GChannelConved;
//    blurImg.Slice({null, null, {{2, 3}} }) = RChannelConved;
//    blurImg.Slice({null, null, {{3, 4}} }) = RChannel.ones();
    
//    auto blur = (MatrixH<3, uint8_t> )(MatrixH<3, float>::concat(BChannel, GChannelConved , RChannelConved, RChannel.ones(), 2) * 255.0);
//    auto blur = (MatrixH<3, uint8_t> )(result * 255.0);
    
    
//    sl[50, 50].print();
//    [pRender updateBaseImage: blur];
    
//    __block cv::VideoCapture cap(0);
//    
//
//
//    std::cout << "Camera opened successfully.\n";
//
//    __block cv::Mat frame;
//    __block cv::Mat OUTframe;
    
    cap = [[CapReader alloc] initWithCam:0];
    __block MatrixH<3, uint8_t> frame;
//    auto BITCH = [BitchWrapper new];
    
    sleep(2);
//    cap >> frame;
//    void* handle = dlopen("/private/var/tmp/_bazel_adityadude/eb3560479e75155bc8d80d1ef4345636/execroot/mediapipe/bazel-out/darwin_arm64-opt/bin/mediapipe/examples/desktop/libmediapipe_xcode_dylib.dylib",
//                          RTLD_NOW | RTLD_LOCAL);
//    if (!handle) {
//        printf("%s\n", dlerror());
//        return;
//    }
////
//    typedef cv::Mat(*main2)(cv::Mat,cv::Mat);
//
//    __block main2 mm = (main2)dlsym(handle, "main2");
//    if (!mm) {
//        printf("%s\n", dlerror());
//        return;
//    }
//    main2(frame, OUTframe);
    
//
//    cap.~VideoCapture();
    
//    auto cvToMatH = MatrixH<3, uint8_t>::fromCVmat(OUTframe);
//    auto matWithAlpha = MatrixH<3, uint8_t>::concat(cvToMatH, MatrixH<3, uint8_t>::constant({cvToMatH.shape[0], cvToMatH.shape[1], 1}, 255), 2);
//    [pRender updateBaseImage:matWithAlpha];
//    auto r = [BITCH detectInMat:frame width:frame.rows height:frame.cols];
    
    
//    __block int i  = 0;
//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
//        @autoreleasepool {
//            NSTimer *timer = [NSTimer timerWithTimeInterval:(1.0/30.0)
//                                                     repeats:YES
//                                                       block:^(NSTimer * _Nonnull timer) {
//                cap >> frame;
//
//                if (frame.empty()) {
//                    std::cerr << "ERROR: Empty frame.\n";
//                } else if (i > 100){
//                    auto mat = MatrixH<3, uint8_t>::fromCVmat(frame);
//                    auto matWithAlpha = MatrixH<3, uint8_t>::concat(mat, MatrixH<3, uint8_t>::constant({mat.shape[0], mat.shape[1], 1}, 255), 2);
//                    [pRender updateBaseImage:matWithAlpha];
//                }
//                i++;
//            }];
//
//            NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
//            [runLoop addTimer:timer forMode:NSDefaultRunLoopMode];
//            [runLoop run];
//        }
//    });
    __block auto face_landmark_mat = MatrixH<2, float>(478*3);
    face_landmark_mat.shape[0] = 478;
    face_landmark_mat.shape[1] = 3;
    face_landmark_mat.calcStrides();
    
    __block auto hand_landmark_mat = MatrixH<3, float>(126);
    hand_landmark_mat.shape[0] = 2;
    hand_landmark_mat.shape[1] = 21;
    hand_landmark_mat.shape[2] = 3;
    hand_landmark_mat.calcStrides();
    
    __block auto icospNode = std::make_shared<IcosphereNode>(0.005, 1);
    __block auto transformMat = MatrixH<3, float>::repeating({478 + 126}, MatrixH<2, float>::eye(4));
    transformMat.flags |= NON_OWNERSHIP_FLAG;
    

    icospNode->buildInstanceFromBuffer((simd_float4x4*)transformMat.buffer, 478 + 126);
    pRender->_NodesQueuePtr.push_back(icospNode);

    
    
    __block int numOfLandmarksFace = 0;
    __block int numOfLandmarksHand = 0;
//    __block auto alphaMat = MatrixH<3, uint8_t>::constant({(size_t)frame.rows, (size_t)frame.cols, 1}, 255);
    __block auto alphaMat = MatrixH<3, uint8_t>::constant({(size_m)cap->height, (size_m)cap->width, 1}, 255);
    [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
//        cap >> frame;
        [cap read:frame];
//        if (frame.empty()) {
//            std::cerr << "ERROR: Empty frame.\n";
//        } else {
//            main2(frame, OUTframe);
//            facial_landmarks(frame, OUTframe, face_landmark_mat.buffer, numOfLandmarksFace);
//            float axis =  ((face_landmark_mat.Sum(0)[0] / 478) - 0.5);
//            printf("axis %f", axis);
//            Cube->setRot(simd_make_float3(0, 100 * axis, 0));
//            pRender->cam.setMouseEvent( 100* axis, 0, false, false, TransformationMode::Translate);
//            
//            pRender->cam.position.x =   face_landmark_mat.Sum(0)[0];
//            
//            pRender->cam.updateAngles(pRender->cam.position);
//            for (int i = 0; i < numOfLandmarksFace; i++) {
//                transformMat[i, 3, 0] = face_landmark_mat[i, 0] - 0.5;
//                transformMat[i, 3, 1] = -1 * face_landmark_mat[i, 1] + 0.5;
//                transformMat[i, 3, 2] = face_landmark_mat[i, 2];
//            }
//            {
//                Timer time;
//            hand_tracking(frame, OUTframe, hand_landmark_mat.buffer, numOfLandmarksHand);
//            }
            auto mat = frame;
            
            {
//                std::cout << "For creating pos mat \n";
//                Timer time;
                for (int i = 0; i < 21; i++) {
                    transformMat[numOfLandmarksFace + i, 3, 0] = hand_landmark_mat[0, i, 0] - 0.5;
                    transformMat[numOfLandmarksFace + i, 3, 1] = -1 * hand_landmark_mat[0, i, 1] + 0.5;
                    transformMat[numOfLandmarksFace + i, 3, 2] = hand_landmark_mat[0, i, 2];
                    
                    transformMat[numOfLandmarksFace + 21 + i, 3, 0] = hand_landmark_mat[1, i, 0] - 0.5;
                    transformMat[numOfLandmarksFace + 21 + i, 3, 1] = -1 * hand_landmark_mat[1, i, 1] + 0.5;
                    transformMat[numOfLandmarksFace + 21 + i, 3, 2] = hand_landmark_mat[1, i, 2];
                }
            }
            
            {
//                std::cout << "For bulding instances \n";
//                Timer time;
                icospNode->BuildInstances();
            }
            {
                std::cout << "For AadiXLA: ";
                Timer time;
                
//                auto matWithAlpha = MatrixH<3, uint8_t>::concat(mat, alphaMat, 2);
                
                [self->pRender2 updateBaseImage:mat];
            }
//            {
//                std::cout << "For AadiXLA 2.0: ";
//                Timer time;
//
//                
////                [pRender2 updateBaseImage:matWithAlpha];
//            }
//            auto RGB_IMG = mlx::core::zeros({1080, 1920, 3}, mlx::core::uint8);
//            auto alpha_IMG = mlx::core::zeros({1080, 1920, 1}, mlx::core::uint8);
////            {
//                std::cout << "For mlx: ";
//////                Timer time;
////                    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
////                    MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
////                    captureDescriptor.captureObject = GlobalGPUManager.metalDevice;
////            
////                    NSError *error = nil;
////                    if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
////                        NSLog(@"Capture start failed: %@", error);
////                    }
//
//            
//            MatrixH<3, uint8_t> matWithAlpha(1080 * 1920 * 4);
//            matWithAlpha.shape[0] = 1080;
//            matWithAlpha.shape[1] = 1920;
//            matWithAlpha.shape[2] = 4;
//            matWithAlpha.calcStrides();
//            matWithAlpha.copyGPUinplace(matWithAlpha, mat, 0, false);
//            matWithAlpha.copyGPUinplace(matWithAlpha, alphaMat, 3, true);
//
//            auto RGBA = mlx::core::concatenate({RGB_IMG, alpha_IMG}, 2);
//            RGBA.eval();
//
////            [captureManager stopCapture];
//
//            NSLog(@"Capture saved to /tmp/capture.gputrace");
//            }

//        }
    }];
    
}

-(void) RGBCam {
    cap = [[CapReader alloc] initWithCam:0];
    __block MatrixH<3, uint8_t> frame;
    
    
    UIComponent *slider1 = [[UIComponent alloc] init];
    slider1.title = @"Red";
    slider1.type = 0;
    slider1.value = 1;
    slider1.minValue = 0.0;
    slider1.maxValue = 1.0;
    slider1.valueChangedCallback = ^(float newValue) {
        // Do something with the value
    };
    [_components addObject:slider1];
    
    UIComponent *slider2 = [[UIComponent alloc] init];
    slider2.title = @"Blue";
    slider2.type = 0;
    slider2.value = 1;
    slider2.minValue = 0.0;
    slider2.maxValue = 1.0;
    slider2.valueChangedCallback = ^(float newValue) {
        // Do something with the value
    };
    [_components addObject:slider2];
    
    UIComponent *slider3 = [[UIComponent alloc] init];
    slider3.title = @"Green";
    slider3.type = 0;
    slider3.value = 1;
    slider3.minValue = 0.0;
    slider3.maxValue = 1.0;
    slider3.valueChangedCallback = ^(float newValue) {
        // Do something with the value
    };
    [_components addObject:slider3];

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        @autoreleasepool {
            NSTimer *timer = [NSTimer timerWithTimeInterval:(1.0/30.0)
                                                     repeats:YES
                                                       block:^(NSTimer * _Nonnull timer) {
                [cap read:frame];
                auto floatFrame = (MatrixH<3, float>)frame;
                floatFrame = floatFrame * MatrixH<1, float>{slider2.value, slider3.value, slider1.value, 1.0f};
                frame = (MatrixH<3, uint8_t>)floatFrame;

                [pRender updateBaseImage:frame];
            }];

            NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
            [runLoop addTimer:timer forMode:NSDefaultRunLoopMode];
            [runLoop run];
        }
    });
}

-(void) GeometryNodeTesterV2 {
    
    MatrixH<4, float> hello =   MatrixH<4, float>::Range(0, {100, 1, 1, 1});
    auto c = hello.addWithTape(hello, hello);
    
//    MatrixH<1, uint8_t> hellos =   MatrixH<1, uint8_t>::Range(0, {100});
    
    __block auto InstanceTransformPoints = IcosphereNode(1.4, 4);
    __block MatrixH<3, float> InstanceTransformView;
    InstanceTransformView.shape[0] = InstanceTransformPoints.vertexCount;
    InstanceTransformView.shape[1] = 1;
    InstanceTransformView.shape[2] = 3;
    InstanceTransformView.total_size =  InstanceTransformView.accumul(0, 3);
    InstanceTransformView.strides[0] = sizeof(Vertex3D) / sizeof(float);
    InstanceTransformView.strides[1] = 3;
    InstanceTransformView.strides[2] = 1;
    InstanceTransformView.buffer = (float*)InstanceTransformPoints.Verticies;
    InstanceTransformView.flags |= NON_OWNERSHIP_FLAG;
    InstanceTransformView.flags |= NON_CONTIGUOUS_FLAG;
    InstanceTransformView.buildMetalBuffer();

    (InstanceTransformView * MatrixH<3, float>::rand_uniform({(uint)InstanceTransformPoints.vertexCount, 1, 3}, 0.9, 1.5)).print();
    std::cout << ((float*)([InstanceTransformView.metalBuffer contents]))[1601] << "\n";
    
    MatrixH<3, float> InstanceTransformMatricies = MatrixH<3, float>::repeating({ (uint)InstanceTransformPoints.vertexCount}, MatrixH<2, float>::eye(4));
    auto noise = MatrixH<3, float>::noise_texture({(uint)InstanceTransformPoints.vertexCount, 1, 1}, 0.5, 2, 0.5, 2, 0);
    InstanceTransformMatricies.Slice({ null, {{3,4}}, {{0,3}} }) = InstanceTransformView * 2 * noise;
//    InstanceTransformMatricies.Slice({ {{InstanceTransformPoints.vertexCount/2, InstanceTransformPoints.vertexCount}}, {{3,4}}, {{0,3}} }) = InstanceTransformView.Slice({ {{ InstanceTransformPoints.vertexCount/2, InstanceTransformPoints.vertexCount}}, null, null }) * 2;
    
    __block auto spNode = std::make_shared<SphereNode>(0.03, 10, 10);
    spNode->BuildInstances((simd_float4x4*)InstanceTransformMatricies.buffer, InstanceTransformMatricies.shape[0]);
    
    auto icoshperenode = IcosphereNode(1, 1);
    __block auto quadN = std::make_shared<QuadNode>(1, 1);
    
    
    auto RGB = MatrixH<3, float>::noise_texture({1920, 1080, 3}, 2) * 255;
//    MatrixH<3, float> scaled = noise * 255;
//    auto rgb = std::make_unique<MatrixH<3, uint8_t>>(3);
//    std::cout << sizeof(MatrixH<3, float>);
//    RGB.total_size = noise.total_size ;
//    RGB.buffer = new uint8_t[noise.total_size];
//    RGB.buildMetalBuffer();
//    std::memcpy(RGB.shape, noise.shape, sizeof(size_m) * 3);
//    std::memcpy(RGB.strides, noise.strides, sizeof(size_m) * 3);
//    scaled.To(RGB, 0);
//
    auto alphaBase = MatrixH<1, uint8_t>({255});
    __block auto alpha = MatrixH<3, uint8_t>::repeating({1920, 1080}, alphaBase);

    auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, alpha, 2);

    __block MatrixH<3, uint8_t> outputImg;
    
    quadN->texture = noisyImg.ToMTLTexture();
    quadN->isTextured = true;
    
//    pRender->_NodesQueuePtr.push_back(spNode);
    pRender->_NodesQueuePtr.push_back(quadN);
    alpha = MatrixH<3, uint8_t>::repeating({42, 61}, alphaBase);
    
    UIComponent *slider1 = [[UIComponent alloc] init];
    slider1.title = @"Scale";
    slider1.type = 0;
    slider1.value = 0.5;
    slider1.minValue = 0.0;
    slider1.maxValue = 4.0;
    slider1.valueChangedCallback = ^(float newValue) {
        auto noiseS = MatrixH<3, float>::noise_texture({(uint)InstanceTransformPoints.vertexCount, 1, 1}, [_components objectAtIndex:1].value, [_components objectAtIndex:2].value, [_components objectAtIndex:3].value);
        MatrixH<3, float> InstanceTransformMatriciesS = MatrixH<3, float>::repeating({ (uint)InstanceTransformPoints.vertexCount}, MatrixH<2, float>::eye(4));
        InstanceTransformMatriciesS.Slice({ null, {{3,4}}, {{0,3}} }) = InstanceTransformView * [_components objectAtIndex:4].value * noiseS;
        spNode->BuildInstances((simd_float4x4*)InstanceTransformMatriciesS.buffer, InstanceTransformMatriciesS.shape[0]);
        
        auto RGB = noiseS * 255;
        RGB.shape[0] = 42;
        RGB.shape[1] = 61;
        auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, alpha, 2);
        quadN->texture = noisyImg.ToMTLTexture();
        quadN->isTextured = true;
        // Do something with the value
    };
    [_components addObject:slider1];
    
    UIComponent *slider2 = [[UIComponent alloc] init];
    slider2.title = @"Detail";
    slider2.type = 0;
    slider2.value = 2;
    slider2.minValue = 0.0;
    slider2.maxValue = 4.0;
    slider2.valueChangedCallback = ^(float newValue) {
        auto noiseS = MatrixH<3, float>::noise_texture({(uint)InstanceTransformPoints.vertexCount, 1, 1}, [_components objectAtIndex:1].value, [_components objectAtIndex:2].value, [_components objectAtIndex:3].value);
        MatrixH<3, float> InstanceTransformMatriciesS = MatrixH<3, float>::repeating({ (uint)InstanceTransformPoints.vertexCount}, MatrixH<2, float>::eye(4));
        InstanceTransformMatriciesS.Slice({ null, {{3,4}}, {{0,3}} }) = InstanceTransformView * [_components objectAtIndex:4].value * noiseS;
        spNode->BuildInstances((simd_float4x4*)InstanceTransformMatriciesS.buffer, InstanceTransformMatriciesS.shape[0]);
        
        

        auto RGB = noiseS * 255;
        RGB.shape[0] = 42;
        RGB.shape[1] = 61;
        auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, alpha, 2);
        quadN->texture = noisyImg.ToMTLTexture();
        quadN->isTextured = true;
        // Do something with the value
    };
    [_components addObject:slider2];
    
    UIComponent *slider3 = [[UIComponent alloc] init];
    slider3.title = @"Roughness";
    slider3.type = 0;
    slider3.value = 0.5;
    slider3.minValue = 0.0;
    slider3.maxValue = 4.0;
    slider3.valueChangedCallback = ^(float newValue) {
        auto noiseS = MatrixH<3, float>::noise_texture({(uint)InstanceTransformPoints.vertexCount, 1, 1}, [_components objectAtIndex:1].value, [_components objectAtIndex:2].value, [_components objectAtIndex:3].value);
        MatrixH<3, float> InstanceTransformMatriciesS = MatrixH<3, float>::repeating({ (uint)InstanceTransformPoints.vertexCount}, MatrixH<2, float>::eye(4));
        InstanceTransformMatriciesS.Slice({ null, {{3,4}}, {{0,3}} }) = InstanceTransformView * [_components objectAtIndex:4].value * noiseS;
        spNode->BuildInstances((simd_float4x4*)InstanceTransformMatriciesS.buffer, InstanceTransformMatriciesS.shape[0]);
        
//        auto RGB = MatrixH<3, float>::noise_texture({400, 400, 3}, [_components objectAtIndex:1].value, [_components objectAtIndex:2].value, [_components objectAtIndex:3].value) * 255;
//        auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, alpha, 2);
        auto RGB = noiseS * 255;
        RGB.shape[0] = 42;
        RGB.shape[1] = 61;
        auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, alpha, 2);
        quadN->texture = noisyImg.ToMTLTexture();
        quadN->isTextured = true;
        // Do something with the value
    };
    [_components addObject:slider3];
    
    UIComponent *slider4 = [[UIComponent alloc] init];
    slider4.title = @"mult";
    slider4.type = 0;
    slider4.value = 2;
    slider4.minValue = 0.0;
    slider4.maxValue = 4.0;
    slider4.valueChangedCallback = ^(float newValue) {
        auto noiseS = MatrixH<3, float>::noise_texture({(uint)InstanceTransformPoints.vertexCount, 1, 1}, [_components objectAtIndex:1].value, [_components objectAtIndex:2].value, [_components objectAtIndex:3].value);
        MatrixH<3, float> InstanceTransformMatriciesS = MatrixH<3, float>::repeating({ (uint)InstanceTransformPoints.vertexCount}, MatrixH<2, float>::eye(4));
        InstanceTransformMatriciesS.Slice({ null, {{3,4}}, {{0,3}} }) = InstanceTransformView * [_components objectAtIndex:4].value * noiseS;
        spNode->BuildInstances((simd_float4x4*)InstanceTransformMatriciesS.buffer, InstanceTransformMatriciesS.shape[0]);
        
        auto RGB = noiseS * 255;
        RGB.shape[0] = 42;
        RGB.shape[1] = 61;
        RGB.calcStrides();
        auto noisyImg = MatrixH<3, uint8_t>::concat((MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, (MatrixH<3, uint8_t>)RGB, alpha, 2);
        quadN->texture = noisyImg.ToMTLTexture();
        quadN->isTextured = true;
        // Do something with the value
    };
    [_components addObject:slider4];

    
    __block int frame = 0;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        @autoreleasepool {
            NSTimer *timer = [NSTimer timerWithTimeInterval:(1.0/30.0)
                                                     repeats:YES
                                                       block:^(NSTimer * _Nonnull timer) {

            MatrixH<3, uint8_t> outImg = MatrixH<3, uint8_t>::repeating({1080, 1920}, MatrixH<1, uint8_t>{255, 255, 255, 255});
//            MatrixH<3, uint8_t> inImg = MatrixH<3, uint8_t>::withShape({1080, 1920, 4});
            
            auto MTLpair = GlobalGPUManager.create_custom("custom_kernel1", outImg, outImg, 100 * 100);
            auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            auto _dispatchExecutionSize =  MTLSizeMake(1080* 1920, 1, 1);
                float t = (float)frame / 30.0;
            [MTLpair.second setBuffer:outImg.metalBuffer offset:0 atIndex:0];
                [MTLpair.second setBytes:&t length:sizeof(float) atIndex:6];
            [MTLpair.second dispatchThreads:_dispatchExecutionSize
                    threadsPerThreadgroup:_threadsPerThreadgroup];
            [MTLpair.second endEncoding];
            [MTLpair.first commit];
            [MTLpair.first waitUntilCompleted];

                
            
        //    outImg.print();
            [pRender updateBaseImage:outImg];
    
//
//    [pRender updateBaseImage:outImg];
//                NSLog(@"Frame: %d", frame);
                frame++;
            }];

            NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
            [runLoop addTimer:timer forMode:NSDefaultRunLoopMode];
            [runLoop run];
        }
    });
}

-(void) GeometryNodeTester {
    auto TriangleObj = CircleNode(1, 40);
    auto spObj = SphereNode(1, 30, 30);
    __block MatrixH<2, float> stridedView;
    stridedView.buffer = (float*)TriangleObj.Verticies + offsetof(Vertex3D, normal) / sizeof(float);
    stridedView.shape[0] = TriangleObj.vertexCount;
    stridedView.shape[1] = 4;
    stridedView.total_size = TriangleObj.vertexCount * 4;
    stridedView.strides[0] = sizeof(Vertex3D) / sizeof(float);
    stridedView.strides[1] = 1;
    stridedView.flags |= NON_OWNERSHIP_FLAG;
    stridedView.flags |= NON_CONTIGUOUS_FLAG;
//    stridedView.buildMetalBuffer();
//    stridedView.printNonCont();
//    std::cout << TriangleObj.Verticies[0] << "\n";
//    for (int i = 0; i < sizeof(Vertex3D) / sizeof(float); i++) {
//        std::cout << *((float*)&TriangleObj.Verticies[0] + i) << "\n";
//    }
////    stridedView = stridedView + MatrixH<1, float>{1.0, 0.0, 0.0};
//    stridedView.printNonCont();
//    pRender->_NodesQueue.reserve(2);


//    TriangleObj.rotation.z = -135;
//    TriangleObj.position.y += 0.5 + (1/1.415);
//    
//    TriangleObj.buildModelMatrix();
//    auto QuadObj = QuadNode(0.5, 1);
//    
//    auto a = MatrixH<1, int>({1, 2, 3});
    
//
    std::cout << "Actual" << "\n";
    auto arrowObj = ArrowNode(simd_make_float2(0, 0.5), 0.1, BLUE);
    auto Transforms = new simd_float4x4[spObj.vertexCount];

    MatrixH<3, float> TransformStridedView;
    TransformStridedView.buffer = (float*)Transforms;
    TransformStridedView.shape[0] = spObj.vertexCount;
    TransformStridedView.shape[1] = 4;
    TransformStridedView.shape[2] = 4;
    TransformStridedView.total_size = spObj.vertexCount * 16;
    TransformStridedView.strides[0] = 16;
    TransformStridedView.strides[1] = 4;
    TransformStridedView.strides[2] = 1;
    TransformStridedView.flags |= NON_OWNERSHIP_FLAG;
//    TransformStridedView.flags |= NON_CONTIGUOUS_FLAG;
//
    Vertex3D* spObj_verts = static_cast<Vertex3D*>(spObj.Verticies);
    TransformStridedView = MatrixH<3, float>::repeating({(size_m)spObj.vertexCount}, MatrixH<2, float>::eye(4));
    for (int i = 0; i < TransformStridedView.shape[0]; i++) {
        // Translation
        TransformStridedView[i, 3, 0] = spObj_verts[i].position.x;
        TransformStridedView[i, 3, 1] = spObj_verts[i].position.y;
        TransformStridedView[i, 3, 2] = spObj_verts[i].position.z;
        
        // Rotation
        
        TransformStridedView[i, 1, 0] = spObj_verts[i].position.x;
        TransformStridedView[i, 1, 1] = spObj_verts[i].position.y;
        TransformStridedView[i, 1, 2] = spObj_verts[i].position.z;
    }
    std::cout << "View" << "\n";
//    TransformStridedView.printNonCont();

    
    arrowObj.buildInstanceFromBuffer(Transforms, spObj.vertexCount);
    
//    pRender->_NodesQueue.push_back(std::move(spObj));
//    pRender->_NodesQueue.push_back(std::move(TriangleObj));
//    pRender->_NodesQueue.push_back(std::move(arrowObj));
    
    auto cylinderobj = CylinderNode(1, 1, 10);
//    pRender->_NodesQueue.push_back(std::move(cylinderobj));
    
    MatrixH<2, float> pts2 = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
        {-0.5f, 0.5f}
    };
    
    __block auto x = MatrixH<2, float>::Range(-1000, {2000, 1}) / 10;
    
    __block auto LineObj = std::make_shared<LineNode>(MatrixH<2, float>::concat(x, x + (-2.0f), 1));
    pRender->_NodesQueuePtr.push_back(LineObj);
    
    mat A = mat(std::string("A"));
    mat B = mat(std::string("B"));
    mat C = A + B * B + A;
    
    std::cout << C.MSLCode << std::endl;
////    arrowObj.setTailAt(ORIGIN);
//
    auto k = mlx::core::array({ 0.1, 0.2, 0.3, 0.4, 0.5 });
//    testMain();

//    float A = M_PI / 4;
//    
//    auto row1 = simd_make_float4(cos(A), -sin(A), 0, 0);
//    auto row2 = simd_make_float4(sin(A),  cos(A), 0, 0);
//    auto row3 = simd_make_float4(0     , 0      , 1, 0);
//    auto row4 = simd_make_float4(0     , 0      , 0, 1);
//    
//    auto mat1 = simd_matrix_from_rows(row1, row2, row3, row4);
//    
//    
////    arrowObj.modelMatrix = simd_mul(mat1, arrowObj.modelMatrix);
////    arrowObj.buildGlobalMatrix();
//    
//    auto rotationNode = GeometryNode<uint16>(std::move(arrowObj));
//    
//
//    auto arrModifier1 = Modifiers();
//    arrModifier1.MakeModifierArray(10, 10, 10, ORIGIN, simd_make_float3(1, 1, 1));
//    rotationNode.buildInstanceFromModifier(arrModifier1);
//
//    MatrixH<4, float> E = MatrixH<4, float>::zeros({10, 10, 10, 3});
//    for (int i = 0; i < E.shape[0]; i++) {
//        for (int j = 0; j < E.shape[1]; j++) {
//            for (int k = 0; k < E.shape[2]; k++) {
//                auto magnitude = (float)(1/sqrt(i*i + j*j + k*k));
//                E[i, j, k] = ( MatrixH<1, float>({(float)i , (float)j , (float)k }) * magnitude );
//                arrModifier1.transforms[i * 100 + j * 10 +k] = simd_mul(arrModifier1.transforms[i * 100 + j * 10 +k],
//                                                                        simd_matrix(simd_make_float4(E[i, j, k].toSimdFloat3(), 0),
//                                                                                    simd_make_float4(E[i, j, k, 1], -E[i, j, k, 0], 0, 0),
//                                                                                    simd_make_float4(0, 0, 1, 0),
//                                                                                    simd_make_float4(0, 0, 0, 1)));
//            }
//        }
//    }
//    
//    
//    arrowObj.buildInstanceFromModifier(arrModifier1);

    
//    auto a = MatrixH<1, float>({1, 2, 3});
//    auto b = MatrixH<2, float>{{1}, {2}, {3}};
//    a.print();
//    b.print();
//  
//    auto c = a*b;
//    c.print();

//    a.printShape();

//    auto g1 = std::vector<GeometryNode<uint16>>({arrowNode});
//    
//    auto gridNode = GeometryNode<uint16>(arrowObj);
//
//
//    
//    auto arrModifier2 = Modifiers();
//    arrModifier2.MakeModifierArray(5, simd_make_float3(0, 0, 0), simd_make_float3(0, 1, 0));
//    
//    
//    gridNode.buildInstanceFromModifier(arrModifier2);
//    pRender->_NodesQueue.push_back(std::move(rotationNode));
    
    MatrixH<2, float> path = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {3, 1, 1}};
    auto a = path[R(0,3)];
    
    MatrixH<2, float> points = {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    __block auto theta = MatrixH<2, float>::Range(0, {100 * 50, 1}) * (2 * M_PI / 100);
    
    __block auto X_range = theta.zeros();
//    for (int i = 0; i < 100; i++) {
    __block auto Line3dobj = std::make_shared<LineNode3D>(MatrixH<2, float>::concat(5*sin(theta), 5*cos(theta), MatrixH<2, float>::zeros({100 * 50, 1}) + theta, 1), points);
    __block auto arr1 = std::make_shared<ArrowNode>(MatrixH<1, float>{0.0, 1.0, 0.0});
    __block auto arr2 = std::make_shared<ArrowNode>(MatrixH<1, float>{0.0, 1.0, 0.0});
//    }
    pRender->_NodesQueuePtr.push_back(Line3dobj);
    pRender->_NodesQueuePtr.push_back(arr1);
    pRender->_NodesQueuePtr.push_back(arr2);
//    MatrixH<2, float> vecs;
//    path.Derivative(vecs, 0, 3);
//    
////        vecs.print();
//    
//    auto vecs_normalised = vecs / (vecs * vecs).SumNoRed(1).sqrt();
////        vecs_normalised.print();
//
//    MatrixH<2, float> iHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
//    MatrixH<2, float> jHat = MatrixH<2, float>::zeros({path.shape[0] - 1, 3});
////    
//    iHat[0] = simd_normalize( simd_cross(vecs_normalised.toSimdFloat3(0), simd_make_float3(0, 0, 1)) );
//    for (int i = 1; i < path.shape[0]-1; i++) {
//        float sign =  simd_dot(iHat.toSimdFloat3(i-1), (vecs_normalised.toSimdFloat3(i) - vecs_normalised.toSimdFloat3(i-1))) > 0 ? 1: -1;
//        MatrixH<1, float> diff = simd_matH(simd_reflect((vecs_normalised[i] - vecs_normalised[i-1]).toSimdFloat3(), iHat[i-1].toSimdFloat3()));
//        iHat[i] = iHat[i-1] + 1 * sign * diff;
//        
//        iHat[i].print();
//        iHat[i] =  iHat[i] / (iHat[i] * iHat[i]).SumNoRed(0).sqrt();
//        iHat[i].print();
//    }
//    
////    iHat = iHat / (iHat * iHat).SumNoRed(1).sqrt();
//    
//    for (int i = 0; i < path.shape[0] - 1; i++) {
//        jHat[i] = simd_normalize(simd_cross(iHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
//    }
//    for (int i = 0; i < path.shape[0] - 1; i++) {
//        iHat[i] = simd_normalize(simd_cross(jHat.toSimdFloat3(i), vecs_normalised.toSimdFloat3(i)));
//    }
//
//    MatrixH<2, float> conviHat = MatrixH<2, float>::zeros({iHat.shape[0] + 1, iHat.shape[1]});
//    iHat.conv<1>(conviHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
//    
//    MatrixH<2, float> convjHat = MatrixH<2, float>::zeros({jHat.shape[0] + 1, jHat.shape[1]});
//    jHat.conv<1>(convjHat, {1, 1}, 1, ConvMode::Full, ConvModeFull::Extend);
//    
//    conviHat = conviHat / (conviHat * conviHat).SumNoRed(1);
//    convjHat = convjHat / (convjHat * convjHat).SumNoRed(1);
//    
//    auto vecsforpath = ArrowNode(simd_make_float2(1, 0), 0.1);
//    auto ihatforpath = ArrowNode(simd_make_float2(1, 0), 0.1, {1, 0, 0, 1});
//    auto jhatforpath = ArrowNode(simd_make_float2(1, 0), 0.1, {0, 1, 0, 1});
//    
//    MatrixH<4, float> transformsforpath = MatrixH<4, float>::repeating({path.shape[0], 3}, MatrixH<2, float>::eye(4));
//    transformsforpath.flags |= OWNERSHIP_FLAG;
//
//    
//    for (int i = 0; i < path.shape[0]-1; i++) {
//        
//        transformsforpath[i, 0, 3] = MatrixH<1, float>::concat(path[i], MatrixH<1, float>{1}, 0);
//        transformsforpath[i, 1, 3] = MatrixH<1, float>::concat(path[i], MatrixH<1, float>{1}, 0);
//        transformsforpath[i, 2, 3] = MatrixH<1, float>::concat(path[i], MatrixH<1, float>{1}, 0);
//        
//        transformsforpath[i, 0, 0] = MatrixH<1, float>::concat(vecs[i], MatrixH<1, float>{0}, 0);
//        transformsforpath[i, 0, 1] = MatrixH<1, float>::concat({-1 * vecs[i, 1], vecs[i, 0], vecs[i, 2]}, MatrixH<1, float>{0}, 0);
//        
//        transformsforpath[i, 1, 0] = MatrixH<1, float>::concat(conviHat[i], MatrixH<1, float>{0}, 0);
//        transformsforpath[i, 1, 1] = MatrixH<1, float>::concat({-1 * conviHat[i, 1], conviHat[i, 0], conviHat[i, 2]}, MatrixH<1, float>{0}, 0);
//        
//        transformsforpath[i, 2, 0] = MatrixH<1, float>::concat(convjHat[i], MatrixH<1, float>{0}, 0);
////        transformsforpath[i, 2, 1] = MatrixH<1, float>::concat({-1 * convjHat[i, 1], convjHat[i, 0], convjHat[i, 2]}, MatrixH<1, float>{0}, 0);
//    }
//    
//    transformsforpath[path.shape[0]-1, 0, 3] = MatrixH<1, float>::concat(path[path.shape[0]-1], MatrixH<1, float>{1}, 0);
//    transformsforpath[path.shape[0]-1, 1, 3] = MatrixH<1, float>::concat(path[path.shape[0]-1], MatrixH<1, float>{1}, 0);
//    transformsforpath[path.shape[0]-1, 2, 3] = MatrixH<1, float>::concat(path[path.shape[0]-1], MatrixH<1, float>{1}, 0);
//    
//    transformsforpath[path.shape[0]-1, 0, 0] = MatrixH<1, float>::concat(vecs[path.shape[0]-2], MatrixH<1, float>{0}, 0);
//    transformsforpath[path.shape[0]-1, 0, 1] = MatrixH<1, float>::concat({-1 * vecs[path.shape[0]-2, 1], vecs[path.shape[0]-2, 0], vecs[path.shape[0]-2, 2]}, MatrixH<1, float>{0}, 0);
//    
//    transformsforpath[path.shape[0]-1, 1, 0] = MatrixH<1, float>::concat(conviHat[path.shape[0]-1], MatrixH<1, float>{0}, 0);
//    
//    transformsforpath[path.shape[0]-1, 2, 0] = MatrixH<1, float>::concat(convjHat[path.shape[0]-1], MatrixH<1, float>{0}, 0);
//    
//    transformsforpath.print();
//    
//    auto drforpath = transformsforpath.SliceCopy({ null, {{0, 1}} });
//    drforpath.flags |= OWNERSHIP_FLAG;
//    
//    auto ihatpath = transformsforpath.SliceCopy({ null, {{1, 2}} });
//    ihatpath.flags |= OWNERSHIP_FLAG;
//    
//    auto jhatpath = transformsforpath.SliceCopy({ null, {{2, 3}} });
//    jhatpath.flags |= OWNERSHIP_FLAG;
//    
//    
//    vecsforpath.buildInstanceFromBuffer((simd_float4x4*)drforpath.buffer, path.shape[0] );
//    ihatforpath.buildInstanceFromBuffer((simd_float4x4*)ihatpath.buffer, path.shape[0] );
//    jhatforpath.buildInstanceFromBuffer((simd_float4x4*)jhatpath.buffer, path.shape[0] );
//    
//    pRender->_NodesQueue.push_back(vecsforpath);
//    pRender->_NodesQueue.push_back(ihatforpath);
//    pRender->_NodesQueue.push_back(jhatforpath);
    
    std::cout << "" ;
    
    __block int frame = 0;
    [NSTimer scheduledTimerWithTimeInterval:(1.0)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
//        stridedView = stridedView + MatrixH<1, float>{0.01, 0.0, 0.0};
////////        self->pRender->_NodesQueue[0].childNodes[0].instanceMatricies[0] = RotationZ(0.1 * frame);
////////        self->pRender->_NodesQueue[0].childNodes[0].BuildInstances();
////        self->pRender->_NodesQueue[0].rotation.z += 0.1;
////        self->pRender->_NodesQueue[0].buildModelMatrix();
//        Line3dobj->UpdatePoints(MatrixH<2, float>::concat(5*sin(theta), 5*cos(theta), MatrixH<2, float>::zeros({100 * 5, 1}) + theta * sin((float)frame / 10), 1), points);
        
        
        auto Y_range = theta.zeros();
        float t1 = -M_PI/2 + (float)frame ;
        float t2 = -M_PI/2 - (float)frame ;
        
        if (frame < Y_range.shape[0]) {
            if (frame + 100 > Y_range.shape[0]) {
                int latter =  Y_range.shape[0] - frame;
                
                auto w1 = theta.Slice({ {{frame, frame + latter}} });
                Y_range[R(frame, frame+latter)] = cos(w1 + t2);
            } else {
                auto w1 = theta.Slice({ {{frame, frame + 100}} });
                Y_range[R(frame, frame+100)] = cos(w1 + t2);
            }
            
        }
        
        LineObj->UpdatePoints(MatrixH<2, float>::concat(theta, Y_range, 1));
        arr1->updateNode({cos(t1), sin(t1),0});
        arr2->updateNode({cos(t2), sin(t2),0}, arr1->vec);
        frame++;
    }];
    
}

-(void) MatTester {
    GlobalGPUManager.initTypeCasting(0, 0);
    MatrixH<1, float> a = MatrixH<1, float>::Range(0, {20});
//    a.Sum(0).print();
//    (a+a).print();
//    ((MatrixH<1, int>)a).print();
    
    MatrixH<1, simd_float2> pts = {
        simd_make_float2(0.0f, 0.0f),
        simd_make_float2(1.0f, 0.0f),
        simd_make_float2(1.0f, 1.0f),
    };
    
    auto c = MatrixH<3, float>::Range(0.0, {3, 4, 2});
    c.print();
    ((MatrixH<2, float>)a).print();
//    c.Sum(2).print();
//    auto d = (MatrixH<1, simd_float2>)c;
//    d.print();
    
    (c.Transpose({1, 0, 2})).print();
    
    auto A = MatrixH<2, float>::eye(3, 3, -1);
    A.print();
    
    mlx::core::array mlxArray = mlx::core::array({1.0f, 2.0f, 3.0f, 4.0f, 4.0f});
    auto f = [](const mlx::core::array& x) -> mlx::core::array {
        return (x * x);
    };
    
    mlx::core::array output = f(mlxArray);
    std::cout << output << "\n";
    
    auto f_and_grad = mlx::core::grad(f);
    auto result = mlx::core::vmap(f_and_grad)(mlxArray);
    std::cout << result << std::endl;
    
    auto Mat1 = MatrixH<2, float>::constant({3, 4}, 1);
    auto Mat2 = MatrixH<2, float>::constant({4, 3}, 1);
    auto Mat3 = Mat1.Dot(Mat2);
    Mat3.print();
    
//    auto l1 = Line(pts);
    
    
//    try {
//        auto [value, grad] = f_and_grad(mlxArray);
//        std::cout << grad << std::endl;
//    } catch (const std::exception& e) {
//        std::cout << "Standard exception: " << e.what() << "\n";
//    } catch (const std::runtime_error& e) {
//        std::cout << "Runtime error: " << e.what() << "\n";
//    } catch (...) {
//        std::cout << "Unknown exception occurred\n";
//    }
    
//    std::cout << grad << std::endl;
    
//    auto x = std::make_unique<Var>("x");
//    auto expr = build_expr(x);

//    std::string metal_code = generate_metal_function(expr);

//    std::cout << metal_code << std::endl;
    
//    MatrixH<1, simd_float2> d = MatrixH<1, simd_float2>::constant({3}, simd_make_float2(0, 0));
//    c.To(d, 0);

//    d.print();
    
}

- (void) TextTesselator {
    Shape<uint16> HelloWorldText = Text3D();
//    for (int i = 0; i < pts.total_size; i++) {
//        auto c = Circle(0.05, 10, simd_make_float3(1, 1, 1));
//        c.position.xy += pts[i];
//        pRender->_objectQueue.push_back(c);
//    }
    pRender->_objectQueue.push_back(HelloWorldText);
}

-(void) TeselatorTester {

    
    MatrixH<1, simd_float2> pts = {
        simd_make_float2(0.0f, 0.0f),
        simd_make_float2(1.0f, 0.0f),
        simd_make_float2(1.0f, 1.0f),
        simd_make_float2(0.0f, 1.0f),
        simd_make_float2(-0.5f, 0.5f)
    };
    MatrixH<2, float> pts2 = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
        {-0.5f, 0.5f}
    };
    Shape<uint16> HelloWorldText = Text3D();
    Shape<uint16> polygon = ConvexPolygon(pts);
    Shape<uint16> polygonConcave = ConcavePolygon(pts);
    polygon.dragable = true;
    pRender->_objectQueue.push_back(polygon);
    
//    auto Mat = pts2;
//
    __strong auto strongSelf = self;
    for (int i = 0; i < pts.total_size; i++) {
        auto c = Circle(0.05, 10, simd_make_float3(1, 1, 1));
        c.position.xy += pts[i];
        c.dragable = true;
        c.transformChangeCallbacks.push_back([i, pts, strongSelf](simd_float3 newPos) {
            auto Mat = pts;
            memcpy(Mat(i), &newPos, sizeof(simd_float2));
            ConcavePolygon::updateShape(Mat, strongSelf->pRender->_objectQueue[0]);
        });
        pRender->_objectQueue.push_back(c);
    }


    
    
    MatrixH<1, float> p = {1, 1, 4, 3, 4};
    MatrixH<1, simd_float2>d;
    
    pts.Derivative(d, 0, 1);
    d.print();
    std::cout << "\n";
//    [NSTimer scheduledTimerWithTimeInterval:(1.0/60)
//                                    repeats:YES
//                                      block:^(NSTimer * _Nonnull timer) {
//    }];
}

-(void) MicTesting123 {
    __block MatrixH<1, int16_t> sample = MatrixH<1, int16_t>::constant({200}, 0);
    __block auto tt2 = Cube(0.5);
//    __block auto tt3 = Text3D(@"Tissue Paper", @"Helvetica", 10, 10, simd_make_float4(1.0, 1.0, 1.0, 1.0));
    __block auto Mic = [[MicReader alloc] initWithMic:0];
//    pRender->_objectQueue.push_back(tt3);
    [NSTimer scheduledTimerWithTimeInterval:(1.0/60)
                                    repeats:YES
                                      block:^(NSTimer * _Nonnull timer) {
        [Mic read:sample];
        if (sample.buffer && sample.total_size > 0) {
            // Your audio processing code
            [self processAudioSample:sample];
        }
        
        auto img = (MatrixH<3, uint8_t>)sample;
        
//        std::cout << (img.total_size) << "\n";
        img.shape[2] = 4;
        img.shape[1] = 16;
        img.shape[0] = 8;
//        img.print();
//        img.printShape();
//        [self->pRender updateBaseImage:img];
    }];
}
-(void) Logic {
    __block MatrixH<3, uint8_t> img = MatrixH<3, uint8_t>::fromImage();
    __block MatrixH<3, uint8_t> imgSa = MatrixH<3, uint8_t>::fromImage();

    
    
    __block MatrixH<3, uint8_t> concImg;
    __block MatrixH<3, int> detectections;
    __block MatrixH<2, float> mat = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    mat.print();
    
    
    cap = [[CapReader alloc] initWithCam:0];
    __block MatrixH<3, uint8_t> img2 = MatrixH<3, uint8_t>::constant({1080, 1920, 4}, 0);
    
    img2.drawText("AAADDII", {10, 10}, {255, 255, 255, 255}, 100);
    img2.invertImg(true);
    auto tt = Triangle(1, {1, 0, 0, 1});
    pRender->_objectQueue.push_back(tt);
    
    auto tt2 = Cube(0.05);
    
//    pRender->_objectQueue.push_back(tt2);
    

    
    tt2.position.x -= 0.8;
    
    auto Modifier = ArrayModifier();
//        Modifier.dPosition.y += 0.5;
//        Modifier.dRotation.z += 360.0 / 5;
        
//        auto tt2Arr = Modifier.MakeModifierArrayRadial(tt2, 5);
        
    Modifier.dPosition.x += 0.2;
    auto tt2Arr =  Modifier.MakeModifierArray(tt2, 4);
        
        
    auto Modifier2 = ArrayModifier();
    Modifier2.dRotation.z = 360.0 / 10;
    tt2Arr = Modifier2.MakeModifierSuperArrayRadial(tt2Arr, 10);
    pRender->_objectQueueInstanced.push_back(tt2Arr);
        
//        [NSThread sleepForTimeInterval:1.0];
//        [cap read:img];
    [pRender2 updateBaseImage:img];
        
    __block auto HandAI = HandDetector();
    sleep(2);
    
    [NSTimer scheduledTimerWithTimeInterval:(1.0/30)
                                        repeats:YES
                                          block:^(NSTimer * _Nonnull timer) {
            
            // Use the dereferenced heap object.
        self->pRender->_objectQueueInstanced[0].shape.rotation += 0.1;

        [self->cap read:img];
        
        
        HandAI.detectHands(img, detectections, false);
        if (detectections.shape[0] > 0) {
            self->pRender->_objectQueueInstanced[0].shape.rotation = HandAI.relativeDistance(detectections, 0) * 0.5;
        }
        HandAI.drawHands(img, detectections, false);
        
        img.invertImg(false);
        concImg = MatrixH<3, uint8_t>::blend(img2, img);
        
        concImg.chromaKeyImg({-1, -1, -1});
        img.invertImg(false);
        
        concImg = MatrixH<3, uint8_t>::blend(concImg, img);
        
        [self->pRender2 updateBaseImage:concImg];
    }];
}
    
- (void) updateCam:(CGSize) drag :(TransformationMode)Mode :(int)viewNumber {
//    if (viewNumber == 1) {
//        if (Mode != TransformationMode::Zoom) {
//            pRender->cam.updatePosition(simd_make_float2(drag.width / _view1.frame.size.width, drag.height / _view1.frame.size.height), Mode);
//    //        std::cout << _view1.frame.size.width << ", " << _view1.frame.size.height << "\n";
//        }
//        else {
//            pRender->cam.updatePosition(simd_make_float2(drag.width , drag.height), Mode);
//        }
//
//        pRender2->_objectQueue[0].position = pRender->cam.position;
//        pRender2->_objectQueue[1].position = pRender->cam.target;
//        Pipe::updateBuffers((pRender->cam.position - pRender->cam.target) / 10, 0.01, 10, simd_make_float3(1, 0, 0), pRender2->_objectQueueInstanced[0].shape);
//        updateDottedLine((pRender->cam.position - pRender->cam.target), 10, pRender2->_objectQueueInstanced[0].transform);
//    } else if (viewNumber == 2){
//        if (Mode != TransformationMode::Zoom) {
//            pRender2->cam.updatePosition(simd_make_float2(drag.width / _view2.frame.size.width, drag.height / _view2.frame.size.height), Mode);
//    //        std::cout << _view1.frame.size.width << ", " << _view1.frame.size.height << "\n";
//        }
//        else {
//            pRender2->cam.updatePosition(simd_make_float2(drag.width , drag.height), Mode);
//        }
//    } else {
//        std::cout << "Incorrect View No." << "\n";
//    }


}

- (void) updateCamProjection:(bool) OrthoPara {
//    pRender->cam.OrthoPara = OrthoPara;
    pRender->cam.updated = false;

}

-(void) updateCamOLD:(int)viewNumber {
    pRender->cam.updateOLD();
    if (viewNumber == 1) {
        pRender->cam.updateOLD();
    } else if (viewNumber == 2) {
        pRender2->cam.updateOLD();
    } else {
        std::cout << "Incorrect View No." << "\n";
    }
}

- (void)processAudioSample:(MatrixH<1, int16_t>&)sample {
    // Example: Calculate RMS value to verify we're getting real data
    if (sample.buffer && sample.total_size > 0) {
        long long sum = 0;
        for (int i = 0; i < sample.total_size; i++) {
            sum += (long long)sample.buffer[i] * sample.buffer[i];
        }
        double rms = sqrt((double)sum / sample.total_size);
        
        // Print RMS value every 30 frames to monitor audio levels
        static int frameCount = 0;
        if (++frameCount % 30 == 0) {
            NSLog(@"Audio RMS: %.2f", rms);
        }
    }
}
#endif

@end
//class Intelligence
//};








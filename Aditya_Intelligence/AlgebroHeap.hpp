//
//  AlgebroHeap.hpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 04/02/25.
//

#ifndef AlgebroHeap_hpp
#define AlgebroHeap_hpp

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <simd/simd.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <CoreServices/CoreServices.h>
#import <CoreText/CoreText.h>
#import "Utilities.hpp"
#import <CoreVideo/CoreVideo.h>

//#include <UniformTypeIdentifiers/UniformTypeIdentifiers.h>



template <typename T>
struct RectH {
    T X, Y, width, height;
    RectH(std::initializer_list<T> data) {

        if (data.size() != 4) {
            std::cerr << "Error: too many or too few elements \n";
            return;
        }

        auto it = data.begin();
        X = *it++;
        Y = *it++;
        width = *it++;
        height = *it;
    }

    RectH operator+ (simd_int2 pt) const {
        return Rect{X + pt.x, Y + pt.y, width, height};
    }
    RectH operator- (simd_int2 pt) const {
        return Rect{X - pt.x, Y - pt.y, width, height};
    }
    

    RectH operator* (simd_float2 pt) const {
        return Rect{X, Y, width * pt.x, height * pt.y};
    }
    
    void print() const {
        std::cout << "Rect(X: " << X << ", Y: " << Y
                  << ", Width: " << width << ", Height: " << height << ")\n";
    }
    
    CGRect toCGRect() {
        return CGRectMake(X, Y, width, height);
    }
};


enum class IMGFormat : uint32_t {
    RGBA = 0,
    BGRA = 1,
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i < vec.size() - 1) os << ", "; // Add a comma except for the last element
    }
    os << "]";
    return os;
}

template <typename Type>
class MatrixH {
public:
    Type* values;
    std::vector<size_t> shape;
    size_t total_size;
    int flags = 0;
    
    MatrixH() : values(nullptr), total_size(0) {
        std::cout << "created"  << "\n";
    }
    
    
//    template <typename T>
//    MatrixH(std::initializer_list<T> init_list) {
//        shape = inferShape(init_list);   // Compute shape dynamically
//        total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
//        values = new Type[total_size];   // Allocate memory
//        size_t runs = 0;
//        flatten(init_list, values, runs);      // Copy values into 1D array
//    }
//    

    
    
    MatrixH(std::initializer_list<Type> init_list) {
        total_size = init_list.size();
        shape = {total_size};
        values = new Type[total_size];   // Allocate memory
//        std::copy(init_list.begin(), init_list.end(), values);
        memcpy(values, init_list.begin(), sizeof(Type) * total_size);
    }
    
    CVPixelBufferRef createPixelBufferFromMat();
    
    static size_t accMul(const std::vector<size_t> shape, int start, int end) {
        size_t totalSize = 1;
        for (int i = start; i < end; i++) {
            totalSize *= shape[i];
        }
        return totalSize;
    }
    
    static size_t accAdd(const std::vector<size_t> shape) {
        size_t totalSize =0;
        for (int i = 0; i < shape.size(); i++) {
            totalSize += shape[i];
        }
        return totalSize;
    }
    int NamedWindow(const char* name, int flags) const;
    int test();
    static MatrixH<uint8_t> fromVideo(const char* vidPath);
    void AIDragCallback( const char* name, void (*cCallback)(simd_float2)) const;
    static MatrixH concat1D(std::function<void(const std::vector<MatrixH<Type>>&)> closure, const std::vector<size_t> compSize) {
        
        MatrixH<Type> result = MatrixH<Type>();
        result.shape = compSize;
        
        size_t newD = 0;
        
        for (int i = 0; i < closure.size(); i++) {
            newD += closure[i].shape[0];
        }

        
        result.shape[0] = newD;
        result.total_size = accMul(result.shape);
        result.values = new Type[result.total_size];
        
        
        for (int i = 0; i < closure.size(); i++) {
            for (int j = 1; j < compSize.size(); j++) {
                if (compSize[j] != closure[i].shape[j]) {
                    
                }
            }
        }
        
        
    }
    
    void drawPath(const MatrixH<float>& path, const MatrixH<int>& colour) {
        // shape of path => {n, 2}
        // shape of colour => {4} RGBA
        
        
    }
    
    void drawLine(simd_int2 pixelP1, simd_int2 pixelP2, int dim, const MatrixH<uint8_t>& colour) {

        
        int X0; int Y0;
        
        simd_float2 diff = simd_make_float2(pixelP2.x - pixelP1.x, pixelP2.y - pixelP1.y);
        
        if (pixelP1.x - pixelP2.x > 0) {
            X0 = pixelP2.x;
        } else {
            X0 = pixelP1.x;
        }
        
        if (pixelP1.y - pixelP2.y > 0) {
            Y0 = pixelP2.y;
        } else {
            Y0 = pixelP1.y;
        }
        
        int width = abs(pixelP1.x - pixelP2.x);
        int height = abs(pixelP1.y - pixelP2.y);
        
        for (int x = 0; x < width + 0; x++) {
            for (int y = 0; y < height + 0; y++) {
                auto xy = simd_make_float2(y, -x) / simd_length(diff);
                
                if (abs(simd::dot(diff, xy)) < 10) {
                    appentMat(y+Y0, x+X0,colour);
                }
                
            }
        }
        
    }
    
    void drawLinef(simd_float2 p1, simd_float2 p2, int dim) {
        auto pixelP1 = simd_make_int2(p1.x * shape[dim+1], p1.y * shape[dim+0]);
        auto pixelP2 = simd_make_int2(p2.x * shape[dim+1], p2.y * shape[dim+0]);
        
        int X0; int Y0;
        
        simd_float2 diff = simd_make_float2(pixelP2.x - pixelP1.x, pixelP2.y - pixelP1.y);
        
        if (pixelP1.x - pixelP2.x > 0) {
            X0 = pixelP2.x;
        } else {
            X0 = pixelP1.x;
        }
        
        if (pixelP1.y - pixelP2.y > 0) {
            Y0 = pixelP2.y;
        } else {
            Y0 = pixelP1.y;
        }
        
        int width = abs(pixelP1.x - pixelP2.x);
        int height = abs(pixelP1.y - pixelP2.y);
        
        for (int x = 0; x < width + 0; x++) {
            for (int y = 0; y < height + 0; y++) {
                auto xy = simd_make_float2(y, -x) / simd_length(diff);
                
                if (abs(simd::dot(diff, xy)) < 10) {
                    appentMat(y+Y0, x+X0,{255, 0, 0, 255});
                }
                
            }
        }
        
    }
    
    void TranslateH(int dim, int amount) {
        if (amount > 0) {
            PadnTrim(dim, amount, 0, 0, amount);
        }
        if (amount < 0) {
            PadnTrim(dim, 0, -amount, -amount, 0);
        }
    }
    
    void Translate2D_H(int dim, int X, int Y) {
        if (X >= 0 && Y >= 0) {
            PadnTrim2D(dim, Y, 0, 0, Y, X, 0, 0, X);
        }
        else if (X <= 0 && Y <= 0) {
            PadnTrim2D(dim, 0, -Y, -Y, 0, 0, -X, -X, 0);
        }
        else if (X < 0 && Y > 0) {
            PadnTrim2D(dim, Y, 0, 0, Y,  0, -X, -X, 0);
        }
        else if (X > 0 && Y < 0) {
            PadnTrim2D(dim, 0, -Y, -Y, 0, X, 0, 0, X);
        }
    }
    
    static MatrixH concat(std::function<void(const std::vector<MatrixH<Type>>&)> closure, const std::vector<size_t> compSize, int dim) {
        
        MatrixH<Type> result = MatrixH<Type>();
        result.shape = compSize;
        
        size_t newD = 0;
        
        for (int i = 0; i < closure.size(); i++) {
            newD += closure[i].shape[dim];
        }
        for (int i = 0; i < compSize.size(); i++) {
            if (compSize[i] == 0) {
                result.shape[i] = newD;
            }
        }
        result.total_size = accMul(result.shape);
        
        
        for (int i = 0; i < closure.size(); i++) {
            for (int j = 0; j < compSize.size(); j++) {
                if (compSize[j] != closure[i].shape[j+1]) {
                    
                }
            }
        }
        
        
    }
    
    void play();
    dispatch_group_t saveVideo(const char* outputPath);
    
    static MatrixH GaussianBlur(float sigma, size_t m, size_t n) {
        MatrixH result =  MatrixH<float>();
        result.shape = std::vector<size_t>({m, n});
        result.total_size = m*n;
        result.values = new float[m*n];
        int halfSizeX = (static_cast<int>(m) - 1) / 2;
        int halfSizeY = (static_cast<int>(n) - 1) / 2;
        int sum = 0;
        
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                int x = i - halfSizeX;
                int y = j - halfSizeY;
                result.values[i*n + j] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                sum += result.values[i*n + j];
                
            }
        }
        
//        for (int i = 0; i < m; ++i)
//            for (int j = 0; j < n; ++j)
//                result.values[i*n + j] /= sum;

        return result;
    }
    void imshow();
    
    void transpose() {
        Type temp[total_size];
        memcpy(temp, values, sizeof(Type) * total_size);
        int blockSize = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<size_t>());
        std::cout << "blockkkk" << blockSize << "\n";
        for (int i =0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                memcpy(values + blockSize*(j * shape[0] + i), temp + blockSize*(i * shape[1] + j), sizeof(Type) * blockSize);
            }
        }
        
        std::swap(shape[0], shape[1]);
    }
    
    void flip(int dim) {
        for (int i = 0; i < shape[dim] / 2; i++) {
            swapD(dim, i, shape[dim]-i-1);
        }
    }

    void rotate(int times) {
        times = times % 4;

        if (abs(times) == 4) {
            return;
        }

        flip(0);
        transpose();
    }
    static MatrixH<uint8_t> fromCam(int camNo, IMGFormat format);
    static MatrixH fromBufferCopy(Type* dataBuffer, const std::vector<size_t>& dims) {
        MatrixH result =  MatrixH<Type>();
        size_t total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        Type* values = new Type[total_size];  // Allocate memory
        memcpy(values, dataBuffer, sizeof(Type) * total_size);
        
        result.total_size = total_size;
        result.shape = dims;
        result.values = values;
        
        return result;
    }
    

    
    MatrixH operator[](size_t i) {
        MatrixH result = MatrixH<Type>();
        result.shape = std::vector<size_t>(shape.begin() + 1, shape.end());;
        result.total_size = total_size / shape[0];
        
        result.values = values + i * result.total_size;
        result.flags = flags | 1;
        return result;
    }
    
    Type* operator()(size_t i) {
        Type* value = values + i * total_size / shape[0];
        return value;
    }
    
    Type* operator()(size_t i, size_t j) {
        Type* value = values + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1]));
        return value;
    }
    
    Type* operator()(size_t i, size_t j, size_t k) {
        Type* value = values + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])) + k * (total_size / (shape[0] * shape[1] * shape[2]));
        return value;
    }
    void appentCGpoint(size_t i, CGPoint pt) {
        values[i * (total_size / shape[0]) + 0] = pt.x;
        values[i * (total_size / shape[0]) + 1] = pt.y;
    }
    
    void appentMat(size_t i, MatrixH<Type>& pt) {
        if (pt.total_size != total_size / shape[0]) {
            std::cerr << "appendMat: Invalid Dimensions" << "\n";
            return;
        }
        
        memcpy(values + i * (total_size / shape[0]), pt.values, sizeof(Type) * pt.total_size);
    }
    
    void appentMat(size_t i, size_t j, const MatrixH<Type>& pt) {
        if (pt.total_size != total_size / (shape[0] * shape[1])) {
            std::cerr << "appendMat: Invalid Dimensions" << "\n";
            return;
        }

        memcpy(values + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])), pt.values, sizeof(Type) * pt.total_size);
    }
    
    void appentCGpoint(size_t i, size_t j, CGPoint pt) {
        values[i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])) + 0] = pt.x;
        values[i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])) + 1] = pt.y;
    }
    
    void cvtColor_BGRA2RGBA() {
        for (int i = 0; i < total_size / 4; i++) {
            uint8_t b = values[4*i];
            values[4*i] = values[4*i+2];
            values[4*i+2] = b;
        }
    }
    ~MatrixH() {
        if (flags & 0) {
//            delete [] values;
//            std::cout << "deleted" << "\n";
        }
        delete [] values;
//        std::cout << "deleted" << "\n";
    }
    void swapD(int dim, int s0, int s1) {
        if (s0 > shape[dim] && s1 > shape[dim]) {
            fprintf(stderr, "MATRIX: exceding dimension of  %lu not able to use.\n", shape[dim]);
            return;
        }
        
        int stride = accMul(shape, dim, shape.size());
        int blockSize = accMul(shape, dim+1, shape.size());
        int noOfOpp = accMul(shape, 0, dim);
        if (dim == 0) {stride  = blockSize; }
        Type* temp = new Type[blockSize];
        
        if (blockSize == 1) {
            int b0 =  blockSize*s0;
            int b1 = blockSize*s1;
            for (int i = 0; i < noOfOpp; i++) {
                *(temp) =  *(values + (i * stride + b0));
                *(values + (i * stride + b0)) =  *(values + (i * stride + b1));
                *(values + (i * stride + b1))=  *temp;
            }
            return;
        }
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(temp, values + (i * stride + blockSize*s0), sizeof(Type)* blockSize);
            memcpy(values + (i * stride + blockSize*s0), values + (i * stride + blockSize*s1), sizeof(Type) * blockSize);
            memcpy(values + (i * stride + blockSize*s1), temp, sizeof(Type) * blockSize);
        }
        delete[] temp;
    }
    
    void swapDS(int dim, int s0, int s1) {
        if (s0 > shape[dim] && s1 > shape[dim]) {
            fprintf(stderr, "MATRIX: exceding dimension of  %lu not able to use.\n", shape[dim]);
            return;
        }
        
        int stride = accMul(shape, dim, shape.size());
        int blockSize = accMul(shape, dim+1, shape.size());
        int noOfOpp = accMul(shape, 0, dim);
        if (dim == 0) {stride  = blockSize; }
        Type temp[blockSize];

        int b0 =  blockSize*s0;
        int b1 = blockSize*s1;
        {
            Timer time;
            for (int i = 0; i < noOfOpp; i++) {
                *(temp) =  *(values + (i * stride + b0));
                *(values + (i * stride + b0)) =  *(values + (i * stride + b1));
                *(values + (i * stride + b1))=  *temp;
            }
        }

    }
    
    static MatrixH fromBuffer(Type* dataBuffer, const std::vector<size_t>& dims) {
        MatrixH result = MatrixH<Type>();
        size_t total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        
        result.total_size = total_size;
        result.shape = dims;
        result.values = dataBuffer;
        
        return result;
    }
    
    static MatrixH constant(Type constant, std::initializer_list<size_t> dims) {
        MatrixH<Type> result = MatrixH<Type>();
        result.shape = dims;
        result.total_size = std::accumulate(result.shape.begin(), result.shape.end(), 1, std::multiplies<size_t>());
        result.values = new Type[result.total_size];
        std::fill(result.values, result.values + result.total_size, constant);
        return result;
    }
    
    template <typename NestedList>
    static MatrixH array(NestedList& init_list) {
        MatrixH result = MatrixH<Type>();
        
        result.shape = inferShape(init_list);   // Compute shape dynamically
        result.total_size = std::accumulate(result.shape.begin(), result.shape.end(), 1, std::multiplies<size_t>());
        result.values = new Type[result.total_size];   // Allocate memory
        flatten(init_list, result.values);      // Copy values into 1D array
        
        return result;
    }
    
    template <typename T>
    struct is_initializer_list : std::false_type {};

    template <typename T>
    struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
    
    template <typename T>
    static std::vector<size_t> inferShape(const std::initializer_list<T>& list) {
        std::vector<size_t> shape = { list.size() };
        
        if constexpr (is_initializer_list<T>::value) {
            auto subshape = inferShape(*list.begin());
            shape.insert(shape.end(), subshape.begin(), subshape.end());
        }
        return shape;
    }
    
    
    MatrixH<Type> AddGPU(MatrixH<Type> &other);
    
    template <typename T>
    static void flatten(const std::initializer_list<T>& list, Type* buffer, size_t& runs) {
        for (const auto& element : list) {
            if constexpr (is_initializer_list<T>::value) {
                flatten(element, buffer, runs);  // Recursive flattening
            } else {
                *(buffer+runs) = element;       // Store element in linear buffer
                runs++;
            }
        }
    }
    
    template <typename T>
    MatrixH<T> padding(int paddingT , int paddingB, int paddingL , int paddingR) {
        int newW = shape[1] + paddingL + paddingR;
        int newH = shape[0] + paddingT + paddingB;
        T* paddedValues = new T[newH * newW * 4];
        if (paddingT == paddingB) {
            for (int i = 0; i < paddingT; i++) {
                for (int j = 0; j < newW; j++) {
                    paddedValues[4 * (i*newW + j) + 0] = 0;
                    paddedValues[4 * (i*newW + j) + 1] = 0;
                    paddedValues[4 * (i*newW + j) + 2] = 0;
                    paddedValues[4 * (i*newW + j) + 3] = 255;
                    
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 0] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 1] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 2] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 3] = 255;
                }
            }
        } else {
            for (int j = 0; j < newW; j++) {
                for (int i = 0; i < paddingT; i++) {
                    paddedValues[4 * (i*newW + j) + 0] = 0;
                    paddedValues[4 * (i*newW + j) + 1] = 0;
                    paddedValues[4 * (i*newW + j) + 2] = 0;
                    paddedValues[4 * (i*newW + j) + 3] = 255;
                }
                for (int i = 0; i < paddingB; i++) {
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 0] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 1] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 2] = 0;
                    paddedValues[4 * ((shape[0] + paddingT + i)*newW + j) + 3] = 255;
                }
            }
        }
        if (paddingL == paddingR) {
            for (int i = 0; i < paddingL; i++) {
                for (int j = 0; j < newH; j++) {
                    paddedValues[4 * (j*newW + i) + 0] = 0;
                    paddedValues[4 * (j*newW + i) + 1] = 0;
                    paddedValues[4 * (j*newW + i) + 2] = 0;
                    paddedValues[4 * (j*newW + i) + 3] = 255;
                    
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 0] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 1] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 2] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 3] = 255;
                }
            }
        } else {
            for (int j = 0; j < newH; j++) {
                for (int i = 0; i < paddingL; i++) {
                    paddedValues[4 * (j*newW + i) + 0] = 0;
                    paddedValues[4 * (j*newW + i) + 1] = 0;
                    paddedValues[4 * (j*newW + i) + 2] = 0;
                    paddedValues[4 * (j*newW + i) + 3] = 255;
                }
                for (int i = 0; i < paddingR; i++) {
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 0] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 1] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 2] = 0;
                    paddedValues[4 * (j*newW + (paddingL + shape[1]) + i) + 3] = 255;
                }
            }
        }
        

        
        for (int i = 0; i < shape[0]; i++) {
            memcpy(paddedValues + (4 * (paddingT+i) * newW) + 4*paddingL, values + 4*i*shape[1], shape[1] * 4);
        }
        
        auto result = MatrixH<uint8_t>();
        result.values = paddedValues;
        result.shape = std::vector<size_t>({static_cast<unsigned long>(newH) , static_cast<unsigned long>(newW), 4});
        result.total_size = newH * newW * 4;
        return result;
    }
    void showImage(char* name) const;
    int cvWaitKey(int maxWait) const;
    
//    void operator=(MatrixH<Type> other) {
//        
//        if (total_size == other.total_size && shape.size() == other.shape.size()) {
//            memcpy(values, other.values, total_size * sizeof(Type));
//            delete [] other.values;
//            std::cout << "Copied" << "\n";
//        } else {
//            // Handle resizing if necessary
//            if (flags & 0) {
//                delete [] values;
//            }
//            
//            values = new Type[other.total_size]; // Allocate new memory
//            memcpy(values, other.values, other.total_size * sizeof(Type));
//            total_size = other.total_size;
//            shape = other.shape; // Assuming shape is std::vector<int>
////            delete [] values;
//        }
//    }
    
//    MatrixH& operator=(MatrixH&& other) noexcept {
//        if (this != &other) {
//            delete [] values;  // Free existing memory.
//            values = other.values;
//            total_size = other.total_size;
//            shape = std::move(other.shape);
//            flags = other.flags;
//            
//            other.values = nullptr;
//            other.total_size = 0;
//        }
//        return *this;
//    }
    
    // Optionally delete the copy constructor/assignment operator if you don't need them:
//    MatrixH(const MatrixH&) = default;
//    MatrixH& operator=(const MatrixH&) = delete;
    
    
    
    void operator=(Type other) {
        std::fill(values, values + total_size, other);
    }
    
    template <typename T>
    MatrixH<Type> ConvG(MatrixH<T> &kernel);
    
    template <typename T, typename L>
    MatrixH<T> convC(MatrixH<L> kernel) {
        if (kernel.shape[0] % 2 == 0 && kernel.shape[1] % 2 == 0) {
            std::cerr << "Kernel of wrong shape" << std::endl;
        }
        int paddingX = (kernel.shape[1] - 1) / 2;
        int paddingY = (kernel.shape[0] - 1) / 2;
        
        int newW = shape[1] + 2*paddingX;
        int newH = shape[0] + 2*paddingY;
        
        T* Newvalues = new T[shape[0] * shape[1]];
        T* paddedValues = padding<T>(paddingX, paddingX, paddingY, paddingY).values;
        
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        float sum4 = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                sum1 = sum2 = sum3 = sum4 = 0;
//                std::cout <<-static_cast<int>((kernel.shape[1]-1)/2) << "\n";
                for (int k = 0; k < static_cast<int>(kernel.shape[0]); k++) {
                    for (int l = 0; l < static_cast<int>(kernel.shape[1]); l++) {
                        int offX = (l-static_cast<int>((kernel.shape[1]-1)/2));
                        int offY = (k-static_cast<int>((kernel.shape[0]-1)/2));
//                        std::cout << "k " << k << " l " << l << " value :"<< kernel.values[k*kernel.shape[1] + l] <<
//                        " actual " <<  static_cast<int> (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 0]) <<" "
//                        <<  static_cast<int> (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 1]) << " "
//                        <<  static_cast<int> (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 2]) <<" "
//                        << static_cast<int> (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 3]) <<  "\n";
//                        
//                        std::cout << (int)paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 0] << " ";
//                        std::cout << (int)paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 1] << " ";
//                        std::cout << (int)paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 2] << " ";
//                        std::cout << (int)paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 3] << " \n";
//                        
//                        std::cout <<  kernel.values[k*kernel.shape[1] + l] << " ";
//                        std::cout <<  kernel.values[k*kernel.shape[1] + l] << " ";
//                        std::cout <<  kernel.values[k*kernel.shape[1] + l] << " ";
//                        std::cout <<  kernel.values[k*kernel.shape[1] + l] << " \n";
//                        
//                        std::cout << (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 0] * kernel.values[k*kernel.shape[1] + l]) << " ";
//                        std::cout << (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 1] * kernel.values[k*kernel.shape[1] + l]) << " ";
//                        std::cout << (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 2] * kernel.values[k*kernel.shape[1] + l]) << " ";
//                        std::cout << (paddedValues[4*((i+paddingY+offX)*newW + paddingX + j+offY) + 3] * kernel.values[k*kernel.shape[1] + l]) << " \n";
                        
                        sum1 += static_cast<float> (paddedValues[4*((i+paddingY+offY)*newW + paddingX + j+offX) + 0]) * kernel.values[k*kernel.shape[1] + l];
                        sum2 += static_cast<float> (paddedValues[4*((i+paddingY+offY)*newW + paddingX + j+offX) + 1]) * kernel.values[k*kernel.shape[1] + l];
                        sum3 += static_cast<float> (paddedValues[4*((i+paddingY+offY)*newW + paddingX + j+offX) + 2]) * kernel.values[k*kernel.shape[1] + l];
                        sum4 += static_cast<float> (paddedValues[4*((i+paddingY+offY)*newW + paddingX + j+offX) + 3]) * kernel.values[k*kernel.shape[1] + l];
//                        std::cout << sum1 << " " << sum2 << " " << sum3 << " " << sum4 << "\n \n";
                        
                    }
                }

                Newvalues[4 * (i*shape[1] + j) + 0] = std::clamp( (int)sum1, 0, 255 );
                Newvalues[4 * (i*shape[1] + j) + 1] = std::clamp( (int)sum2, 0, 255 );
                Newvalues[4 * (i*shape[1] + j) + 2] = std::clamp( (int)sum3, 0, 255 );
                Newvalues[4 * (i*shape[1] + j) + 3] = 255;
                
//                Newvalues[4 * (i*shape[1] + j) + 0] = paddedValues[4*((i+paddingY+0)*newW + paddingX + j) + 0];
//                Newvalues[4 * (i*shape[1] + j) + 1] = paddedValues[4*((i+paddingY+0)*newW + paddingX + j) + 1];
//                Newvalues[4 * (i*shape[1] + j) + 2] = paddedValues[4*((i+paddingY+0)*newW + paddingX + j) + 2];
//                Newvalues[4 * (i*shape[1] + j) + 3] = paddedValues[4*((i+paddingY+0)*newW + paddingX + j) + 3];
            }
        }

        
        auto result = MatrixH<uint8_t>();
        result.values = Newvalues;
        result.shape = std::vector<size_t>({(shape[0]) , (shape[1]), 4});
        result.total_size = (shape[0]) * (shape[1] ) * 4;
        return result;
    }
    
    static MatrixH<uint8_t> fromImage() {
        
        CFStringRef path = CFStringCreateWithCString(NULL, "/Users/adityadude/Documents/IMG_1278.JPG", kCFStringEncodingUTF8);
        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
        CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(url);
        CFRelease(path);
        if (!cgImage) {
            std::cerr << "Failed to create CGImage" << std::endl;
//            return;
        }
        size_t width = CGImageGetWidth(cgImage);
        size_t height = CGImageGetHeight(cgImage);
        size_t bytesPerRow = 4 * width;
        void *data = malloc(bytesPerRow * height);
        CGContextRef context = CGBitmapContextCreate(data, width, height, 8, bytesPerRow,
                                                     CGImageGetColorSpace(cgImage),
                                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
        CGContextRelease(context);
        CGImageRelease(cgImage);
        
        uint8_t* pixelData = static_cast<uint8_t*>(data);
        

        
        MatrixH result = MatrixH<uint8_t>();
        result.values = pixelData;
        result.shape = std::vector<size_t>({height, width, 4});
        result.total_size = width * height * 4;
        return result;
    }
    
    void Pad(int dim, int left, int right) {
        size_t oldDim = shape[dim];
        size_t Oldstride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        shape[dim] = shape[dim] + left + right;
        total_size = accMul(shape, 0, shape.size());
        Type* paddedValues = new Type[total_size];
        
        size_t stride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        size_t blocksize = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1, 1, std::multiplies<size_t>());
        if (dim == 0 ) { stride = blocksize;}
        size_t noOfOpp = std::accumulate(shape.begin(), shape.begin()+dim, 1, std::multiplies<size_t>());
        
        for (int i = 0; i < noOfOpp; i++) {
            for (int j = 0; j < left; j ++) {
                memset(paddedValues + i * stride + j * blocksize, 0, sizeof(Type) * blocksize);
            }
    
            memcpy(paddedValues + i * stride +  left * blocksize, values + (i) * Oldstride, sizeof(Type) * blocksize * oldDim);
            
            for (int j = 0; j < right; j ++) {
                memset(paddedValues + i * stride +  (left + oldDim + j) * blocksize , 0, sizeof(Type) * blocksize);
            }
        }
        
        delete[] values;
        values = paddedValues;
    }
    
    void Trim(int dim, int ltrim, int rtrim) {
        size_t oldDim = shape[dim];
        size_t Oldstride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        shape[dim] = shape[dim]  - ltrim - rtrim;
        total_size = accMul(shape);
        Type* paddedValues = new Type[total_size];
        
        size_t stride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        size_t blocksize = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1, 1, std::multiplies<size_t>());
        if (dim == 0 ) { stride = blocksize;}
        size_t noOfOpp = std::accumulate(shape.begin(), shape.begin()+dim, 1, std::multiplies<size_t>());
        
        for (int i = 0; i < noOfOpp; i++) {
            memcpy(paddedValues + i * stride * blocksize, values + (i) * Oldstride + ltrim * blocksize, sizeof(Type) * blocksize * (oldDim - ltrim - rtrim));
        }
        
        delete[] values;
        values = paddedValues;
    }
    
    static MatrixH<Type> Range(Type start, std::initializer_list<size_t> dims) {
        MatrixH<Type> result = MatrixH<Type>();
        result.shape = dims;
        result.total_size = std::accumulate(result.shape.begin(), result.shape.end(), 1, std::multiplies<size_t>());
        result.values = new Type[result.total_size];
        
        for (int i = 0; i < result.total_size; i++) {
            
            result.values[i] = i+start;
        }
        return result;
    }
    
    
    
    void PadnTrim(int dim, int left,int ltrim, int right,  int rtrim) {
        size_t oldDim = shape[dim];
        size_t Oldstride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        
        shape[dim] = shape[dim] + left + right - ltrim - rtrim;
        total_size = accMul(shape, 0, shape.size());
        Type* paddedValues = new Type[total_size];
        
        size_t stride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        size_t blocksize = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1, 1, std::multiplies<size_t>());
        if (dim == 0 ) { stride = blocksize;}
        
        size_t noOfOpp = std::accumulate(shape.begin(), shape.begin()+dim, 1, std::multiplies<size_t>());
        
        for (int i = 0; i < noOfOpp; i++) {
            for (int j = 0; j < left; j ++) {
                memset(paddedValues + i * stride + j * blocksize, 0, sizeof(Type) * blocksize);
            }
            
            // use oldblocksize
            memcpy(paddedValues + i * stride +  left * blocksize, values + (i) * Oldstride + ltrim * blocksize, sizeof(Type) * blocksize * (oldDim - ltrim - rtrim));
            
            for (int j = 0; j < right; j ++) {
                memset(paddedValues + i * stride +  (left + oldDim - ltrim - rtrim + j) * blocksize , 0, sizeof(Type) * blocksize);
            }
        }
        
        delete[] values;
        values = paddedValues;
    }
    
    void PadnTrim2D(int dim, int left, int ltrim, int right,  int rtrim, int left2, int ltrim2, int right2,  int rtrim2) {
        
        size_t oldDim = shape[dim];
        
//
        
        size_t Oldstride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        size_t Oldblocksize = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1, 1, std::multiplies<size_t>());
        
        shape[dim] = shape[dim] + left + right - ltrim - rtrim;
        shape[dim+1] = shape[dim+1] + left2 + right2 - ltrim2 - rtrim2;
        total_size = accMul(shape, 0, shape.size());
        Type* paddedValues = new Type[total_size];
        
        size_t stride = std::accumulate(shape.begin()+dim, shape.end(), 1, std::multiplies<size_t>());
        
        size_t blocksize = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1, 1, std::multiplies<size_t>());
        
        size_t blocksize2 = total_size / std::accumulate(shape.begin(), shape.begin()+dim+1+1, 1, std::multiplies<size_t>());
        if (dim == 0 ) { stride = blocksize;}
        size_t noOfOpp = std::accumulate(shape.begin(), shape.begin()+dim, 1, std::multiplies<size_t>());
        
        for (int i = 0; i < noOfOpp; i++) {
            for (int j = 0; j < left; j ++) {
                memset(paddedValues + i * stride + j * blocksize, 0, sizeof(Type) * blocksize);
            }
            
            for (int j = 0; j < oldDim - ltrim - rtrim; j++) {
                
                memset(paddedValues + i * stride + j * blocksize, 0, sizeof(Type) * blocksize2 * (left2 + ltrim2));
                
                memcpy(paddedValues + (i * stride) + (blocksize * left) + (j * blocksize) + (left2 * blocksize2), values + (i * Oldstride) + (ltrim * Oldblocksize) + (j * Oldblocksize) + (ltrim2 * blocksize2), sizeof(Type) * (Oldblocksize - (ltrim2 + rtrim2) * blocksize2));
                
//                memcpy(paddedValues + i * stride +  (left + j) * blocksize + (left2 * blocksize2), values + (i) * Oldstride + (ltrim + j) * Oldblocksize
//                       + (ltrim2 * blocksize2), sizeof(Type) * (Oldblocksize - (ltrim2 + rtrim2) * blocksize2) );
                
                
                memset(paddedValues + i * stride +  (left + j) * blocksize + (left2 * blocksize2) + (Oldblocksize - (ltrim2 + rtrim2) * blocksize2), 0, sizeof(Type) * blocksize2 * (right2 + rtrim2));
            }


            for (int j = 0; j < right; j ++) {
                memset(paddedValues + i * stride +  (left + oldDim - ltrim - rtrim + j) * blocksize , 0, sizeof(Type) * blocksize);
            }
        }
        
        delete[] values;
        
        values = paddedValues;
    }
    
    void drawRect(RectH<int> rect, MatrixH<Type> element) {
        for (int i = 0; i < shape.size() - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        
        
        if (rect.X + rect.width > shape[1] || rect.Y + rect.height > shape[0]) {
            std::cerr << "Error Dimensions excedeError Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        
        Type* rowBuffer = new Type[rect.width * element.total_size];
        
        for (size_t i = 0; i < rect.width; i++) {
            memcpy(values + rect.Y * widthsize + (rect.X + i) * elementSize, element.values, element.total_size * sizeof(Type));
        }
        
        for (int j = rect.Y+1; j < rect.Y + rect.height; j++) {
            memcpy(values + j * widthsize + rect.X * elementSize , values + rect.Y * widthsize + rect.X * elementSize, element.total_size * rect.width * sizeof(Type));
        }
    }
    
    simd_float2 Normalise(simd_int2 deviceCoord) {
        simd_float2 size = simd_make_float2(shape[1], shape[0]);
        
        return simd_make_float2((float)deviceCoord.x, (float)deviceCoord.y) / size;
    }
    
    void drawRect(simd_int2 p1, simd_int2 p2, MatrixH<Type> element) {
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
            memcpy(values + Y * widthsize + (X + i) * elementSize, element.values, element.total_size * sizeof(Type));
        }
        
        for (int j = Y+1; j < Y + height; j++) {
            memcpy(values + j * widthsize + X * elementSize , values + Y * widthsize + X * elementSize, element.total_size * width * sizeof(Type));
        }
    }
    
    void RectangleDetect(MatrixH<int>& detections);
    void HandsDetect(MatrixH<int>& detections, bool all_pts);
    
    
    
    void drawHands(MatrixH<int>& hands, bool lines) {
        static MatrixH<uint8_t> RED = {255, 0, 0, 255};
        static MatrixH<uint8_t> GREEN = {0, 255, 255, 255};
        
        for (int i= 0; i < hands.shape[0]; i++) {
            
            for (int j= 0; j < hands.shape[1]; j++) {
                if (i == 0) {
                    drawElipse({*hands(i, j, 0), *hands(i, j, 1), 50, 50}, RED);
                } else {
                    drawElipse({*hands(i, j, 0), *hands(i, j, 1), 50, 50}, GREEN);
                }
                
            }
            
            if (lines){
                for (int j= 0; j < 5; j++) {
                    for (int k= 0; k< 3; k++) {
                        drawLine(simd_make_int2(*hands(i, k*5 +j , 0), *hands(i, k*5 +j, 1)), simd_make_int2(*hands(i, (k+1)*5 +j, 0), *hands(i, (k+1)*5 +j, 1)), 0, GREEN);
                    }
                }
            }
            

        }
    }
    
    
    void drawRectStroked(RectH<int> rect, MatrixH<Type> element, float stroke) {
        int strokeInt = shape[0] * stroke;
        for (int i = 0; i < shape.size() - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        
        
        if (rect.X + rect.width > shape[1] || rect.Y + rect.height > shape[0]) {
            std::cerr << "Error Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        
        Type* rowBuffer = new Type[rect.width * element.total_size];
        
        for (size_t i = 0; i < rect.width; i++) {
            memcpy(values + rect.Y * widthsize + (rect.X + i) * elementSize, element.values, element.total_size * sizeof(Type));
        }
        
        for (int j = rect.Y+1; j < rect.Y + strokeInt; j++) {
            memcpy(values + j * widthsize + rect.X * elementSize , values + rect.Y * widthsize + rect.X * elementSize, element.total_size * rect.width * sizeof(Type));
        }
        
        for (int j = rect.Y + strokeInt; j < rect.Y + rect.height; j++) {
            memcpy(values + j * widthsize + rect.X * elementSize , values + rect.Y * widthsize + rect.X * elementSize, element.total_size * strokeInt * sizeof(Type));
            memcpy(values + j * widthsize + (rect.X + rect.width - strokeInt) * elementSize , values + rect.Y * widthsize + rect.X * elementSize, element.total_size * strokeInt * sizeof(Type));
        }
        
        for (int j = rect.Y + rect.height - strokeInt; j < rect.Y + rect.height; j++) {
            memcpy(values + j * widthsize + rect.X * elementSize , values + rect.Y * widthsize + rect.X * elementSize, element.total_size * rect.width * sizeof(Type));
        }
    }
    void FaceDetect(MatrixH<int>& detections);

    
    void drawElipse(const RectH<int>& rect, const MatrixH<Type>& element) {
        for (int i = 0; i < shape.size() - 2; i++) {
            if (shape[i+2] != element.shape[i]) {
                std::cerr << "Elipse: Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
                std::cerr << shape << " != " << element.shape << "\n";
                return;
            }
        }
        if (rect.X + rect.width > shape[1] || rect.Y + rect.height > shape[0]) {
            std::cerr << "Error Dimensions excede \n";
            return;
        }
        
        
        size_t widthsize = total_size / shape[0];
        size_t elementSize = total_size / (shape[0] * shape[1]);
        auto centre = simd_make_float2(rect.X + (rect.width / 2.0), rect.Y + (rect.height / 2.0));
        auto rad = simd_make_float2(rect.width, rect.height) / 2;
        
        for (int i = rect.X; i < rect.X + rect.width; i ++) {
            for (int j = rect.Y; j < rect.Y + rect.height; j ++) {
                auto coord = simd_make_float2(i, j);
                float S1 = simd_dot(((coord - centre) / rad), ((coord - centre) / rad)) - 1.0;
                
                if (S1 < 0.0) {
                    memcpy(values + j * widthsize + i * elementSize , element.values, element.total_size * sizeof(Type));
                }
            }
        }
    }
    
    void drawText(char* text, MatrixH<int> point, const MatrixH<int>& colour, float fontSize);
    
    void saveImage() {
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        if (!colorSpace) {
            fprintf(stderr, "Failed to create color space!\n");
        }
        CGContextRef context = CGBitmapContextCreate(
            values,                     // Data buffer (NULL for CoreGraphics to allocate)
                                                     shape[1],                    // Width
                                                     shape[0],                   // Height
            8,                        // Bits per component
                                                     shape[1] * 4,                // Bytes per row (RGBA, 4 bytes per pixel)
            colorSpace,               // Color space
            kCGImageAlphaPremultipliedLast // Bitmap info
        );

        if (!context) {
            fprintf(stderr, "Failed to create bitmap context!\n");
            CGColorSpaceRelease(colorSpace);
        }
        
        CGImageRef image = CGBitmapContextCreateImage(context);
        if (!image) {
            fprintf(stderr, "Failed to create image from context!\n");
            CGContextRelease(context);
            CGColorSpaceRelease(colorSpace);
        }
        
        CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, CFSTR("/Users/adityadude/Desktop/output1.png"), kCFURLPOSIXPathStyle, false);
        CGImageDestinationRef destination = CGImageDestinationCreateWithURL(url, kUTTypePNG, 1, NULL);
        if (destination) {
            CGImageDestinationAddImage(destination, image, NULL);
            if (!CGImageDestinationFinalize(destination)) {
                fprintf(stderr, "Failed to write image to file!\n");
            }
            CFRelease(destination);
        } else {
            fprintf(stderr, "Failed to create image destination!\n");
        }
    }
//    ~MatrixH() { delete[] values; }
    
    void print() const {
        if (shape.size() == 1) {
            std::cout << "{ ";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << (int)values[i] << " ,";
            }
            std::cout << "} \n";
        }
        else if (shape.size() == 2) {
            std::cout << "{ ";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << (int)values[shape[1] * i + j] << " ";
                }
                std::cout << "} \n";
            }
            std::cout << "} \n";
        } else if (shape.size() == 3) {
            for (size_t i = 0; i < shape[0]; i++) {
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << "{ ";
                    for (size_t k = 0; k < shape[2]; k++) {
                        std::cout << (int)values[shape[2]*(shape[1] * i + j) + k] << " ";
                    }
                    std::cout << "} ";
                }
                std::cout << std::endl;
            }
        } else if (shape.size() == 4) {
            for (size_t l = 0; l < shape[0]; l++) {
                for (size_t i = 0; i < shape[1]; i++) {
                    for (size_t j = 0; j < shape[2]; j++) {
                        std::cout << "{ ";
                        for (size_t k = 0; k < shape[3]; k++) {
                            std::cout << (int)values[shape[3]*(shape[2]*(shape[1] * l + i) + j)  + k] << " ";
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
    
    
};

template <typename type>
class ShapeWrapper {
public:
    class Shape;
    Shape* pShapes;
    size_t count;
};

class CamReaderWrapper {
public:
    CamReaderWrapper(int camNo = 0);
    ~CamReaderWrapper() {
        
    }

    bool grabFrame();
    void retrieveFrame(MatrixH<uint8_t>& frame); // adjust signature as needed
    int startCaptureDevice(int cameraNum);
    void stopCaptureDevice();

private:
    // Use Pimpl to hide the Objective-C++ implementation
    class CamReader;
    CamReader* pCamReader;
};


class MetalWrapper {
public:
    MetalWrapper();
    ~MetalWrapper() {
        
    }
    void Render(MatrixH<uint8_t>& layer);
//    std::vector<Shape<uint8_t>> shapes;
private:
    class Renderer;
    Renderer* pRenderer;
    
    

private:
    // Use Pimpl to hide the Objective-C++ implementation
    class CamReader;
    CamReader* pCamReader;
};
static MatrixH<uint8_t> RED = {255, 0, 0, 255};





#endif /* AlgebroHeap_hpp */

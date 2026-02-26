//#include "Mods/GPUManager.h"
//#include "Mods/GeometryNode.h"
//#include "Mods/Utils.h"
//
//GeometryNode<uint16> cube(nil, 0, nil, 0);
////
////  TestTracer.cpp
////  WorldOf3D
////
////  Created by Aditya Dudeja on 03/11/25.
////
//
//#ifndef TestTracer_h
//#define TestTracer_h
//
//#include <iostream>
#include <vector>
//#include <string>

//#include <memory>
//#include <sstream>
//#include <map>
//#include <functional>
//
//// Forward declarations
//class TracedExpr;
//class Tensor;
//class TensorAccess;
//
//// Index variable (I, J, K, etc.)
//class IndexVar {
//public:
//    std::string name;
//    int offset;
//    
//    IndexVar(const std::string& name, int offset = 0)
//        : name(name), offset(offset) {}
//    
//    IndexVar operator+(int n) const {
//        return IndexVar(name, offset + n);
//    }
//    
//    IndexVar operator-(int n) const {
//        return IndexVar(name, offset - n);
//    }
//    
//    std::string toString() const {
//        if (offset == 0) return name;
//        if (offset > 0) return name + "+" + std::to_string(offset);
//        return name + std::to_string(offset);
//    }
//    
//    std::string toMSL() const {
//        if (offset == 0) return "gid";
//        if (offset > 0) return "(gid + " + std::to_string(offset) + ")";
//        return "(gid - " + std::to_string(std::abs(offset)) + ")";
//    }
//};
//
//// Tensor as a view over a buffer
//class Tensor {
//public:
//    std::vector<size_t> shape;
//    std::vector<size_t> strides;
//    size_t offset;
//    std::string buffer;
//    std::string name;
//    size_t ndim;
//    
//    Tensor(const std::vector<size_t>& shape,
//           const std::string& name = "tensor",
//           const std::vector<size_t>& strides = {},
//           size_t offset = 0)
//        : shape(shape), offset(offset), name(name), ndim(shape.size()) {
//        
//        buffer = "buffer_" + name;
//        
//        if (strides.empty()) {
//            this->strides = computeDefaultStrides(shape);
//        } else {
//            this->strides = strides;
//        }
//    }
//    
//    // Row-major (C-contiguous) strides
//    static std::vector<size_t> computeDefaultStrides(const std::vector<size_t>& shape) {
//        std::vector<size_t> strides(shape.size());
//        size_t stride = 1;
//        for (int i = shape.size() - 1; i >= 0; i--) {
//            strides[i] = stride;
//            stride *= shape[i];
//        }
//        return strides;
//    }
//    
//    // Create a view with different shape
//    Tensor view(const std::vector<size_t>& newShape) const {
//        return Tensor(newShape, name, computeDefaultStrides(newShape), offset);
//    }
//    
//    // Transpose creates a view with swapped strides
//    Tensor transpose(size_t dim0, size_t dim1) const {
//        auto newStrides = strides;
//        auto newShape = shape;
//        std::swap(newStrides[dim0], newStrides[dim1]);
//        std::swap(newShape[dim0], newShape[dim1]);
//        return Tensor(newShape, name, newStrides, offset);
//    }
//    
//    // Slice creates a view
//    Tensor slice(size_t dim, size_t start, size_t end = 0) const {
//        auto newShape = shape;
//        if (end == 0) end = shape[dim];
//        size_t newOffset = offset + start * strides[dim];
//        newShape[dim] = end - start;
//        return Tensor(newShape, name, strides, newOffset);
//    }
//};
//
//// Expression types
//enum class ExprType {
//    CONSTANT,
//    TENSOR_ACCESS,
//    BINARY_OP,
//    ARRAY
//};
//
//// Tensor access (tensor[I] or tensor[I, J])
//class TensorAccess {
//public:
//    std::shared_ptr<Tensor> tensor;
//    std::vector<IndexVar> indices;
//    
//    TensorAccess(std::shared_ptr<Tensor> tensor, const std::vector<IndexVar>& indices)
//        : tensor(tensor), indices(indices) {}
//    
//    // Check if this access needs inner loops (fewer indices than dimensions)
//    bool needsInnerLoop() const {
//        return indices.size() < tensor->ndim;
//    }
//    
//    size_t getNumInnerDims() const {
//        return tensor->ndim - indices.size();
//    }
//    
//    std::string toMSL(const std::vector<std::string>& innerLoopVars = {}) const {
//        std::stringstream ss;
//        ss << tensor->buffer << "[";
//        
//        if (tensor->offset > 0) {
//            ss << tensor->offset << " + ";
//        }
//        
//        std::vector<std::string> terms;
//        
//        // Handle explicit indices
//        for (size_t i = 0; i < indices.size(); i++) {
//            const auto& idx = indices[i];
//            size_t stride = tensor->strides[i];
//            
//            if (stride == 1) {
//                terms.push_back(idx.toMSL());
//            } else {
//                terms.push_back(idx.toMSL() + " * " + std::to_string(stride));
//            }
//        }
//        
//        // Handle inner dimensions with loop variables
//        for (size_t i = 0; i < innerLoopVars.size(); i++) {
//            size_t dimIdx = indices.size() + i;
//            size_t stride = tensor->strides[dimIdx];
//            
//            if (stride == 1) {
//                terms.push_back(innerLoopVars[i]);
//            } else {
//                terms.push_back(innerLoopVars[i] + " * " + std::to_string(stride));
//            }
//        }
//        
//        for (size_t i = 0; i < terms.size(); i++) {
//            if (i > 0) ss << " + ";
//            ss << terms[i];
//        }
//        
//        ss << "]";
//        return ss.str();
//    }
//};
//
//// Traced expression node
//class TracedExpr : public std::enable_shared_from_this<TracedExpr> {
//public:
//    ExprType type;
//    std::vector<std::shared_ptr<TracedExpr>> children;
//    std::string opSymbol; // +, -, *, /
//    float constantValue;
//    std::vector<float> arrayValue;
//    std::shared_ptr<TensorAccess> accessPtr;
//    
//    TracedExpr(ExprType type) : type(type), constantValue(0) {}
//    
//    static std::shared_ptr<TracedExpr> constant(float val) {
//        auto expr = std::make_shared<TracedExpr>(ExprType::CONSTANT);
//        expr->constantValue = val;
//        return expr;
//    }
//    
//    static std::shared_ptr<TracedExpr> array(const std::vector<float>& vals) {
//        auto expr = std::make_shared<TracedExpr>(ExprType::ARRAY);
//        expr->arrayValue = vals;
//        return expr;
//    }
//    
//    static std::shared_ptr<TracedExpr> access(std::shared_ptr<TensorAccess> acc) {
//        auto expr = std::make_shared<TracedExpr>(ExprType::TENSOR_ACCESS);
//        expr->accessPtr = acc;
//        return expr;
//    }
//    
//    std::shared_ptr<TracedExpr> binaryOp(const std::string& op, std::shared_ptr<TracedExpr> other) {
//        auto expr = std::make_shared<TracedExpr>(ExprType::BINARY_OP);
//        expr->opSymbol = op;
//        expr->children = {shared_from_this(), other};
//        return expr;
//    }
//    
//    bool hasArrayConstant() const {
//        if (type == ExprType::ARRAY) return true;
//        for (const auto& child : children) {
//            if (child->hasArrayConstant()) return true;
//        }
//        return false;
//    }
//    
//    bool needsInnerLoop() const {
//        if (type == ExprType::TENSOR_ACCESS) {
//            return accessPtr->needsInnerLoop();
//        }
//        for (const auto& child : children) {
//            if (child->needsInnerLoop()) return true;
//        }
//        return false;
//    }
//    
//    size_t getMaxInnerDims() const {
//        if (type == ExprType::TENSOR_ACCESS) {
//            return accessPtr->getNumInnerDims();
//        }
//        size_t maxDims = 0;
//        for (const auto& child : children) {
//            maxDims = std::max(maxDims, child->getMaxInnerDims());
//        }
//        return maxDims;
//    }
//    
//    std::string toMSL(const std::vector<std::string>& innerLoopVars = {}) const {
//        switch (type) {
//            case ExprType::CONSTANT:
//                return std::to_string(constantValue);
//            
//            case ExprType::TENSOR_ACCESS:
//                return accessPtr->toMSL(innerLoopVars);
//            
//            case ExprType::ARRAY:
//                return arrayValue.empty() ? "0" : std::to_string(arrayValue[0]);
//            
//            case ExprType::BINARY_OP: {
//                std::string left = children[0]->toMSL(innerLoopVars);
//                std::string right = children[1]->toMSL(innerLoopVars);
//                return left + " " + opSymbol + " " + right;
//            }
//        }
//        return "unknown";
//    }
//};
//
//// Operator overloads for TracedExpr
//inline std::shared_ptr<TracedExpr> operator-(std::shared_ptr<TracedExpr> left, std::shared_ptr<TracedExpr> right) {
//    return left->binaryOp("-", right);
//}
//
//inline std::shared_ptr<TracedExpr> operator+(std::shared_ptr<TracedExpr> left, std::shared_ptr<TracedExpr> right) {
//    return left->binaryOp("+", right);
//}
//
//inline std::shared_ptr<TracedExpr> operator*(std::shared_ptr<TracedExpr> left, std::shared_ptr<TracedExpr> right) {
//    return left->binaryOp("*", right);
//}
//
//inline std::shared_ptr<TracedExpr> operator/(std::shared_ptr<TracedExpr> left, std::shared_ptr<TracedExpr> right) {
//    return left->binaryOp("/", right);
//}
//
//// Assignment (result[I] = expr)
//struct Assignment {
//    std::shared_ptr<TensorAccess> target;
//    std::shared_ptr<TracedExpr> expr;
//};
//
//// Traced tensor proxy
//class TracedTensor {
//public:
//    std::shared_ptr<Tensor> tensor;
//    
//    TracedTensor(std::shared_ptr<Tensor> tensor) : tensor(tensor) {}
//    
//    std::shared_ptr<TracedExpr> operator[](const IndexVar& idx) {
//        auto acc = std::make_shared<TensorAccess>(tensor, std::vector<IndexVar>{idx});
//        return TracedExpr::access(acc);
//    }
//    
//    std::shared_ptr<TracedExpr> at(const IndexVar& idx) {
//        return (*this)[idx];
//    }
//    
//    std::shared_ptr<TracedExpr> at(const IndexVar& idx1, const IndexVar& idx2) {
//        auto acc = std::make_shared<TensorAccess>(tensor, std::vector<IndexVar>{idx1, idx2});
//        return TracedExpr::access(acc);
//    }
//};
//
//// MSL Kernel Generator
//class MSLKernelGenerator {
//private:
//    int bufferIndex;
//    std::map<std::string, std::shared_ptr<Tensor>> bufferMap;
//    
//    void collectBuffers(std::shared_ptr<Tensor> tensor) {
//        if (bufferMap.find(tensor->buffer) == bufferMap.end()) {
//            bufferMap[tensor->buffer] = tensor;
//        }
//    }
//    
//    void collectBuffersFromExpr(std::shared_ptr<TracedExpr> expr) {
//        if (expr->type == ExprType::TENSOR_ACCESS) {
//            collectBuffers(expr->accessPtr->tensor);
//        }
//        for (const auto& child : expr->children) {
//            collectBuffersFromExpr(child);
//        }
//    }
//    
//public:
//    MSLKernelGenerator() : bufferIndex(0) {}
//    
//    std::string generate(const Assignment& assignment, const std::string& indexVarName) {
//        // Collect all buffers
//        collectBuffers(assignment.target->tensor);
//        collectBuffersFromExpr(assignment.expr);
//        
//        std::stringstream code;
//        code << "kernel void vmapedFunc(\n";
//        
//        // Generate buffer parameters
//        for (const auto& pair : bufferMap) {
//            const std::string& bufferName = pair.first;
//            const std::shared_ptr<Tensor>& tensor = pair.second;
//            bool isResult = tensor == assignment.target->tensor;
//            const char* qualifier = isResult ? "device float*" : "device const float*";
//            code << "    " << qualifier << " " << bufferName
//                 << " [[buffer(" << bufferIndex++ << ")]],\n";
//        }
//        
//        code << "    uint gid [[thread_position_in_grid]]\n";
//        code << ") {\n";
//        
//        // Check if we need inner loops for accessing multidimensional tensors with fewer indices
//        if (assignment.target->needsInnerLoop() || assignment.expr->needsInnerLoop()) {
//            size_t numInnerDims = std::max(
//                assignment.target->getNumInnerDims(),
//                assignment.expr->getMaxInnerDims()
//            );
//            
//            // Generate nested loops for inner dimensions
//            std::vector<std::string> loopVars;
//            std::string indent = "    ";
//            
//            for (size_t d = 0; d < numInnerDims; d++) {
//                std::string loopVar = std::string(1, 'j' + d);
//                loopVars.push_back(loopVar);
//                
//                // Get the size of this inner dimension from the tensor shape
//                size_t dimIdx = assignment.target->tensor->ndim - numInnerDims + d;
//                size_t dimSize = assignment.target->tensor->shape[dimIdx];
//                
//                code << indent << "for (uint " << loopVar << " = 0; "
//                     << loopVar << " < " << dimSize << "; "
//                     << loopVar << "++) {\n";
//                indent += "    ";
//            }
//            
//            // Generate assignment with loop variables
//            code << indent << assignment.target->toMSL(loopVars)
//                 << " = " << assignment.expr->toMSL(loopVars) << ";\n";
//            
//            // Close loops
//            for (size_t d = 0; d < numInnerDims; d++) {
//                indent = indent.substr(0, indent.length() - 4);
//                code << indent << "}\n";
//            }
//        } else {
//            // Simple assignment without inner loops
//            code << "    " << assignment.target->toMSL()
//                 << " = " << assignment.expr->toMSL() << ";\n";
//        }
//        
//        code << "}\n";
//        
//        return code.str();
//    }
//};
//
//// Main tracing function
//std::string traceMSL(
//    std::function<Assignment(const IndexVar&, TracedTensor&, TracedTensor&)> fn,
//    const std::vector<size_t>& shape,
//    size_t axis = 0
//) {
//    IndexVar indexVar(std::string(1, 'I' + axis));
//    
//    auto inputTensor = std::make_shared<Tensor>(shape, "matrix");
//    auto outputTensor = std::make_shared<Tensor>(shape, "result");
//    
//    TracedTensor tracedInput(inputTensor);
//    TracedTensor tracedOutput(outputTensor);
//    
//    Assignment assignment = fn(indexVar, tracedInput, tracedOutput);
//    
//    MSLKernelGenerator generator;
//    return generator.generate(assignment, indexVar.name);
//}
//
//// Helper to create assignment
//inline Assignment assign(TracedTensor& result, const IndexVar& idx, std::shared_ptr<TracedExpr> expr) {
//    auto acc = std::make_shared<TensorAccess>(result.tensor, std::vector<IndexVar>{idx});
//    return Assignment{acc, expr};
//}
//
//inline Assignment assign(TracedTensor& result, const IndexVar& idx1, const IndexVar& idx2,
//                  std::shared_ptr<TracedExpr> expr) {
//    auto acc = std::make_shared<TensorAccess>(result.tensor, std::vector<IndexVar>{idx1, idx2});
//    return Assignment{acc, expr};
//}
//
//
//// Examples
//int testMain() {
//    std::cout << "Example 1: result[I] = mat[I] - mat[I-1]\n";
//    std::cout << std::string(50, '=') << "\n";
//    auto msl1 = traceMSL([](const IndexVar& I, TracedTensor& mat, TracedTensor& result) {
//        return assign(result, I, mat[I] - mat[I - 1]);
//    }, {10});
//    std::cout << msl1 << "\n\n";
//    
//    std::cout << "Example 2: result[I] = mat[I+1] - 1\n";
//    std::cout << std::string(50, '=') << "\n";
//    auto msl2 = traceMSL([](const IndexVar& I, TracedTensor& mat, TracedTensor& result) {
//        return assign(result, I, mat[I + 1] - TracedExpr::constant(1));
//    }, {10});
//    std::cout << msl2 << "\n\n";
//    
//    std::cout << "Example 3: 2D tensor - result[I] = mat[I+1] - {1, 1, 1}\n";
//    std::cout << std::string(50, '=') << "\n";
//    auto msl3 = traceMSL([](const IndexVar& I, TracedTensor& mat, TracedTensor& result) {
//        return assign(result, I, mat[I + 1] - TracedExpr::array({1, 1, 1}));
//    }, {10, 3});
//    std::cout << msl3 << "\n\n";
//    
//    std::cout << "Example 4: 2D tensor with single index - result[I] = mat[I+1] - mat[I]\n";
//    std::cout << std::string(50, '=') << "\n";
//    auto msl4 = traceMSL([](const IndexVar& I, TracedTensor& mat, TracedTensor& result) {
//        return assign(result, I, mat[I] - mat[I-1]);
//    }, {10, 5, 6});
//    std::cout << msl4 << "\n";
//    
//    return 0;
//}
//#endif /* TestTracer_h */



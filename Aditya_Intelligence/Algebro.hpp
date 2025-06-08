//
//  Algebro.hpp
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 03/02/25.
//

#ifndef Algebro_hpp
#define Algebro_hpp

#include <stdio.h>

void trial();

template <typename T>
class Test {
public:
    void Tester();
};



template <size_t M, size_t N>
class Matrix {
public:
    float values[M*N];
    static constexpr int rows = M;
    static constexpr int cols = N;
    
    static Matrix<M, N> zeros();
    
    static Matrix<M, N> ones();
    
    static Matrix<M, N> constant(float constant);
    
    static Matrix<M, N> Range(float start);
    
    Matrix<M, N> operator+(const Matrix<M, N>& other) const;
    Matrix<M, N> operator-(const Matrix<M, N>& other) const;
    Matrix<M, N> operator*(const Matrix<M, N>& other) const;
    Matrix<M, N> operator/(const Matrix<M, N>& other) const;
    
    Matrix<M, N> operator+(const float scalar) const;
    Matrix<M, N> operator-(const float scalar) const;
    Matrix<M, N> operator*(const float scalar) const;
    Matrix<M, N> operator/(const float scalar) const;
    
    template <size_t Q>
    Matrix<M, Q> dot(Matrix<N, Q>& B);
    
    Matrix<M*N,1> flatten();
    
    template <size_t P, size_t Q>
    Matrix<P, Q> reshape();
    
    void print();
};


#endif /* Algebro_hpp */

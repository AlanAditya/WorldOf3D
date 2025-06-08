//
//  Fibonacci.cpp
//  Algorithms
//
//  Created by Manoj Kumar on 14/12/24.
//

#include "Fibonacci.hpp"
#include <iostream>

int fibnac(int n) {
    int a = 0;
    int b = 1;
    int temp = 0;
    while (n > 0) {
        temp = a + b;
        a = b;
        b = temp;
        std::cout << temp << std::endl;
        n--;
    }
    return 0;
}
void printBinary(int number) {
    std::cout << "Binary representation of " << number << ": ";
    for (int i = sizeof(int) * 8 - 1; i >= 0; --i) {  // Iterate from most significant bit to least
        std::cout << ((number >> i) & 1);  // Extract the i-th bit
    }
    std::cout << std::endl;
}


int overflow() {
    uint a = 999999999;
    a=a+1099999999;
    a=a+10000000;
    a=a+100000;
    a=a+10000;
    a=a+30000000;
    a=a+1000000;
    a=a+5000000;
    a=a+1300000;
    a=a+19990;
    a=a+53659;
    a=a+1;
    a=1;
//    a=16;
    
    unsigned char* bytes = reinterpret_cast<unsigned char *>(&a);
    a = 0x80000000;
    a=a+2;
//    for (std::size_t i = sizeof(int)-1; i+1 > 0; i--) {
//        std::cout << ((a) & i) << "\n";
//        
////        printf("%02x ", bytes[i]);  // Print each byte in hexadecimal
//    }
    std::cout <<  a << "\n";
    printBinary(a);
    return 0;
}

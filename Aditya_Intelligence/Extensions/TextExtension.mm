//
//  TextExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 23/02/25.
//

#import <Foundation/Foundation.h>
#include "AlgebroHeap.hpp"

#include <CoreGraphics/CoreGraphics.h>

// Function to create a CGColor from RGBA values
CGColorRef createCGColor(float r, float g, float b, float a) {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGFloat components[] = {r, g, b, a};
    CGColorRef color = CGColorCreate(colorSpace, components);
    CGColorSpaceRelease(colorSpace);
    return color;  // Remember to CFRelease when done using it
}

CGColorRef createCGColorFromMatrixH(const MatrixH<int> &colorMat) {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGFloat components[] = {colorMat.values[0] / 255.0, colorMat.values[1] / 255.0, colorMat.values[2] / 255.0, colorMat.values[3] / 255.0};
    CGColorRef color = CGColorCreate(colorSpace, components);
    CGColorSpaceRelease(colorSpace);
    return color;  // Remember to CFRelease when done using it
}

template <typename Type>
void MatrixH<Type>::drawText(char* text, MatrixH<int> point, const MatrixH<int>& colorMat, float fontSize) {
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
        values, shape[1], shape[0], 8 * sizeof(Type), sizeof(Type) * widthsize, colorSpace, kCGImageAlphaPremultipliedLast
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
    
    CGContextSetTextPosition(context, point.values[0], point.values[1]);
    CGContextSetTextDrawingMode(context, kCGTextFillClip);
    
    // Draw text
    CTLineDraw(line, context);
}

template void MatrixH<uint8_t>::drawText(char* text, MatrixH<int> point, const MatrixH<int>& colorMat, float fontSize);

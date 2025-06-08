//
//  StringConverter.m
//  WorldOf3D
//
//  Created by Manoj Kumar on 05/06/24.
//

//#import <Foundation/Foundation.hpp>
//#include "StringConverter.hpp"

//NS::String* stringToNSString(const std::string& str) {
//    @autoreleasepool {
//        // Create an NSString from the std::string
//        NSString* nsString = [NSString stringWithUTF8String:str.c_str()];
//        // Convert NSString to NS::String (assuming they are bridgeable)
//        NS::String* nsCppString = reinterpret_cast<NS::String*>(nsString);
//        return nsCppString;
//    }
//}
#import <Foundation/Foundation.h>
#include "StringConverter.hpp"

NS::String* stringToNSString(const std::string& str) {
    @autoreleasepool {
        NSString* NSstr = [NSString stringWithUTF8String:str.c_str()];
        return (NS::String*)CFBridgingRetain(NSstr);;
    }
}

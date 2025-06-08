//
//  ViewImporter.m
//  ImageProcessing
//
//  Created by Manoj Kumar on 17/08/24.
//

//#import <Foundation/Foundation.h>
//#include "ImageProcessing-Swift.h"
//#include <AppKit/AppKit.h>

//NSView* create() {
////    NSView *pSwiftView = ImageProcessing::HelloView();
//    return pSwiftView;
//}

#include <Cocoa/Cocoa.h>
#include "ImageProcessing-Swift.h"
#include <SwiftUI/SwiftUI.h>
#include <MetalKit/MetalKit.h>
#include <Metal/Metal.h>
#include <AppKit/AppKit.hpp>
#import <MetalKit/MetalKit.hpp>
#import "ViewImporterBridge.hpp"
#include <iostream>

//#include "ViewImporterBridge.hpp"



// Functionto create an NSView with a NSTextField displaying "Hello, World"
NSView* createHelloWorldView() {
    // Create a frame for the NSView
    NSRect frame = NSMakeRect(0, 0, 200, 100);
    
    // Create an NSView
    NSView* view = [[NSView alloc] initWithFrame:frame];
    
    // Create a NSTextField with the text "Hello, World"
    NSTextField* textField = [[NSTextField alloc] initWithFrame:NSMakeRect(10, 10, 180, 80)];
    [textField setStringValue:@"Hello, World"];
    [textField setBezeled:NO];
    [textField setDrawsBackground:NO];
    [textField setEditable:NO];
    [textField setSelectable:NO];
    
    // Add the NSTextField to the NSView
    [view addSubview:textField];
    
    return view;
}

//NSView* createHelloWorldSwiftView(NSView* view) {
//    NSView *swiftUIViewController = [MySwiftUIViewControllerBridge createSwiftUIViewControllerWithView:view];
//    return swiftUIViewController;
//}

NSView* add_arrays(const float* inA,
                const float* inB,
                float* result,
                int length)
{
    for (int index = 0; index < length ; index++)
    {
        result[index] = inA[index] + inB[index];
    }
    NSRect frame = NSMakeRect(0, 0, 200, 100);
    
    // Create an NSView
    NSView* view = [[NSView alloc] initWithFrame:frame];
    return view;
}

//@implementation SwiftUI_C_Plus_plus : NSObject
////{
////    int NoOfViews = 8;
////}
//
//- (NSView*) ViewExchanger: (NSView*) MTL
//{
//    MySwiftUIViewController *swiftUIViewController = [[MySwiftUIViewController alloc] initWithView:MTL];
//    return swiftUIViewController.view;
//}
//
//@end


//void SwiftUI_C_Plus_plus::createHelloWorldSwiftViewMTK(NSView* metalView) {
//    MySwiftUIViewController *swiftUIViewController = [[MySwiftUIViewController alloc] initWithView:metalView];
////    return swiftUIViewController.view;
//}

//NSView* createHelloWorldSwiftViewMTK(NS::View *metalView) {
//    NSView *SwiftCompatible = static_cast<NSView*>(metalView);
//    MySwiftUIViewController *swiftUIViewController = [[MySwiftUIViewController alloc] initWithView:SwiftCompatible];
////    NSView *swiftUIViewController = [MySwiftUIViewControllerBridge createSwiftUIViewControllerWithView:SwiftCompatible];
//    swiftUIViewController.objcData;
//    return swiftUIViewController.view;
//}
//struct ExchangePackage { Object3D objec; NS::View* view; };
NSString* stdStringToNSString(const std::string& str) {
    return [NSString stringWithUTF8String:str.c_str()];
}

Object3D deserializeObject3D(NSData *data) {
    Object3D object;

    // Ensure the data length matches the size of Object3D
    if (data.length == sizeof(Object3D)) {
        // Deserialize NSData into Object3D
        [data getBytes:&object length:sizeof(Object3D)];
        std::cout << object.location.x;
//        NSString *nsString = stdStringToNSString(object.id);
//        NSLog(@"%@", nsString);
        // Convert id from C-style string to std::string
        object.id = std::string(object.id);
    } else {
        // Handle the case where data length doesn't match the expected size
        throw std::runtime_error("NSData size does not match Object3D size");
    }

    return object;
}

void processArray(NSArray *objcArray, Object3D *storeObj) {
    int index = 0;
    for (NSData *data in objcArray) {
        Object3D obj = deserializeObject3D(data);
        storeObj[index] = obj;
        index++;
    }
}



void printNSArray(NSArray *array) {
    // Iterate over the NSArray
    for (id element in array) {
        // Check the type of the element and print accordingly
        if ([element isKindOfClass:[NSString class]]) {
            NSString *str = (NSString *)element;
            std::cout << [str UTF8String] << std::endl;
        } else if ([element isKindOfClass:[NSNumber class]]) {
            NSNumber *num = (NSNumber *)element;
            std::cout << [num stringValue].UTF8String << std::endl;
        } else {
            std::cout << "Unknown type" << std::endl;
        }
    }
}

ExchangePackage SwiftCPP_Bridge(NS::View *metalView) {
    NSView *SwiftCompatible = static_cast<NSView*>(metalView);
    MySwiftUIViewController *swiftUIViewController = [[MySwiftUIViewController alloc] initWithView:SwiftCompatible];
//    NSView *swiftUIViewController = [MySwiftUIViewControllerBridge createSwiftUIViewControllerWithView:SwiftCompatible];
//    swiftUIViewController.objcData;
    
    NS::View *customView = (__bridge NS::View *)swiftUIViewController.view;
    
    NSArray *data = swiftUIViewController.objcData;
    Object3D* OBJECTS = new Object3D[data.count];
    processArray(data, OBJECTS);
    NSString *nsString = stdStringToNSString(OBJECTS[0].id);
    NSLog(@"%@", nsString);
    ExchangePackage package = { OBJECTS, customView };
    printNSArray(data);
    return package;
}

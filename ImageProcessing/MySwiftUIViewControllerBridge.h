//
//  MySwiftUIViewControllerBridge.h
//  WorldOf3D
//
//  Created by Manoj Kumar on 20/08/24.
//

#ifndef MySwiftUIViewControllerBridge_h
#define MySwiftUIViewControllerBridge_h

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

@class MySwiftUIViewController;

@interface MySwiftUIViewControllerBridge : NSObject

// Method to create an instance of MySwiftUIViewController
+ (NSView *)createSwiftUIViewControllerWithView:(NSView *)view;

@end

#endif /* MySwiftUIViewControllerBridge.h */

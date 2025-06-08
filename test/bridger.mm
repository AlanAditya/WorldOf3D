//
//  bridger.m
//  test
//
//  Created by Manoj Kumar on 11/01/25.
//

#import <Foundation/Foundation.h>
#import "AppKit/AppKit.h"
#import <Cocoa/Cocoa.h>
#include "AppKit/AppKitPrivate.hpp"
#include <Foundation/NSObject.hpp>
#include <CoreGraphics/CGGeometry.h>
#import <MetalKit/MetalKit.hpp>
#include "Header.hpp"

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

void RunApp() {
    sleep(1);
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            NSApplication *app = [NSApplication sharedApplication];
            NSRect frame = NSMakeRect(100, 100, 800, 600);
            NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                           styleMask:(NSWindowStyleMaskTitled |
                                                                      NSWindowStyleMaskClosable |
                                                                      NSWindowStyleMaskResizable)
                                                             backing:NSBackingStoreBuffered
                                                               defer:NO];
            [window setTitle:@"Hello, Cocoa!"];
            [window makeKeyAndOrderFront:app];
            [app setActivationPolicy:NSApplicationActivationPolicyRegular];
            // Run the event loop
            [app run];
        }
    });
}

#import <thread>

@interface ImageWindowController : NSObject
@property (strong) NSWindow *window;
+ (instancetype)sharedInstance;
- (void)showImage:(NSString *)imagePath;
@end

@implementation ImageWindowController

+ (instancetype)sharedInstance {
    static ImageWindowController *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[ImageWindowController alloc] init];
    });
    return sharedInstance;
}

- (void)showImage:(NSString *)imagePath {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!self.window) {
            NSRect frame = NSMakeRect(100, 100, 800, 600);
            self.window = [[NSWindow alloc] initWithContentRect:frame
                                                      styleMask:(NSWindowStyleMaskTitled |
                                                                 NSWindowStyleMaskClosable |
                                                                 NSWindowStyleMaskResizable)
                                                        backing:NSBackingStoreBuffered
                                                          defer:NO];
            [self.window setTitle:@"Image Viewer"];
        }

        NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePath];
        if (!image) {
            NSLog(@"Failed to load image: %@", imagePath);
            return;
        }

        NSImageView *imageView = [[NSImageView alloc] initWithFrame:self.window.contentView.bounds];
        [imageView setImage:image];
        [imageView setImageScaling:NSImageScaleProportionallyUpOrDown];
        [self.window setContentView:imageView];

        [self.window makeKeyAndOrderFront:nil];
    });
}

@end

void imshow() {
    static bool appInitialized = false;
    char* imagePath = "/Users/adityadude/Desktop/output1.png";
    if (!appInitialized) {
        appInitialized = true;
        std::thread([] {
            @autoreleasepool {
                NSApplication *app = [NSApplication sharedApplication];
                [app setActivationPolicy:NSApplicationActivationPolicyRegular];
                [NSApp activateIgnoringOtherApps:YES];
                [app run];  // Runs event loop in background thread
            }
        }).detach();
        
        // Allow time for NSApplication to initialize before showing image
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    [[ImageWindowController sharedInstance] showImage:[NSString stringWithUTF8String:imagePath]];
}

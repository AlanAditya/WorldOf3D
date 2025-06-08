//
//  main.m
//  Video Editor
//
//  Created by Manoj Kumar on 26/01/25.
//

#include <iostream>
//extern "C" {
//#include <libavformat/avformat.h>
//#include <libavcodec/avcodec.h>
//#include <libswscale/swscale.h>
//}

#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import "Header.h"
#import <Video_Editor-Swift.h>

@interface SimpleView : NSView
@end



int main(int argc, const char * argv[]) {
    
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        CGRect frame = CGRectMake(100, 100, 1000, 1000);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame styleMask:NSWindowStyleMaskClosable | NSWindowStyleMaskResizable | NSWindowStyleMaskTitled backing:NSBackingStoreBuffered defer:false];
        [window setTitle:@"TUSHHHHUI"];
        [window makeKeyAndOrderFront:nil];
        
//        NSView* view = [[SimpleView alloc] initWithFrame:frame];
//        [window setContentView:view];
        
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MTKView* MetalView = [[MTKView alloc] initWithFrame:frame device:device];
        
        [MetalView setColorPixelFormat: MTLPixelFormatBGRA8Unorm_sRGB];
        [MetalView setClearColor:MTLClearColorMake(1.0, 1.0, 1.0, 1.0)];
        [MetalView setClearDepth:1.0f];
        [MetalView setDepthStencilPixelFormat:MTLPixelFormatDepth16Unorm];
        
        MetalViewDelegate* Mydelegate = [MetalViewDelegate new];
        Mydelegate->CommandQueue = [device newCommandQueue];
        [MetalView setDelegate: Mydelegate];
        [window setContentView:MetalView];
        EditorViewController *viewController = [[EditorViewController alloc] init];
        MySwiftUIViewController *MySVC = [[MySwiftUIViewController alloc] init];
        MainViewWrapper *main = [[MainViewWrapper alloc] init];
        [window setContentView: [main getView]];
//        window.contentViewController = MySVC;
//        window.contentViewController = viewController;
        
        [app run];
        // Setup code that might create autoreleased objects goes here.
    }
    return NSApplicationMain(argc, argv);
}

@implementation SimpleView

-(instancetype) initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    return self;
}

-(void) drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    CGContextRef context = NSGraphicsContext.currentContext.CGContext;
    
    auto Rect1 = CGRectMake(0, 0, 100, 100);
    CGContextSetRGBFillColor(context, 1.0, 0.0, 0.0, 1.0);
    CGContextFillRect(context, Rect1);
    
    auto Rect2 = CGRectMake(25, 25, 50, 50);
    CGContextSetRGBFillColor(context, 0.0, 0.0, 1.0, 1.0);
    CGContextFillRect(context, Rect2);
}

@end

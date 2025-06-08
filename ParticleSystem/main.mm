//
//  main.m
//  ParticleSystem
//
//  Created by Manoj Kumar on 11/02/25.
//

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import "ViewController.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        CGRect Frame = CGRectMake(100, 100, 600, 600);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:Frame styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable  backing:NSBackingStoreBuffered defer:NO];
        [window setTitle:@"Particle System"];
        [window makeKeyAndOrderFront:nil];
        
        id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
        MTKView* view = [[MTKView alloc] initWithFrame:Frame device:metalDevice];
        [view setColorPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
        [view setClearColor:MTLClearColorMake(0.0, 0.0, 0.0, 1.0)];
        [view setClearDepth:1.0];
        [view setPreferredFramesPerSecond:120];
        [view setFramebufferOnly: false];
        [view setDepthStencilPixelFormat:MTLPixelFormatDepth16Unorm];
        
        MetalViewController *viewController = [[MetalViewController alloc] initWithDevice:metalDevice origin:window.frame.origin];
        viewController->CommandQueue = [metalDevice newCommandQueue];
        [view setDelegate:viewController];
//        [view setNeedsDisplay:false];
        
        [window setContentView:view];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        [app run];
    }
    return NSApplicationMain(argc, argv);
}


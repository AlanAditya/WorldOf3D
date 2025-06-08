//
//  GUI_Extension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 20/02/25.
//

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import "AlgebroHeap.hpp"
#import <AppKit/AppKit.h>
#include "MetalLayerHPPextension.hpp"

static NSApplication *application = nil;
static NSMutableDictionary *windows = nil;
static bool wasInitialized = false;

@interface DraggableImageView : NSImageView
// Define a function pointer property that takes an NSPoint as the drag value.
//@property (nonatomic, assign) void (*dragCallback)(simd_float2 dragPoint);
@property (nonatomic, assign) NSPoint lastDrag;
@property (nonatomic, assign) std::function<void(simd_float2)> dragCallback;
@property (nonatomic, assign) std::function<void(simd_float2)> scrollCallback;
@property (nonatomic, assign) std::function<void(simd_float2)> magnifyCallback;
@end

@implementation DraggableImageView



- (void)handlePan:(NSPanGestureRecognizer *)gesture {
    NSPoint translation = [gesture translationInView:self];
    if (self.dragCallback) {
        self.dragCallback(simd_make_float2((translation.x - _lastDrag.x) / self.frame.size.width, (translation.y - _lastDrag.y) / self.frame.size.height) * 2);
        _lastDrag = translation;
    }
    if ([gesture state] == NSGestureRecognizerStateEnded) {
        _lastDrag = NSMakePoint(0, 0);
    }
    // Use translation or gesture location to update your UI
}

- (void)handleZoom:(NSMagnificationGestureRecognizer *)gesture {
    CGFloat translation = gesture.magnification;
    if (self.magnifyCallback) {
        self.magnifyCallback(simd_make_float2((translation + 1.0) , (translation + 1.0) ) );
    }
    // Use translation or gesture location to update your UI
}

- (void)mouseDown:(NSEvent *)theEvent {
    printf("Mouse Down I repeat mouse down \n");
}

- (void)scrollWheel:(NSEvent *)scroll {
    if (self.scrollCallback) {
        self.scrollCallback(simd_make_float2( ((scroll.scrollingDeltaX) / self.frame.size.width), ((scroll.scrollingDeltaY) / self.frame.size.height)));
    }
}

@end



int initSystem(int , char **) {
    wasInitialized = true;
    
    application = [NSApplication sharedApplication];
    windows = [[NSMutableDictionary alloc] init];
    [application setActivationPolicy:NSApplicationActivationPolicyRegular];
    return 0;
}

static NSWindow* getWindow(const char *name) {
    NSWindow* retval = nil;
    @autoreleasepool {
        NSString* winName = [NSString stringWithFormat:@"%s", name];
        retval = [windows valueForKey:winName];
    }
    return retval;
}

template <typename Type>
void MatrixH<Type>::AIDragCallback( const char* name, void (*cCallback)(simd_float2)) const
{
    
    NSWindow *window = nil;
    if(name == NULL)
        std::cerr << "NULL window name" << "\n";
    @autoreleasepool{
        window = getWindow(name);
        if(window) {
            [(DraggableImageView *)[window contentView] setDragCallback:cCallback];

        }
    }
}

template void MatrixH<uint8_t>::AIDragCallback( const char* name, void (*cCallback)(simd_float2)) const;

int startWindowThread() {
    return 0;
}

void destroyWindow(const char* name) {
    @autoreleasepool {
        NSWindow* window = getWindow(name);
        if (window) {
            if ([window styleMask] & NSWindowStyleMaskFullScreen) {
                [window toggleFullScreen:nil];
            }
            [window close];
            [windows removeObjectForKey:[NSString stringWithFormat:@"%s", name]];
        }
    }
}

void destroyAllWindows() {
    @autoreleasepool {
        NSDictionary* list = [NSDictionary dictionaryWithDictionary:windows];
        for (NSString* key in list) {
            destroyWindow([key cStringUsingEncoding:NSASCIIStringEncoding]);
        }
    }
}

template <typename Type>
int MatrixH<Type>::NamedWindow(const char* name, int flags) const {
    if (!wasInitialized) {
        initSystem(0, 0);
    }
    @autoreleasepool {
        sleep(1);
        NSWindow* window = getWindow(name);
        if (window) {
            return 0;
            
        }
        
        NSScreen* screen = [NSScreen mainScreen];
        NSString* windowName = [NSString stringWithFormat:@"%s", name];
        
        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable;
        
        CGFloat windowWidth = [NSWindow minFrameWidthWithTitle:windowName styleMask:style];
        NSRect contentRect = NSMakeRect(100, 100, 600,600);
        if (screen) {
            NSRect displayScreen = [screen visibleFrame];
            contentRect.origin.y = displayScreen.size.height - 20;
        }
        
        window = [[NSWindow alloc] initWithContentRect:contentRect styleMask:style backing:NSBackingStoreBuffered defer:YES screen:screen];
        [window setFrameTopLeftPoint:contentRect.origin];
        [window setContentView: [[DraggableImageView alloc] init]];
        [application activate];
        
        [window makeKeyAndOrderFront:nil];
        [application activateIgnoringOtherApps:YES];
        [application setPresentationOptions:NSApplicationPresentationDefault];
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        [window setHasShadow: true];
        [window setAcceptsMouseMovedEvents:YES];
        [window setTitle:windowName];
        [windows setValue:window forKey:windowName];
        
        
        return (int)[windows count] - 1;
    }
}

template int MatrixH<uint8_t>::NamedWindow(const char* name, int flags) const;

template <typename Type>
void MatrixH<Type>::showImage(char* name) const {
    if (!values) {
        return;
    }
    @autoreleasepool {
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        if (!colorSpace) {
            fprintf(stderr, "Failed to create color space!\n");
        }
        CGContextRef context = CGBitmapContextCreate(
            values,                     // Data buffer (NULL for CoreGraphics to allocate)
                                                     shape[1],                    // Width
                                                     shape[0],                   // Height
            8,                        // Bits per component
                                                     shape[1] * 4,                // Bytes per row (RGBA, 4 bytes per pixel)
            colorSpace,               // Color space
            kCGImageAlphaPremultipliedLast // Bitmap info
        );

        if (!context) {
            fprintf(stderr, "Failed to create bitmap context!\n");
            CGColorSpaceRelease(colorSpace);
        }
        
        CGImageRef image = CGBitmapContextCreateImage(context);
        if (!image) {
            fprintf(stderr, "Failed to create image from context!\n");
            CGContextRelease(context);
            CGColorSpaceRelease(colorSpace);
        }
        NSWindow* window = getWindow(name);

        if (!window) {
            NamedWindow(name, 0);
            window = getWindow(name);
            NSImage *img = [[NSImage alloc] initWithCGImage:image size:NSMakeSize(shape[1], shape[0])];
            [(DraggableImageView *)[window contentView] setImage: img];
        }
        if (window) {
            NSImage *img = [[NSImage alloc] initWithCGImage:image size:NSMakeSize(shape[1], shape[0])];
            [(DraggableImageView *)[window contentView] setImage: img];
        }
        CGContextRelease(context);
        CGImageRelease(image);
    }
}

template void MatrixH<uint8_t>::showImage(char* name) const;

template <typename Type>
int MatrixH<Type>::cvWaitKey(int maxWait) const
{
    int returnCode = -1;
    @autoreleasepool{
        double start = [[NSDate date] timeIntervalSince1970];

        while(true) {
//            if(([[NSDate date] timeIntervalSince1970] - start) * 1000 >= maxWait && maxWait>0)
//                break;
            NSEvent *event = [application nextEventMatchingMask:NSEventMaskAny
            untilDate://[NSDate dateWithTimeIntervalSinceNow: 1./100]
                              NULL
            inMode:NSDefaultRunLoopMode
            dequeue:YES];

            if([event type] == NSEventTypeKeyDown && [[event characters] length]) {
                returnCode = [[event characters] characterAtIndex:0];
                break;
            }

            [application sendEvent:event];
            [application updateWindows];

            [NSThread sleepForTimeInterval:1.0f/1000];
            break;
        }
        return returnCode;
    }
}

template int MatrixH<uint8_t>::cvWaitKey(int maxWait) const;


template <typename Type>
int MatrixH<Type>::test() {
    sleep(1);

    NSApplication* app = [NSApplication sharedApplication];
    
    wasInitialized = true;
    windows = [[NSMutableDictionary alloc] init];
    
    NSRect frame = NSMakeRect(100, 100, 1920 , 1080 );
    NSWindow* window = [[NSWindow alloc] initWithContentRect:frame styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable backing:NSBackingStoreBuffered defer:false];
    [window setTitle:@"Hello Cocoa"];
    [window makeKeyAndOrderFront:nil];
    
    auto prender = RendererHPP();
    auto tt = Cube(.1);
    prender.objectQueue.push_back(tt);
    
    tt.instanceCount = 125;
    simd_float4x4* grid = new simd_float4x4[125];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                grid[i * 25 + j * 5 + k] = Translation(simd_make_float3((i-2)*0.25f, (j-2)*0.25f,  k * 0.25f));
                std::cout << grid[i * 16 + j * 4 + k] << "\n";
            }
        }
    }
    auto cubeArr = ArrayShape(tt, grid);
    prender.objectQueueArray.push_back(cubeArr);
    
    
    std::cout << grid[0] << "\n";
    
    
    auto viewImg =[ DraggableImageView new];
    auto *panGesture = [[NSPanGestureRecognizer alloc] initWithTarget:viewImg action:@selector(handlePan:)];
    auto *pinchGesture = [[NSMagnificationGestureRecognizer alloc] initWithTarget:viewImg action:@selector(handleZoom:)];
    [viewImg addGestureRecognizer:pinchGesture];
    [viewImg addGestureRecognizer:panGesture];
    
    auto dragCaller = [&prender](simd_float2 pos) {
        prender.camera.updatePosition(pos, TransformationMode::Translate);
        std::cout << "Dragged to: " << pos << "\n";
    };
    
    auto scrollCaller = [&prender](simd_float2 pos) {
        prender.camera.updatePosition(pos, TransformationMode::Orbit);
        std::cout << "Scroll to: " << pos << "\n";
        
    };
    
    auto pinchCaller = [&prender](simd_float2 pos) {
        prender.camera.updatePosition(pos, TransformationMode::Zoom);
        std::cout << "ZOmom to: " << pos << "\n";
    };
    [viewImg setDragCallback: dragCaller];
    [viewImg setScrollCallback:scrollCaller];
    [viewImg setMagnifyCallback:pinchCaller];
    [window setContentView:  viewImg];
    
    
    [windows setValue:window forKey:@"Hello Cocoa"];
    
    [app setPresentationOptions:NSApplicationPresentationDefault];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    
    NSTimeInterval lastDrawTime = 0;
    NSTimeInterval targetFrameTime = 1.0/30.0;
    
    auto cap = CamReaderWrapper(0);
    
    MatrixH<int> detections;

    
    MatrixH<uint8> frameCam;
    MatrixH<uint8> frameRender;
    prender.clear = true;
    
    bool mRunning= true;
    while (mRunning) {
        @autoreleasepool {
            NSEvent* event = NULL;

            
            event = [app nextEventMatchingMask:NSEventMaskAny untilDate:[NSDate dateWithTimeIntervalSinceNow:0.01] inMode: NSDefaultRunLoopMode dequeue:true];
            if (event.type == NSEventTypeLeftMouseDragged) {
                NSLog(@"%lu", event.type);
            }
            BOOL shiftPressed = ([event modifierFlags] & NSEventModifierFlagShift) != 0;
            if (shiftPressed && [event keyCode] == 8) { // 'C' key has keyCode 8 on macOS
                prender.metalCapture = true;
            }
            
            [app sendEvent:event];
            
            NSTimeInterval currentTime = [NSDate timeIntervalSinceReferenceDate];
            if (currentTime - lastDrawTime >= targetFrameTime) {
                cap.retrieveFrame(frameCam);
                
                frameCam.flip(1);
                frameCam.HandsDetect(detections, false);
                frameCam.cvtColor_BGRA2RGBA();
                if (detections.shape[0] > 0) {
//                    frameCam.drawHands(detections, false);
                    float lenth = sqrt((*detections(0, 0, 0) - *detections(0, 1, 0))*(*detections(0, 0, 0) - *detections(0, 1, 0)) + (*detections(0, 0, 1) - *detections(0, 1, 1))*(*detections(0, 0, 1) - *detections(0, 1, 1)));
                    prender.objectQueueArray[0].shape.rotation.y = lenth * 0.1;
                    prender.objectQueueArray[0].shape.rotation.x = lenth * 0.1;
//                    frameCam.drawLine(simd_make_int2(*detections(0, 0, 0), *detections(0, 0, 1)), simd_make_int2(*detections(0, 1, 0), *detections(0, 1, 1)), 0, {255, 0, 0, 255});
                }
                
                if (detections.shape[0] > 1) {
                    frameCam.drawHands(detections, false);
                    float lenth = ((*detections(1, 0, 0) - *detections(1, 1, 0)) * (*detections(1, 0, 0) - *detections(1, 1, 0)) + (*detections(1, 0, 1) - *detections(1, 1, 1)) * (*detections(1, 0, 1) - *detections(1, 1, 1)));
                    
                    prender.camera.setPosition(simd_make_float2(lenth * 0.001, lenth * 0.001), TransformationMode::Orbit);

//                    frameCam.drawLine(simd_make_int2(*detections(0, 0, 0), *detections(0, 0, 1)), simd_make_int2(*detections(0, 1, 0), *detections(0, 1, 1)), 0, {255, 0, 0, 255});
                }
                
                prender.Render(frameRender);
                frameRender.showImage("Hello Cocoa");
                frameCam.showImage("Hello Cam");
                
                [app updateWindows];
                lastDrawTime = currentTime;
            }
            
            
//            [NSThread sleepForTimeInterval:(1.0f / 200)];
        }
    }
    return 0;
}

template int MatrixH<uint8_t>::test();

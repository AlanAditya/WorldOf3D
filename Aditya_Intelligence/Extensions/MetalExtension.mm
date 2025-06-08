//
//  MetalExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 08/02/25.
//

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#include "Algebro.hpp"
#import "AlgebroHeap.hpp"




template <typename T>
void MatrixH<T>::imshow() {
    const char* imagePath = "/Users/adityadude/Desktop/output1.png";
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
    
    NSLog(@"Run Window");
    sleep(1);
//    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            NSLog(@"Run Window");
            
            NSApplication *app = [NSApplication sharedApplication];
            
            // Create the window
            NSRect frame = NSMakeRect(100, 100, 800, 600);
            NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                           styleMask:(NSWindowStyleMaskTitled |
                                                                      NSWindowStyleMaskClosable |
                                                                      NSWindowStyleMaskResizable)
                                                             backing:NSBackingStoreBuffered
                                                               defer:NO];
            [window setTitle:@"TUSHHHHUI"];
            
            // Load the image
//            NSString *imagePathString = [NSString stringWithUTF8String:imagePath];
//            NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePathString];
//            if (!image) {
//                NSLog(@"Failed to load image: %s", imagePath);
//                return;
//            }
            NSImage *img = [[NSImage alloc] initWithCGImage:image size:NSMakeSize(shape[1], shape[0])];
            // Create an NSImageView to display the image
            NSImageView *imageView = [[NSImageView alloc] initWithFrame:frame];
            [imageView setImage:img];
            [imageView setImageScaling:NSImageScaleProportionallyUpOrDown];
            
            // Set the view inside the window
            [window setContentView:imageView];
            [window makeKeyAndOrderFront:nil];
            [app activateIgnoringOtherApps:YES];
            [app setActivationPolicy:NSApplicationActivationPolicyRegular];
            [app run];
        }
}



template void MatrixH<uint8_t>::imshow();

#import <AVFoundation/AVFoundation.h>

CGImageRef imageFromAVCapturePhoto(AVCapturePhoto *photo) {
    NSData *jpegData = [photo fileDataRepresentation];
    if (!jpegData) return NULL;

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)jpegData);
    CGImageRef imageRef = CGImageCreateWithJPEGDataProvider(provider, NULL, true, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);

    return imageRef;
}

static uint8_t *imageBuffer = NULL;
static size_t imageWidth = 0, imageHeight = 0, imageBytesPerRow = 0;
static BOOL photoCaptured = NO;

// ✅ Function to Capture an Image
uint8_t* capturePhoto(size_t *width, size_t *height, size_t *bytesPerRow) {
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPresetPhoto;

    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (!device) {
        NSLog(@"No camera found!");
        return NULL;
    }

    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) {
        NSLog(@"Error creating camera input: %@", error.localizedDescription);
        return NULL;
    }

    AVCapturePhotoOutput *photoOutput = [[AVCapturePhotoOutput alloc] init];

    if ([session canAddInput:input]) {
        NSLog(@"input Added");
        [session addInput:input];
    }
    if ([session canAddOutput:photoOutput]) {
        NSLog(@"output Added");
        [session addOutput:photoOutput];
    }

    [session startRunning];

    // ✅ Capture a Photo
    AVCapturePhotoSettings *photoSettings = [AVCapturePhotoSettings photoSettings];
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    [photoOutput capturePhotoWithSettings:photoSettings delegate:(id<AVCapturePhotoCaptureDelegate>)[^(
        AVCapturePhotoOutput *output, AVCapturePhoto *photo, NSError *error) {
        
        if (error) {
            NSLog(@"Error capturing photo: %@", error.localizedDescription);
            dispatch_semaphore_signal(semaphore);
            return;
        }

        // ✅ Convert photo to raw pixel buffer
        CGImageRef imageRef = imageFromAVCapturePhoto(photo);
        imageWidth = CGImageGetWidth(imageRef);
        imageHeight = CGImageGetHeight(imageRef);
        imageBytesPerRow = imageWidth * 4;
        size_t bufferSize = imageBytesPerRow * imageHeight;

        if (!imageBuffer) {
            imageBuffer = (uint8_t *)malloc(bufferSize);
        }

        CGContextRef context = CGBitmapContextCreate(
            imageBuffer,
            imageWidth,
            imageHeight,
            8,
            imageBytesPerRow,
            CGImageGetColorSpace(imageRef),
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );

        CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), imageRef);
        CGContextRelease(context);
        CGImageRelease(imageRef);

        photoCaptured = YES;
        dispatch_semaphore_signal(semaphore);
    } copy]];

    // ✅ Wait until the photo is captured
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

    [session stopRunning];

    *width = imageWidth;
    *height = imageHeight;
    *bytesPerRow = imageBytesPerRow;
    return imageBuffer;
}


//template <typename T>
//MatrixH<uint8_t> MatrixH<T>::fromCam() {
//    auto result = MatrixH<uint8_t>();
//    sleep(2);
//    result.values = capturePhoto(&imageWidth, &imageHeight, &imageBytesPerRow);
//    result.shape = std::vector<size_t>({imageWidth, imageHeight, 4});
//    result.total_size = imageWidth * imageHeight;
//    result.print();
//    return result;
//    
//}
//
//template MatrixH<uint8_t> MatrixH<uint8_t>::fromCam();

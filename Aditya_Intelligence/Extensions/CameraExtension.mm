//
//  CameraExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 14/02/25.
//

#import <Foundation/Foundation.h>
#include "Algebro.hpp"
#import <AVFoundation/AVFoundation.h>
#import "AlgebroHeap.hpp"


@interface CaptureDelegateV1 : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    NSCondition* mHasNewFrame;
    CVPixelBufferRef mGrabbedPixels;
    CVImageBufferRef mCurrentImageBuffer;
    size_t currentSize;
}

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection;
-(BOOL) grabImageUntilDate: (NSDate *)limit;
-(BOOL) updateImage;
-(void) getImage: (uint8_t*)buffer;

@end

@implementation CaptureDelegateV1

- (id)init {
    self = [super init];
    mHasNewFrame = [[NSCondition alloc] init];
    mGrabbedPixels = NULL;
    mCurrentImageBuffer = NULL;
    currentSize = 0;
    return self;
}

-(void) dealloc {
    CVBufferRelease(mGrabbedPixels);
    CVBufferRelease(mCurrentImageBuffer);
}

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    (void)output;
    (void)sampleBuffer;
    (void)connection;
    
    
    CVImageBufferRef image = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVBufferRetain(image);
    NSLog(@"output Captures");

    [mHasNewFrame lock];
    
    CVBufferRelease(mCurrentImageBuffer);
    mCurrentImageBuffer = image;
    
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mCurrentImageBuffer);
    NSLog(@"Pixel Format: 0x%X", pixelFormat);
    
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(mCurrentImageBuffer);
    NSLog(@"Bytes Per Row: %zu", bytesPerRow);
    
    
    CVPixelBufferLockBaseAddress(mCurrentImageBuffer, kCVPixelBufferLock_ReadOnly);
    for (int i = 0; i< 160; i++) {
        std::cout << (int)(reinterpret_cast<uint8*>(CVPixelBufferGetBaseAddress(mCurrentImageBuffer)))[i] << "\n";
    }
    
    CVPixelBufferUnlockBaseAddress(mCurrentImageBuffer, kCVPixelBufferLock_ReadOnly);

    [mHasNewFrame signal];
    [mHasNewFrame unlock];
}

-(BOOL) grabImageUntilDate:(NSDate *)limit {
    BOOL isGrabbed = false;
    [mHasNewFrame lock];
    
    if (mGrabbedPixels) {
        CVBufferRelease(mGrabbedPixels);
    }
    
    if ([mHasNewFrame waitUntilDate:limit]) {
        isGrabbed = true;
        mGrabbedPixels = CVBufferRetain(mCurrentImageBuffer);
    }
    
    [mHasNewFrame unlock];
    
    return isGrabbed;
}

-(BOOL) updateImage: (uint8_t*) buffer {
    if ( ! mGrabbedPixels ) {
        return false;
    }
    
    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uint8_t *baseaddress = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    size_t height = CVPixelBufferGetHeight(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);
    NSLog(@"Reached");
    
    bool res = false;
    
    if (rowBytes != 0) {
        size_t dataSize = rowBytes * height;  // Calculate total bytes required
//        for (int i = 0; i< 16; i++) {
//            std::cout << (int)baseaddress[i] << "\n";
//        }
        memcpy(buffer, baseaddress, dataSize);
        res = YES;

    } else {
        fprintf(stderr, "OpenCV: rowBytes == 0 or unknown pixel format 0x%08X\n", pixelFormat);
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    mGrabbedPixels = NULL;

    return res;
}

-(void) getImage: (uint8_t*)buffer {
    
}


@end




void cameraBufferCapture(MatrixH<uint8_t> &mat, int cameraNum, IMGFormat format) {
    @autoreleasepool {
        AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if (status == AVAuthorizationStatusDenied)
        {
            fprintf(stderr, "MATRIX: camera access has been denied. ...\n");
        }
        else if (status != AVAuthorizationStatusAuthorized) {
            fprintf(stderr, "MATRIX: not authorized to capture video (status %ld), requesting...\n", status);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL) {}];
            if ([NSThread isMainThread]) {
                [[NSRunLoop mainRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
            } else {
                fprintf(stderr, "MATRIX: can not spin main run loop from other thread, set ");
            }
        }
        
        
        AVCaptureDeviceDiscoverySession *session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[ AVCaptureDeviceTypeExternal, AVCaptureDeviceTypeBuiltInWideAngleCamera ] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
        
        
        NSArray *devices = session.devices;
        for (int i = 0; i< devices.count; i++) {
            NSLog(@"%@", ((AVCaptureDevice*)(devices[i])).description);
            if ([((AVCaptureDevice*)(devices[i])).activeFormat.supportedColorSpaces containsObject:@(AVCaptureColorSpace_sRGB)]) {
                NSLog(@"This camera supports sRGB color space, so background effects should work.");
            } else {
                NSLog(@"Warning: Background effects may not be available.");
            }
        }
        
        if ( devices.count == 0 ) {
            fprintf(stderr, "MATRIX: AVFoundation didn't find any attached Video Input Devices!\n");
        }
        
        if (cameraNum < 0 || cameraNum > devices.count) {
            fprintf(stderr, "MATRIX: out device of bound (0-%ld): %d\n", devices.count-1, cameraNum);
        }
        
        AVCaptureDevice* mCaptureDevice = devices[cameraNum];
        if ( ! mCaptureDevice ) {
            fprintf(stderr, "MATRIX: device %d not able to use.\n", cameraNum);
        }
        
        NSError *error = nil;
        AVCaptureDeviceInput* mCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:mCaptureDevice error:&error];
        if ( error ) {
            fprintf(stderr, "MATRIX: error in [AVCaptureDeviceInput initWithDevice:error:]\n");
            NSLog(@"OpenCV: %@", error.localizedDescription);
        }
        
        CaptureDelegateV1* mCapture = [[CaptureDelegateV1 alloc] init];
        AVCaptureVideoDataOutput* mCaptureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
        
        dispatch_queue_t queue = dispatch_queue_create("cameraQueue", DISPATCH_QUEUE_SERIAL);
        [mCaptureVideoDataOutput setSampleBufferDelegate: mCapture queue: queue];
        
        NSMutableDictionary *pixelBufferOptionsL = [mCaptureVideoDataOutput.videoSettings mutableCopy];
        NSLog(@"%@", pixelBufferOptionsL);
        
        for (NSString *key in pixelBufferOptionsL) {
            NSLog(@"%@: %@", key, pixelBufferOptionsL[key]);
        }
        
        int width = 0;
        int height = 0;
        while ( true ) {
            // auto matching
            pixelBufferOptionsL[(id)kCVPixelBufferWidthKey]  = @(1.0*width);
            pixelBufferOptionsL[(id)kCVPixelBufferHeightKey] = @(1.0*height);
            mCaptureVideoDataOutput.videoSettings = pixelBufferOptionsL;

            // compare matched size and my options
            CMFormatDescriptionRef format = mCaptureDevice.activeFormat.formatDescription;
            CMVideoDimensions deviceSize = CMVideoFormatDescriptionGetDimensions(format);
            if ( deviceSize.width == width && deviceSize.height == height ) {
                break;
            }

            // fit my options to matched size
            width = deviceSize.width;
            height = deviceSize.height;
        }
        
//        std::cout << width << height << "\n";
        mat.values  = new uint8_t[width * height * 4];
        mat.shape = std::vector<size_t>({(size_t)height, (size_t)width, 4});
        mat.total_size = width * height * 4;
        OSType pixelFormat;
        switch (format) {
            case IMGFormat::RGBA:
                pixelFormat = kCVPixelFormatType_32ARGB;
                break;
                
            case IMGFormat::BGRA:
                pixelFormat = kCVPixelFormatType_32BGRA;
                break;
        }
        
            //OSType pixelFormat = kCVPixelFormatType_422YpCbCr8;
        NSDictionary *pixelBufferOptions;
        if (width > 0 && height > 0) {
            pixelBufferOptions =
                @{
                    (id)kCVPixelBufferWidthKey:  @(1.0*width),
                    (id)kCVPixelBufferHeightKey: @(1.0*height),
                    (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
                };
        } else {
            pixelBufferOptions =
                @{
                    (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
                };
        }
        mCaptureVideoDataOutput.videoSettings = pixelBufferOptions;
        mCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = YES;
    
        // create session
        AVCaptureSession* mCaptureSession = [[AVCaptureSession alloc] init];
        mCaptureSession.sessionPreset = AVCaptureSessionPresetMedium;
        [mCaptureSession addInput: mCaptureDeviceInput];
        [mCaptureSession addOutput: mCaptureVideoDataOutput];
        
        NSArray *supportedFormats = [mCaptureVideoDataOutput.availableVideoCVPixelFormatTypes copy];
        NSLog(@"Supported pixel formats: %@", supportedFormats);
        
        [mCaptureSession startRunning];
        [NSThread sleepForTimeInterval:0.5];
        NSDate *limit = [NSDate dateWithTimeIntervalSinceNow: 1];
        if ( [mCapture grabImageUntilDate: limit] ) {
            bool value = [mCapture updateImage:mat.values];
            std::cout << "stat" << value << "\n";
        }
    }
}

template <typename T>
MatrixH<uint8_t> MatrixH<T>::fromCam(int camNo, IMGFormat format) {
    auto result = MatrixH<uint8_t>();
    cameraBufferCapture(result, camNo, format);
    return result;
}

template MatrixH<uint8_t> MatrixH<uint8_t>::fromCam(int camNo, IMGFormat format);

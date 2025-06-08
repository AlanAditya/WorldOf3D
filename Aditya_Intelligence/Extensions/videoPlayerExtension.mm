//
//  videoPlayerExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 19/02/25.
//

#import <Foundation/Foundation.h>
#include "AlgebroHeap.hpp"
#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>




CVPixelBufferRef createPixelBuffer(uint8_t *frameData, size_t width, size_t height, size_t channels) {
    NSDictionary *pixelAttributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
    
    CVPixelBufferRef pixelBuffer;
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)pixelAttributes, &pixelBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    void *bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);

    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
     
     // Assuming channels is the number of bytes per pixel (should be 4 for BGRA)
     size_t copyBytesPerRow = width * channels;
     
     // Copy row by row to respect the pixel buffer's stride.
     for (size_t row = 0; row < height; row++) {
         memcpy((uint8_t *)bufferAddress + row * bytesPerRow,
                frameData + row * copyBytesPerRow,
                copyBytesPerRow);
     }
    
//    memcpy(bufferAddress, frameData, width * height * channels); // Copy raw frame data

    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    return pixelBuffer;
}

CMSampleBufferRef createSampleBuffer(CVPixelBufferRef pixelBuffer, CMTime pts) {
    CMVideoFormatDescriptionRef videoInfo;
    CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, &videoInfo);

    CMSampleTimingInfo timing = { .duration = CMTimeMake(1, 30), .presentationTimeStamp = pts, .decodeTimeStamp = kCMTimeInvalid };
    CMSampleBufferRef sampleBuffer;
    CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixelBuffer, videoInfo, &timing, &sampleBuffer);

    CFRelease(videoInfo);
    return sampleBuffer;
}

@interface VideoView : NSView
@property (nonatomic, strong) AVSampleBufferDisplayLayer *videoLayer;
@end

@implementation VideoView
- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.wantsLayer = YES;
        
        // Initialize the AVSampleBufferDisplayLayer
        _videoLayer = [[AVSampleBufferDisplayLayer alloc] init];
        _videoLayer.videoGravity = AVLayerVideoGravityResizeAspect;
        [self.layer addSublayer:_videoLayer];
    }
    return self;
}

// Ensure the video layer resizes with the view
- (void)resizeSubviewsWithOldSize:(NSSize)oldSize {
    [super resizeSubviewsWithOldSize:oldSize];
    NSLog(@"called");
    self.subviews.firstObject.frame = self.bounds;
}
@end

template <typename T>
void MatrixH<T>::play() {
    sleep(1);
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
            
            
            // Set the view inside the window
            VideoView *videoView = [[VideoView alloc] initWithFrame:frame];
            
            // Create AVSampleBufferDisplayLayer
            AVSampleBufferDisplayLayer *videoLayer = [[AVSampleBufferDisplayLayer alloc] init];
            videoLayer.videoGravity = AVLayerVideoGravityResizeAspect;
            videoLayer.frame = videoView.bounds;
            videoView.wantsLayer = YES;
            [videoView.layer addSublayer:videoLayer];

            size_t frames = shape[0];
            size_t height = shape[1];
            size_t width = shape[2];
            size_t channels = shape[3];
            
            
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                for (int i = 0; i < frames; i++) {
                    uint8_t *frameData = values + (i * height * width * channels);

                    CVPixelBufferRef pixelBuffer = createPixelBuffer(frameData, width, height, channels);
                    if (!pixelBuffer) {std::cout << "errrrrrr"; continue;}

                    CMTime pts = CMTimeMake(i, 30); // 30 FPS
                    CMSampleBufferRef sampleBuffer = createSampleBuffer(pixelBuffer, pts);
                        if (!sampleBuffer) {std::cout << "errrrrrr"; continue;}
                    
                    if (sampleBuffer) {
                        CFRetain(sampleBuffer);
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [[videoLayer sampleBufferRenderer] enqueueSampleBuffer:sampleBuffer];
                        });
                    }

                    CFRelease(sampleBuffer);
                    CVPixelBufferRelease(pixelBuffer);
                    
                    usleep(33333); // Simulating 30 FPS (1/30 sec)
                }
            });
            


            [window setContentView:videoView];
            [window makeKeyAndOrderFront:nil];
            [app activateIgnoringOtherApps:YES];
            [app setActivationPolicy:NSApplicationActivationPolicyRegular];
            [app run];
        }
}


template void MatrixH<uint8_t>::play();

template<typename T>
dispatch_group_t MatrixH<T>::saveVideo(const char* outputPath) {
    size_t frames = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];
    size_t channels = shape[3];
    
    dispatch_group_t group = dispatch_group_create();
    @autoreleasepool {
        NSLog(@"Saving Video...");

        NSString *filePath = [NSString stringWithUTF8String:outputPath];
        NSURL *outputURL = [NSURL fileURLWithPath:filePath];
        
        NSFileManager *fileManager = [NSFileManager defaultManager];
        if ([fileManager fileExistsAtPath:filePath]) {
            NSError *removeError = nil;
            [fileManager removeItemAtPath:filePath error:&removeError];
            if (removeError) {
                NSLog(@"Failed to remove existing file: %@", removeError);
                return group;
            }
        }
        
        // Setup AVAssetWriter
        NSError *error = nil;
        AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:outputURL fileType:AVFileTypeQuickTimeMovie error:&error];

        if (error) {
            NSLog(@"Error creating writer: %@", error);
            return group;
        }

        NSDictionary *videoSettings = @{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height),
            AVVideoScalingModeKey: AVVideoScalingModeResize
        };

        AVAssetWriterInput *writerInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:videoSettings];
        writerInput.expectsMediaDataInRealTime = YES;
        writerInput.performsMultiPassEncodingIfSupported = NO;

        NSDictionary *bufferAttributes = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferWidthKey: @(width),
            (id)kCVPixelBufferHeightKey: @(height),
        };

        AVAssetWriterInputPixelBufferAdaptor *adaptor = [[AVAssetWriterInputPixelBufferAdaptor alloc] initWithAssetWriterInput:writerInput sourcePixelBufferAttributes:bufferAttributes];

        if ([writer canAddInput:writerInput]) {
            [writer addInput:writerInput];
        } else {
            NSLog(@"Failed to add input");
            return group;
        }

        [writer startWriting];
        [writer startSessionAtSourceTime:kCMTimeZero];

        dispatch_queue_t queue = dispatch_queue_create("video_encoding_queue", NULL);
        
        
        dispatch_group_enter(group);
        
        [writerInput requestMediaDataWhenReadyOnQueue:queue usingBlock:^{
            int frameIndex = 0;
            while (frameIndex < frames) {
                if ([writerInput isReadyForMoreMediaData]) {
                    uint8_t *frameData = values + (frameIndex * height * width * channels);
                    CVPixelBufferRef pixelBuffer = createPixelBuffer(frameData, width, height, channels);

                    if (pixelBuffer) {
                        CMTime pts = CMTimeMake(frameIndex, 30); // 30 FPS
                        [adaptor appendPixelBuffer:pixelBuffer withPresentationTime:pts];
                        CVPixelBufferRelease(pixelBuffer);
                    }

                    frameIndex++;
                }
            }

            [writerInput markAsFinished];
            [writer finishWritingWithCompletionHandler:^{
                dispatch_group_leave(group);
                NSLog(@"Video saved to %@", filePath);
            }];
        }];
//        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }
    return group;
}

template dispatch_group_t MatrixH<uint8_t>::saveVideo(const char* outputPath);

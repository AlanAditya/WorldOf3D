#pragma once
#import <AVFoundation/AVFoundation.h>
#import "../matrix.h"

// The shared protocol so you can easily swap between Microphone and Files later
@protocol AudioCapture <NSObject>
- (void)getAudioTensor:(matrix&)out_matrix samplesNeeded:(int)samplesNeeded;
@end


// The Microphone Capture Class
@interface MicrophoneAudioCapture : NSObject <AudioCapture, AVCaptureAudioDataOutputSampleBufferDelegate>

// AVFoundation Components
@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) dispatch_queue_t audioQueue;

- (instancetype)init;
- (void)start;
- (void)stop;

// Protocol Implementation
- (void)getAudioTensor:(matrix&)out_matrix samplesNeeded:(int)samplesNeeded;

@end

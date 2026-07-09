#import "microphone.h"

// =======================================================
// C++ Ring Buffer (Private to this file)
// =======================================================
constexpr size_t AUDIO_RING_BUFFER_SIZE = 44100;
struct LockFreeRingBuffer {
    float buffer[AUDIO_RING_BUFFER_SIZE];
    std::atomic<int> read_head;
    std::atomic<int> write_head;
    
    void push(float* incoming_data, int num_samples) {
        int current_write = write_head.load(std::memory_order_relaxed);
        int current_read = read_head.load(std::memory_order_acquire);
        int used = (current_write - current_read + AUDIO_RING_BUFFER_SIZE) % AUDIO_RING_BUFFER_SIZE;
        int space = (AUDIO_RING_BUFFER_SIZE - (used + 1));
        if (num_samples > space) {
            NSLog(@"[AUDIO WARN] Ring Buffer Overflow! Dropping %d samples.", num_samples);
            return;
        }
        
        size_t first_sweep = AUDIO_RING_BUFFER_SIZE - current_write;
        size_t second_sweep = 0;
        if (first_sweep > num_samples) {first_sweep = num_samples;}
        else { second_sweep = num_samples - first_sweep; }
        
        memcpy(buffer + current_write, incoming_data, first_sweep * sizeof(float));
        memcpy(buffer, incoming_data + first_sweep, second_sweep * sizeof(float));
        write_head.store((current_write+num_samples) % AUDIO_RING_BUFFER_SIZE, std::memory_order_release);
    }
    
    void pop(float* out_data, int requested_samples) {
        int current_write = write_head.load(std::memory_order_acquire);
        int current_read = read_head.load(std::memory_order_relaxed);
        size_t used = (current_write - current_read + AUDIO_RING_BUFFER_SIZE) % AUDIO_RING_BUFFER_SIZE;
        NSLog(@"To read %zu", used);
        size_t first_sweep = AUDIO_RING_BUFFER_SIZE - current_read;
        size_t second_sweep = 0;
        
        size_t to_read = std::min<size_t>(requested_samples, used);
        
        if (first_sweep > to_read) first_sweep = to_read;
        else second_sweep = to_read - first_sweep;
        
        memcpy(out_data, buffer + current_read, first_sweep * sizeof(float));
        memcpy(out_data + first_sweep, buffer, second_sweep * sizeof(float));
        if (requested_samples > used) {
            memset(out_data + first_sweep + second_sweep, 0, (requested_samples - used) * sizeof(float));
        }
        read_head.store((current_read + to_read) % AUDIO_RING_BUFFER_SIZE);
    }
};


// =======================================================
// Objective-C Implementation
// =======================================================
@implementation MicrophoneAudioCapture {
    // Private instance variables go here
    LockFreeRingBuffer ringBuffer;
    std::function<matrix(const matrix&)> int16_to_float32;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        self.captureSession = [[AVCaptureSession alloc] init];
        NSError* error = nil;
//        AVCaptureDevice* audioDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeMicrophone mediaType:AVMediaTypeAudio position:AVCaptureDevicePositionFront];
        AVCaptureDevice* audioDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
        AVCaptureDeviceInput* audioInput = [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:&error];
        if (error) {
            NSLog(@"Failed to get mic: %@", error.localizedDescription);
            return nil;
        }
        if ([self.captureSession canAddInput:audioInput]) {
            [self.captureSession addInput:audioInput];
        }
        AVCaptureAudioDataOutput* audioOutput = [[AVCaptureAudioDataOutput alloc] init];
        NSDictionary* audioSettings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVLinearPCMBitDepthKey: @32,
            AVLinearPCMIsFloatKey: @YES,
            AVLinearPCMIsNonInterleavedKey: @NO
        };
#if !TARGET_OS_IPHONE
        [audioOutput setAudioSettings:audioSettings];
#endif
        self.audioQueue = dispatch_queue_create("MicrophoneCaptureQueue", DISPATCH_QUEUE_SERIAL);
        [audioOutput setSampleBufferDelegate:self queue:self.audioQueue];
    
        if ([self.captureSession canAddOutput:audioOutput]) {
            [self.captureSession addOutput:audioOutput];
        }
        [self start];
    }
    return self;
}

- (void)start {
    [self.captureSession startRunning];
}

- (void)stop {
    [self.captureSession stopRunning];
}

// -------------------------------------------------------
// The Push: Apple's Real-Time Background Thread
// -------------------------------------------------------
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    AudioBufferList audioBufferList;
    CMBlockBufferRef blockBuffer = NULL;
    OSStatus status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, NULL, &audioBufferList, sizeof(audioBufferList), NULL, NULL, kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, &blockBuffer);
    if (status != noErr) {
        NSLog(@"Failed to extract audio buffer");
        if (blockBuffer) CFRelease(blockBuffer);
        return;
    }

#if TARGET_OS_IPHONE
    // ---------------------------------------------
    // iOS ONLY CODE
    // (e.g., handling 16-bit Int audio from iPhone mic)
    // ---------------------------------------------
    int16_t* rawAudio = (int16_t*)audioBufferList.mBuffers[0].mData;
    int total_samples = audioBufferList.mBuffers[0].mDataByteSize / sizeof(int16_t);
    
    // We can't push int16 into a float ringBuffer, so we quickly convert on the CPU.
    // This takes ~0.001ms (infinitely faster than a GPU dispatch!)
    float tempBuffer[total_samples];
    for (int i = 0; i < total_samples; i++) {
        tempBuffer[i] = (float)rawAudio[i] / 32768.0f;
    }
    
    // Safely push to the ring buffer!
    ringBuffer.push(tempBuffer, total_samples);
#else
    // ---------------------------------------------
    // MACOS ONLY CODE
    // (e.g., setting the 32-bit float audioSettings dict)
    // ---------------------------------------------
    float* audioData = (float*)audioBufferList.mBuffers[0].mData;
    // numFloats = sample_size * channels
    int numFloats = audioBufferList.mBuffers[0].mDataByteSize / sizeof(float);
    ringBuffer.push(audioData, numFloats);
#endif

    
    if (blockBuffer) {
        CFRelease(blockBuffer);
    }
}

// -------------------------------------------------------
// The Pull: Your 60 FPS Main Video Loop
// -------------------------------------------------------
- (void)getAudioTensor:(matrix&)out_matrix samplesNeeded:(int)samplesNeeded {
    NSLog(@"got called");
    if (!out_matrix.buffer) {
        out_matrix.buffer = new uint8_t[samplesNeeded * dtype_size(out_matrix.type)];
        out_matrix.total_size = samplesNeeded;
        out_matrix.shape()[0] = samplesNeeded / 2;
        out_matrix.shape()[1] = 2;
        out_matrix.calcStrides();
    }
    // 1. Pop 'samplesNeeded' floats from LockFreeRingBuffer into out_matrix.buffer
    ringBuffer.pop((float*)out_matrix.buffer, samplesNeeded);
}

@end

//
//  CamReaderExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 26/02/25.
//

#import <Foundation/Foundation.h>
#import "AlgebroHeap.hpp"
#import <AVFoundation/AVFoundation.h>

@interface CaptureDelegate : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    uint8_t* buffer;
    NSCondition* mHasNewFrame;
    CVPixelBufferRef mGrabbedPixels;
    CVImageBufferRef mCurrentImageBuffer;
    size_t currentSize;
}

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection;
-(bool) grabImageUntilDate:(NSDate*)limit;
-(bool)updateImage;
-(void) getImage:(MatrixH<uint8_t>*)Mat;

@end



class CamReaderWrapper::CamReader {
public:
    CamReader(int camNo = 0);
    ~CamReader();
    bool grabFrame();
    bool retrieveFrame( MatrixH<uint8_t>& outputArray );
    int startCaptureDevice(int cameraNum);
    void stopCaptureDevice();
        
private:
    AVCaptureSession* mCaptureSession;
    AVCaptureDeviceInput* mCaptureDeviceInput;
    AVCaptureVideoDataOutput* mCaptureVideoDataOutput;
    AVCaptureDevice* mCaptureDevice;
    CaptureDelegate* mCapture;
    bool grabFrame(double timeOut);
    int camNum;
    int started;
    int width;
    int height;
};

bool CamReaderWrapper::CamReader::retrieveFrame(MatrixH<uint8_t>& outputArray) {
    
    NSDate *limit = [NSDate dateWithTimeIntervalSinceNow: 1];
    
    
    auto status  = [mCapture grabImageUntilDate: limit];
    
    [mCapture getImage:&outputArray];

    return true;
}

bool CamReaderWrapper::CamReader::grabFrame(double timeOut) {

    bool isGrabbed = false;
    NSDate *limit = [NSDate dateWithTimeIntervalSinceNow: timeOut];
    if ( [mCapture grabImageUntilDate: limit] ) {
        isGrabbed = [mCapture updateImage];
    }
    
    return isGrabbed;
}

@implementation CaptureDelegate

- (id)init {
    self = [super init];
    mHasNewFrame = [[NSCondition alloc] init];
    mCurrentImageBuffer = NULL;
    mGrabbedPixels = NULL;
    currentSize = 0;
    return self;
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

- (void)getImage:(MatrixH<uint8_t>*) Mat {
    
    if ( ! mGrabbedPixels ) {
        std::cout << "error grabbed pixels Null";
    }
    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uint8_t *baseaddress = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));
    
    std::vector<size_t> shape = { CVPixelBufferGetHeight(mGrabbedPixels), CVPixelBufferGetWidth(mGrabbedPixels), 4 };
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);
    
    bool res = false;
    if (rowBytes != 0 && (pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_422YpCbCr8)) {
        
        if (Mat->values) {
            Mat->shape = shape;
            Mat->total_size = shape[0] * shape[1] * shape[2];
            memcpy(Mat->values, baseaddress, shape[0] * shape[1] * shape[2]);
        } else {
            Mat->shape = shape;
            Mat->total_size = shape[0] * shape[1] * shape[2];
            Mat->values = new uint8_t[Mat->total_size];
            memcpy(Mat->values, baseaddress, shape[0] * shape[1] * shape[2]);
        }
        
        
        
        res = true;
               
    } else {
        
        fprintf(stderr, "MatrixH: rowBytes == 0 or unknown pixel format 0x%08X\n", pixelFormat);
        *Mat = MatrixH<uint8_t>();
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    
    mGrabbedPixels = NULL;
}

- (bool)updateImage {
    if ( ! mGrabbedPixels ) {
        return false;
    }
    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uint8_t *baseaddress = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));
    
    std::vector<size_t> shape = { CVPixelBufferGetWidth(mGrabbedPixels), CVPixelBufferGetHeight(mGrabbedPixels), 4 };
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);
    
    bool res = false;
    if (rowBytes != 0 && (pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_422YpCbCr8)) {
//        mOutputArray = MatrixH<uint8_t>().fromBuffer(baseaddress, shape);
        res = true;
               
    } else {
        fprintf(stderr, "MatrixH: rowBytes == 0 or unknown pixel format 0x%08X\n", pixelFormat);
//        mOutputArray = MatrixH<uint8_t>();
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    mGrabbedPixels = NULL;

    return res;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    (void)captureOutput;
    (void)sampleBuffer;
    (void)connection;

    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVBufferRetain(imageBuffer);
    [mHasNewFrame lock];

    CVBufferRelease(mCurrentImageBuffer);
    mCurrentImageBuffer = imageBuffer;
    [mHasNewFrame broadcast];

    [mHasNewFrame unlock];

}

@end

void CamReaderWrapper::retrieveFrame(MatrixH<uint8_t>& frame) {

    
    pCamReader->retrieveFrame(frame);

}

CamReaderWrapper::CamReaderWrapper(int CamNo) {
    pCamReader = new CamReader(CamNo);
}

CamReaderWrapper::CamReader::~CamReader() {
    stopCaptureDevice();
}

CamReaderWrapper::CamReader::CamReader(int camNo) {
    mCaptureSession = nil;
    mCaptureDevice = nil;
    mCaptureDeviceInput = nil;
    mCaptureVideoDataOutput = nil;
    mCapture = nil;
    
    camNum = camNo;
    
    if ( ! startCaptureDevice(camNum) ) {
        fprintf(stderr, "MatrixH: camera failed to properly initialize!\n");
        started = 0;
    } else {
        started = 1;
    }
}

void CamReaderWrapper::CamReader::stopCaptureDevice() {
    NSLog(@"Stopping Session Now");
    [mCaptureSession stopRunning];
}

int CamReaderWrapper::CamReader::startCaptureDevice(int cameraNum) {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];

    if (status == AVAuthorizationStatusDenied)
    {
        fprintf(stderr, "MATRIX: camera access has been denied. ...\n");
        return 0;
    }
    else if (status != AVAuthorizationStatusAuthorized) {
        fprintf(stderr, "MATRIX: not authorized to capture video (status %ld), requesting...\n", status);
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL) {}];
        if ([NSThread isMainThread]) {
            [[NSRunLoop mainRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
        } else {
            fprintf(stderr, "MATRIX: can not spin main run loop from other thread, set ");
        }
        return 0;
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
    mCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:mCaptureDevice error:&error];
    if ( error ) {
        fprintf(stderr, "MATRIX: error in [AVCaptureDeviceInput initWithDevice:error:]\n");
        NSLog(@"OpenCV: %@", error.localizedDescription);
    }
    
    mCapture = [[CaptureDelegate alloc] init];
    mCaptureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    
    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", DISPATCH_QUEUE_SERIAL);
    [mCaptureVideoDataOutput setSampleBufferDelegate: mCapture queue: queue];
    
    NSMutableDictionary *pixelBufferOptionsL = [mCaptureVideoDataOutput.videoSettings mutableCopy];
    
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
    
    std::cout << "Initilizing Cam with Width: " << width << " and Height: " << height << "\n";
    
    OSType pixelFormat;
    pixelFormat = kCVPixelFormatType_32BGRA;
    
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
    
    mCaptureSession = [[AVCaptureSession alloc] init];
    mCaptureSession.sessionPreset = AVCaptureSessionPresetMedium;
    [mCaptureSession addInput: mCaptureDeviceInput];
    [mCaptureSession addOutput: mCaptureVideoDataOutput];
    
    [mCaptureSession startRunning];
    [NSThread sleepForTimeInterval:0.5];

    grabFrame(1);
    
    
    return 1;
}

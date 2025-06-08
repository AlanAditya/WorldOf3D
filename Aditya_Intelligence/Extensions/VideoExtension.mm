//
//  VideoExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 19/02/25.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#include "AlgebroHeap.hpp"

template <typename Type>
MatrixH<uint8_t> MatrixH<Type>::fromVideo(const char* vidPath) {
//    const char* vidPath = "/Users/adityadude/Downloads/WhatsApp Video 2025-01-01 at 14.49.11.mp4";
    NSString *filePath = [NSString stringWithUTF8String:vidPath];
    
    NSURL* url = [NSURL fileURLWithPath:filePath];
    AVAsset* asset = [AVAsset assetWithURL:url];
    __block AVAssetTrack* videoTrack;
    
    [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> * videoArray, NSError * _Nullable) {
        videoTrack = videoArray.firstObject;
    }];
    AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:nil];
    NSDictionary* outputSettings = @{
            (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)  // 4-channel (BGRA)
    };
    
    AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
    [reader addOutput:trackOutput];
    [reader startReading];
    
    size_t frameCount = (size_t)(CMTimeGetSeconds(asset.duration) * videoTrack.nominalFrameRate);
    size_t width = (size_t)videoTrack.naturalSize.width;
    size_t height = (size_t)videoTrack.naturalSize.height;
    size_t channels = 4; // BGRA format has 4 channels
    std::cout << "data " << frameCount <<" "<< height <<" " <<width<< " "<<channels;
    uint8_t* values = new uint8_t[width*height*frameCount*channels];
    NSLog(@"CMTime %f", CMTimeGetSeconds(asset.duration));
    
    size_t frameIndex = 0;
    while ([reader status] == AVAssetReaderStatusReading) {
        CMSampleBufferRef sampleBuffer = [trackOutput copyNextSampleBuffer];
        if (!sampleBuffer) {std::cout << "error "; break; }
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        std::cout << (width * channels) << "\n";
        std::cout << (bytesPerRow)<< "\n";
        std::cout << (width * channels * sizeof(Type))<< "\n";
        for (size_t y = 0; y < height; y++) {
            memcpy(values + (frameIndex * height * width * channels) + (y * width * channels),
                   baseAddress + (y * bytesPerRow),
                   width * channels * sizeof(Type));
        }
        frameIndex++;
        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    }
    
    MatrixH<uint8_t> result = MatrixH();
    result.shape = {frameCount, height, width, channels};
    result.total_size = width*height*frameCount*channels;
    result.values = values;
    return result;
}

template MatrixH<uint8_t> MatrixH<uint8_t>::fromVideo(const char* vidPath);

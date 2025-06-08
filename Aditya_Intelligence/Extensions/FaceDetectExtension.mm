//
//  FaceDetectExtension.m
//  Aditya_Intelligence
//
//  Created by Manoj Kumar on 28/02/25.
//

#import <Foundation/Foundation.h>
#include "AlgebroHeap.hpp"
#import <Vision/Vision.h>





template <typename Type>
CVPixelBufferRef MatrixH<Type>::createPixelBufferFromMat() {
    
    size_t width = shape[1];
    size_t height = shape[0];
    size_t channels = shape[2];
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
                values + row * copyBytesPerRow,
                copyBytesPerRow);
     }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    return pixelBuffer;
}

template <typename Type>
void MatrixH<Type>::RectangleDetect(MatrixH<int>& detections) {
    CVPixelBufferRef pixelBuffer = createPixelBufferFromMat();
    if (!pixelBuffer) {
        NSLog(@"Pixel buffer creation failed.");
        return;
    }
    
    VNDetectRectanglesRequest *faceReq = [[VNDetectRectanglesRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
        if (error) {
            NSLog(@"Face detection error: %@", error);
            return;
        }
        
        NSArray *observations = request.results;
        if (detections.values) {
            delete [] detections.values;
        }
        
        
        detections.values = new int[observations.count * 4];
        detections.shape = {observations.count, 4};
        detections.total_size = observations.count * 4;
        int index = 0;
        for (VNRectangleObservation *face in observations) {
            // The boundingBox is in normalized coordinates (origin at bottom-left).
            NSLog(@"Detected face at bounding box: %@", NSStringFromRect(face.boundingBox));
            
            detections.values[index * 4 + 0] = (int)(face.boundingBox.origin.x * shape[1]);
            detections.values[index * 4 + 1] = (int)( ((1.0-face.boundingBox.origin.y) - face.boundingBox.size.height) * shape[0]);
            detections.values[index * 4 + 2] = (int)(face.boundingBox.size.width * shape[1]);
            detections.values[index * 4 + 3] = (int)(face.boundingBox.size.height * shape[0]);
            
            drawRectStroked({(int)(face.boundingBox.origin.x * shape[1]),  (int)( ((1.0-face.boundingBox.origin.y) - face.boundingBox.size.height) * shape[0]), (int)(face.boundingBox.size.width * shape[1]), (int)(face.boundingBox.size.height * shape[0])}, {255, 255, 255, 255}, 0.01);
            index++;
        }
        NSLog(@"Error performing face detection: ");
    }];
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
    NSError *error = nil;
    [handler performRequests:@[faceReq] error:&error];
    if (error) {
        NSLog(@"Error performing face detection: %@", error);
    }
    
    CVPixelBufferRelease(pixelBuffer);
    

}

template <typename Type>
void MatrixH<Type>::FaceDetect(MatrixH<int>& detections) {
    CVPixelBufferRef pixelBuffer = createPixelBufferFromMat();
    if (!pixelBuffer) {
        NSLog(@"Pixel buffer creation failed.");
        return;
    }
    
    VNDetectFaceRectanglesRequest *faceReq = [[VNDetectFaceRectanglesRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
        if (error) {
            NSLog(@"Face detection error: %@", error);
            return;
        }
        NSArray *observations = request.results;
        
        if (detections.values) {
            delete [] detections.values;
        }
        
        
        detections.values = new int[observations.count * 4];
        detections.shape = {observations.count, 4};
        detections.total_size = observations.count * 4;
        int index = 0;
        for (VNFaceObservation *face in observations) {
            
            // The boundingBox is in normalized coordinates (origin at bottom-left).
            NSLog(@"Detected face at bounding box: %@", NSStringFromRect(face.boundingBox));
            
            detections.values[index * 4 + 0] = (int)(face.boundingBox.origin.x * shape[1]);
            detections.values[index * 4 + 1] = (int)( ((1.0-face.boundingBox.origin.y) - face.boundingBox.size.height) * shape[0]);
            detections.values[index * 4 + 2] = (int)(face.boundingBox.size.width * shape[1]);
            detections.values[index * 4 + 3] = (int)(face.boundingBox.size.height * shape[0]);
            
            drawRectStroked({(int)(face.boundingBox.origin.x * shape[1]),  (int)( ((1.0-face.boundingBox.origin.y) - face.boundingBox.size.height) * shape[0]), (int)(face.boundingBox.size.width * shape[1]), (int)(face.boundingBox.size.height * shape[0])}, {255, 255, 255, 255}, 0.01);
            index++;
        }
        NSLog(@"Error performing face detection: ");
    }];
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
    NSError *error = nil;
    [handler performRequests:@[faceReq] error:&error];
    if (error) {
        NSLog(@"Error performing face detection: %@", error);
    }
    
    CVPixelBufferRelease(pixelBuffer);
    

}

static MatrixH<int> pt = {0, 0};

template <typename Type>
void MatrixH<Type>::HandsDetect(MatrixH<int>& detections, bool all_pts) {
    CVPixelBufferRef pixelBuffer = createPixelBufferFromMat();
    if (!pixelBuffer) {
        NSLog(@"Pixel buffer creation failed.");
        return;
    }
    
    size_t obsPerFinger;
    size_t width = shape[1];
    size_t height = shape[0];
    
    if (all_pts) {obsPerFinger = 20;} else { obsPerFinger = 5;}
    VNDetectHumanHandPoseRequest *handReq = [[VNDetectHumanHandPoseRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
        if (error) {
            NSLog(@"Hand detection error: %@", error);
            return;
        }
        NSArray<VNHumanHandPoseObservation *> *observations = request.results;
        
        
        if (detections.total_size !=  observations.count * obsPerFinger * 2) {
            delete [] detections.values;
            detections.values = new int[observations.count * obsPerFinger * 2];
            detections.shape = {observations.count, obsPerFinger, 2};
            detections.total_size = observations.count * obsPerFinger * 2;
            
        }
        if (!detections.values) {
            detections.values = new int[observations.count * obsPerFinger * 2];
            detections.shape = {observations.count, obsPerFinger, 2};
            detections.total_size = observations.count * obsPerFinger * 2;
        }
        
        std::cout << detections.shape[0] <<" finger"<< "\n";
        

        int index = 0;
        for (VNHumanHandPoseObservation *point in observations) {
            
            NSError *landmarkError = nil;
            VNRecognizedPoint *thumbTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbTip error:&landmarkError];
                    if (landmarkError) {
                        NSLog(@"Error retrieving thumb tip landmark: %@", landmarkError);
                        continue;
                    }
            *detections(index, 0, 0) = (int)(thumbTip.location.x * width);
            *detections(index, 0, 1) = (int)( ((1.0-thumbTip.location.y)) * height);
            
            
                    // Similarly, you can retrieve other landmarks, e.g.,
            VNRecognizedPoint *indexTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexTip error:&landmarkError];
                    if (landmarkError) {
                        NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                        continue;
                    }
            *detections(index, 1, 0) = (int)(indexTip.location.x * width);
            *detections(index, 1, 1) = (int)( ((1.0-indexTip.location.y)) * height);
            
            VNRecognizedPoint *middleTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleTip error:&landmarkError];
                    if (landmarkError) {
                        NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                        continue;
                    }
            *detections(index, 2, 0) = (int)(middleTip.location.x * width);
            *detections(index, 2, 1) = (int)( ((1.0-middleTip.location.y)) * height);
            
            VNRecognizedPoint *ringTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingTip error:&landmarkError];
                    if (landmarkError) {
                        NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                        continue;
                    }
            *detections(index, 3, 0) = (int)(ringTip.location.x * width);
            *detections(index, 3, 1) = (int)( ((1.0-ringTip.location.y)) * height);
            
            VNRecognizedPoint *pinkyTip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleTip error:&landmarkError];
                    if (landmarkError) {
                        NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                        continue;
                    }
            *detections(index, 4, 0) = (int)(pinkyTip.location.x * width);
            *detections(index, 4, 1) = (int)( ((1.0-pinkyTip.location.y)) * height);
            
            
            if (all_pts) {
                VNRecognizedPoint *thumbDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 5, 0) = (int)(thumbDip.location.x * width);
                *detections(index, 5, 1) = (int)( ((1.0-thumbDip.location.y)) * height);
                
                VNRecognizedPoint *indexDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexDIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 6, 0) = (int)(indexDip.location.x * width);
                *detections(index, 6, 1) = (int)( ((1.0-indexDip.location.y)) * height);
                
                VNRecognizedPoint *middleDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleDIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 7, 0) = (int)(middleDip.location.x * width);
                *detections(index, 7, 1) = (int)( ((1.0-middleDip.location.y)) * height);
                
                VNRecognizedPoint *ringDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingDIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 8, 0) = (int)(ringDip.location.x * width);
                *detections(index, 8, 1) = (int)( ((1.0-ringDip.location.y)) * height);
                
                VNRecognizedPoint *pinkyDip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleDIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 9, 0) = (int)(pinkyDip.location.x * width);
                *detections(index, 9, 1) = (int)( ((1.0-pinkyDip.location.y)) * height);
                
                VNRecognizedPoint *thumbPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbMP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 10, 0) = (int)(thumbPip.location.x * width) ;
                *detections(index, 10, 1) = (int)( ((1.0-thumbPip.location.y)) * height);
                
                VNRecognizedPoint *indexPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexPIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 11, 0) = (int)(indexPip.location.x * width);
                *detections(index, 11, 1) = (int)( ((1.0-indexPip.location.y)) * height);
                
                VNRecognizedPoint *middlePip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddlePIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 12, 0) = (int)(middlePip.location.x * width);
                *detections(index, 12, 1) = (int)( ((1.0-middlePip.location.y)) * height);
                
                VNRecognizedPoint *ringPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingPIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 13, 0) = (int)(ringPip.location.x * width);
                *detections(index, 13, 1) = (int)( ((1.0-ringPip.location.y)) * height);
                
                VNRecognizedPoint *pinkyPip = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittlePIP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 14, 0) = (int)(pinkyPip.location.x * width);
                *detections(index, 14, 1) = (int)( ((1.0-pinkyPip.location.y)) * height);
                
                VNRecognizedPoint *thumbMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameThumbCMC error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 15, 0) = (int)(thumbMCP.location.x * width) ;
                *detections(index, 15, 1) = (int)( ((1.0-thumbMCP.location.y)) * height) ;
                
                VNRecognizedPoint *indexMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameIndexMCP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 16, 0) = (int)(indexMCP.location.x * width) ;
                *detections(index, 16, 1) = (int)( ((1.0-indexMCP.location.y)) * height) ;
                
                VNRecognizedPoint *middleMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameMiddleMCP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 17, 0) = (int)(middleMCP.location.x * width);
                *detections(index, 17, 1) = (int)( ((1.0-middleMCP.location.y)) * height);
                
                VNRecognizedPoint *ringMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameRingMCP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 18, 0) = (int)(ringMCP.location.x * width);
                *detections(index, 18, 1) = (int)( ((1.0-ringMCP.location.y)) * height);
                
                VNRecognizedPoint *pinkyMCP = [point recognizedPointForKey:VNHumanHandPoseObservationJointNameLittleMCP error:&landmarkError];
                        if (landmarkError) {
                            NSLog(@"Error retrieving index tip landmark: %@", landmarkError);
                            continue;
                        }
                *detections(index, 19, 0) = (int)(pinkyMCP.location.x * width);
                *detections(index, 19, 1) = (int)( ((1.0-pinkyMCP.location.y)) * height);
                
                
                
            }

            // The boundingBox is in normalized coordinates (origin at bottom-left).
//            NSLog(@"Detected face at bounding box: %@", NSStringFromRect(face.boundingBox));
            
//            *pt(0) = (int)(thumbTip.location.x * shape[1]);
//            *pt(1) = (int)( ((1.0-thumbTip.location.y)) * shape[0]);
//            detections.appentMat(index, 0, pt);
//            
//            *pt(0) = (int)(indexTip.location.x * shape[1]);
//            *pt(1) = (int)( ((1.0-indexTip.location.y)) * shape[0]);
//            detections.appentMat(index, 1, pt);
//            
//            *pt(0) = (int)(middleTip.location.x * shape[1]);
//            *pt(1) = (int)( ((1.0-middleTip.location.y)) * shape[0]);
//            detections.appentMat(index, 2, pt);
//            
//            // Ring
//            *pt(0) = (int)(ringTip.location.x * shape[1]);
//            *pt(1) = (int)((1.0 - ringTip.location.y) * shape[0]);
//            detections.appentMat(index, 3, pt);
//
//            // Pinky
//            *pt(0) = (int)(pinkyTip.location.x * shape[1]);
//            *pt(1) = (int)((1.0 - pinkyTip.location.y) * shape[0]);
//            detections.appentMat(index, 4, pt);
//            
//            drawLine(simd_make_int2(*detections(index, 0, 0), *detections(0, 0, 1)), simd_make_int2(*detections(index, 1, 0), *detections(0, 1, 1)), 0, RED);
//            

            
//            printf("width: %i", (int)(thumbTip.location.x * shape[1]));
//            printf("height: %i", (int)( ((1.0-thumbTip.location.y)) * shape[0]));
//            detections[0].print();
//            drawElipse({(int)(thumbTip.location.x * shape[1]),  (int)( ((1.0-thumbTip.location.y)) * shape[0]), 50, 50}, {255, 255, 255, 255});
//            
//            drawElipse({(int)(indexTip.location.x * shape[1]),  (int)( ((1.0-indexTip.location.y)) * shape[0]), 50, 50}, {255, 255, 255, 255});
//            drawElipse({(int)(indexDip.location.x * shape[1]),  (int)( ((1.0-indexDip.location.y)) * shape[0]), 50, 50}, {255, 255, 255, 255});
//            drawElipse({(int)(indexPip.location.x * shape[1]),  (int)( ((1.0-indexPip.location.y)) * shape[0]), 50, 50}, {255, 255, 255, 255});
//            drawElipse({(int)(indexMCP.location.x * shape[1]),  (int)( ((1.0-indexMCP.location.y)) * shape[0]), 50, 50}, {255, 255, 255, 255});
//            
//            drawElipse({(int)(middleTip.location.x * shape[1]),  (int)( ((1.0-middleTip.location.y)) * shape[0]), 100, 100}, {255, 255, 255, 255});
            index++;
        }
    }];
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
    NSError *error = nil;
    [handler performRequests:@[handReq] error:&error];
    if (error) {
        NSLog(@"Error performing face detection: %@", error);
    }
    
    CVPixelBufferRelease(pixelBuffer);

}

template void MatrixH<uint8_t>::RectangleDetect(MatrixH<int>& detections);
template void MatrixH<uint8_t>::FaceDetect(MatrixH<int>& detections);
template void MatrixH<uint8_t>::HandsDetect(MatrixH<int>& detections, bool all_pts);
template CVPixelBufferRef MatrixH<uint8_t>::createPixelBufferFromMat();

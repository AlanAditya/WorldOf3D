////
////  FaceLandmarkWrapper.mm
////  WorldOf3D
////
////  Created by Aditya Dudeja on 12/12/25.
////
//
//#ifndef FaceLandmarkWrapper_h
//#define FaceLandmarkWrapper_h
//// FaceLandmarkWrapper.mm
//#import "Bitch.h"
// 
//@implementation FaceLandmarkResult
//@end
//
////@interface BitchWrapper: NSObject {
////    std::unique_ptr<mediapipe::CalculatorGraph> _graph;
////    std::unique_ptr<mediapipe::OutputStreamPoller> _poller;
////    BOOL _initialized;
////    int64_t _frame_timestamp;
////}
////@end
//#include <fstream>
//namespace mediapipe {
//
//typedef std::function<absl::Status(const std::string&, std::string*)>
//    ResourceProviderFn;
//
//// Returns true if files are provided via a custom resource provider.
//bool HasCustomGlobalResourceProvider();
//
//// Overrides the behavior of GetResourceContents.
//void SetCustomGlobalResourceProvider(ResourceProviderFn fn);
//
//}
//constexpr char kInputStream[] = "input_video";
//constexpr char kOutputStream[] = "output_video";
//constexpr char kWindowName[] = "MediaPipe";
//
//static std::string g_model_root = "/private/var/tmp/_bazel_adityadude/eb3560479e75155bc8d80d1ef4345636/execroot/mediapipe/bazel-out/darwin_arm64-opt/bin/";
//
//absl::Status MyResourceProvider(const std::string& path, std::string* output) {
//    std::string full_path = g_model_root + path;
//
//    std::ifstream file(full_path, std::ios::binary);
//    if (!file.is_open()) {
//        return absl::NotFoundError("Model not found: " + full_path);
//    }
//
//    output->assign(std::istreambuf_iterator<char>(file),
//                   std::istreambuf_iterator<char>());
//    return absl::OkStatus();
//}
//
//void SetupMediaPipeResourceProvider() {
//    mediapipe::SetCustomGlobalResourceProvider(MyResourceProvider);
//}
//
//@implementation BitchWrapper
//
//- (instancetype)init {
//    self = [super init];
//    if (self) {
//        _initialized = false;
//        
////        NSString *faceDetectionModel = @"/private/var/tmp/_bazel_adityadude/eb3560479e75155bc8d80d1ef4345636/execroot/mediapipe/bazel-out/darwin_arm64-opt/bin/mediapipe/modules/face_detection/face_detection_short_range.tflite";
////        NSString *faceLandmarkModel = @"/private/var/tmp/_bazel_adityadude/eb3560479e75155bc8d80d1ef4345636/execroot/mediapipe/bazel-out/darwin_arm64-opt/bin/mediapipe/modules/face_landmark/face_landmark.tflite";
//        
//        NSString *path = @"/private/var/tmp/_bazel_adityadude/eb3560479e75155bc8d80d1ef4345636/execroot/mediapipe/bazel-out/darwin_arm64-opt/bin/";
//        setenv("MEDIAPIPE_RESOURCE_DIR", [path UTF8String], 1);
////        NSString *contents = @R"(
////            input_stream: "input_video"
////            output_stream: "multi_face_landmarks"
////            
////            # Convert ImageFrame to Image
////            node {
////              calculator: "ToImageCalculator"
////              input_stream: "IMAGE_CPU:input_video"
////              output_stream: "IMAGE:input_video_image"
////            }
////            
////            node {
////              calculator: "FlowLimiterCalculator"
////              input_stream: "input_video_image"
////              input_stream: "FINISHED:multi_face_landmarks"
////              input_stream_info: {
////                tag_index: "FINISHED"
////                back_edge: true
////              }
////              output_stream: "throttled_input_video"
////            }
////            
////            node {
////              calculator: "FaceLandmarkFrontCpu"
////              input_stream: "IMAGE:throttled_input_video"
////              output_stream: "LANDMARKS:multi_face_landmarks"
////            }
////        )";
//        
////        contents = @R"(
////        input_stream: "input_video"
////
////        # Output image with rendered results. (ImageFrame)
////        output_stream: "output_video"
////        # Collection of detected/processed faces, each represented as a list of
////        # landmarks. (std::vector<NormalizedLandmarkList>)
////        output_stream: "multi_face_landmarks"
////
////        # Throttles the images flowing downstream for flow control. It passes through
////        # the very first incoming image unaltered, and waits for downstream nodes
////        # (calculators and subgraphs) in the graph to finish their tasks before it
////        # passes through another image. All images that come in while waiting are
////        # dropped, limiting the number of in-flight images in most part of the graph to
////        # 1. This prevents the downstream nodes from queuing up incoming images and data
////        # excessively, which leads to increased latency and memory usage, unwanted in
////        # real-time mobile applications. It also eliminates unnecessarily computation,
////        # e.g., the output produced by a node may get dropped downstream if the
////        # subsequent nodes are still busy processing previous inputs.
////        node {
////          calculator: "FlowLimiterCalculator"
////          input_stream: "input_video"
////          input_stream: "FINISHED:output_video"
////          input_stream_info: {
////            tag_index: "FINISHED"
////            back_edge: true
////          }
////          output_stream: "throttled_input_video"
////        }
////
////        # Defines side packets for further use in the graph.
////        node {
////          calculator: "ConstantSidePacketCalculator"
////          output_side_packet: "PACKET:0:num_faces"
////          output_side_packet: "PACKET:1:with_attention"
////          node_options: {
////            [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
////              packet { int_value: 1 }
////              packet { bool_value: true }
////            }
////          }
////        }
////
////        # Subgraph that detects faces and corresponding landmarks.
////        node {
////          calculator: "FaceLandmarkFrontCpu"
////          input_stream: "IMAGE:throttled_input_video"
////          input_side_packet: "NUM_FACES:num_faces"
////          input_side_packet: "WITH_ATTENTION:with_attention"
////          output_stream: "LANDMARKS:multi_face_landmarks"
////          output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
////          output_stream: "DETECTIONS:face_detections"
////          output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
////        }
////
////        # Subgraph that renders face-landmark annotation onto the input image.
////        node {
////          calculator: "FaceRendererCpu"
////          input_stream: "IMAGE:throttled_input_video"
////          input_stream: "LANDMARKS:multi_face_landmarks"
////          input_stream: "NORM_RECTS:face_rects_from_landmarks"
////          input_stream: "DETECTIONS:face_detections"
////          output_stream: "IMAGE:output_video"
////        }
////)";
//
//        SetupMediaPipeResourceProvider();
//        NSString *faceDetectionModel = [[NSBundle mainBundle] pathForResource:@"face_detection_short_range"
//                                                                        ofType:@"tflite"
//                                                                   inDirectory:@"mediapipe/modules/face_detection"];
//        NSString *faceLandmarkModel = [[NSBundle mainBundle] pathForResource:@"face_landmark"
//                                                                      ofType:@"tflite"
//                                                                 inDirectory:@"mediapipe/modules/face_landmark"];
//        
//        if (!faceDetectionModel || !faceLandmarkModel) {
//            NSLog(@"❌ Model files not found in bundle!");
//            NSLog(@"Face detection: %@", faceDetectionModel);
//            NSLog(@"Face landmark: %@", faceLandmarkModel);
//            
//            // List what IS in the bundle
//            NSString *bundlePath = [[NSBundle mainBundle] resourcePath];
//            NSLog(@"Bundle path: %@", bundlePath);
//            NSFileManager *fm = [NSFileManager defaultManager];
//            NSArray *contents = [fm contentsOfDirectoryAtPath:bundlePath error:nil];
//            NSLog(@"Bundle contents: %@", contents);
//            
////            return nil;
//        }
////        
////        NSLog(@"✅ Found models:");
////        NSLog(@"  Face detection: %@", faceDetectionModel);
////        NSLog(@"  Face landmark: %@", faceLandmarkModel);
//        
//        // Load and parse graph config
//        std::string config_contents;
////        NSString *contents = [NSString stringWithContentsOfFile:configPath
////                                                       encoding:NSUTF8StringEncoding
////                                                          error:nil];
//
////        if (!contents) {
////            NSLog(@"Failed to load graph config");
////            return nil;
////        }
//        NSString *graphPath = @"/Users/adityadude/mediapipe/mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt";
//
//        NSError *error = nil;
//        NSString *graphContents = [NSString stringWithContentsOfFile:graphPath
//                                                           encoding:NSUTF8StringEncoding
//                                                              error:&error];
//        
//        if (!graphContents) {
//            NSLog(@"❌ Failed to load graph: %@", error);
//            return nil;
//        }
//        
//        config_contents = std::string([graphContents UTF8String]);
//        
//        mediapipe::CalculatorGraphConfig config;
//        if (!mediapipe::ParseTextProto<mediapipe::CalculatorGraphConfig>(
//                config_contents, &config)) {
//            NSLog(@"Failed to parse graph config");
//            return nil;
//        }
////        // Initialize graph
//        _graph = std::make_unique<mediapipe::CalculatorGraph>();
//        auto status = _graph->Initialize(config);
////
//        if (!status.ok()) {
//            std::string error_msg(status.message());
//            NSLog(@"Failed to initialize graph: %s", error_msg.c_str());
//            return nil;
//        }
//        
//        auto poller_or = _graph->AddOutputStreamPoller("output_video");
//
//        status = _graph->StartRun({});
//        if (!status.ok()) {
//            std::string error_msg(status.message());
//            NSLog(@"Failed to start graph: %s", error_msg.c_str());
//            return nil;
//        }
//        if (!poller_or.ok()) {
//            NSLog(@"Failed to setup poller: %s", poller_or.status().ToString().c_str());
//            return nil;
//        }
//
//        
//        _poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_or.value()));
//        
//        _initialized = true;
//        NSLog(@"✅ Face landmark detector initialized");
//    }
//    
//    return self;
//}
////
////- (FaceLandmarkResult *)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer {
////    FaceLandmarkResult *result = [[FaceLandmarkResult alloc] init];
////    result.detected = false;
////    result.confidence = 0.0f;
////    result.landmarks = @[];
////    
////    if (!_initialized || !_graph || !_poller) {
////        NSLog(@"❌ Detector not initialized");
////        return result;
////    }
////    
////    // Convert CVPixelBuffer to cv::Mat
////    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
////    
////    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
////    size_t width = CVPixelBufferGetWidth(pixelBuffer);
////    size_t height = CVPixelBufferGetHeight(pixelBuffer);
////    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
////    
////    cv::Mat frame(static_cast<int>(height), static_cast<int>(width),
////                  CV_8UC4, baseAddress, bytesPerRow);
////    
////    cv::Mat rgb_frame;
////    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGRA2RGB);
////    
////    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
////    
////    // Process with MediaPipe
////    return [self detectInMat:rgb_frame width:width height:height];
////}
////
//////- (FaceLandmarkResult *)detectInImage:(NSImage *)image {
//////    // Convert NSImage to cv::Mat
//////    CGImageRef cgImage = [image CGImageForProposedRect:NULL context:nil hints:nil];
//////    
//////    size_t width = CGImageGetWidth(cgImage);
//////    size_t height = CGImageGetHeight(cgImage);
//////    
//////    cv::Mat mat(static_cast<int>(height), static_cast<int>(width), CV_8UC4);
//////    
//////    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//////    CGContextRef context = CGBitmapContextCreate(mat.data, width, height, 8,
//////                                                 mat.step, colorSpace,
//////                                                 kCGImageAlphaPremultipliedLast);
//////    CGColorSpaceRelease(colorSpace);
//////    
//////    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
//////    CGContextRelease(context);
//////    
//////    cv::Mat rgb_mat;
//////    cv::cvtColor(mat, rgb_mat, cv::COLOR_RGBA2RGB);
//////    
//////    return [self detectInMat:rgb_mat width:width height:height];
//////}
//////
//- (FaceLandmarkResult *)detectInMat:(cv::Mat)camera_frame width:(size_t)width height:(size_t)height {
//    FaceLandmarkResult *result = [[FaceLandmarkResult alloc] init];
//    result.detected = false;
//    
//    if (!_initialized || !_graph || !_poller) {
//        return result;
//    }
//    
////    // Convert to MediaPipe ImageFrame
////    auto input_frame = std::make_unique<mediapipe::ImageFrame>(
////        mediapipe::ImageFormat::SRGB,
////        frame.cols, frame.rows,
////        mediapipe::ImageFrame::kDefaultAlignmentBoundary
////    );
////    
////    cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
////    frame.copyTo(input_mat);
//    
//    // Create timestamp
////    _frame_timestamp++;
////    auto timestamp = mediapipe::Timestamp(_frame_timestamp);
//////    auto packets = mediapipe::MakePacket<mediapipe::ImageFrame> (
//////                                                                std::move(*input_frame)
//////                                                            ).At(mediapipe::Timestamp(_frame_timestamp));
////    // Wrap Mat into an ImageFrame.
////       auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
////           mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
////           mediapipe::ImageFrame::kDefaultAlignmentBoundary);
////       cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
////       frame.copyTo(input_frame_mat);
////
////       // Send image packet into the graph.
////       size_t frame_timestamp_us =
////           (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
////    auto status = _graph->AddPacketToInputStream(
////           "input_video", mediapipe::Adopt(input_frame.release())
////                             .At(mediapipe::Timestamp(frame_timestamp_us)));
////    
////    if (!status.ok()) {
////        std::string error_msg(status.message());
////        // Log the status CODE and message explicitly
////        NSLog(@"❌ Failed to add packet. Status Code: %d, Message: %s",
////              status.code(), // e.g., 3 for INVALID_ARGUMENT
////              error_msg.c_str());
//////        return result;
////    }
////    
////    // Poll for output
////    mediapipe::Packet packet;
////    if (!_poller->Next(&packet)) {
////        // No output available yet
////        return result;
////    }
////    
////    if (packet.IsEmpty()) {
////        return result;
////    }
////    NSLog(@"Hello ??");
//    
//    // Wrap Mat into an ImageFrame.
//        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
//            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
//            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
//        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
//        camera_frame.copyTo(input_frame_mat);
//
//        // Send image packet into the graph.
//        size_t frame_timestamp_us =
//            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
//        auto status = (_graph->AddPacketToInputStream(
//            kInputStream, mediapipe::Adopt(input_frame.release())
//                            .At(mediapipe::Timestamp(frame_timestamp_us))));
//        if (!status.ok()) {
//            std::string error_msg(status.message());
//            // Log the status CODE and message explicitly
//            NSLog(@"❌ Failed to add packet. Status Code: %d, Message: %s",
//                  status.code(), // e.g., 3 for INVALID_ARGUMENT
//                  error_msg.c_str());
//    //        return result;
//        }
//        // Get the graph result packet, or stop if that fails.
//        mediapipe::Packet packet;
//        if (!_poller->Next(&packet)) NSLog(@"F google F nis");
//        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
//    
//    // Extract landmarks
////    try {
////        const auto& multi_face_landmarks =
////            packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
////        
////        if (!multi_face_landmarks.empty()) {
////            const auto& face = multi_face_landmarks[0];
////            
////            NSMutableArray *landmarks = [NSMutableArray array];
////            for (const auto& landmark : face.landmark()) {
////                CGPoint point = CGPointMake(
////                    landmark.x() * width,
////                    landmark.y() * height
////                );
////                [landmarks addObject:[NSValue valueWithPoint:point]];
////            }
////            
////            result.landmarks = landmarks;
////            result.detected = true;
////            result.confidence = 1.0f;
////            
////            NSLog(@"✅ Detected face with %lu landmarks", (unsigned long)landmarks.count);
////        }
////    } catch (const std::exception& e) {
////        NSLog(@"❌ Exception extracting landmarks: %s", e.what());
////    }
////    
//    return result;
//}
////
//////- (void)dealloc {
////////    if (_graph) {
////////        auto status = _graph->CloseAllInputStreams();
////////        if (!status.ok()) {
////////            std::string error_msg(status.message());
////////            NSLog(@"Warning: Error closing input streams: %s", error_msg.c_str());
////////        }
////////        
////////        status = _graph->WaitUntilDone();
////////        if (!status.ok()) {
////////            std::string error_msg(status.message());
////////            NSLog(@"Warning: Error waiting for graph: %s", error_msg.c_str());
////////        }
////////    }
////////    _poller.reset();
////////    _graph.reset();
//////}
////
////
//@end
//
//#endif /* FaceLandmarkWrapper_h */
//    

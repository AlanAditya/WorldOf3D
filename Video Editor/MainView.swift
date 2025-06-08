//
//  MainView.swift
//  Video Editor
//
//  Created by Manoj Kumar on 29/01/25.
//

import SwiftUI
import UniformTypeIdentifiers
import AVFoundation

public struct MainView: View {
    public var body: some View {

            VSplitView {
                HSplitView {
                    Color.red
                    Color.blue
                    Color.green
                }
                TimeLineView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
    }
}

struct TimeLineView: View {
    @State private var droppedVideos: [VideoClip] = []  // Stores dropped videos
    @State private var scaleAnchor: UnitPoint = .center
    @State private var scale: Float = 1.0
    @State private var oldscale: Float = 1.0
    @State private var mouseLocation: CGPoint = .zero
    
    var body: some View {
        VStack {
            Text("TimeLine View")
                .font(.headline)
            
            ScrollView(.horizontal) {
                HStack {
                    ForEach(droppedVideos) { video in
                        VideoClipView(video: video, scale: $scale)
                    }
                }
                
                .frame(height: 100)
                .background(Color.black.opacity(0.2))
                .cornerRadius(8)
                .padding()
                .scaleEffect(x: CGFloat(scale), y: 1, anchor: scaleAnchor)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: 120)
        .background(
            GeometryReader { geometry in
                Color.blue
                    .contentShape(Rectangle())  // Ensures mouse tracking works
                    .onHover { hovering in
                        if hovering {
                            trackMouse(in: geometry)
                        }
                    }
            }
        )
        .onDrop(of: [.movie, .fileURL], isTargeted: nil) { providers in
            handleDrop(providers: providers)
        }

        .gesture(
            MagnificationGesture()
                .onChanged { value in
                    scale = Float(value) * oldscale
                }
                .onEnded { _ in
                    withAnimation {
                        oldscale = scale // Limits zoom between 1x and 3x
                    }
                }
        )

    }
    
    private func trackMouse(in geometry: GeometryProxy) {
        NSEvent.addLocalMonitorForEvents(matching: .mouseMoved) { event in
            DispatchQueue.main.async {
                let location = event.locationInWindow
                let localPosition = CGPoint(
                    x: location.x - geometry.frame(in: .global).origin.x,
                    y: location.y - geometry.frame(in: .global).origin.y
                )
                mouseLocation = localPosition
                print(mouseLocation)
            }
            return event
        }
    }

    // MARK: - Update Scale Anchor Based on Mouse Location
    private func updateScaleAnchor(from location: CGPoint) {
        scaleAnchor = UnitPoint(
            x: location.x / 1000,  // Normalize width (Assumes 1000px width, adjust as needed)
            y: 1.0 // Keeps Y fixed as timeline is horizontal
        )
    }
    
    // MARK: - Handle Drop Action
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) {
                provider.loadItem(forTypeIdentifier: UTType.movie.identifier, options: nil) { (item, error) in
                    if let url = item as? URL {
                        DispatchQueue.main.async {
                            let newVideo = VideoClip(id: UUID(), url: url)
                            droppedVideos.append(newVideo)
                        }
                    }
                }
            } else if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { (data, error) in
                    if let urlData = data as? Data,
                       let urlString = String(data: urlData, encoding: .utf8),
                       let url = URL(string: urlString) {
                        DispatchQueue.main.async {
                            let newVideo = VideoClip(id: UUID(), url: url)
                            droppedVideos.append(newVideo)
                        }
                    }
                }
            }
        }
        return true
    }
}

// MARK: - Video Representation (Rectangle in Timeline)
struct VideoClip: Identifiable {
    let id: UUID
    let name: String
    let url: URL
    let duration: Double  // Duration in seconds
    let thumbnail: NSImage?
    
    init(id: UUID = UUID(), url: URL) {
        self.id = id
        self.url = url
        self.name = url.lastPathComponent
        self.duration = VideoClip.getVideoDuration(from: url)
        self.thumbnail = VideoClip.generateThumbnail(for: url)
    }
    
    // MARK: - Get Video Duration
    private static func getVideoDuration(from url: URL) -> Double {
        let asset = AVAsset(url: url)
        return CMTimeGetSeconds(asset.duration)
    }
    
    // MARK: - Generate Video Thumbnail
    private static func generateThumbnail(for url: URL) -> NSImage? {
        let asset = AVAsset(url: url)
        let imageGenerator = AVAssetImageGenerator(asset: asset)
        imageGenerator.appliesPreferredTrackTransform = true
        
        do {
            let time = CMTime(seconds: 1, preferredTimescale: 600)  // Capture at 1s mark
            let cgImage = try imageGenerator.copyCGImage(at: time, actualTime: nil)
            return NSImage(cgImage: cgImage, size: NSSize(width: 100, height: 60))
        } catch {
            print("Thumbnail generation failed: \(error)")
            return nil
        }
    }
}

struct VideoClipView: View {
    let video: VideoClip
    @Binding var scale: Float
    let sensitivity: Float = 10
    
    var body: some View {
        VStack {
            if let thumbnail = video.thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .scaledToFit()
                    .frame(width: Double(1 * sensitivity) * video.duration, height: 60)
                    .cornerRadius(5)
            } else {
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: Double(1 * sensitivity) * video.duration, height: 60)
                    .cornerRadius(5)
            }
            
            Text(video.name)
                .font(.caption)
                .lineLimit(1)
            
            Text("\(Int(video.duration)) sec")
                .font(.caption2)
                .foregroundColor(.gray)
        }
        .cornerRadius(5)
        .padding(.horizontal, 0)
        .background(.blue)
        .frame(width: Double(1 * sensitivity) * video.duration, height: 60)
    }
}

@objc class MainViewWrapper: NSObject {
    @MainActor @objc func getView() -> NSView {
        return NSHostingView(rootView: MainView().body)
    }
}

@objcMembers public class MySwiftUIViewController: NSViewController {
    @objc public override func viewDidLoad() {
        super.viewDidLoad()
//        let wrappedNSView = NSViewWrapper(nsView: customView)
        let hostingController = NSHostingController(rootView: MainView().body)
        
        addChild(hostingController)
        view.addSubview(hostingController.view)
        hostingController.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hostingController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingController.view.topAnchor.constraint(equalTo: view.topAnchor),
            hostingController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
}

#Preview {
    MainView()
}

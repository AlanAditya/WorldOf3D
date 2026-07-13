//
//  ContentView.swift
//  AdityaIntelligenceProMax
//
//  Created by Manoj Kumar on 09/03/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject var inteligence = Intelligence(CGRect(x: 0, y: 0, width: 300, height: 300))
    @State var oldDrag: CGSize = .zero
    @State var panDrag: CGSize = .zero
    var body: some View {
        VStack {
            
            MTKViewWrapper(intel: inteligence, viewNo: 1)
//                .gesture(DragGesture().onChanged { value in
//                    print(value.translation)
//                    inteligence.updateCam(value.translation, TransformationMode.Orbit);
//                })
                .gesture(DragGesture()
                    .onChanged { value in
                        
                        inteligence.updateCam(CGSize(width: ((value.translation.width - oldDrag.width) / 1.0), height: ((value.translation.height - oldDrag.height) / 1.0)), TransformationMode.Orbit, 1)
                        oldDrag = value.translation
                    }
                    .onEnded { _ in
                        oldDrag = .zero
                        inteligence.updateCamOLD(1)
                    })
                .gesture(MagnificationGesture().onChanged { value in
                    print("Zoom scale: \(value)")
                    inteligence.updateCam(CGSize(width: value, height: value), TransformationMode.Zoom, 1)
                }
                .onEnded({_ in
                    print("Zoom endereço:")
                    inteligence.updateCamOLD(1);
                }))
//                .overlay(
//                    TwoFingerPanGesture { translation in
//                        print("Two-finger pan translation: \(translation)")
//                        // Call your updateCam with the pan transformation
//                        inteligence.updateCam(CGSize(width: translation.x, height: translation.y), TransformationMode.Translate)
//                    }
//                )
            MTKViewWrapper(intel: inteligence, viewNo: 2)

        }
        .onAppear {
            inteligence.memoryTestLogic()
        }
        .padding()
    }
}


#if os(macOS)
struct MTKViewWrapper: NSViewRepresentable {
    var intel: Intelligence
    var viewNo: Int;

    func makeNSView(context: Context) -> MTKView {
        if viewNo == 1 {
            return intel.view1
        } else {
            return intel.view2
        }
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
        // Update the view if needed.
    }
}
#endif

#if os(iOS)
struct MTKViewWrapper: UIViewRepresentable {
    var intel: Intelligence
    var viewNo: Int;
    func makeUIView(context: Context) -> MTKView {
        if viewNo == 1 {
            return intel.view1
        } else {
            return intel.view2
        }
        
    }
    
    func updateUIView(_ nsView: MTKView, context: Context) {
        
        // Update the view if needed.
    }
}
#endif

#Preview {
    ContentView()
}
#if os(iOS)
// Custom two‐finger pan gesture recognizer
class TwoFingerPanGestureRecognizer: UIPanGestureRecognizer {
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        if let allTouches = event.allTouches, allTouches.count != 2 {
            // If not exactly two touches, fail the gesture.
            self.state = .failed
        }
        super.touchesBegan(touches, with: event)
    }
}

// SwiftUI wrapper for the two‐finger pan gesture recognizer
struct TwoFingerPanGesture: UIViewRepresentable {
    var onPan: (CGPoint) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPan: onPan)
    }

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.backgroundColor = .clear

        let panGR = TwoFingerPanGestureRecognizer(target: context.coordinator,
                                                    action: #selector(Coordinator.handlePan(_:)))
        view.addGestureRecognizer(panGR)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Nothing to update.
    }

    class Coordinator: NSObject {
        var onPan: (CGPoint) -> Void

        init(onPan: @escaping (CGPoint) -> Void) {
            self.onPan = onPan
        }

        @objc func handlePan(_ gesture: TwoFingerPanGestureRecognizer) {
            if gesture.state == .changed {
                let translation = gesture.translation(in: gesture.view)
                onPan(translation)
            }
        }
    }
}
#endif

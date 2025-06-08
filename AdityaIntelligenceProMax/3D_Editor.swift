//
//  3D_Editor.swift
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 08/06/25.
//

import SwiftUI

struct _3D_Editor: View {
    @State var inteligence = Intelligence(CGRect(x: 0, y: 0, width: 300, height: 300))
    @State var oldDrag: CGSize = .zero
    @State var panDrag: CGSize = .zero
    @State var scrollEvent: Any?
    @State var keyMonitor: Any?
    @State var pressedKeys = Set<UInt16>()
    @State var cameraTimer: NSObject? = nil
    var body: some View {
        MTKViewWrapper(intel: inteligence!, viewNo: 1)
//            .gesture(DragGesture()
//                .onChanged { value in
//                    inteligence!.updateCam(CGSize(width: ((value.translation.width - oldDrag.width) / 1.0), height: ((value.translation.height - oldDrag.height) / 1.0)), TransformationMode.Orbit, 1)
//                    oldDrag = value.translation
//                }
//                .onEnded { _ in
//                    oldDrag = .zero
//                    inteligence!.updateCamOLD(1)
//                })
//            .gesture(MagnificationGesture().onChanged { value in
//                print("Zoom scale: \(value)")
//                inteligence!.updateCam(CGSize(width: value, height: value), TransformationMode.Zoom, 1)
//            }
//            .onEnded({_ in
//                print("Zoom endereço:")
//                inteligence!.updateCamOLD(1);
//            }))
            .onAppear {
                inteligence?.textTesselator()
//                scrollEvent = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel ) { scroll in
//                    inteligence!.updateCam(CGSize(width: scroll.scrollingDeltaX, height: scroll.scrollingDeltaY), TransformationMode.Translate, 1)
//                    return scroll
//                }
//                NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
//                    if (event.keyCode == 31) {
//                        inteligence?.updateCamProjection(false)
//                    } else if (event.keyCode == 35) {
//                        inteligence?.updateCamProjection(true)
//                        
//                    } else {
//                        pressedKeys.insert(event.keyCode)
//                    }
//                    
//                    return nil
//                }
//
//                NSEvent.addLocalMonitorForEvents(matching: [.keyUp]) { event in
//                    pressedKeys.remove(event.keyCode)
//                    return nil
//                }
//                // Use Timer instead of CADisplayLink
//                cameraTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 120.0, repeats: true) { _ in
//                    updateCamera()
//                }
//
////                keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
////                    switch event.keyCode {
////                    case 123:
////                        inteligence!.updateCam(CGSize(width: -10, height: 0), TransformationMode.Translate)
////                        return nil
////                    case 124:
////                        inteligence!.updateCam(CGSize(width: 10, height: 0), TransformationMode.Translate)
////                        return nil
////                    case 125:
////                        inteligence!.updateCam(CGSize(width: 0, height: 10), TransformationMode.Translate)
////                        return nil
////                    case 126:
////                        inteligence!.updateCam(CGSize(width: 0, height: -10), TransformationMode.Translate)
////                        return nil
////                    default:
////                        break
////                    }
////                    return event
//                }
            }
//        MTKViewWrapper(intel: inteligence!, viewNo: 2)
//            .gesture(DragGesture()
//                .onChanged { value in
//                    inteligence!.updateCam(CGSize(width: ((value.translation.width - oldDrag.width) / 1.0), height: ((value.translation.height - oldDrag.height) / 1.0)), TransformationMode.Orbit, 2)
//                    oldDrag = value.translation
//                }
//                .onEnded { _ in
//                    oldDrag = .zero
//                    inteligence!.updateCamOLD(2)
//                })
//            .gesture(MagnificationGesture().onChanged { value in
//                print("Zoom scale: \(value)")
//                inteligence!.updateCam(CGSize(width: value, height: value), TransformationMode.Zoom, 2)
//            }
//            .onEnded({_ in
//                print("Zoom endereço:")
//                inteligence!.updateCamOLD(2);
//            }))
    }
    
    func updateCamera() {
        var delta = CGSize.zero
        
        if pressedKeys.contains(123) { // Left
            delta.width -= 2
        }
        if pressedKeys.contains(124) { // Right
            delta.width += 2
        }
        if pressedKeys.contains(125) { // Down
            delta.height += 2
        }
        if pressedKeys.contains(126) { // Up
            delta.height -= 2
        }

        if delta != .zero {
            inteligence?.updateCam(delta, TransformationMode.Translate, 1)
        }
    }

    
}

#Preview {
    _3D_Editor()
}

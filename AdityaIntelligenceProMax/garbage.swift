//
//  garbage.swift
//  WorldOf3D
//
//  Created by Aditya Dudeja on 16/11/25.
//

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

//func updateCamera() {
//        var delta = CGSize.zero
//
//        if pressedKeys.contains(123) { // Left
//            delta.width -= 2
//        }
//        if pressedKeys.contains(124) { // Right
//            delta.width += 2
//        }
//        if pressedKeys.contains(125) { // Down
//            delta.height += 2
//        }
//        if pressedKeys.contains(126) { // Up
//            delta.height -= 2
//        }
//
//        if delta != .zero {
//            inteligence?.updateCam(delta, TransformationMode.Translate, 1)
//        }
//    }

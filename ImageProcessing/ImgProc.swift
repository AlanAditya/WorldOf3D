//
//  ImgProc.swift
//  ImageProcessing
//
//  Created by Manoj Kumar on 16/08/24.
//

import Cocoa
import Foundation
import SwiftUI
import PhotosUI
import AppKit
import MetalKit
import Metal
//import Cocoa


public struct ImagePickerView: View {
    @State private var selectedImages = [PhotosPickerItem]()
    @State private var isPresented = true
    weak private var host: NSViewController? = nil
    
    public init() {
        print("Hello World")
    }
    public var body: some View {
        Text("Loading")
            .photosPicker(isPresented: $isPresented, selection: $selectedImages)
            .onChange(of: selectedImages) {
//                host?.dismiss(<#T##NSViewController#>)
                Task {
                    for selection in selectedImages {
                        guard let data = try? await selection.loadTransferable(type: Data.self),
                              let uiImage = NSImage(data: data)
                        else { continue }
//                        self.loadImage(uiImage)
                    }
                }
            }
    }
}

//public struct PhotoPickerView: View {
//    @State public var image: CGImage?
//    @State public var isShowingPhotoPicker = false
//    
//    public init() { }
//    
//    public var body: some View {
//        VStack {
//            if  let image == image {
//                Image(image, scale: 1, label: Text("Img"))
//                    .resizable()
//                    .scaledToFit()
//                    .frame(maxWidth: 300, maxHeight: 300)
//            } else {
//                Text("No Image Selected")
//                    .font(.headline)
//            }
//
//            Button("Select Photo") {
//                isShowingPhotoPicker = true
//            }
//            .padding()
//            .fileImporter(
//                isPresented: $isShowingPhotoPicker,
//                allowedContentTypes: [.image],
//                allowsMultipleSelection: false
//            ) { result in
//                switch result {
//                case .success(let urls):
//                    if let url = urls.first {
////                        loadImage(from: url)
//                    }
//                case .failure(let error):
//                    print("Error selecting photo: \(error.localizedDescription)")
//                }
//            }
//        }
//        .frame(width: 400, height: 400)
//    }
//
//    func loadImage(from url: URL) {
//        if let nsImage = CGImageSourceCreateWithURL(url as CFURL, nil) {
//            self.image = CGImageSourceCreateImageAtIndex(nsImage, 0, nil)!
//        } else {
//            print("Failed to load image from URL: \(url)")
//        }
//    }
//}

public struct HelloView: View {
    weak private var host: NSViewController? = nil
    
    public init() {
        print("Hello World")
    }
    public var body: some View {
        Text("Hello")
            .frame(width: 100, height: 100)
    }
    
//    public func present() -> NSView {
//        var bodyB: some View {
//            Text("Hello")
//                .frame(width: 100, height: 100)
//        }
//        return NSHostingView(rootView: bodyB)
//    }
}

public struct HelloViewKKKKK: View {
    weak private var host: NSViewController? = nil
    
    public init() {
        print("Hello World")
    }
    public var body: some View {
        Text("Hello")
            .frame(width: 100, height: 100)
    }
    
//    public func present() -> NSView {
//        var bodyB: some View {
//            Text("Hello")
//                .frame(width: 100, height: 100)
//        }
//        return NSHostingView(rootView: bodyB)
//    }
}


//import SwiftUI
//
//@objc public class ImageProcessing: NSObject {
//
//}

struct NSViewWrapper: NSViewRepresentable {
    let nsView: NSView

    func makeNSView(context: Context) -> NSView {
        return nsView
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        // Update the view if needed
    }
}

struct itemView: View {
    @State var objec: Object3D
    var body: some View {
        Text(objec.shape.rawValue)
        VStack {
            HStack {
                subItemView(value: objec.location.x, title: "x")
                subItemView(value: objec.location.y, title: "y")
                subItemView(value: objec.location.z, title: "z")
            }
            HStack {
                subItemView(value: objec.rotation.x, title: "x")
                subItemView(value: objec.rotation.y, title: "y")
                subItemView(value: objec.rotation.z, title: "z")
            }
            HStack {
                subItemView(value: objec.scale.x, title: "x")
                subItemView(value: objec.scale.y, title: "y")
                subItemView(value: objec.scale.z, title: "z")
            }
        }
    }
}

struct subItemView: View {
    @State var value: Float
    let title: String
    var body: some View {
        TextField(title, text: Binding(
                get: {String(value)},
                set: { val in
                    if let FloatValue = Float(val) {
                        value = FloatValue
                    }
                }))
    }
}


struct SideBarContents: View {
    @State var ObjData: [Object3D]
    var body: some View {
        List {
            ForEach(ObjData, id: \.id) { objec in
                itemView(objec: objec)
            }
        }
    }
}


public struct MainWindows: View {
    @State var ObjDataMain: [Object3D]
    let customView: NSView
    public var body: some View {
        NavigationSplitView {
            SideBarContents(ObjData: ObjDataMain)
        } detail: {
            NSViewWrapper(nsView: customView)
        }

//        .frame(width: 400, height: 400)
    }
    
    public static func createView() -> NSView {
        let contentView = HelloView()
        let hostingView = NSHostingView(rootView: contentView.body)
        return hostingView
    }
}
extension NSData {
    // Method to create NSData from Object3D
    convenience init(object3D: Object3D) {
        var mutableObject3D = object3D
        self.init(bytes: &mutableObject3D, length: MemoryLayout<Object3D>.size)
    }
    
    // Method to retrieve Object3D from NSData
    var object3DValue: Object3D {
        var object = Object3D(shape: .cube) // Default initialization
        self.getBytes(&object, length: MemoryLayout<Object3D>.size)
        return object
    }
}
@objcMembers public class MySwiftUIViewController: NSViewController {
    let customView: NSView
    public var Data = [
        Object3D(shape: .cube),
        Object3D(shape: .sphere)
    ]
    
    @objc public var objcData: NSArray {
        print(Data[0].id)
        return Data.map { NSData(object3D: $0) } as NSArray
    }
    
    @objc init(view: NSView) {
        self.customView = view
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    @objc public override func viewDidLoad() {
        super.viewDidLoad()
//        let wrappedNSView = NSViewWrapper(nsView: customView)
        let swiftUIView = MainWindows(ObjDataMain: Data, customView: customView)
        let hostingController = NSHostingController(rootView: swiftUIView)
        
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


//
//@objc public class MySwiftUIViewControllerBridge: NSObject {
//    
//    @objc public static func createSwiftUIViewControllerWithView(_ view: NSView) -> NSView {
//        return MySwiftUIViewController(view: view).view
//    }
//}

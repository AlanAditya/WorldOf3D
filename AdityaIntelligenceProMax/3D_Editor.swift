//
//  3D_Editor.swift
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 08/06/25.
//

import SwiftUI

//
//struct ComponentList: RandomAccessCollection, ObservableObject {
//    typealias Element = UIComponent
//    typealias Index = Int
//
//    let intel: Intelligence
//    
//    @Published let array: NSMutableArray
//    
//    var startIndex: Int { 0 }
//    var endIndex: Int { array.count }
//    
//    private var cancellable: AnyCancellable?
//    
//    init(intel: Intelligence) {
//        self.intel = intel
//        self.array = intel.components
//    }
//    subscript(position: Int) -> UIComponent {
//        array.object(at: position) as! UIComponent
//    }
//}
import Combine

// Make this a class so ObservableObject works
final class ComponentList: ObservableObject, RandomAccessCollection {
    typealias Element = UIComponent
    typealias Index = Int

    let intel: Intelligence                      // Objective-C owner (must be NSObject/KVO-compliant)
    @Published private(set) var items: [UIComponent] = []

    // RandomAccessCollection requirements
    var startIndex: Int { items.startIndex }
    var endIndex: Int { items.endIndex }
    var count: Int { items.count }
    subscript(position: Int) -> UIComponent { items[position] }

    private var cancellable: AnyCancellable?

    init(intel: Intelligence) {
        self.intel = intel
        // initial snapshot from NSMutableArray -> [UIComponent]
        if let comps = intel.components as? [UIComponent] {
            self.items = comps
        } else if let nsarr = intel.components as? NSMutableArray {
            self.items = nsarr.compactMap { $0 as? UIComponent }
        }

        // KVO publisher: observe the Obj-C property 'components'.
        // This requires 'components' be KVO-compliant (NSObject subclass and exposed as @property).
        cancellable = intel.publisher(for: \.components)
            .receive(on: DispatchQueue.main)   // UI updates on main
            .sink { [weak self] newVal in
                guard let self = self else { return }
                // copy into Swift array snapshot
                if let arr = newVal as? [UIComponent] {
                    self.items = arr
                } else if let nsarr = newVal as? NSMutableArray {
                    self.items = nsarr.compactMap { $0 as? UIComponent }
                } else {
                    self.items = []
                }
            }
    }

    // Helper to update a UIComponent property and notify Obj-C owner.
    // This ensures the Obj-C array is mutated in a KVO-friendly way.
    func updateValue(at index: Int, newValue: Float) {
        guard index >= 0 && index < items.count else { return }
        let comp = items[index]

        // mutate the underlying Obj-C model in a KVO-friendly way.
        // Ideally Intelligence exposes methods to mutate its components so it wraps willChange/didChange.
        // Example call:
        intel.willChangeValue(forKey: "components")
        comp.value = newValue
        // if the Obj-C container needs actual replacement, update it:
        // intel.components.replaceObject(at: index, with: comp)
        intel.didChangeValue(forKey: "components")
        // local snapshot will be refreshed from KVO sink once didChange fires.
        // Optionally update the local snapshot immediately for smoother UI:
        var copy = items
        copy[index] = comp
        self.items = copy
    }
}

struct ControlPanel: View {
    let intel: Intelligence  // Just use 'let' for reference types
    @StateObject var list: ComponentList
    init(intel: Intelligence) {
        self.intel = intel
        _list = StateObject(wrappedValue: ComponentList(intel: intel))
    }

    
    var body: some View {
        List {
            ForEach(0..<list.count, id: \.self) { i in
                let c = list[i]
                if c.type == 0 {
                    Text("\(c.title): \(c.value, specifier: "%.2f")")
                    Slider(
                        value: Binding(
                            get: { c.value },
                            set: { newValue in
                                c.value = newValue
                                c.valueChangedCallback(newValue);
                                list.updateValue(at: i, newValue: newValue)
                                                            print(c.value)
                            }
                        ),
                        in: c.minValue ... c.maxValue,
                    )
                    .compositingGroup()
                    
                }
            }
        }
        
    }
}

#if os(macOS)
struct NStoSwift: NSViewRepresentable {
    let nsView: NSView
    func makeNSView(context: Context) -> NSView {
        return nsView
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        
        // Update the view if needed.
    }
}
#endif

#if os(iOS)
struct NStoSwift: UIViewRepresentable {
    let nsView: UIView
    func makeUIView(context: Context) -> UIView {
        return nsView
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        
        // Update the view if needed.
    }
}
#endif
struct _3D_Editor: View {
    @State var inteligence = Intelligence(CGRect(x: 0, y: 0, width: 300, height: 300))
    @State var oldDrag: CGSize = .zero
    @State var panDrag: CGSize = .zero
    @State var scrollEvent: Any?
    @State var keyMonitor: Any?
    @State var pressedKeys = Set<UInt16>()
    @State var cameraTimer: NSObject? = nil
    @State var pos: CGFloat = 0
    var body: some View {
        #if os(macOS)
        HSplitView {
//            MTKViewWrapper(intel: inteligence!, viewNo: 2)
            MTKViewWrapper(intel: inteligence!, viewNo: 1)
                .layoutPriority(1)
                .onAppear {
//                    let matrix = MatrixH<_CInt_1, CFloat>([1.0, 2.0, 3.0] as [CFloat])
                    
                    inteligence?.vec_field()
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
            VSplitView {
                
                NStoSwift(nsView: inteligence!.sidePanel.view)
                ControlPanel(intel: inteligence!)
                    .frame(minWidth: 220, maxWidth: .infinity)
                    .compositingGroup()
                
//                    .layoutPriority(1)
//                ContentViewBenchmark()
            }
        
        }
#endif
#if os(iOS)
MTKViewWrapper(intel: inteligence!, viewNo: 1)
    .layoutPriority(1)
    .onAppear {
        //                    let matrix = MatrixH<_CInt_1, CFloat>([1.0, 2.0, 3.0] as [CFloat])
        
        inteligence?.iOS_Depth()
    }
#endif
    }
}

#Preview {
    _3D_Editor()
}

#if os(macOS)
@objc class SwiftTextHost : NSObject {
    @objc static func make(_ text: String) -> NSView {
        let host = NSHostingView(
            rootView:
                Text(text)
                    .font(.system(size: 14))
                    .frame(maxWidth: .infinity, alignment: .leading)   // prevents centering
                    .fixedSize(horizontal: false, vertical: true)     // natural height
                    .padding(.vertical, 4)                             // breathing space
        )

        host.translatesAutoresizingMaskIntoConstraints = false

        // Prevent HostingView from expanding vertically
        host.setContentCompressionResistancePriority(.required, for: .vertical)
        host.setContentHuggingPriority(.required, for: .vertical)

        return host
    }
}


struct BenchmarkView: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let viewC = Benchmark()
        return viewC.view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        // Nothing: benchmark runs only once
    }
}

let viewCount = 5000
let iterations = 50

struct Item: Identifiable {
    let id = UUID()
    var text: String
}

@MainActor
final class Model: ObservableObject {
    @Published var items = (0..<viewCount).map { _ in Item(text: "Test") }

    func runBenchmark() {
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            for _ in 0..<500 {
                let idx = Int.random(in: 0..<viewCount)
                items[idx].text = "Updated"
            }
        }

        let end = CFAbsoluteTimeGetCurrent()
        print("SwiftUI Benchmark: \(end - start) s")
    }
}

struct ContentViewBenchmark: View {
    @StateObject var model = Model()

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading) {
                ForEach(model.items) { item in
                    Text(item.text)
                        .frame(height: 18, alignment: .topLeading)
                }
            }
        }
        .onAppear { model.runBenchmark() }
    }
}
#endif

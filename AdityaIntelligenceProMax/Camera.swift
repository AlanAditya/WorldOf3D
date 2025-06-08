//import AppKit
//import Metal
//import MetalKit
//import simd
//
//// MARK: - Camera System
//class Camera {
//    var position: simd_float3 = simd_float3(0, 5, 10)
//    var target: simd_float3 = simd_float3(0, 0, 0)
//    var up: simd_float3 = simd_float3(0, 1, 0)
//    
//    // Orbit parameters
//    var distance: Float = 10.0
//    var azimuth: Float = 0.0  // Horizontal rotation
//    var elevation: Float = 0.3 // Vertical rotation (radians)
//    
//    // Camera limits
//    let minDistance: Float = 0.10
//    let maxDistance: Float = 100.0
//    let minElevation: Float = -Float.pi / 2 + 0.1
//    let maxElevation: Float = Float.pi / 2 - 0.1
//    
//    func updatePosition() {
//        // Convert spherical coordinates to cartesian
//        let x = distance * cos(elevation) * cos(azimuth)
//        let y = distance * sin(elevation)
//        let z = distance * cos(elevation) * sin(azimuth)
//        
//        position = target + simd_float3(x, y, z)
//    }
//    
//    func getViewMatrix() -> simd_float4x4 {
//        return lookAt(eye: position, center: target, up: up)
//    }
//    
//    func orbit(deltaAzimuth: Float, deltaElevation: Float) {
//        azimuth += deltaAzimuth
//        elevation -= deltaElevation
//        
//        // Clamp elevation to prevent flipping
//        elevation = max(minElevation, min(maxElevation, elevation))
//        
//        updatePosition()
//    }
//    
//    func zoom(factor: Float) {
//        distance *= factor
////        distance = max(minDistance, min(maxDistance, distance))
//        updatePosition()
//    }
//    
//    func pan(deltaX: Float, deltaY: Float) {
//        // Get camera's right and up vectors
//        let forward = normalize(target - position)
//        let right = normalize(cross(forward, up))
//        let upVector = cross(right, forward)
//        
//        // Move target and position together
//        let panVector = right * deltaX + upVector * deltaY
//        target += panVector
//        updatePosition()
//    }
//}
//
//// MARK: - Matrix Math Utilities
//func lookAt(eye: simd_float3, center: simd_float3, up: simd_float3) -> simd_float4x4 {
//    let z = normalize(eye - center)
//    let x = normalize(cross(up, z))
//    let y = cross(z, x)
//    
//    let translation = simd_float4x4(
//        [1, 0, 0, 0],
//        [0, 1, 0, 0],
//        [0, 0, 1, 0],
//        [-dot(x, eye), -dot(y, eye), -dot(z, eye), 1]
//    )
//    
//    let rotation = simd_float4x4(
//        [x.x, y.x, z.x, 0],
//        [x.y, y.y, z.y, 0],
//        [x.z, y.z, z.z, 0],
//        [0, 0, 0, 1]
//    )
//    
//    return rotation * translation
//}
//
//func perspectiveProjection(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
//    let f = 1.0 / tan(fovy * 0.5)
//    return simd_float4x4(
//        [f / aspect, 0, 0, 0],
//        [0, f, 0, 0],
//        [0, 0, (far + near) / (near - far), -1],
//        [0, 0, (2 * far * near) / (near - far), 0]
//    )
//}
//
//// Matrix inverse implementation
//func matrixInverse(_ matrix: simd_float4x4) -> simd_float4x4 {
//    let m = matrix
//    
//    // Calculate 2x2 determinants for the first row
//    let a2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2]
//    let a1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1]
//    let a1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1]
//    let a0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0]
//    let a0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0]
//    let a0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0]
//    let a2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2]
//    let a1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1]
//    let a1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1]
//    let a2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2]
//    let a1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1]
//    let a1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
//    let a0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0]
//    let a0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0]
//    let a0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0]
//    let a0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0]
//    let a0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0]
//    let a0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0]
//    
//    // Calculate the determinant
//    let det = m[0][0] * (m[1][1] * a2323 - m[1][2] * a1323 + m[1][3] * a1223) -
//              m[0][1] * (m[1][0] * a2323 - m[1][2] * a0323 + m[1][3] * a0223) +
//              m[0][2] * (m[1][0] * a1323 - m[1][1] * a0323 + m[1][3] * a0123) -
//              m[0][3] * (m[1][0] * a1223 - m[1][1] * a0223 + m[1][2] * a0123)
//    
//    if abs(det) < 1e-8 {
//        // Return identity matrix if determinant is too small
//        return simd_float4x4(1.0)
//    }
//    
//    let invDet = 1.0 / det
//    
//    return simd_float4x4(
//        [
//            invDet * (m[1][1] * a2323 - m[1][2] * a1323 + m[1][3] * a1223),
//            invDet * -(m[0][1] * a2323 - m[0][2] * a1323 + m[0][3] * a1223),
//            invDet * (m[0][1] * a2313 - m[0][2] * a1313 + m[0][3] * a1213),
//            invDet * -(m[0][1] * a2312 - m[0][2] * a1312 + m[0][3] * a1212)
//        ],
//        [
//            invDet * -(m[1][0] * a2323 - m[1][2] * a0323 + m[1][3] * a0223),
//            invDet * (m[0][0] * a2323 - m[0][2] * a0323 + m[0][3] * a0223),
//            invDet * -(m[0][0] * a2313 - m[0][2] * a0313 + m[0][3] * a0213),
//            invDet * (m[0][0] * a2312 - m[0][2] * a0312 + m[0][3] * a0212)
//        ],
//        [
//            invDet * (m[1][0] * a1323 - m[1][1] * a0323 + m[1][3] * a0123),
//            invDet * -(m[0][0] * a1323 - m[0][1] * a0323 + m[0][3] * a0123),
//            invDet * (m[0][0] * a1313 - m[0][1] * a0313 + m[0][3] * a0113),
//            invDet * -(m[0][0] * a1312 - m[0][1] * a0312 + m[0][3] * a0112)
//        ],
//        [
//            invDet * -(m[1][0] * a1223 - m[1][1] * a0223 + m[1][2] * a0123),
//            invDet * (m[0][0] * a1223 - m[0][1] * a0223 + m[0][2] * a0123),
//            invDet * -(m[0][0] * a1213 - m[0][1] * a0213 + m[0][2] * a0113),
//            invDet * (m[0][0] * a1212 - m[0][1] * a0212 + m[0][2] * a0112)
//        ]
//    )
//}
//
//// MARK: - Shader Uniforms
//struct Uniforms {
//    var modelViewProjectionMatrix: simd_float4x4
//    var inverseModelViewProjectionMatrix: simd_float4x4
//    var cameraPosition: simd_float3
//    var gridScale: Float
//}
//
//// MARK: - Metal Renderer
//class InfiniteGridRenderer: NSObject {
//    private var device: MTLDevice!
//    private var commandQueue: MTLCommandQueue!
//    private var renderPipelineState: MTLRenderPipelineState!
//    private var uniformBuffer: MTLBuffer!
//    private var depthStencilState: MTLDepthStencilState!
//    
//    private let camera = Camera()
//    private var gridScale: Float = 1.0
//    
//    override init() {
//        super.init()
//        setupMetal()
//    }
//    
//    private func setupMetal() {
//        // Initialize Metal device and command queue
//        guard let device = MTLCreateSystemDefaultDevice() else {
//            fatalError("Metal is not supported on this device")
//        }
//        self.device = device
//        self.commandQueue = device.makeCommandQueue()!
//        
//        // Create uniform buffer
//        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride,
//                                        options: .storageModeShared)!
//        
//        // Setup render pipeline and depth state
//        setupRenderPipeline()
//        setupDepthState()
//    }
//    
//    private func setupRenderPipeline() {
//        // Define the shader source directly in Swift
//        let shaderSource = """
//        #include <metal_stdlib>
//        using namespace metal;
//
//        struct Uniforms {
//            float4x4 modelViewProjectionMatrix;
//            float4x4 inverseModelViewProjectionMatrix;
//            float3 cameraPosition;
//            float gridScale;
//        };
//
//        struct VertexOut {
//            float4 position [[position]];
//            float3 worldPosition;
//            float3 viewDirection;
//        };
//
//        vertex VertexOut gridVertexShader(uint vertexID [[vertex_id]],
//                                         constant Uniforms& uniforms [[buffer(0)]]) {
//            // Generate fullscreen quad vertices
//            float2 positions[4] = {
//                float2(-1.0, -1.0),
//                float2( 1.0, -1.0),
//                float2(-1.0,  1.0),
//                float2( 1.0,  1.0)
//            };
//            
//            float2 pos = positions[vertexID];
//            
//            VertexOut out;
//            out.position = float4(pos, 0.0, 1.0);
//            
//            // Calculate world position by unprojecting screen coordinates
//            float4x4 invMVP = uniforms.inverseModelViewProjectionMatrix;
//            float4 nearPoint = invMVP * float4(pos.x, pos.y, -1.0, 1.0);
//            float4 farPoint = invMVP * float4(pos.x, pos.y, 1.0, 1.0);
//            
//            nearPoint /= nearPoint.w;
//            farPoint /= farPoint.w;
//            
//            out.viewDirection = normalize(farPoint.xyz - nearPoint.xyz);
//            out.worldPosition = nearPoint.xyz;
//            
//            return out;
//        }
//
//        fragment float4 gridFragmentShader(VertexOut in [[stage_in]],
//                                          constant Uniforms& uniforms [[buffer(0)]]) {
//            // Ray-plane intersection to find ground position
//            float3 rayDir = normalize(in.viewDirection);
//            float3 rayOrigin = uniforms.cameraPosition;
//            
//            // Intersect with XZ plane (y = 0)
//            float t = -rayOrigin.y / rayDir.y;
//            
//            if (t < 0.0) {
//                discard_fragment();
//            }
//            
//            float3 worldPos = rayOrigin + rayDir * t;
//            
//            // Scale the grid
//            float2 coord = worldPos.xz * uniforms.gridScale;
//            
//            // Create grid lines using derivative-based anti-aliasing
//            float2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
//            float line = min(grid.x, grid.y);
//            
//            // Major grid lines every 10 units
//            float2 majorGrid = abs(fract(coord * 0.1 - 0.5) - 0.5) / fwidth(coord * 0.1);
//            float majorLine = min(majorGrid.x, majorGrid.y);
//            
//            // Create axis lines (X and Z axes in different colors)
//            float xAxis = abs(worldPos.z) < 0.1 ? 1.0 : 0.0;
//            float zAxis = abs(worldPos.x) < 0.1 ? 1.0 : 0.0;
//            
//            // Combine lines
//            float gridIntensity = 1.0 - min(line, 1.0);
//            float majorGridIntensity = 1.0 - min(majorLine, 1.0);
//            
//            // Distance-based fade
//            float distance = length(worldPos - uniforms.cameraPosition);
//            float fade = exp(-distance * 0.02);
//            float nearFade = smoothstep(0.0, 5.0, distance); // Fade out very close grid
//            
//            // Grid colors
//            float3 gridColor = float3(0.3, 0.3, 0.4);
//            float3 majorGridColor = float3(0.5, 0.5, 0.6);
//            float3 xAxisColor = float3(1.0, 0.3, 0.3); // Red for X axis
//            float3 zAxisColor = float3(0.3, 0.3, 1.0); // Blue for Z axis
//            
//            // Combine colors
//            float3 finalColor = gridColor;
//            float totalAlpha = gridIntensity;
//            
//            // Add major grid
//            if (majorGridIntensity > 0.0) {
//                finalColor = mix(finalColor, majorGridColor, majorGridIntensity * 0.7);
//                totalAlpha = max(totalAlpha, majorGridIntensity * 0.8);
//            }
//            
//            // Add axis lines
//            if (xAxis > 0.0) {
//                finalColor = mix(finalColor, xAxisColor, 0.8);
//                totalAlpha = max(totalAlpha, 0.9);
//            }
//            if (zAxis > 0.0) {
//                finalColor = mix(finalColor, zAxisColor, 0.8);
//                totalAlpha = max(totalAlpha, 0.9);
//            }
//            
//            // Apply fading
//            float alpha = totalAlpha * fade * nearFade;
//            
//            // Ensure minimum visibility for main axes
//            if (xAxis > 0.0 || zAxis > 0.0) {
//                alpha = max(alpha, 0.3 * fade);
//            }
//            
//            return float4(finalColor, alpha);
//        }
//        """
//        
//        // Create library from source
//        do {
//            let library = try device.makeLibrary(source: shaderSource, options: nil)
//            let vertexFunction = library.makeFunction(name: "gridVertexShader")!
//            let fragmentFunction = library.makeFunction(name: "gridFragmentShader")!
//            
//            let pipelineDescriptor = MTLRenderPipelineDescriptor()
//            pipelineDescriptor.vertexFunction = vertexFunction
//            pipelineDescriptor.fragmentFunction = fragmentFunction
//            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
//            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
//            
//            // Enable blending for grid fade effect
//            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
//            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
//            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
//            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
//            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
//            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
//            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
//            
//            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
//        } catch {
//            fatalError("Failed to create render pipeline state: \(error)")
//        }
//    }
//    
//    private func setupDepthState() {
//        let depthStateDescriptor = MTLDepthStencilDescriptor()
//        depthStateDescriptor.depthCompareFunction = .less
//        depthStateDescriptor.isDepthWriteEnabled = true
//        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)
//    }
//    
//    func render(in view: MTKView) {
//        guard let drawable = view.currentDrawable,
//              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
//        
//        // Clear to dark background
//        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)
//        renderPassDescriptor.colorAttachments[0].loadAction = .clear
//        
//        // Update uniforms
//        updateUniforms(viewSize: view.drawableSize)
//        
//        // Create command buffer and encoder
//        let commandBuffer = commandQueue.makeCommandBuffer()!
//        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
//        
//        // Set pipeline state and uniforms
//        renderEncoder.setRenderPipelineState(renderPipelineState)
//        renderEncoder.setDepthStencilState(depthStencilState)
//        renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 0)
//        renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
//        
//        // Draw fullscreen quad (grid will be generated in shader)
//        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
//        
//        renderEncoder.endEncoding()
//        commandBuffer.present(drawable)
//        commandBuffer.commit()
//    }
//    
//    private func updateUniforms(viewSize: CGSize) {
//        let aspect = Float(viewSize.width / viewSize.height)
//        let projectionMatrix = perspectiveProjection(fovy: Float.pi / 4, aspect: aspect, near: 0.1, far: 1000.0)
//        let viewMatrix = camera.getViewMatrix()
//        let mvpMatrix = projectionMatrix * viewMatrix
//        let invMVPMatrix = matrixInverse(mvpMatrix)
//        
//        let uniformsPointer = uniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
//        uniformsPointer.pointee.modelViewProjectionMatrix = mvpMatrix
//        uniformsPointer.pointee.inverseModelViewProjectionMatrix = invMVPMatrix
//        uniformsPointer.pointee.cameraPosition = camera.position
//        uniformsPointer.pointee.gridScale = gridScale
//    }
//    
//    // MARK: - Mouse and Gesture Handling
//    func handleMouseDrag(deltaX: Float, deltaY: Float, isRightMouseButton: Bool, isShiftPressed: Bool) {
//        let sensitivity: Float = 0.01
//        
//        if isRightMouseButton || isShiftPressed {
//            // Pan camera
//            camera.pan(deltaX: -deltaX * sensitivity, deltaY: deltaY * sensitivity)
//        } else {
//            // Orbit camera
//            camera.orbit(deltaAzimuth: deltaX * sensitivity, deltaElevation: -deltaY * sensitivity)
//        }
//    }
//    
//    func handleScrollWheel(deltaY: Float) {
//        let zoomFactor = 1.0 + (deltaY * 0.01)
//        camera.zoom(factor: zoomFactor)
//    }
//    
//    func handleMagnification(magnification: Float) {
//        gridScale *= (1.0 + magnification * 0.1)
//        gridScale = max(0.1, min(10.0, gridScale))
//    }
//}
//
//// MARK: - Custom Metal View for macOS
//class InfiniteGridMetalView: MTKView {
//    private var renderer: InfiniteGridRenderer!
//    private var trackingArea: NSTrackingArea?
//    
//    override init(frame frameRect: NSRect, device: MTLDevice?) {
//        super.init(frame: frameRect, device: device)
//        setupView()
//    }
//    
//    required init(coder: NSCoder) {
//        super.init(coder: coder)
//        setupView()
//    }
//    
//    private func setupView() {
//        device = MTLCreateSystemDefaultDevice()
//        colorPixelFormat = .bgra8Unorm
//        depthStencilPixelFormat = .depth32Float
//        clearDepth = 1.0
//        
//        renderer = InfiniteGridRenderer()
//        delegate = renderer
//        
//        // Setup tracking area for mouse events
//        updateTrackingAreas()
//    }
//    
//    override func updateTrackingAreas() {
//        super.updateTrackingAreas()
//        
//        if let trackingArea = trackingArea {
//            removeTrackingArea(trackingArea)
//        }
//        
//        trackingArea = NSTrackingArea(
//            rect: bounds,
//            options: [.activeInKeyWindow, .mouseMoved, .enabledDuringMouseDrag],
//            owner: self,
//            userInfo: nil
//        )
//        
//        if let trackingArea = trackingArea {
//            addTrackingArea(trackingArea)
//        }
//    }
//    
//    // MARK: - Mouse Event Handling
//    override func mouseDragged(with event: NSEvent) {
//        let deltaX = Float(event.deltaX)
//        let deltaY = Float(event.deltaY)
//        let isShiftPressed = event.modifierFlags.contains(.shift)
//        
//        renderer.handleMouseDrag(deltaX: deltaX, deltaY: deltaY, isRightMouseButton: false, isShiftPressed: isShiftPressed)
//    }
//    
//    override func rightMouseDragged(with event: NSEvent) {
//        let deltaX = Float(event.deltaX)
//        let deltaY = Float(event.deltaY)
//        
//        renderer.handleMouseDrag(deltaX: deltaX, deltaY: deltaY, isRightMouseButton: true, isShiftPressed: false)
//    }
//    
//    override func scrollWheel(with event: NSEvent) {
//        let deltaY = Float(event.scrollingDeltaY)
//        renderer.handleScrollWheel(deltaY: deltaY)
//    }
//    
//    override func magnify(with event: NSEvent) {
//        let magnification = Float(event.magnification)
//        renderer.handleMagnification(magnification: magnification)
//    }
//    
//    // Accept first responder to receive key events
//    override var acceptsFirstResponder: Bool {
//        return true
//    }
//}
//
//// MARK: - MTKViewDelegate Extension
//extension InfiniteGridRenderer: MTKViewDelegate {
//    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
//        // Handle resize if needed
//    }
//    
//    func draw(in view: MTKView) {
//        render(in: view)
//    }
//}
//
//// MARK: - Window Controller
//class InfiniteGridWindowController: NSWindowController {
//    convenience init() {
//        let window = NSWindow(
//            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
//            styleMask: [.titled, .closable, .miniaturizable, .resizable],
//            backing: .buffered,
//            defer: false
//        )
//        
//        self.init(window: window)
//        
//        window.title = "Infinite Grid - Metal Renderer"
//        window.center()
//        
//        let metalView = InfiniteGridMetalView(frame: window.contentView!.bounds)
//        metalView.autoresizingMask = [.width, .height]
//        window.contentView = metalView
//        
//        // Add instructions to window
//        addInstructionsToWindow()
//    }
//    
//    private func addInstructionsToWindow() {
//        guard let window = window else { return }
//        
//        let instructionsView = NSTextView(frame: NSRect(x: 10, y: 10, width: 300, height: 120))
//        instructionsView.string = """
//        Controls:
//        • Left mouse drag: Orbit camera
//        • Right mouse drag or Shift+drag: Pan camera
//        • Scroll wheel: Zoom in/out
//        • Trackpad pinch: Adjust grid scale
//        """
//        instructionsView.isEditable = false
//        instructionsView.backgroundColor = NSColor.black.withAlphaComponent(0.7)
//        instructionsView.textColor = NSColor.white
//        instructionsView.font = NSFont.systemFont(ofSize: 12)
//        
//        window.contentView?.addSubview(instructionsView)
//    }
//}
//
//// MARK: - App Delegate
//class AppDelegate: NSObject, NSApplicationDelegate {
//    var windowController: InfiniteGridWindowController?
//    
//    func applicationDidFinishLaunching(_ notification: Notification) {
//        windowController = InfiniteGridWindowController()
//        windowController?.showWindow(nil)
//    }
//    
//    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
//        return true
//    }
//}
//
//// MARK: - Main Entry Point
//
////class Camera3D {
////    simd_float3 oldPosition = {0.0, -5.0, -10.0};
////
////
////    simd_float3 up = {0, 1, 0};
////    simd_float3 right = {1, 0, 0};
////    simd_float3 forward = {0, 0, 1};
////    
////    float azimuthalAngle = 0;
////    float polarAngle = 0;
////    
////
////    
////    float r = 1;
////    float d = 1;
////    float aspectRatio = 1;
////    
////    // Camera Limits
////    float farP = 1000;
////    float nearP = 0.1;
////    float minPolarAngle = -M_PI / 2 + 0.1;
////    float maxPolarAngle = M_PI / 2 - 0.1;
////    
////    bool AxisUpdated = false;
////    float scale = 1;
////
////    float sensitivity = 1;
////public:
////    bool updated = false;
////    bool OrthoPara = 1;
////    simd_float3 position = {0.0, -5.0, -10.0};
////    simd_float3 target = {0, 0, 0};
////    simd_float4x4 viewMatrix;
////    Camera3D() {
////        simd_float3 vec = target - position;
////        
////        forward = simd::normalize(vec);
////        right = simd::normalize(simd::cross(forward, up));
////        up = simd::normalize(simd::cross(right, forward));
////        AxisUpdated = true;
////    }
////    
////    void updateOLD() {
////        oldPosition = position;
////        scale = 1;
////    }
////    
////    void updateMatrix() {
////        if (!updated) {
////            simd_float3 zAxis = simd::normalize( target - position );
////            simd_float3 xAxis = simd::normalize(simd::cross(up, zAxis));
////
////            simd_float3 yAxis = simd::normalize(simd::cross(zAxis, xAxis));
////            
////            // dot is used because first the rotation is undone then
////            simd_float4 row0 = {xAxis.x, xAxis.y, xAxis.z, -simd::dot(xAxis, position)};
////            simd_float4 row1 = {yAxis.x, yAxis.y, yAxis.z, -simd::dot(yAxis, position)};
////            simd_float4 row2 = {zAxis.x, zAxis.y, zAxis.z, -simd::dot(zAxis, position)};
////            simd_float4 row3 = {0,      0,      0,      1         };
////
////            if (OrthoPara) {
////                d = tan(0.5 * M_PI * 0.25);
////                
////                r = d * aspectRatio;
////                
////                float k = farP - nearP;
////                
////                simd_float4 row4 = {1/r,  0,        0,                           0};
////                simd_float4 row5 = {0,        1/d,  0,                           0};
////                simd_float4 row6 = {0,        0,        (farP+nearP)/-k,  -2 * nearP * farP * (1/k)};
////                simd_float4 row7 = {0.0f,     0.0f,     -1.0f,                        0};
////                
////                simd_float4x4 lookat =  simd_matrix_from_rows(row0, row1, row2, row3);
////                simd_float4x4 Perspective = simd_matrix_from_rows(row4, row5, row6, row7);
////                
////                viewMatrix = simd_mul(Perspective, lookat);
////                
////                updated = true;
////            } else {
////                float k = farP - nearP;
////                simd_float4 row4 = {2/r,  0,    0,    0};
////                simd_float4 row5 = {0,    2/d,  0,    0};
////                simd_float4 row6 = {0,    0,    1/k,  -nearP * (1/k)};
////                simd_float4 row7 = {0.0f, 0.0f, 0.0f, static_cast<float>(0.33 * simd_length(target - position))};
////                simd_float4x4 lookat =  simd_matrix_from_rows(row0, row1, row2, row3);
////                simd_float4x4 Orthographic = simd_matrix_from_rows(row4, row5, row6, row7);
////                viewMatrix = simd_mul(Orthographic, lookat);
////                updated = true;
////            }
//// 
////        }
////    }
////    
////
////    void handleMouseEvents(float deltaX, float deltaY, bool isRightMouseButton, bool isShiftPressed, TransformationMode eventType) {
////        float sensitivity = 0.01f;
////        
////        if (isRightMouseButton || eventType == TransformationMode::Translate) {
////            // Pan camera
////            if (!AxisUpdated) {
////                simd_float3 vec = target - position;
////                
////                forward = simd::normalize(vec);
////                right = simd::normalize(simd::cross(forward, up));
////                up = simd::normalize(simd::cross(right, forward));
////                AxisUpdated = true;
////            }
////            auto dist = 1;
////            target += (up * deltaY + right * deltaX) * sensitivity * sensitivity * dist;
////            updateOLD();
////        } else if (eventType == TransformationMode::Orbit){
////            // Orbit camera
////            azimuthalAngle += deltaY * sensitivity;
////            polarAngle     += deltaX * sensitivity;
////
////            // Update up vector based on azimuthal angle
////            if (std::cos(azimuthalAngle) < 0.0f) {
////                up = simd::float3{0, -1, 0};
////            } else {
////                up = simd::float3{0, 1, 0};
////            }
////
////            // Calculate camera distance
////            float distance = simd::length(target - position);
////
////            // Optionally clamp polar angle
////            // polarAngle = std::fmax(0.01f, std::fmin(static_cast<float>(M_PI) - 0.01f, polarAngle));
////            std::cout << up << "\n";
////            // Calculate new camera position
////            float x = target.x + distance * std::sin(polarAngle) * std::cos(azimuthalAngle) ;
////            float y = target.y + distance * std::sin(azimuthalAngle) ;
////            float z = target.z + distance * -std::cos(polarAngle) * std::cos(azimuthalAngle);
////
////            // Update camera position
////            position = simd::float3{x, y, z};
////            updateOLD(); // Assuming it's defined
////            AxisUpdated = false;
////        } else if (eventType == TransformationMode::Zoom) {
////            position = target + (oldPosition - target) * (1 / deltaX );
////        }
////        updated = false;
////        
////    }
////    
////    void updateAspectRatio(float ratio) {
////        aspectRatio = ratio;
////        r = ratio * d;
////        updated = false;
////    }
////};

import AppKit
import Metal
import MetalKit
import simd

// MARK: - Camera System
class Camera {
    var position: simd_float3 = simd_float3(0, 5, 10)
    var target: simd_float3 = simd_float3(0, 0, 0)
    var up: simd_float3 = simd_float3(0, 1, 0)
    
    // Orbit parameters
    var distance: Float = 10.0
    var azimuth: Float = 0.0  // Horizontal rotation
    var elevation: Float = 0.3 // Vertical rotation (radians)
    
    // Camera limits
    let minDistance: Float = 0.10
    let maxDistance: Float = 100.0
    let minElevation: Float = -Float.pi / 2 + 0.1
    let maxElevation: Float = Float.pi / 2 - 0.1
    
    func updatePosition() {
        // Convert spherical coordinates to cartesian
        let x = distance * cos(elevation) * cos(azimuth)
        let y = distance * sin(elevation)
        let z = distance * cos(elevation) * sin(azimuth)
        
        position = target + simd_float3(x, y, z)
    }
    
    func getViewMatrix() -> simd_float4x4 {
        return lookAt(eye: position, center: target, up: up)
    }
    
    func orbit(deltaAzimuth: Float, deltaElevation: Float) {
        azimuth += deltaAzimuth
        elevation -= deltaElevation
        
        // Clamp elevation to prevent flipping
        elevation = max(minElevation, min(maxElevation, elevation))
        
        updatePosition()
    }
    
    func zoom(factor: Float) {
        distance *= factor
        updatePosition()
    }
    
    func pan(deltaX: Float, deltaY: Float) {
        // Get camera's right and up vectors
        let forward = normalize(target - position)
        let right = normalize(cross(forward, up))
        let upVector = cross(right, forward)
        
        // Move target and position together
        let panVector = right * deltaX + upVector * deltaY
        target += panVector
        updatePosition()
    }
}

// MARK: - Matrix Math Utilities
func lookAt(eye: simd_float3, center: simd_float3, up: simd_float3) -> simd_float4x4 {
    let z = normalize(eye - center)
    let x = normalize(cross(up, z))
    let y = cross(z, x)
    
    let translation = simd_float4x4(
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-dot(x, eye), -dot(y, eye), -dot(z, eye), 1]
    )
    
    let rotation = simd_float4x4(
        [x.x, y.x, z.x, 0],
        [x.y, y.y, z.y, 0],
        [x.z, y.z, z.z, 0],
        [0, 0, 0, 1]
    )
    
    return rotation * translation
}

func perspectiveProjection(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let f = 1.0 / tan(fovy * 0.5)
    return simd_float4x4(
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0]
    )
}

func translationMatrix(_ translation: simd_float3) -> simd_float4x4 {
    return simd_float4x4(
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [translation.x, translation.y, translation.z, 1]
    )
}

func scaleMatrix(_ scale: simd_float3) -> simd_float4x4 {
    return simd_float4x4(
        [scale.x, 0, 0, 0],
        [0, scale.y, 0, 0],
        [0, 0, scale.z, 0],
        [0, 0, 0, 1]
    )
}

// Matrix inverse implementation
func matrixInverse(_ matrix: simd_float4x4) -> simd_float4x4 {
    let m = matrix
    
    // Calculate 2x2 determinants for the first row
    let a2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2]
    let a1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1]
    let a1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1]
    let a0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0]
    let a0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0]
    let a0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0]
    let a2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2]
    let a1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1]
    let a1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1]
    let a2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2]
    let a1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1]
    let a1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    let a0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0]
    let a0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0]
    let a0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0]
    let a0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0]
    let a0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0]
    let a0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    
    // Calculate the determinant
    let det = m[0][0] * (m[1][1] * a2323 - m[1][2] * a1323 + m[1][3] * a1223) -
              m[0][1] * (m[1][0] * a2323 - m[1][2] * a0323 + m[1][3] * a0223) +
              m[0][2] * (m[1][0] * a1323 - m[1][1] * a0323 + m[1][3] * a0123) -
              m[0][3] * (m[1][0] * a1223 - m[1][1] * a0223 + m[1][2] * a0123)
    
    if abs(det) < 1e-8 {
        // Return identity matrix if determinant is too small
        return simd_float4x4(1.0)
    }
    
    let invDet = 1.0 / det
    
    return simd_float4x4(
        [
            invDet * (m[1][1] * a2323 - m[1][2] * a1323 + m[1][3] * a1223),
            invDet * -(m[0][1] * a2323 - m[0][2] * a1323 + m[0][3] * a1223),
            invDet * (m[0][1] * a2313 - m[0][2] * a1313 + m[0][3] * a1213),
            invDet * -(m[0][1] * a2312 - m[0][2] * a1312 + m[0][3] * a1212)
        ],
        [
            invDet * -(m[1][0] * a2323 - m[1][2] * a0323 + m[1][3] * a0223),
            invDet * (m[0][0] * a2323 - m[0][2] * a0323 + m[0][3] * a0223),
            invDet * -(m[0][0] * a2313 - m[0][2] * a0313 + m[0][3] * a0213),
            invDet * (m[0][0] * a2312 - m[0][2] * a0312 + m[0][3] * a0212)
        ],
        [
            invDet * (m[1][0] * a1323 - m[1][1] * a0323 + m[1][3] * a0123),
            invDet * -(m[0][0] * a1323 - m[0][1] * a0323 + m[0][3] * a0123),
            invDet * (m[0][0] * a1313 - m[0][1] * a0313 + m[0][3] * a0113),
            invDet * -(m[0][0] * a1312 - m[0][1] * a0312 + m[0][3] * a0112)
        ],
        [
            invDet * -(m[1][0] * a1223 - m[1][1] * a0223 + m[1][2] * a0123),
            invDet * (m[0][0] * a1223 - m[0][1] * a0223 + m[0][2] * a0123),
            invDet * -(m[0][0] * a1213 - m[0][1] * a0213 + m[0][2] * a0113),
            invDet * (m[0][0] * a1212 - m[0][1] * a0212 + m[0][2] * a0112)
        ]
    )
}

// MARK: - Shader Uniforms
struct GridUniforms {
    var modelViewProjectionMatrix: simd_float4x4
    var inverseModelViewProjectionMatrix: simd_float4x4
    var cameraPosition: simd_float3
    var gridScale: Float
}

struct ObjectUniforms {
    var modelViewProjectionMatrix: simd_float4x4
    var modelMatrix: simd_float4x4
    var normalMatrix: simd_float3x3
    var color: simd_float3
}

// MARK: - Vertex Structure
struct Vertex {
    var position: simd_float3
    var normal: simd_float3
}

// MARK: - Scene Object
struct SceneObject {
    var position: simd_float3
    var scale: simd_float3
    var color: simd_float3
    
    init(position: simd_float3 = simd_float3(0, 0, 0),
         scale: simd_float3 = simd_float3(1, 1, 1),
         color: simd_float3 = simd_float3(0.8, 0.6, 0.4)) {
        self.position = position
        self.scale = scale
        self.color = color
    }
    
    func getModelMatrix() -> simd_float4x4 {
        return translationMatrix(position) * scaleMatrix(scale)
    }
}

class InfiniteGridRenderer: NSObject {
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    
    // Grid rendering
    private var gridRenderPipelineState: MTLRenderPipelineState!
    private var gridUniformBuffer: MTLBuffer!
    
    // Object rendering
    private var objectRenderPipelineState: MTLRenderPipelineState!
    private var objectUniformBuffer: MTLBuffer!
    private var cubeVertexBuffer: MTLBuffer!
    private var cubeIndexBuffer: MTLBuffer!
    
    private var depthStencilState: MTLDepthStencilState!
    private var gridDepthStencilState: MTLDepthStencilState!
    
    private let camera = Camera()
    private var gridScale: Float = 1.0
    
    // Scene objects
    private var objects: [SceneObject] = []
    
    override init() {
        super.init()
        setupMetal()
        setupScene()
    }
    
    private func setupMetal() {
        // Initialize Metal device and command queue
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Create uniform buffers
        gridUniformBuffer = device.makeBuffer(length: MemoryLayout<GridUniforms>.stride,
                                            options: .storageModeShared)!
        objectUniformBuffer = device.makeBuffer(length: MemoryLayout<ObjectUniforms>.stride,
                                              options: .storageModeShared)!
        
        // Setup render pipelines and resources
        setupGridRenderPipeline()
        setupObjectRenderPipeline()
        setupDepthState()
        setupCubeGeometry()
    }
    
    private func setupScene() {
        // Add a cube at the center (0, 0.5, 0) - positioned at y=0.5 so it sits on the grid
        objects.append(SceneObject(position: simd_float3(0, 0.0, 0),
                                 scale: simd_float3(1, 1, 1),
                                 color: simd_float3(0.9, 0.3, 0.3)))
//        
//        // Add some additional cubes for demonstration
//        objects.append(SceneObject(position: simd_float3(3, 0.9, 0),
//                                 scale: simd_float3(0.8, 1.5, 0.8),
//                                 color: simd_float3(0.3, 0.9, 0.3)))
//        
//        objects.append(SceneObject(position: simd_float3(-2, 1, 2),
//                                 scale: simd_float3(1.2, 0.5, 1.2),
//                                 color: simd_float3(0.3, 0.3, 0.9)))
//        
//        objects.append(SceneObject(position: simd_float3(0, 0.25, -3),
//                                 scale: simd_float3(0.5, 0.5, 0.5),
//                                 color: simd_float3(0.9, 0.9, 0.3)))
//        
//        // Initialize camera to look at the scene
//        camera.position = simd_float3(5, 3, 5)  // Position camera away from origin
    }
    
    private func setupCubeGeometry() {
        // Define cube vertices with normals
        let vertices: [Vertex] = [
            // Front face
            Vertex(position: simd_float3(-0.5, -0.5,  0.5), normal: simd_float3(0, 0, 1)),
            Vertex(position: simd_float3( 0.5, -0.5,  0.5), normal: simd_float3(0, 0, 1)),
            Vertex(position: simd_float3( 0.5,  0.5,  0.5), normal: simd_float3(0, 0, 1)),
            Vertex(position: simd_float3(-0.5,  0.5,  0.5), normal: simd_float3(0, 0, 1)),
            
            // Back face
            Vertex(position: simd_float3(-0.5, -0.5, -0.5), normal: simd_float3(0, 0, -1)),
            Vertex(position: simd_float3(-0.5,  0.5, -0.5), normal: simd_float3(0, 0, -1)),
            Vertex(position: simd_float3( 0.5,  0.5, -0.5), normal: simd_float3(0, 0, -1)),
            Vertex(position: simd_float3( 0.5, -0.5, -0.5), normal: simd_float3(0, 0, -1)),
            
            // Top face
            Vertex(position: simd_float3(-0.5,  0.5, -0.5), normal: simd_float3(0, 1, 0)),
            Vertex(position: simd_float3(-0.5,  0.5,  0.5), normal: simd_float3(0, 1, 0)),
            Vertex(position: simd_float3( 0.5,  0.5,  0.5), normal: simd_float3(0, 1, 0)),
            Vertex(position: simd_float3( 0.5,  0.5, -0.5), normal: simd_float3(0, 1, 0)),
            
            // Bottom face
            Vertex(position: simd_float3(-0.5, -0.5, -0.5), normal: simd_float3(0, -1, 0)),
            Vertex(position: simd_float3( 0.5, -0.5, -0.5), normal: simd_float3(0, -1, 0)),
            Vertex(position: simd_float3( 0.5, -0.5,  0.5), normal: simd_float3(0, -1, 0)),
            Vertex(position: simd_float3(-0.5, -0.5,  0.5), normal: simd_float3(0, -1, 0)),
            
            // Right face
            Vertex(position: simd_float3( 0.5, -0.5, -0.5), normal: simd_float3(1, 0, 0)),
            Vertex(position: simd_float3( 0.5,  0.5, -0.5), normal: simd_float3(1, 0, 0)),
            Vertex(position: simd_float3( 0.5,  0.5,  0.5), normal: simd_float3(1, 0, 0)),
            Vertex(position: simd_float3( 0.5, -0.5,  0.5), normal: simd_float3(1, 0, 0)),
            
            // Left face
            Vertex(position: simd_float3(-0.5, -0.5, -0.5), normal: simd_float3(-1, 0, 0)),
            Vertex(position: simd_float3(-0.5, -0.5,  0.5), normal: simd_float3(-1, 0, 0)),
            Vertex(position: simd_float3(-0.5,  0.5,  0.5), normal: simd_float3(-1, 0, 0)),
            Vertex(position: simd_float3(-0.5,  0.5, -0.5), normal: simd_float3(-1, 0, 0))
        ]
        
        let indices: [UInt16] = [
            0,  1,  2,    0,  2,  3,    // front
            4,  5,  6,    4,  6,  7,    // back
            8,  9, 10,    8, 10, 11,    // top
            12, 13, 14,   12, 14, 15,   // bottom
            16, 17, 18,   16, 18, 19,   // right
            20, 21, 22,   20, 22, 23    // left
        ]
        
        cubeVertexBuffer = device.makeBuffer(bytes: vertices,
                                           length: vertices.count * MemoryLayout<Vertex>.stride,
                                           options: .storageModeShared)!
        
        cubeIndexBuffer = device.makeBuffer(bytes: indices,
                                          length: indices.count * MemoryLayout<UInt16>.stride,
                                          options: .storageModeShared)!
    }
    
    private func setupGridRenderPipeline() {
        // Grid shader source - FIXED VERSION
        let gridShaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct GridUniforms {
            float4x4 modelViewProjectionMatrix;
            float4x4 inverseModelViewProjectionMatrix;
            float3 cameraPosition;
            float gridScale;
        };

        struct GridVertexOut {
            float4 position [[position]];
            float3 worldPosition;
            float3 viewDirection;
        };

        vertex GridVertexOut gridVertexShader(uint vertexID [[vertex_id]],
                                             constant GridUniforms& uniforms [[buffer(0)]]) {
            // Generate fullscreen quad vertices
            float2 positions[4] = {
                float2(-1.0, -1.0),
                float2( 1.0, -1.0),
                float2(-1.0,  1.0),
                float2( 1.0,  1.0)
            };
            
            float2 pos = positions[vertexID];
            
            GridVertexOut out;
            out.position = float4(pos, 0.999999, 1.0); // Push grid to far plane to render behind objects
            
            // Calculate world position by unprojecting screen coordinates
            float4x4 invMVP = uniforms.inverseModelViewProjectionMatrix;
            float4 nearPoint = invMVP * float4(pos.x, pos.y, 0.0, 1.0); // Use z=0 instead of -1
            float4 farPoint = invMVP * float4(pos.x, pos.y, 1.0, 1.0);
            
            nearPoint /= nearPoint.w;
            farPoint /= farPoint.w;
            
            out.viewDirection = normalize(farPoint.xyz - nearPoint.xyz);
            out.worldPosition = nearPoint.xyz;
            
            return out;
        }

        fragment float4 gridFragmentShader(GridVertexOut in [[stage_in]],
                                          constant GridUniforms& uniforms [[buffer(0)]]) {
            // Ray-plane intersection to find ground position
            float3 rayDir = normalize(in.viewDirection);
            float3 rayOrigin = uniforms.cameraPosition;
            
            // Intersect with XZ plane (y = 0)
            float t = -rayOrigin.y / rayDir.y;
            
            // Discard if intersection is behind camera or ray doesn't hit plane
            if (t < 0.0 || rayDir.y >= 0.0) {
                discard_fragment();
            }
            
            float3 worldPos = rayOrigin + rayDir * t;
            
            // Scale the grid
            float2 coord = worldPos.xz * uniforms.gridScale;
            
            // Create grid lines using derivative-based anti-aliasing
            float2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
            float line = min(grid.x, grid.y);
            
            // Major grid lines every 10 units
            float2 majorGrid = abs(fract(coord * 0.1 - 0.5) - 0.5) / fwidth(coord * 0.1);
            float majorLine = min(majorGrid.x, majorGrid.y);
            
            // Create axis lines (X and Z axes in different colors)
            float xAxis = abs(worldPos.z) < 0.1 ? 1.0 : 0.0;
            float zAxis = abs(worldPos.x) < 0.1 ? 1.0 : 0.0;
            
            // Combine lines
            float gridIntensity = 1.0 - min(line, 1.0);
            float majorGridIntensity = 1.0 - min(majorLine, 1.0);
            
            // Distance-based fade
            float distance = length(worldPos - uniforms.cameraPosition);
            float fade = exp(-distance * 0.01); // Reduced fade rate
            float nearFade = smoothstep(0.0, 3.0, distance); // Reduced near fade distance
            
            // Grid colors
            float3 gridColor = float3(0.3, 0.3, 0.4);
            float3 majorGridColor = float3(0.5, 0.5, 0.6);
            float3 xAxisColor = float3(1.0, 0.3, 0.3); // Red for X axis
            float3 zAxisColor = float3(0.3, 0.3, 1.0); // Blue for Z axis
            
            // Combine colors
            float3 finalColor = gridColor;
            float totalAlpha = gridIntensity * 0.6; // Reduced base alpha
            
            // Add major grid
            if (majorGridIntensity > 0.0) {
                finalColor = mix(finalColor, majorGridColor, majorGridIntensity * 0.7);
                totalAlpha = max(totalAlpha, majorGridIntensity * 0.5);
            }
            
            // Add axis lines
            if (xAxis > 0.0) {
                finalColor = mix(finalColor, xAxisColor, 0.8);
                totalAlpha = max(totalAlpha, 0.7);
            }
            if (zAxis > 0.0) {
                finalColor = mix(finalColor, zAxisColor, 0.8);
                totalAlpha = max(totalAlpha, 0.7);
            }
            
            // Apply fading
            float alpha = totalAlpha * fade * nearFade;
            
            // Ensure minimum visibility for main axes
            if (xAxis > 0.0 || zAxis > 0.0) {
                alpha = max(alpha, 0.2 * fade);
            }
            
            return float4(finalColor, alpha);
        }
        """
        
        // Create library from source
        do {
            let library = try device.makeLibrary(source: gridShaderSource, options: nil)
            let vertexFunction = library.makeFunction(name: "gridVertexShader")!
            let fragmentFunction = library.makeFunction(name: "gridFragmentShader")!
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
            gridRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create grid render pipeline state: \(error)")
        }
    }
    
    private func setupObjectRenderPipeline() {
        // Object shader source
        let objectShaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct ObjectUniforms {
            float4x4 modelViewProjectionMatrix;
            float4x4 modelMatrix;
            float3x3 normalMatrix;
            float3 color;
        };

        struct Vertex {
            float3 position [[attribute(0)]];
            float3 normal [[attribute(1)]];
        };

        struct ObjectVertexOut {
            float4 position [[position]];
            float3 worldPosition;
            float3 worldNormal;
            float3 color;
        };

        vertex ObjectVertexOut objectVertexShader(Vertex in [[stage_in]],
                                                 constant ObjectUniforms& uniforms [[buffer(1)]]) {
            ObjectVertexOut out;
            
            float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
            out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
            out.worldPosition = worldPos.xyz;
            out.worldNormal = normalize(uniforms.normalMatrix * in.normal);
            out.color = uniforms.color;
            
            return out;
        }

        fragment float4 objectFragmentShader(ObjectVertexOut in [[stage_in]]) {
            // Simple lighting calculation
            float3 lightDirection = normalize(float3(0.3, 1.0, 0.5));
            float3 normal = normalize(in.worldNormal);
            
            float diffuse = max(dot(normal, lightDirection), 0.0);
            float ambient = 0.3;
            
            float3 finalColor = in.color * (ambient + diffuse * 0.7);
            
            return float4(finalColor, 1.0);
        }
        """
        
        // Create library from source
        do {
            let library = try device.makeLibrary(source: objectShaderSource, options: nil)
            let vertexFunction = library.makeFunction(name: "objectVertexShader")!
            let fragmentFunction = library.makeFunction(name: "objectFragmentShader")!
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            
            // Setup vertex descriptor
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float3
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            
            vertexDescriptor.attributes[1].format = .float3
            vertexDescriptor.attributes[1].offset = MemoryLayout<simd_float3>.stride
            vertexDescriptor.attributes[1].bufferIndex = 0
            
            vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
            vertexDescriptor.layouts[0].stepRate = 1
            vertexDescriptor.layouts[0].stepFunction = .perVertex
            
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            
            objectRenderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create object render pipeline state: \(error)")
        }
    }
    
    private func setupDepthState() {
        // Regular depth state for objects
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .less
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)
        
        // Special depth state for grid - don't write to depth buffer
        let gridDepthStateDescriptor = MTLDepthStencilDescriptor()
        gridDepthStateDescriptor.depthCompareFunction = .lessEqual
        gridDepthStateDescriptor.isDepthWriteEnabled = false // Key fix: don't write depth for grid
        gridDepthStencilState = device.makeDepthStencilState(descriptor: gridDepthStateDescriptor)
    }
    
    func render(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
        
        // Clear to dark background
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 1.0
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        
        // Get view and projection matrices
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = perspectiveProjection(fovy: Float.pi / 4, aspect: aspect, near: 0.1, far: 1000.0)
        let viewMatrix = camera.getViewMatrix()
        let viewProjectionMatrix = projectionMatrix * viewMatrix
        
        // First render the grid (in the background, depth at far plane)
        renderGrid(renderEncoder: renderEncoder, viewSize: view.drawableSize)
        
        // Then render objects (they will appear in front of the grid)
        renderObjects(renderEncoder: renderEncoder, viewProjectionMatrix: viewProjectionMatrix)
        debugCameraAndMatrices(viewSize: view.drawableSize)
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func renderObjects(renderEncoder: MTLRenderCommandEncoder, viewProjectionMatrix: simd_float4x4) {
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setRenderPipelineState(objectRenderPipelineState)
        renderEncoder.setVertexBuffer(cubeVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(objectUniformBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(objectUniformBuffer, offset: 0, index: 0)
        
        for object in objects {
            // Update object uniforms
            let modelMatrix = object.getModelMatrix()
            let mvpMatrix = viewProjectionMatrix * modelMatrix
            let normalMatrix = simd_float3x3(
                simd_float3(modelMatrix[0][0], modelMatrix[0][1], modelMatrix[0][2]),
                simd_float3(modelMatrix[1][0], modelMatrix[1][1], modelMatrix[1][2]),
                simd_float3(modelMatrix[2][0], modelMatrix[2][1], modelMatrix[2][2])
            )
            
            let uniformsPointer = objectUniformBuffer.contents().bindMemory(to: ObjectUniforms.self, capacity: 1)
            uniformsPointer.pointee.modelViewProjectionMatrix = mvpMatrix
            uniformsPointer.pointee.modelMatrix = modelMatrix
            uniformsPointer.pointee.normalMatrix = normalMatrix
            uniformsPointer.pointee.color = object.color
            
            // Draw the cube
            renderEncoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: 36, // 6 faces * 2 triangles * 3 vertices
                indexType: .uint16,
                indexBuffer: cubeIndexBuffer,
                indexBufferOffset: 0
            )
        }
    }
    
    private func renderGrid(renderEncoder: MTLRenderCommandEncoder, viewSize: CGSize) {
        // Use special depth state for grid
        renderEncoder.setDepthStencilState(gridDepthStencilState)
        
        // Update grid uniforms
        let aspect = Float(viewSize.width / viewSize.height)
        let projectionMatrix = perspectiveProjection(fovy: Float.pi / 4, aspect: aspect, near: 0.1, far: 1000.0)
        let viewMatrix = camera.getViewMatrix()
        let mvpMatrix = projectionMatrix * viewMatrix
        let invMVPMatrix = matrixInverse(mvpMatrix)
        
        let uniformsPointer = gridUniformBuffer.contents().bindMemory(to: GridUniforms.self, capacity: 1)
        uniformsPointer.pointee.modelViewProjectionMatrix = mvpMatrix
        uniformsPointer.pointee.inverseModelViewProjectionMatrix = invMVPMatrix
        uniformsPointer.pointee.cameraPosition = camera.position
        uniformsPointer.pointee.gridScale = gridScale
        
        // Set grid pipeline state and uniforms
        renderEncoder.setRenderPipelineState(gridRenderPipelineState)
        renderEncoder.setVertexBuffer(gridUniformBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentBuffer(gridUniformBuffer, offset: 0, index: 0)
        
        // Draw fullscreen quad (grid will be generated in shader)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    
    // MARK: - Public Methods for Adding Objects
    func addCube(at position: simd_float3, scale: simd_float3 = simd_float3(1, 1, 1), color: simd_float3 = simd_float3(0.8, 0.6, 0.4)) {
        objects.append(SceneObject(position: position, scale: scale, color: color))
    }
    
    func clearObjects() {
        objects.removeAll()
    }
    
    func getObjects() -> [SceneObject] {
        return objects
    }
    
    // MARK: - Mouse and Gesture Handling
    func handleMouseDrag(deltaX: Float, deltaY: Float, isRightMouseButton: Bool, isShiftPressed: Bool) {
        let sensitivity: Float = 0.01
        
        if isRightMouseButton || isShiftPressed {
            // Pan camera
            camera.pan(deltaX: -deltaX * sensitivity, deltaY: deltaY * sensitivity)
        } else {
            // Orbit camera
            camera.orbit(deltaAzimuth: deltaX * sensitivity, deltaElevation: -deltaY * sensitivity)
        }
    }
    
    func handleScrollWheel(deltaY: Float) {
        let zoomFactor = 1.0 + (deltaY * 0.01)
        camera.zoom(factor: zoomFactor)
    }
    
    func handleMagnification(magnification: Float) {
        gridScale *= (1.0 + magnification * 0.1)
        gridScale = max(0.1, min(10.0, gridScale))
    }
    
    // MARK: - Debug Methods
    func debugCameraAndMatrices(viewSize: CGSize) {
        let aspect = Float(viewSize.width / viewSize.height)
        let projectionMatrix = perspectiveProjection(fovy: Float.pi / 4, aspect: aspect, near: 0.1, far: 1000.0)
        let viewMatrix = camera.getViewMatrix()
        let viewProjectionMatrix = projectionMatrix * viewMatrix
        
        print("Camera Position: \(camera.position)")
        print("View Matrix: \(viewMatrix)")
        print("Projection Matrix: \(projectionMatrix)")
        print("ViewProjection Matrix: \(viewProjectionMatrix)")
        
        // Test a cube at origin
        if let firstObject = objects.first {
            let modelMatrix = firstObject.getModelMatrix()
            let mvpMatrix = viewProjectionMatrix * modelMatrix
            print("First Object Position: \(firstObject.position)")
            print("Model Matrix: \(modelMatrix)")
            print("MVP Matrix: \(mvpMatrix)")
            
            // Transform a test vertex
            let testVertex = simd_float4(0, 0, 0, 1) // origin
            let transformedVertex = mvpMatrix * testVertex
            print("Test vertex (0,0,0,1) -> \(transformedVertex)")
            print("After w-divide: \(transformedVertex.z / transformedVertex.w)")
        }
    }
}

// MARK: - Custom Metal View for macOS
class InfiniteGridMetalView: MTKView {
    private var renderer: InfiniteGridRenderer!
    private var trackingArea: NSTrackingArea?
    
    override init(frame frameRect: NSRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
        setupView()
    }
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }
    
    private func setupView() {
        device = MTLCreateSystemDefaultDevice()
        colorPixelFormat = .bgra8Unorm
        depthStencilPixelFormat = .depth32Float
        clearDepth = 1.0
        
        renderer = InfiniteGridRenderer()
        delegate = renderer
        
        // Setup tracking area for mouse events
        updateTrackingAreas()
    }
    
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        
        if let trackingArea = trackingArea {
            removeTrackingArea(trackingArea)
        }
        
        trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .enabledDuringMouseDrag],
            owner: self,
            userInfo: nil
        )
        
        if let trackingArea = trackingArea {
            addTrackingArea(trackingArea)
        }
    }
    
    // MARK: - Mouse Event Handling
    override func mouseDragged(with event: NSEvent) {
        let deltaX = Float(event.deltaX)
        let deltaY = Float(event.deltaY)
        let isShiftPressed = event.modifierFlags.contains(.shift)
        
        renderer.handleMouseDrag(deltaX: deltaX, deltaY: deltaY, isRightMouseButton: false, isShiftPressed: isShiftPressed)
    }
    
    override func rightMouseDragged(with event: NSEvent) {
        let deltaX = Float(event.deltaX)
        let deltaY = Float(event.deltaY)
        
        renderer.handleMouseDrag(deltaX: deltaX, deltaY: deltaY, isRightMouseButton: true, isShiftPressed: false)
    }
    
    override func scrollWheel(with event: NSEvent) {
        let deltaY = Float(event.scrollingDeltaY)
        renderer.handleScrollWheel(deltaY: deltaY)
    }
    
    override func magnify(with event: NSEvent) {
        let magnification = Float(event.magnification)
        renderer.handleMagnification(magnification: magnification)
    }
    
    override func keyDown(with event: NSEvent) {
        // Add keyboard shortcuts for adding/removing cubes
        switch event.keyCode {
        case 15: // 'R' key - add random cube
            let randomX = Float.random(in: -5...5)
            let randomZ = Float.random(in: -5...5)
            let randomY = Float.random(in: 0.5...2.0)
            let randomScale = Float.random(in: 0.5...1.5)
            let randomColor = simd_float3(
                Float.random(in: 0.3...1.0),
                Float.random(in: 0.3...1.0),
                Float.random(in: 0.3...1.0)
            )
            renderer.addCube(at: simd_float3(randomX, randomY, randomZ),
                           scale: simd_float3(randomScale, randomScale, randomScale),
                           color: randomColor)
        case 8: // 'C' key - clear all objects
            renderer.clearObjects()
        default:
            super.keyDown(with: event)
        }
    }
    
    // Accept first responder to receive key events
    override var acceptsFirstResponder: Bool {
        return true
    }
}

// MARK: - MTKViewDelegate Extension
extension InfiniteGridRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }
    
    func draw(in view: MTKView) {
        render(in: view)
    }
}

// MARK: - Window Controller
class InfiniteGridWindowController: NSWindowController {
    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        
        self.init(window: window)
        
        window.title = "Infinite Grid with Cubes - Metal Renderer"
        window.center()
        
        let metalView = InfiniteGridMetalView(frame: window.contentView!.bounds)
        metalView.autoresizingMask = [.width, .height]
        window.contentView = metalView
        
        // Make the metal view first responder for keyboard events
        window.makeFirstResponder(metalView)
        
        // Add instructions to window
        addInstructionsToWindow()
    }
    
    private func addInstructionsToWindow() {
        guard let window = window else { return }
        
        let instructionsView = NSTextView(frame: NSRect(x: 10, y: 10, width: 350, height: 160))
        instructionsView.string = """
        Controls:
        • Left mouse drag: Orbit camera
        • Right mouse drag or Shift+drag: Pan camera
        • Scroll wheel: Zoom in/out
        • Trackpad pinch: Adjust grid scale
        
        Keyboard:
        • R: Add random cube
        • C: Clear all cubes
        
        The red cube at (0,0,0) marks the center of the grid.
        """
        instructionsView.isEditable = false
        instructionsView.backgroundColor = NSColor.black.withAlphaComponent(0.7)
        instructionsView.textColor = NSColor.white
        instructionsView.font = NSFont.systemFont(ofSize: 12)
        
        window.contentView?.addSubview(instructionsView)
    }
}

// MARK: - App Delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var windowController: InfiniteGridWindowController?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        windowController = InfiniteGridWindowController()
        windowController?.showWindow(nil)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// MARK: - Main Entry PointAttachments[0].pixelFormat = .bgra8Unorm
           

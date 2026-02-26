//
//  GeometryNode.mm
//  WorldOf3D
//
//  Created by Aditya Dudeja on 08/02/26.
//


//
//  GeometryNode.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 10/12/25.
//

#ifndef GeometryNode_h
#define GeometryNode_h

@import Metal;
@import simd;
@import Utils;
@import GPUManager;
@import std_core.type_traits.is_base_of;
@import MetalKit;

#include <iostream>
#include <vector>
#import <Foundation/Foundation.h>
#import <ModelIO/ModelIO.h>

@import Camera3D;
class Modifiers {
public:
    bool ownership = true;
    simd_float4x4* transforms = nil;
    uint32_t noOfTransforms = 0;

    Modifiers() {

    }

    void MakeModifierArray(int count, simd_float3 start, simd_float3 spacing) {
        if (count != noOfTransforms) {
            if (transforms) { delete [] transforms; }
            transforms = new simd_float4x4[count];
            noOfTransforms = count;
        }

        for (int i = 0; i < count; i++) {
            transforms[i] = Translation(start + i * spacing);
        }
    }
    
    void MakeModifierArray( int m, int n, simd_float3 start, simd_float3 spacing) {
        if (m*n != noOfTransforms) {
            if (transforms) { delete [] transforms; }
            transforms = new simd_float4x4[m*n];
            noOfTransforms = m*n;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                transforms[i*n + j] = Translation(start + (simd_make_float3(i, j, 0) * spacing));
            }
        }
    }
    
    void MakeModifierArray( int m, int n, int l, simd_float3 start, simd_float3 spacing) {
        if (m*n*l != noOfTransforms) {
            if (transforms) { delete [] transforms; }
            transforms = new simd_float4x4[m*n*l];
            noOfTransforms = m*n*l;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < l; k++) {
                    transforms[i*n*l + j*l + k] = Translation(start + (simd_make_float3(i, j, k) * spacing));
                }
            }
        }
    }

    ~Modifiers() {
        if (ownership && transforms != nil) {
            delete [] transforms;
        }
    }
};


static uint32_t reservedSystemRenderPipelines = 1;

enum class VertexType { Vertex3D, Point };

template<typename T>
class GeometryNode {
public:
    std::vector<GeometryNode<T>> childNodes;
    
    void* Verticies = nil;
    int vertexCount;
    
    T* indices = nil;
    int indexCount;
    
    id<MTLBuffer> vertexBuffer = nil;
    id<MTLBuffer> indexBuffer = nil;
    id<MTLBuffer> transformBuffer = nil;
    
    simd_float3 position= {0.0, 0.0, 0.0};
    simd_float3 scale = {1.0, 1.0, 1.0};
    simd_float3 rotation = {0.0, 0.0, 0.0};
    
    simd_float4x4 modelMatrix = Identity();
    simd_float4x4 globalMatrix = Identity();
    
    simd_float4x4* instanceMatricies = nil;
    simd_float4x4* parentInstanceMatricies = nil;
    simd_float4x4* preMulparentInstanceMatricies = nil;
    
    uint32_t instances = 1;
    uint32_t parentInstances = 0;
    
    VertexType vertexType = VertexType::Vertex3D;
    RenderPipelineType renderPipelineType = RenderPipelineType::Predefined;

    bool update = true;
    bool isTextured = false;
    bool visible = true;
    bool ownership = true;
    
    int RenderStateNo = 0;
    typedef void (^RenderStateBlock)(id<MTLRenderCommandEncoder> encoder);
    RenderStateBlock renderStateBlock = nil;
    
    id<MTLTexture> texture = nil;
    
    MTLPrimitiveType drawType = MTLPrimitiveTypeTriangleStrip;
    std::string name = "Node";
    
    GeometryNode(void* Verticies, int vertexCount, T* indices, int indexCount): childNodes() {
        this->Verticies = Verticies;
        this->vertexCount = vertexCount;
        this->indices = indices;
        
        this->indexCount = indexCount;
        this->instanceMatricies = new simd_float4x4[instances];
        this->preMulparentInstanceMatricies = new simd_float4x4[instances];
        for (int i = 0; i < instances; i++) {
            this->instanceMatricies[i] = Identity();
            this->preMulparentInstanceMatricies[i] = Identity();
        }
    }
    
//    GeometryNode(std::vector<GeometryNode<T>>& nodes) {
//        this->vertexCount = 0;
//        this->indexCount = 0;
//        this->childNodes = nodes;
//        this->vertexBuffer = nil;
//        for (int i = 0; i  < childNodes.size(); i++) {
//            childNodes[i].parentInstances = instances;
//            childNodes[i].parentInstanceMatricies = preMulparentInstanceMatricies;
//        }
//    }
//
//    GeometryNode(std::vector<GeometryNode<T>>&& nodes) {
//
//        this->vertexCount = 0;
//        this->indexCount = 0;
//        this->childNodes = std::move(nodes);
//        this->vertexBuffer = nil;
//        for (int i = 0; i < childNodes.size(); i++) {
//            childNodes[i].parentInstances = instances;
//            childNodes[i].parentInstanceMatricies = preMulparentInstanceMatricies;
//        }
//    }
    // CRITICAL: This variadic constructor MUST exclude single GeometryNode arguments!
    // Without sizeof...(Children) > 1, the compiler may choose THIS constructor instead
    // of the copy/move constructor when pushing to std::vector, causing catastrophic bugs:
    //
    // What happens WITHOUT the constraint:
    // 1. vector.push_back(arrowNode) tries to construct a new GeometryNode
    // 2. Compiler sees this variadic template as a "perfect match" for forwarding
    // 3. It calls GeometryNode(arrowNode) using THIS constructor instead of copy ctor
    // 4. The arrowNode gets interpreted as a CHILD node to be added
    // 5. This creates phantom nested children and loses all the original geometry data
    // 6. Child nodes end up with childNodes.size() > 0 when they should be leaf nodes
    // 7. parentInstances gets reset to default values, breaking instancing
    //
    // The fix: Require sizeof...(Children) > 1 to force copy/move ctors for single arguments
    // This ensures vector operations use the correct constructors and preserve object state
    //
    
    // Also messed up entire instancing as when appending arrow Node with instance 3 and Triangle and Quad with parent instance 3
    // this constructor was called instead of copy constructor so what happend was when vector tried to copy triange and quad it called this which created a new object which had parent instance 0 and in this construct u can see it changes the parent instance to the new parent which has 0 so the 3 vanished and this changed it 0;
    template<typename... Children, typename = std::enable_if_t<(std::is_base_of_v<GeometryNode<T>, std::decay_t<Children>> && ...)>>
    GeometryNode(Children&&... children) requires (sizeof...(Children) > 1){
        childNodes.reserve(sizeof...(children));
        (childNodes.push_back(std::forward<Children>(children)), ...);
        this->vertexCount = 0;
        this->indexCount = 0;
        for (int i = 0; i < childNodes.size(); i++) {
            childNodes[i].parentInstances = instances;
            childNodes[i].parentInstanceMatricies = preMulparentInstanceMatricies;
        }
    }
    
    template<typename... Children>
    void AddNodes(Children&&... children) {
        // we need to reserve capacity otherwise copy will also be called
        // the first push_back is a move, the second and subsequent pushes may trigger a vector reallocation as it may run out of space so it will copy all contents to a new buffer:
        // This gaurentees there wont be more tha one copy per AddNodes call as in the begining we tell the vector to reserve the adequete space if it has well and good if not there will be one reallocation
        childNodes.reserve(childNodes.size() + sizeof...(children));
        uint32_t prevSize = childNodes.size();
        (childNodes.push_back(std::forward<Children>(children)), ...);
        int totalInstances = instances;
        if (parentInstances == 0) {
#ifdef SAFE_MODE
            if (parentInstanceMatricies) { std::cerr << "parent instance = 0 but has parent instance matrix buffer" << "\n"; }
#endif
            
        } else {
            totalInstances *= parentInstances;
        }
        for (int i = prevSize; i < childNodes.size(); i++) {
//            childNodes[i].buildGlobalMatrix(modelMatrix);
            childNodes[i].BuildInstances(preMulparentInstanceMatricies,  totalInstances);
        }
        
    }
    // If no parent as in parentInstances = 0; then a node is not req to have a parentInstanceMaricies and a premultipliedinstance matricies its transform buffer will have just its instances
    void buildBuffers(id<MTLDevice> metalDevice) {
        if (!vertexBuffer || !indexBuffer || update ) {
            uint32_t size_of_verts = VertexType::Vertex3D == vertexType ? sizeof(Vertex3D) : sizeof(Point3D);
            vertexBuffer = [metalDevice newBufferWithBytesNoCopy:Verticies length:vertexCount * size_of_verts  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
            }];
            
            indexBuffer = [metalDevice newBufferWithBytesNoCopy:indices length:indexCount * sizeof(T)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
            }];
            
//            if (!preMulparentInstanceMatricies) {
//                preMulparentInstanceMatricies = new simd_float4x4[1];
//                if (instances != 1) {
//                    std::cerr << " preMulInstance matrix uninitialised with more than one instance." << "\n";
//                    throw;
//                }
//                preMulparentInstanceMatricies[0] = Identity();
//            }
            int totalInstance = instances;
            if (parentInstances != 0) {
                totalInstance *= parentInstances;
            }
//            if (parentInstances == 0) {
//                if (!instanceMatricies) {
//                    instanceMatricies = new simd_float4x4[1];
//                }
//                transformBuffer = [metalDevice newBufferWithBytesNoCopy:instanceMatricies length:instances * sizeof(simd_float4x4)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//                }];
//            } else {
            transformBuffer = [metalDevice newBufferWithBytesNoCopy:preMulparentInstanceMatricies length:totalInstance * sizeof(simd_float4x4)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
                }];
//            }
            
            std::cout << "buffer built" << "\n";
        }
        for (int i = 0; i<childNodes.size(); i++) {
            childNodes[i].buildBuffers(metalDevice);
        }
        update = false;
    }
    
    void setPos(simd_float3 newPosition) {
        modelMatrix = Translation(newPosition - position) * modelMatrix;
        position = newPosition;
        buildGlobalMatrix();

    }
    void setRot(simd_float3 newRotation) {
        modelMatrix = Translation( position) * RotationX( newRotation.x ) * RotationY( newRotation.y) * RotationZ( newRotation.z) * Scale( scale );
        rotation = newRotation;
        buildGlobalMatrix();
    }
    void setScale(simd_float3 newScale) {
        modelMatrix = modelMatrix * Scale(newScale / scale);
        scale = newScale;
        buildGlobalMatrix();
    }
    void buildModelMatrix() {
        modelMatrix = Translation(position) * RotationX( rotation.x) * RotationY( rotation.y) * RotationZ( rotation.z) * Scale( scale );
        buildGlobalMatrix();
    }
    
    void applyModalMatrix() {
        position= {0.0, 0.0, 0.0};
        scale = {1.0, 1.0, 1.0};
        rotation = {0.0, 0.0, 0.0};
        buildModelMatrix();
    }
    // builds global mat for the child Nodes and itself
    void buildGlobalMatrix(simd_float4x4 newGlobalMat) {
//        globalMatrix = newGlobalMat;
//        simd_float4x4 newGlobalMatToPass = simd_mul( globalMatrix, modelMatrix );
//        for (int i=0; i<childNodes.size(); i++) {
//            childNodes[i].buildGlobalMatrix(newGlobalMatToPass);
//
//        }
        BuildInstances();
    }
    // builds global mat for the child Nodes
    void buildGlobalMatrix(int start = 0) {
//        simd_float4x4 newGlobalMatToPass = simd_mul( globalMatrix, modelMatrix);
//        for (int i=0; i<childNodes.size(); i++) {
//            childNodes[i].buildGlobalMatrix(newGlobalMatToPass);
//        }
        BuildInstances();
    }
    
    void BuildInstances(simd_float4x4* parentMatricies, int _Parentinstances) {
        
        if (parentInstances != _Parentinstances) {
            parentInstances = _Parentinstances;
            if (preMulparentInstanceMatricies) {delete [] preMulparentInstanceMatricies;}
            preMulparentInstanceMatricies = new simd_float4x4[parentInstances * instances];
            update = true;
        }
        parentInstanceMatricies = parentMatricies;
        
        
//        if (!instanceMatricies) {
//            if (instances != 1) {
//                std::cerr << "instance matrix uninitialised with more than one instance." << "\n";
//                throw;
//            }
//            instanceMatricies = new simd_float4x4[1];
//            *instanceMatricies = Identity();
//
//        }
        
//        if (!preMulparentInstanceMatricies) {
//            if (instances * parentInstances != 1) {
//                std::cerr << "preMulparentInstanceMatricies instance matrix uninitialised with more than one instance." << "\n";
//                throw;
//            }
//            instanceMatricies = new simd_float4x4[1];
//            *instanceMatricies = Identity();
//
//        }
        
//        if (!parentMatricies) {
//
//            std::cout << "FIX NEEDED: No Parent Matrix";
//            parentInstanceMatricies = new simd_float4x4[1];
//            *parentInstanceMatricies = Identity();
//        }
        
        BuildInstances();
    }
    
    void BuildInstances() {
        if (parentInstances == 0) {
            for (int j = 0; j < instances; j++) {
                preMulparentInstanceMatricies[j] = simd_mul(instanceMatricies[j], modelMatrix);
            }
            for (int i = 0; i < childNodes.size(); i++) {
                childNodes[i].BuildInstances(preMulparentInstanceMatricies, instances);
            }
        } else {
            for (int i = 0; i < parentInstances; i++) {
                for (int j = 0; j < instances; j++) {
                    preMulparentInstanceMatricies[i * instances + j] = simd_mul(parentInstanceMatricies[i] , simd_mul(instanceMatricies[j], modelMatrix));
                }
            }
            
            for (int i = 0; i < childNodes.size(); i++) {
                childNodes[i].BuildInstances(preMulparentInstanceMatricies, parentInstances * instances);
            }
        }
    }
    
    void buildInstanceFromModifier(Modifiers& m) {
        delete [] instanceMatricies;
        instanceMatricies = m.transforms;
        if (m.noOfTransforms != instances) {
            delete [] preMulparentInstanceMatricies;
            int totalInstances = m.noOfTransforms;
            if (parentInstances != 0) { totalInstances *= parentInstances; }
            preMulparentInstanceMatricies = new simd_float4x4[totalInstances];
            update = true;
        }
        instances = m.noOfTransforms;
        m.ownership = false;
        BuildInstances();
    }
    
    void buildInstanceFromBuffer(simd_float4x4* transformBuffer, int noOfTransforms) {
        delete [] instanceMatricies;
        instanceMatricies = transformBuffer;
        if (noOfTransforms != instances) {
            delete [] preMulparentInstanceMatricies;
            int totalInstances = noOfTransforms;
            if (parentInstances != 0) { totalInstances *= parentInstances; }
            preMulparentInstanceMatricies = new simd_float4x4[totalInstances];
            update = true;
        }
        
        instances = noOfTransforms;
        BuildInstances();
    }

    static GeometryNode<T> BuildGeoNodeFromModel(std::string path) {
        NSString* fullPath = [NSString stringWithUTF8String:path.c_str()];
        NSString* name = [fullPath stringByDeletingPathExtension];
        NSString* extension = [fullPath pathExtension];
        
        NSURL* meshURL = [[NSBundle mainBundle] URLForResource:name
                                          withExtension:extension];
        
        
        
        MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init ];
        // Store position in attribute[0]
        mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
        mtlVertexDescriptor.attributes[0].offset = 0;
        mtlVertexDescriptor.attributes[0].bufferIndex = 0;

        // Store texture coordinates in attribute[1]
        mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
        mtlVertexDescriptor.attributes[1].offset = offsetof(Vertex3D, colour);
        mtlVertexDescriptor.attributes[1].bufferIndex = 0;

        
        mtlVertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
        mtlVertexDescriptor.attributes[2].offset = offsetof(Vertex3D, textureCoordinates);
        mtlVertexDescriptor.attributes[2].bufferIndex = 0;
        
        mtlVertexDescriptor.attributes[3].format = MTLVertexFormatFloat3;
        mtlVertexDescriptor.attributes[3].offset = offsetof(Vertex3D, normal);
        mtlVertexDescriptor.attributes[3].bufferIndex = 0;
        
        // Set stride to twice the bytes per float2.
        mtlVertexDescriptor.layouts[0].stride = sizeof(Vertex3D);
        mtlVertexDescriptor.layouts[0].stepRate = 1;
        mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
        
        
        
        MDLVertexDescriptor* MeshDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor);
        
        MeshDescriptor.attributes[0].name = MDLVertexAttributePosition;
        MeshDescriptor.attributes[1].name = @"primvars:displayColor";
        MeshDescriptor.attributes[2].name = MDLVertexAttributeTextureCoordinate;
        MeshDescriptor.attributes[3].name = MDLVertexAttributeNormal;
        auto meshbufferallocator = [[MTKMeshBufferAllocator alloc] initWithDevice:GlobalGPUManager.metalDevice];

        MDLAsset* asset = [[MDLAsset alloc] initWithURL:meshURL vertexDescriptor:MeshDescriptor bufferAllocator:meshbufferallocator];
        GeometryNode<T> rootNode{nil, 0, nil, 0};
        NSArray<MDLObject *> *children = [asset childObjectsOfClass:MDLMesh.class];
        for (NSUInteger meshIndex = 0; meshIndex < children.count; meshIndex++) {
            MDLObject *object = children[meshIndex];
            if (![object isKindOfClass:[MDLMesh class]]) {
                NSLog(@"Warning: Object at index %lu is not an MDLMesh, skipping", (unsigned long)meshIndex);
                continue;
            }
            MDLMesh *mdlMesh = (MDLMesh *)object;
            mdlMesh.vertexDescriptor = MeshDescriptor;
            NSError *error = nil;
            MTKMesh *mtkMesh = [[MTKMesh alloc] initWithMesh:mdlMesh
                                                      device:GlobalGPUManager.metalDevice
                                                       error:&error];
            if (error) {
                NSLog(@"Error creating MTKMesh: %@", error.localizedDescription);
                continue;
            }
            
            printf("Vertex count: %lu \n", (unsigned long)mtkMesh.vertexCount);
            printf("Submesh Count: %lu \n", (unsigned long)mtkMesh.submeshes.count);
            
            for (NSUInteger submeshIndex = 0; submeshIndex < mtkMesh.submeshes.count; submeshIndex++) {
                MDLSubmesh* mdlSubmesh = mdlMesh.submeshes[submeshIndex];
                MTKSubmesh* mtkSubmesh = mtkMesh.submeshes[submeshIndex];
                auto submeshNode = GeometryNode<T>( nil, 0, nil, 0 );
                
                MTKMeshBuffer* vertexBuffer = mtkMesh.vertexBuffers[0];
                NSUInteger vertexDataSize = vertexBuffer.length;
                void* vertexData = vertexBuffer.map.bytes;
                
                // Allocate memory for GeometryNode to own
                void* ownedVertexData = malloc(vertexDataSize);
                memcpy(ownedVertexData, vertexData, vertexDataSize);
                
                submeshNode.Verticies = ownedVertexData;
                submeshNode.vertexCount = (int)mtkMesh.vertexCount;
                
                // Create Metal buffer that points to owned data (no copy, no deallocator)
                submeshNode.vertexBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:ownedVertexData
                                                                     length:vertexDataSize
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                
                // Get index data from MDL buffer
                id<MDLMeshBuffer> mdlIndexBuffer = mtkSubmesh.indexBuffer;
                std::cout << mtkSubmesh.indexBuffer.offset << "\n";
                NSUInteger indexDataSize = mdlIndexBuffer.length;
                void* indexData = mdlIndexBuffer.map.bytes;
                
                // Allocate memory for GeometryNode to own
                void* ownedIndexData = malloc(indexDataSize);
                memcpy(ownedIndexData, indexData, indexDataSize);
                
                submeshNode.indices = static_cast<T*>(ownedIndexData);
                submeshNode.indexCount = (int)mtkSubmesh.indexCount;
                
                // Create Metal buffer that points to owned data (no copy, no deallocator)
                submeshNode.indexBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:ownedIndexData
                                                                    length:indexDataSize
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                
                submeshNode.drawType = mtkSubmesh.primitiveType;
                submeshNode.update = false;
                submeshNode.visible = true;
                // Handle texture
                if (mdlSubmesh.material) {
                    MDLMaterial* material = mdlSubmesh.material;
                    MDLMaterialProperty* baseColorProperty =
                        [material propertyWithSemantic:MDLMaterialSemanticBaseColor];
                    
                    if (baseColorProperty && baseColorProperty.type == MDLMaterialPropertyTypeTexture) {
                        NSURL* textureURL = baseColorProperty.URLValue;
                        
                        MTKTextureLoader* textureLoader = [[MTKTextureLoader alloc] initWithDevice:GlobalGPUManager.metalDevice];
                        NSError* texError = nil;
                        submeshNode.texture = [textureLoader newTextureWithContentsOfURL:textureURL
                                                                                 options:nil
                                                                                   error:&texError];
                        if (!texError && submeshNode.texture) {
                            submeshNode.isTextured = true;
                        }
                    }
                }
                MDLMaterial* material = mdlSubmesh.material;
                MDLMaterialProperty* baseColorProperty =
                    [material propertyWithSemantic:MDLMaterialSemanticBaseColor];
                NSURL* textureURL;
                if (baseColorProperty) {
                    if (baseColorProperty.type == MDLMaterialPropertyTypeTexture) {
                        textureURL = baseColorProperty.URLValue;
                    } 
                    else if (baseColorProperty.type == MDLMaterialPropertyTypeString) {
                        if (baseColorProperty.stringValue.length > 0) {                                 
                            textureURL = [NSURL fileURLWithPath:baseColorProperty.stringValue];
                        }
                    } else if (baseColorProperty.type == MDLMaterialPropertyTypeURL) {
                            textureURL = baseColorProperty.URLValue;
                    } else if (baseColorProperty.type == MDLMaterialPropertyTypeNone) {
                             // Fallback: checks for string or URL value even if type is None
                    if (baseColorProperty.URLValue) {
                        textureURL = baseColorProperty.URLValue;
                    } else if (baseColorProperty.stringValue.length > 0) {
                        // Construct URL assuming relative path or filename
                        // We might need to resolve it relative to the model, but here lets try simple conversion
                        textureURL = [NSURL fileURLWithPath:baseColorProperty.stringValue];
                    }
                    }
                }

                MTKTextureLoader* textureLoader = [[MTKTextureLoader alloc] initWithDevice:GlobalGPUManager.metalDevice];

                submeshNode.texture = [textureLoader newTextureWithContentsOfURL:[[NSBundle mainBundle] URLForResource:@"Texture_4_ffffff"
                                                                                             withExtension:@"png"]
                                                                             options:nil
                                                                               error:&error];
                submeshNode.isTextured = true;
//                submeshNode.vertexBuffer = vertexBuffer.buffer;
//                submeshNode.vertexBuffer
//                submeshNode.vertexCount = (int)mtkMesh.vertexCount;
//                
//                submeshNode.indexBuffer = mtkSubmesh.indexBuffer.buffer;
//                submeshNode.indexCount = (int)mtkSubmesh.indexCount;
//                
//                submeshNode.Verticies = submeshNode.vertexBuffer.contents;
//                submeshNode.indices = static_cast<unsigned short*>(submeshNode.indexBuffer.contents);
//                submeshNode.drawType = mtkSubmesh.primitiveType;
//                submeshNode.visible = true;
//                submeshNode.update = false;
//                submeshNode.ownership = false;
                rootNode.AddNodes(std::move(submeshNode));
            }
            // TODO: Extract vertex/index data from mtkMesh to fill GeometryNode if needed
        }
        
        return rootNode;
    }
    
    
    void draw(id<MTLRenderCommandEncoder> cmdEncoder, id<MTLDevice> metalDevice, id<MTLRenderPipelineState> __strong (&predefinedStates)[3], NSMutableArray<id<MTLRenderPipelineState>>* customStates, Camera3D* cam, int& active_state) {
        if (vertexCount == 0 || indexCount == 0 || !visible)  {
            for (int i=0; i<childNodes.size(); i++) {
                childNodes[i].draw(cmdEncoder, metalDevice, predefinedStates, customStates, cam, active_state);
            }
            return;
        }
        buildBuffers(metalDevice);
        if (!instanceMatricies) {
            instanceMatricies = new simd_float4x4[1];
            if (instances != 1) {
                std::cerr << "instance matrix uninitialised with more than one instance." << "\n";
                throw;
            }
            instanceMatricies[0] = Identity();
        }
        int totalInstance = instances;
        if (parentInstances != 0) {
            totalInstance *= parentInstances;
        }
        if (renderPipelineType == RenderPipelineType::Custom) {
            if (active_state != 1000 + RenderStateNo-1) {
                [cmdEncoder setRenderPipelineState:[customStates objectAtIndex:RenderStateNo-1]];
                active_state = 1000 + RenderStateNo-1;
            }
            renderStateBlock(cmdEncoder);
            [cmdEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&modelMatrix length:sizeof(simd_float4x4) atIndex:1];
//           [cmdEncoder setVertexBytes:&globalMatrix length:sizeof(simd_float4x4) atIndex:2];

            [cmdEncoder setVertexBuffer:transformBuffer offset:0 atIndex:3];
            [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
            if (isTextured) {
                [cmdEncoder setFragmentTexture:texture atIndex:0];
            }
            [cmdEncoder drawIndexedPrimitives: drawType indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer indexBufferOffset:0 instanceCount:totalInstance];
            [cmdEncoder setRenderPipelineState:predefinedStates[0]];
        } else {
            if (RenderStateNo == static_cast<int>(PredefinedRenderPipelineState::Mesh)) {
                if (active_state != static_cast<int>(PredefinedRenderPipelineState::Mesh)) {
                    [cmdEncoder setRenderPipelineState:predefinedStates[0]];
                    active_state = 0;
                }
                [cmdEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
                [cmdEncoder setVertexBytes:&modelMatrix length:sizeof(simd_float4x4) atIndex:1];
                [cmdEncoder setVertexBuffer:transformBuffer offset:0 atIndex:3];
                [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
                if (isTextured) {
                    [cmdEncoder setFragmentTexture:texture atIndex:0];
                }
                [cmdEncoder drawIndexedPrimitives: drawType indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer indexBufferOffset:0 instanceCount:totalInstance];
//            [cmdEncoder setVertexBytes:&globalMatrix length:sizeof(simd_float4x4) atIndex:2];
            } else if (RenderStateNo == static_cast<int>(PredefinedRenderPipelineState::PointCloud)) {
                if (active_state != static_cast<int>(PredefinedRenderPipelineState::PointCloud)) {
                    [cmdEncoder setRenderPipelineState:predefinedStates[1]];
                    active_state = 1;
                }
                [cmdEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
                [cmdEncoder setVertexBytes:&modelMatrix length:sizeof(simd_float4x4) atIndex:1];
                [cmdEncoder setVertexBuffer:transformBuffer offset:0 atIndex:3];
                // [cmdEncoder drawIndexedPrimitives: drawType indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer indexBufferOffset:0 instanceCount:totalInstance];
                [cmdEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:vertexCount instanceCount:totalInstance];
            } else if (RenderStateNo == static_cast<int>(PredefinedRenderPipelineState::Billboard)) {
                if (active_state != static_cast<int>(PredefinedRenderPipelineState::Billboard)) {
                    [cmdEncoder setRenderPipelineState:predefinedStates[2]];
                    active_state = 2;
                }
                [cmdEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
                [cmdEncoder setVertexBytes:&modelMatrix length:sizeof(simd_float4x4) atIndex:1];
                [cmdEncoder setVertexBuffer:transformBuffer offset:0 atIndex:3];
                [cmdEncoder setVertexBytes: &cam->position length:sizeof(simd_float3) atIndex:4];
                [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
                if (isTextured) {
                    [cmdEncoder setFragmentTexture:texture atIndex:0];
                }
                [cmdEncoder drawIndexedPrimitives: drawType indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer indexBufferOffset:0 instanceCount:totalInstance];
//            [cmdEncoder setVertexBytes:&globalMatrix length:sizeof(simd_float4x4) atIndex:2];
            }
        }
            
        
        for (int i=0; i<childNodes.size(); i++) {
            childNodes[i].draw(cmdEncoder, metalDevice, predefinedStates, customStates, cam, active_state);
        }
    }
    
    GeometryNode<T>( GeometryNode<T>&& other) {
        std::cout << "Moved" << "\n";
        this->~GeometryNode();
        
        Verticies = other.Verticies;
        vertexBuffer = other.vertexBuffer;
        other.Verticies = nullptr;
        other.vertexBuffer = nullptr;
        indices = other.indices;
        other.indices = nullptr;
        
        vertexCount = other.vertexCount;
        indexCount = other.indexCount;
        instances = other.instances;
        parentInstances = other.parentInstances;
        
        instanceMatricies = other.instanceMatricies;
        other.instanceMatricies = nullptr;
        
        preMulparentInstanceMatricies = other.preMulparentInstanceMatricies;
        other.preMulparentInstanceMatricies = nullptr;
        
        parentInstanceMatricies = other.parentInstanceMatricies;
        other.parentInstanceMatricies = nullptr;
        
        position = other.position;
        rotation = other.rotation;
        scale = other.scale;
        
        globalMatrix = other.globalMatrix;
        modelMatrix = other.modelMatrix;
        
        drawType = other.drawType;
        childNodes = std::move(other.childNodes);
        
        isTextured = other.isTextured;
        texture = other.texture;
        renderStateBlock = other.renderStateBlock;
        RenderStateNo = other.RenderStateNo;
        ownership = other.ownership;
        
        
        visible = other.visible;
        name = std::move(other.name);
        vertexType = other.vertexType;
        renderPipelineType = other.renderPipelineType;
        other.~GeometryNode();
    }
  
    // the first push_back is a move, the second and subsequent pushes may trigger a vector reallocation as it may run out of space so it will copy all contents to a new buffer:
    GeometryNode<T>(const GeometryNode<T>& other) {
        std::cout << "Copied" << "\n";
        
        if (vertexCount != other.vertexCount) {
            if (Verticies && ownership) { delete [] Verticies; }
            vertexCount = other.vertexCount;
            if (vertexType == VertexType::Vertex3D) {
                Verticies = new Vertex3D[vertexCount];
            } else {
                Verticies = new Point3D[vertexCount];
            }
            
        }
        
        uint32_t size_of_verts = VertexType::Vertex3D == vertexType ? sizeof(Vertex3D) : sizeof(Point3D);
        if (Verticies) {memcpy(Verticies, other.Verticies, vertexCount * size_of_verts);}
        
        if (indexCount != other.indexCount) {
            if (indices && ownership) { delete [] indices; }
            indexCount = other.indexCount;
            indices = new T[indexCount];
        }
        
        if (indices) { memcpy(indices, other.indices, indexCount * sizeof(T)); }
        
        if (instances != other.instances) {
            if (instanceMatricies) { delete [] instanceMatricies; }
            delete [] instanceMatricies;
//            instances = other.instances;
            instanceMatricies = new simd_float4x4[other.instances];
        }
        if (instanceMatricies) { memcpy(instanceMatricies, other.instanceMatricies, other.instances * sizeof(simd_float4x4)); }
        
        if (!preMulparentInstanceMatricies) {
            int totalInstance = other.parentInstances == 0 ? other.instances : other.parentInstances * other.instances;
            preMulparentInstanceMatricies = new simd_float4x4[totalInstance];
            memcpy(preMulparentInstanceMatricies, other.preMulparentInstanceMatricies, totalInstance * sizeof(simd_float4x4));
        } else {
            if (instances * parentInstances != other.instances * other.parentInstances) {
                if (preMulparentInstanceMatricies) { delete [] preMulparentInstanceMatricies; }
                int totalInstance = other.parentInstances == 0 ? other.instances : other.parentInstances * other.instances;
                preMulparentInstanceMatricies = new simd_float4x4[totalInstance];
                memcpy(preMulparentInstanceMatricies, other.preMulparentInstanceMatricies, totalInstance * sizeof(simd_float4x4));
            }
        }
        instances = other.instances;
        parentInstances = other.parentInstances;
        parentInstanceMatricies = other.parentInstanceMatricies;
//        if (preMulparentInstanceMatricies) {
//            if (other.preMulparentInstanceMatricies) {
//                memcpy(preMulparentInstanceMatricies, other.preMulparentInstanceMatricies, instances * parentInstances * sizeof(simd_float4x4));
//            } else {
//                preMulparentInstanceMatricies = nil;
//            }
//
//        }


        position = other.position;
        rotation = other.rotation;
        scale = other.scale;
        
        globalMatrix = other.globalMatrix;
        modelMatrix = other.modelMatrix;
        
        drawType = other.drawType;
        isTextured = other.isTextured;
        childNodes = other.childNodes;

        texture = other.texture;
        renderStateBlock = other.renderStateBlock;
        RenderStateNo = other.RenderStateNo;
        
        visible = other.visible;
        name = other.name;
        vertexType = other.vertexType;
        renderPipelineType = other.renderPipelineType;
    }
//
//
//
    // copy assignment
    GeometryNode<T>& operator=(const GeometryNode<T>& other) {
        if (&other == this) { }
        else {
            std::cout << "Copied" << "\n";
            
            if (vertexCount != other.vertexCount) {
                if (Verticies && ownership) { delete [] Verticies; }
                vertexCount = other.vertexCount;
                if (vertexType == VertexType::Vertex3D) {
                    Verticies = new Vertex3D[vertexCount];
                } else {
                    Verticies = new Point3D[vertexCount];
                }
                
            }
            uint32_t size_of_verts = VertexType::Vertex3D == other.vertexType ? sizeof(Vertex3D) : sizeof(Point3D);
            if (Verticies) {memcpy(Verticies, other.Verticies, vertexCount * size_of_verts);}
            
            if (indexCount != other.indexCount) {
                if (indices && ownership) { delete [] indices; }
                indexCount = other.indexCount;
                indices = new T[indexCount];
            }
            
            if (indices) { memcpy(indices, other.indices, indexCount * sizeof(T)); }
            
            if (instances != other.instances) {
                if (instanceMatricies) { delete [] instanceMatricies; }
                delete [] instanceMatricies;
    //            instances = other.instances;
                instanceMatricies = new simd_float4x4[other.instances];
            }
            if (instanceMatricies) { memcpy(instanceMatricies, other.instanceMatricies, other.instances * sizeof(simd_float4x4)); }
            
            if (instances * parentInstances != other.instances * other.parentInstances) {
                if (preMulparentInstanceMatricies) { delete [] preMulparentInstanceMatricies; }
                preMulparentInstanceMatricies = new simd_float4x4[other.instances * other.parentInstances];
            }
            instances = other.instances;
            parentInstances = other.parentInstances;
            parentInstanceMatricies = other.parentInstanceMatricies;
            
            if (preMulparentInstanceMatricies) {
                if (other.preMulparentInstanceMatricies) {
                    memcpy(preMulparentInstanceMatricies, other.preMulparentInstanceMatricies, instances * parentInstances * sizeof(simd_float4x4));
                } else {
                    preMulparentInstanceMatricies = nil;
                }
                
            }


            position = other.position;
            rotation = other.rotation;
            scale = other.scale;
            
            globalMatrix = other.globalMatrix;
            modelMatrix = other.modelMatrix;
            
            drawType = other.drawType;
            childNodes = other.childNodes;
            isTextured = other.isTextured;
            
            texture = other.texture;
            renderStateBlock = other.renderStateBlock;
            RenderStateNo = other.RenderStateNo;
            
            visible = other.visible;
            name = other.name;
            vertexType = other.vertexType;
            renderPipelineType = other.renderPipelineType;
        }
        return *this;
    }
//
    GeometryNode<T>& operator=(GeometryNode<T>&& other) {
        std::cout << "Moved" << "\n";
        this->~GeometryNode();
        
        Verticies = other.Verticies;
        vertexBuffer = other.vertexBuffer;
        other.Verticies = nullptr;
        other.vertexBuffer = nullptr;
        
        indices = other.indices;
        indexBuffer = other.indexBuffer;
        other.indices = nullptr;
        other.indexBuffer = nullptr;
        
        vertexCount = other.vertexCount;
        indexCount = other.indexCount;
        instances = other.instances;
        parentInstances = other.parentInstances;
        
        instanceMatricies = other.instanceMatricies;
        other.instanceMatricies = nullptr;
        
        preMulparentInstanceMatricies = other.preMulparentInstanceMatricies;
        other.preMulparentInstanceMatricies = nullptr;
        
        parentInstanceMatricies = other.parentInstanceMatricies;
        other.parentInstanceMatricies = nullptr;
        
        position = other.position;
        rotation = other.rotation;
        scale = other.scale;
        
        globalMatrix = other.globalMatrix;
        modelMatrix = other.modelMatrix;
        
        drawType = other.drawType;
        isTextured = other.isTextured;
        ownership = other.ownership;
        childNodes = std::move(other.childNodes);
        
        texture = other.texture;
        renderStateBlock = other.renderStateBlock;
        RenderStateNo = other.RenderStateNo;
        
        visible = other.visible;
        name = std::move(other.name);
        vertexType = other.vertexType;
        renderPipelineType = other.renderPipelineType;

        other.~GeometryNode();
        return *this;
    }
    
    ~GeometryNode() {
        if (Verticies && ownership) { delete [] Verticies; }
        if (indices && ownership) { delete [] indices; }
        if (instanceMatricies) { delete [] instanceMatricies; }
        if (preMulparentInstanceMatricies) { delete [] preMulparentInstanceMatricies; }
    }
};

#endif /* GeometryNode_h */


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
@import std_core.type_traits.is_base_of;
#include <iostream>
#include <vector>

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




template<typename T>
class GeometryNode {
public:
    std::vector<GeometryNode<T>> childNodes;
    
    Vertex3D* Verticies = nil;
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
    
    bool update = true;
    bool isTextured = false;
    bool visible = true;
    int RenderStateNo = 0;
    typedef void (^RenderStateBlock)(id<MTLRenderCommandEncoder> encoder);
    RenderStateBlock renderStateBlock = nil;
    
    id<MTLTexture> texture = nil;
    
    MTLPrimitiveType drawType = MTLPrimitiveTypeTriangleStrip;
    std::string name = "Node";
    
    GeometryNode(Vertex3D* Verticies, int vertexCount, T* indices, int indexCount): childNodes() {
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
        uint32 prevSize = childNodes.size();
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

            vertexBuffer = [metalDevice newBufferWithBytesNoCopy:Verticies length:vertexCount * sizeof(Vertex3D)  options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
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
            for (int i = 0; i<childNodes.size(); i++) {
                childNodes[i].buildBuffers(metalDevice);
            }
            
            
            std::cout << "buffer built" << "\n";
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
    
    
    void draw(id<MTLRenderCommandEncoder> cmdEncoder, id<MTLDevice> metalDevice, id<MTLRenderPipelineState> defaultState, NSMutableArray<id<MTLRenderPipelineState>>* customStates) {
        if (vertexCount != 0 && indexCount != 0 && visible)  {
            buildBuffers(metalDevice);
            if (RenderStateNo != 0) {
                [cmdEncoder setRenderPipelineState:[customStates objectAtIndex:RenderStateNo-1]];
                renderStateBlock(cmdEncoder);
            }
            [cmdEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
            [cmdEncoder setVertexBytes:&modelMatrix length:sizeof(simd_float4x4) atIndex:1];
//            [cmdEncoder setVertexBytes:&globalMatrix length:sizeof(simd_float4x4) atIndex:2];
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

            [cmdEncoder setVertexBuffer:transformBuffer offset:0 atIndex:3];
            [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
            if (isTextured) {
                [cmdEncoder setFragmentTexture:texture atIndex:0];
            }
            [cmdEncoder drawIndexedPrimitives: drawType indexCount:indexCount indexType:MTLIndexTypeUInt16 indexBuffer:indexBuffer indexBufferOffset:0 instanceCount:totalInstance];
            if (RenderStateNo != 0) {
                [cmdEncoder setRenderPipelineState:defaultState];
            }
        }
        for (int i=0; i<childNodes.size(); i++) {
            childNodes[i].draw(cmdEncoder, metalDevice, defaultState, customStates);
        }
    }
    
    GeometryNode<T>( GeometryNode<T>&& other) {
        std::cout << "Moved" << "\n";
        this->~GeometryNode();
        
        Verticies = other.Verticies;
        other.Verticies = nullptr;
        
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
        
        visible = other.visible;
        name = std::move(other.name);
        other.~GeometryNode();
    }
  
    // the first push_back is a move, the second and subsequent pushes may trigger a vector reallocation as it may run out of space so it will copy all contents to a new buffer:
    GeometryNode<T>(const GeometryNode<T>& other) {
        std::cout << "Copied" << "\n";
        
        if (vertexCount != other.vertexCount) {
            if (Verticies) { delete [] Verticies; }
            vertexCount = other.vertexCount;
            Verticies = new Vertex3D[vertexCount];
            
        }
        if (Verticies) {memcpy(Verticies, other.Verticies, vertexCount * sizeof(Vertex3D));}
        
        if (indexCount != other.indexCount) {
            if (indices) { delete [] indices; }
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
                if (Verticies) { delete [] Verticies; }
                vertexCount = other.vertexCount;
                Verticies = new Vertex3D[vertexCount];
                
            }
            if (Verticies) {memcpy(Verticies, other.Verticies, vertexCount * sizeof(Vertex3D));}
            
            if (indexCount != other.indexCount) {
                if (indices) { delete [] indices; }
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
        }
        return *this;
    }
//
    GeometryNode<T>& operator=(GeometryNode<T>&& other) {
        std::cout << "Moved" << "\n";
        this->~GeometryNode();
        
        Verticies = other.Verticies;
        other.Verticies = nullptr;
        
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
        isTextured = other.isTextured;
        childNodes = std::move(other.childNodes);
        
        texture = other.texture;
        renderStateBlock = other.renderStateBlock;
        RenderStateNo = other.RenderStateNo;
        
        visible = other.visible;
        name = std::move(other.name);
        
        other.~GeometryNode();
        return *this;
    }
    
    ~GeometryNode() {
        if (Verticies) { delete [] Verticies; }
        if (indices) { delete [] indices; }
        if (instanceMatricies) { delete [] instanceMatricies; }
        if (preMulparentInstanceMatricies) { delete [] preMulparentInstanceMatricies; }
    }
};

#endif /* GeometryNode_h */

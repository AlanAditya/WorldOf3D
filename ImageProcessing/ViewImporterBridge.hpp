//
//  ViewImporterBridge.h
//  WorldOf3D
//
//  Created by Manoj Kumar on 17/08/24.
//

#ifndef ViewImporterBridge_h
#define ViewImporterBridge_h
#include <MetalKit/MetalKit.hpp>
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>

//NS::View* createHelloWorldView();
////NS::View* createHelloWorldSwiftView();
//NS::View* add_arrays(const float* inA,
//                const float* inB,
//                float* result,
//                int length);
//void createHelloWorldSwiftViewMTK(const NS::View* metalView);
enum class Geometry {
    Cube,
    Sphere,
    Donut
};

struct Object3D {
    std::string id;        // UUID as a string
    Geometry shape;        // Geometry enum
    simd_float3 location;  // SIMD3<Float> equivalent
    simd_float3 rotation;  // SIMD3<Float> equivalent
    simd_float3 scale;   
};

struct ExchangePackage { Object3D* objec; NS::View* view; };

//NS::View* createHelloWorldSwiftViewMTK(NS::View* metalView);

ExchangePackage SwiftCPP_Bridge(NS::View *metalView);


//class SwiftUI_C_Plus_plus : public NS::Object
//{
//    public:
//        virtual void createHelloWorldSwiftViewMTK(NS::View* metalView);
//};





#ifdef __cplusplus
extern "C" {
#endif





//void createHelloWorldSwiftViewMTK1(NS::View* metalView) {
//}



#ifdef __cplusplus
}
#endif
#endif /* ViewImporterBridge_h */

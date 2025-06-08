//
//  3DCameraExtension.h
//  WorldOf3D
//
//  Created by Manoj Kumar on 05/03/25.
//

#ifndef _DCameraExtension_h
#define _DCameraExtension_h

#include "AlgebroHeap.hpp"
#include <simd/simd.h>

enum class TransformationMode {
    Orbit,
    Translate,
    Zoom
};

class Camera3D {
    simd_float3 position = {0.0, 0.0, -3.0};
    simd_float3 oldPosition = {0.0, 0.0, -3.0};
    simd_float3 target = {0, 0, 0};

    simd_float3 up = {0, 1, 0};
    simd_float3 right = {1, 0, 0};
    simd_float3 forward = {0, 0, 1};
    
    float farP = 10;
    float nearP = 0.1;
    
    float r = 1;
    float d = 1;
    
    
    bool updated = false;

public:
    simd_float4x4 viewMatrix;
    Camera3D() {
        simd_float3 vec = target - position;
        forward = simd::normalize(vec);
        right = simd::normalize(simd::cross(forward, up));
        up = simd::normalize(simd::cross(right, forward));
    }
    
    void updatePosition(simd_float2 drag, TransformationMode mode) {
        if (mode == TransformationMode::Translate) {
            auto dist = simd::length(target - position);
            position += up * drag.y + right * drag.x;
            target = position + simd_normalize(forward) * dist;
        }
        else if (mode == TransformationMode::Orbit) {
            
            position = (RotationY(drag.x * 100) * RotationX(drag.y * 100) * simd_make_float4(position - target, 1) ).xyz + target;
            
            simd_float3 vec = target - position;
            forward = simd::normalize(vec);
            right = simd::normalize(simd::cross(forward, up));
            up = simd::normalize(simd::cross(right, forward));
        }
        else if (mode == TransformationMode::Zoom) {
            position = target + (position - target) * drag.x;
        }
        updated = false;
    }
    
    void setPosition(simd_float2 drag, TransformationMode mode) {
        if (mode == TransformationMode::Translate) {
            auto dist = simd::length(target - position);
            position = oldPosition + up * drag.y + right * drag.x;
            target = position + simd_normalize(forward) * dist;
        }
        else if (mode == TransformationMode::Orbit) {
            
            simd_float3 vec = position - target;
            std::cout << drag << "\n";
            position = (RotationY(drag.x * 0.01) * RotationX(drag.y * 0.01) * simd_make_float4(vec, 1) ).xyz + target;
            std::cout << position << "\n";
//            vec = target - position;
//            forward = simd::normalize(vec);
//            right = simd::normalize(simd::cross(forward, up));
//            up = simd::normalize(simd::cross(right, forward));
        }
        else if (mode == TransformationMode::Zoom) {
            position = target + (position - target) * drag.x;
        }
        updated = false;
    }
    
    void updateMatrix() {
        if (!updated) {
            simd_float3 zAxis = -simd::normalize(position - target);
            simd_float3 xAxis = simd::normalize(simd::cross(up, zAxis));

            simd_float3 yAxis = simd::cross(zAxis, xAxis);

        //    upVector = yAxis
            // dot is used because first the rotation is undone then
            simd_float4 row0 = {xAxis.x, xAxis.y, xAxis.z, -simd::dot(xAxis, position)};
            simd_float4 row1 = {yAxis.x, yAxis.y, yAxis.z, -simd::dot(yAxis, position)};
            simd_float4 row2 = {zAxis.x, zAxis.y, zAxis.z, -simd::dot(zAxis, position)};
            simd_float4 row3 = {0,      0,      0,      1         };

//            simd_float4 row0 = {right.x, right.y, right.y, -position.x};
//            simd_float4 row1 = {up.x, up.y, up.z, position.y};
//            simd_float4 row2 = {forward.x, forward.y, forward.z, -position.z};
//            simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
            
            float k = farP - nearP;
            simd_float4 row4 = {1/r,  0,    0,    0};
            simd_float4 row5 = {0,    1/d,  0,    0};
            simd_float4 row6 = {0,    0,    1/k,  0};
            simd_float4 row7 = {0.0f, 0.0f, 0.0f, 1.0f};
            simd_float4x4 camMat =  simd_matrix_from_rows(row0, row1, row2, row3);
            simd_float4x4 clipMat = simd_matrix_from_rows(row4, row5, row6, row7);
            viewMatrix = simd_mul(clipMat, camMat);
            updated = true;
        }
    }
};

#endif /* _DCameraExtension_h */

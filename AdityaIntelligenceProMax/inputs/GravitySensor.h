#pragma once
#import <simd/simd.h>

class GravitySensor {
public:
    GravitySensor();
    ~GravitySensor();
    
    // Pulls the current gravity vector. If motion data is not yet available,
    // it returns a default gravity vector, e.g., (0, -1, 0).
    simd_float3 pull();
    
private:
    void* motionManager; // Opaque pointer to CMMotionManager
};

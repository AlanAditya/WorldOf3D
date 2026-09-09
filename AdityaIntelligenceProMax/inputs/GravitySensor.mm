#ifdef OS_TARGET_IPHONE
#import "GravitySensor.h"
#import <CoreMotion/CoreMotion.h>

GravitySensor::GravitySensor() {
    CMMotionManager *manager = [[CMMotionManager alloc] init];
    manager.deviceMotionUpdateInterval = 1.0 / 60.0; // Update at 60Hz
    
    // Start updates immediately upon initialization
    if (manager.isDeviceMotionAvailable) {
        [manager startDeviceMotionUpdates];
    }
    
    // Retain it so it isn't destroyed
    motionManager = (void*)CFBridgingRetain(manager);
}

GravitySensor::~GravitySensor() {
    CMMotionManager *manager = (__bridge_transfer CMMotionManager*)motionManager;
    [manager stopDeviceMotionUpdates];
    manager = nil;
}

simd_float3 GravitySensor::pull() {
    CMMotionManager *manager = (__bridge CMMotionManager*)motionManager;
    
    CMDeviceMotion *motion = manager.deviceMotion;
    if (motion != nil) {
        // CoreMotion provides gravity in the device's reference frame
        return simd_make_float3((float)motion.gravity.x, 
                                (float)motion.gravity.y, 
                                (float)motion.gravity.z);
    }
    
    // Return a default downward vector if data is not available yet
    return simd_make_float3(0.0f, -1.0f, 0.0f);
}
#endif

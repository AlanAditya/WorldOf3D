//
//  Header.h
//  WorldOf3D
//
//  Created by Manoj Kumar on 28/01/25.
//

#ifndef Header_h
#define Header_h

#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>

@interface EditorViewController : NSViewController
@end

@interface MetalViewDelegate : NSObject<MTKViewDelegate>
{
    @public
        id<MTLDevice> metalDevice;
        id<MTLCommandQueue> CommandQueue;
        id<MTLRenderPipelineState> BasicRenderPipelineState;
    }
@end


#endif /* Header_h */

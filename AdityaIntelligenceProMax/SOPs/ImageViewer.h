#pragma once
#include <mutex>
#include "Viewer.cpp"

// A Viewer that shows one BGRA image stretched over its viewport: the renderer's base image
// (updateBaseImageV2), but as a Viewer, so it can be a page, have its own viewport, or be hidden.
// Make it with -[Renderer makeImageViewer] (it needs the renderer's plane pipeline).
//
// update() is safe from any thread (e.g. an applet's capture queue); draw() runs on the render
// thread. Like updateBaseImageV2, the pixels are copied into the texture the GPU may be reading,
// so a frame can show half of the old image and half of the new one.
class ImageViewer : public Viewer {
public:
    // `restore_depth_state` is the renderer's usual one, put back after drawing.
    ImageViewer(id<MTLDevice> device, id<MTLRenderPipelineState> plane_state, id<MTLDepthStencilState> restore_depth_state)
        : device(device), plane_state(plane_state), restore_depth_state(restore_depth_state) {
        interactive = false;   // nothing to orbit
        show_grid = false;
        // The quad sits at the same depth (0.99999) as the renderer's base-image quad, which is
        // drawn first and writes it, so with the usual Less test every pixel of this one failed.
        // It's a background: always drawn, and it leaves the depth alone for whatever comes after.
        MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = NO;
        depth_state = [device newDepthStencilStateWithDescriptor:desc];
    }

    // `layer` is [H, W, 4] BGRA (UInt8, or anything astype can turn into it).
    void update(matrix layer) {
        if (layer.dims != 3 || layer.shape()[2] != 4) {
            throw std::runtime_error("ImageViewer::update: expected [H, W, 4] BGRA");
        }
        if (layer.type != dtype::UInt8) layer = layer.astype(dtype::UInt8);
        if (layer.tape) layer.eval_metal();
        if (layer.total_size == 0 || !layer.buffer) return;

        NSUInteger w = layer.shape()[1], h = layer.shape()[0];
        id<MTLTexture> tex;
        {
            std::lock_guard<std::mutex> guard(lock);
            tex = texture;
        }
        bool fresh = !tex || tex.width != w || tex.height != h;
        if (fresh) {
            MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                                                                            width:w
                                                                                           height:h
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModeShared;
            tex = [device newTextureWithDescriptor:desc];
        }
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:layer.buffer bytesPerRow:w * 4];
        if (fresh) {
            // Swapped in only once it holds a whole image.
            std::lock_guard<std::mutex> guard(lock);
            texture = tex;
        }
    }

    void draw(id<MTLRenderCommandEncoder> cmdEncoder,
              id<MTLDevice> metalDevice,
              id<MTLRenderPipelineState> __strong (&predefinedStates)[7],
              Camera3D* cam,
              int& active_state) override {
        id<MTLTexture> tex;
        {
            std::lock_guard<std::mutex> guard(lock);
            tex = texture;
        }
        if (!tex) return;

        // The renderer's base-image quad: NDC corners + UVs, filling the viewport.
        float vertexData[] = {-1, -1, 0, 1,
                               1, -1, 1, 1,
                              -1,  1, 0, 0,
                               1,  1, 1, 0};
        [cmdEncoder setRenderPipelineState:plane_state];
        [cmdEncoder setDepthStencilState:depth_state];
        [cmdEncoder setVertexBytes:vertexData length:sizeof(vertexData) atIndex:0];
        [cmdEncoder setFragmentTexture:tex atIndex:0];
        [cmdEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [cmdEncoder setDepthStencilState:restore_depth_state];
        active_state = -1;   // not one of the predefined states: the next Viewer must set its own
    }

private:
    id<MTLDevice> device;
    id<MTLRenderPipelineState> plane_state;
    id<MTLDepthStencilState> depth_state;
    id<MTLDepthStencilState> restore_depth_state;
    id<MTLTexture> texture;
    std::mutex lock;
};

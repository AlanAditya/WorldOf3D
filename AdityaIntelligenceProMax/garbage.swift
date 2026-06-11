//
//  garbage.swift
//  WorldOf3D
//
//  Created by Aditya Dudeja on 16/11/25.
//

//            .gesture(DragGesture()
//                .onChanged { value in
//                    inteligence!.updateCam(CGSize(width: ((value.translation.width - oldDrag.width) / 1.0), height: ((value.translation.height - oldDrag.height) / 1.0)), TransformationMode.Orbit, 1)
//                    oldDrag = value.translation
//                }
//                .onEnded { _ in
//                    oldDrag = .zero
//                    inteligence!.updateCamOLD(1)
//                })
//            .gesture(MagnificationGesture().onChanged { value in
//                print("Zoom scale: \(value)")
//                inteligence!.updateCam(CGSize(width: value, height: value), TransformationMode.Zoom, 1)
//            }
//            .onEnded({_ in
//                print("Zoom endereço:")
//                inteligence!.updateCamOLD(1);
//            }))
//        MTKViewWrapper(intel: inteligence!, viewNo: 2)
//            .gesture(DragGesture()
//                .onChanged { value in
//                    inteligence!.updateCam(CGSize(width: ((value.translation.width - oldDrag.width) / 1.0), height: ((value.translation.height - oldDrag.height) / 1.0)), TransformationMode.Orbit, 2)
//                    oldDrag = value.translation
//                }
//                .onEnded { _ in
//                    oldDrag = .zero
//                    inteligence!.updateCamOLD(2)
//                })
//            .gesture(MagnificationGesture().onChanged { value in
//                print("Zoom scale: \(value)")
//                inteligence!.updateCam(CGSize(width: value, height: value), TransformationMode.Zoom, 2)
//            }
//            .onEnded({_ in
//                print("Zoom endereço:")
//                inteligence!.updateCamOLD(2);
//            }))

//func updateCamera() {
//        var delta = CGSize.zero
//
//        if pressedKeys.contains(123) { // Left
//            delta.width -= 2
//        }
//        if pressedKeys.contains(124) { // Right
//            delta.width += 2
//        }
//        if pressedKeys.contains(125) { // Down
//            delta.height += 2
//        }
//        if pressedKeys.contains(126) { // Up
//            delta.height -= 2
//        }
//
//        if delta != .zero {
//            inteligence?.updateCam(delta, TransformationMode.Translate, 1)
//        }
//    }

//void matrix::add_cpu_brodcasted(const matrix &other, matrix &result, EvalType evalType) const {
//    if (evalType == EvalType::EVAL_AUTO) {
//        // lazy evaluation so auto is the default one so
//        // when u type a+b it gets called with auto so it
//        // just calculates the shape strides and stores
//        // them in the primitive
//        auto primit = new AdditionPrimitive(*this, other);
//        primit->desc_a = BroadcastDescriptor::create(result.dims);
//        primit->desc_b = BroadcastDescriptor::create(result.dims);
//        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
//                         primit->desc_a, primit->desc_b, dims, other.dims);
//        result.total_size = result.accumul(0, result.dims);
//        result.tape = primit;
//        return;
//    }
//    
//    // FOR PATH BUILD_TRACE AND EVAL_CPU
//    // for situations where c = a+b; d= c+c
//    // c is an unmaterialised temp so it has no buffer thus both c's where treated differently or for that matter reusing temp nodes recalcuated and allocated everything cause we didnt know they were the same thing but same nodes shared same primitive so we stored some of the properties in the primitive so we can identify if somwhere else memory for c has been allocated then we use the same
//    if (result.tape->out_buffer && !result.buffer) {
//        result.buffer = result.tape->out_buffer;
//        result.metalBuffer = result.tape->out_metal_buffer;
//        result.refCount = result.tape->out_refcount;
//        result.refCount->fetch_add(1);
//        return;
//    } else {
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//        
//        result.tape->out_buffer = (uint8_t*)result.buffer;
//        result.tape->out_metal_buffer = result.metalBuffer;
//        result.tape->out_refcount = result.refCount;
//    }
//    
////
////    if (result.total_size != result.accumul(0, result.dims) || (evalType == EvalType::EVAL_CPU)) {
////        // as during execution of compiled graph we
////        // trust everything that memory has been
////        // allocated shapes are precomputed
////        result.begin_refcount();
////        result.total_size = result.accumul(0, result.dims);
////        result.releaseBuffer();
////        result.buffer =
////        new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
////        if (result.total_size > 10) {
////            result.buildMetalBuffer();
////        }
////    }
////
////    if (evalType == EvalType::BUILD_TRACE) { // compiles the graph in compile i mean finalises
////        // the shape and size and allocates memory so
////        // great for tasks where called repeatedly
////        result.begin_refcount();
////        result.releaseBuffer();
////        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
////        if (result.total_size > 10) {
////            result.buildMetalBuffer();
////        }
////        return;
////    }
//    
//    // addition logic
//    
//    
//    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
//    if (result.dims != fmax(dims, other.dims)) {
//        throw std::invalid_argument("Incompatible dims of the result mat");
//    }
//    
//    size_m *strideA = ((AdditionPrimitive *)result.tape)->desc_a->strides(result.dims);
//    size_m *strideB = ((AdditionPrimitive *)result.tape)->desc_b->strides(result.dims);
//    
//    dispatch_type(type, result.buffer, [&](auto *out_data) {
//        using T = std::decay_t<decltype(*out_data)>;
//        for (int gid = 0; gid < result.total_size; gid++) {
//            // globalIndex for A and B
//            size_t GindexA = 0;
//            size_t GindexB = 0;
//            
//            // axis Index for Result like temp storage for i, j, k, l ... along each
//            // axis for result
//            size_t indexR = 0;
//            
//            int rem = gid;
//            for (int i = 0; i < result.dims; i++) {
//                indexR = rem / result.strides()[i];
//                GindexA += indexR * strideA[i];
//                GindexB += indexR * strideB[i];
//                rem %= result.strides()[i];
//            }
//            
//            out_data[gid] = static_cast<T *>(buffer)[GindexA] +
//            static_cast<T *>(other.buffer)[GindexB];
//        }
//    });
//}

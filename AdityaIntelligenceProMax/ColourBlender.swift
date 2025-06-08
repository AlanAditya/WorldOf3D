//
//  ColourBlender.swift
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 05/06/25.
//

import SwiftUI

struct ColourBlender: View {
    @State var inteligence = Intelligence(CGRect(x: 0, y: 0, width: 300, height: 300))
    @State var R: Double = 1
    @State var G: Double = 1
    @State var B: Double = 1
    var body: some View {
        VStack {
            Slider(value: $R, in: 0...1)
            Slider(value: $G, in: 0...1)
            Slider(value: $B, in: 0...1)
            MTKViewWrapper(intel: inteligence!, viewNo: 1)
            MTKViewWrapper(intel: inteligence!, viewNo: 2)
        }
        .onAppear {
            
//            DispatchQueue.global(qos: .userInitiated).async {
                inteligence?.memoryTestLogic()
//            }
        }
        .onChange(of: R) { oldValue, newValue in
            inteligence?.red = Float(newValue)
        }
        .onChange(of: G) { oldValue, newValue in
            inteligence?.green = Float(newValue)
        }
        .onChange(of: B) { oldValue, newValue in
            inteligence?.blue = Float(newValue)
        }
        .padding()
    }
}






#Preview {
    ContentView()
}

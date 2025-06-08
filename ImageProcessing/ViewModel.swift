//
//  ViewModel.swift
//  ImageProcessing
//
//  Created by Manoj Kumar on 20/08/24.
//

import Foundation
import simd

enum Geometry: String {
    case cube = "Cube"
    case sphere = "Sphere"
    case donut = "Donut"
}

public struct Object3D {
    let id = UUID()
    var shape: Geometry
    var location: SIMD3<Float> = SIMD3<Float>(x: 0, y: 0, z: 0)
    var rotation: SIMD3<Float> = SIMD3<Float>(x: 0, y: 0, z: 0)
    var scale: SIMD3<Float> = SIMD3<Float>(x: 1, y: 1, z: 1)
}
    

var Objects = [
    Object3D(shape: .cube)
]

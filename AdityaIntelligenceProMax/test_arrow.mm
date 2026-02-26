//#import <Foundation/Foundation.h>
////#import "MatrixH.h"
//#import <iostream>
//
//int main() {
//    GlobalGPUManager.initAll();
//    
//    uint32_t grid_x_size = 2;
//    uint32_t grid_y_size = 2;
//    uint32_t grid_z_size = 2;
//
//    auto field = MatrixH<4, float>::constant({grid_x_size, grid_y_size, grid_z_size, 3}, 1.0f);
//    
//    auto transforms_for_arrow = MatrixH<5, float>::repeatingGPU({grid_x_size, grid_y_size, grid_z_size}, MatrixH<2, float>::eye(4));
//    
//    transforms_for_arrow.flags |= NON_OWNERSHIP_FLAG;
//    
//    transforms_for_arrow.Slice({ {null} , {null}, {null} , {{0,1}}, {{0, 3}} }).flatten<3>() = MatrixH<2, float>({ {0, -1, 0}, {1, 0, 0}, {0, 0, 1} }).Dot(std::move(field).flatten<2>().T()).T();
//
//    std::move(transforms_for_arrow).flatten<2>().print();
//    
//    return 0;
//}

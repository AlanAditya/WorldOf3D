//
//  SidePannel.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 08/12/25.
//

#ifndef SidePannel_h
#define SidePannel_h
#include <vector>
#include <memory>

@interface FlippedStackView : NSStackView
@end

@interface SidePannel : NSViewController
@property (nonatomic, strong) FlippedStackView* Vstack;
- (instancetype)initWithVector:(std::vector<std::shared_ptr<GeometryNode<uint16>>>*)vec;
- (void) updateAssetManager;
@end

@interface Benchmark : NSViewController

@end

#endif /* SidePannel_h */

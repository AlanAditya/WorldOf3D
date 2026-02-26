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
#if !TARGET_OS_IPHONE
@interface FlippedStackView : NSStackView
@end

@interface SidePannel : NSViewController
@property (nonatomic, strong) FlippedStackView* Vstack;
- (instancetype)initWithVector:(std::vector<std::shared_ptr<GeometryNode<uint16_t>>>*)vec;
- (void) updateAssetManager;
@end

@interface Benchmark : NSViewController

@end
#endif




#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>

// Forward declare your C++ class if needed
// template <typename T> class GeometryNode;

// 1. Benchmark is just a UIViewController
@interface Benchmark : UIViewController
@end

// 2. SidePannel
@interface SidePannel : UIViewController


@property (nonatomic, strong) UIStackView* Vstack;

// Constructor signature remains identical (C++ works fine here)
- (instancetype)initWithVector:(std::vector<std::shared_ptr<GeometryNode<uint16_t>>>*)vec;

- (void) updateAssetManager;

@end
#endif
#endif /* SidePannel_h */

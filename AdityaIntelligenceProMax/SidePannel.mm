//
//  MatrixH.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 08/12/25.
//


#ifndef MatrixH_h
#define MatrixH_h

@import SwiftUI;

@import GeometryNode;
#import <AdityaIntelligenceProMax-Swift.h>
#import <vector>
#import <string>
#import <iostream>
#import "SidePannel.h"


#if !TARGET_OS_IPHONE
@import Cocoa;
@import AppKit;
@implementation FlippedStackView
- (BOOL)isFlipped { return YES; }
@end

@implementation SidePannel {
    std::vector<std::shared_ptr<GeometryNode<uint16_t>>>* Assets;
}
- (instancetype)initWithVector:(std::vector<std::shared_ptr<GeometryNode<uint16>>>*)vec; {
    self = [super init];
    
    Assets = vec;
    
    return self;
}

- (void)onVisibilityClicked:(NSButton *)sender {
    NSInteger i = sender.tag-3;

    // Modify C++ vector (example: toggle visibility)
    (*Assets)[i]->visible = !(*Assets)[i]->visible;

    // Optional: update icon
    NSString *symbol = (*Assets)[i]->visible ? @"eye" : @"eye.slash";
    sender.image = [NSImage imageWithSystemSymbolName:symbol
                         accessibilityDescription:@"Visibility"];
}

-(void) viewDidLoad {
    [super viewDidLoad];

    std::vector<std::string> assets = {"Hello", "How are U"
    };
    
    NSColor* lightStrip = [NSColor colorWithName:nil dynamicProvider:^NSColor * _Nonnull(NSAppearance* appearance) {
        if ([appearance.name isEqualToString:NSAppearanceNameDarkAqua] ||
            [appearance.name isEqualToString:NSAppearanceNameVibrantDark]) {
            return [NSColor colorWithWhite:0.20 alpha:1.0]; // Dark mode - lighter gray
        } else {
            return [NSColor colorWithWhite:0.95 alpha:1.0]; // Light mode - light gray
        }
    }];

    // Dark strip
    NSColor* darkStrip = [NSColor colorWithName:nil
                             dynamicProvider:^NSColor * _Nonnull(NSAppearance * _Nonnull appearance) {
        if ([appearance.name isEqualToString:NSAppearanceNameDarkAqua] ||
            [appearance.name isEqualToString:NSAppearanceNameVibrantDark]) {
            return [NSColor colorWithWhite:0.15 alpha:1.0]; // Dark mode - darker gray
        } else {
            return [NSColor colorWithWhite:0.90 alpha:1.0]; // Light mode - darker gray
        }
    }];
    
    
    _Vstack = [[FlippedStackView alloc] init];
    _Vstack.translatesAutoresizingMaskIntoConstraints = false;
    _Vstack.orientation = NSUserInterfaceLayoutOrientationVertical;
    _Vstack.spacing = 0;
    
    for (int i = 0; i <  assets.size(); i ++) {
        NSTextField* label = [NSTextField wrappingLabelWithString:[NSString stringWithUTF8String:assets[i].c_str()]];
        label.translatesAutoresizingMaskIntoConstraints = false;
        
        NSView* rowContainer = [[NSView alloc] init];
        rowContainer.translatesAutoresizingMaskIntoConstraints = false;
        rowContainer.wantsLayer = true;
        
        // adding the icon
        // Left icon (geometry indicator)
        NSImageView* geometryIcon = [[NSImageView alloc] init];
        geometryIcon.translatesAutoresizingMaskIntoConstraints = false;
        geometryIcon.image = [NSImage imageWithSystemSymbolName:@"cube.fill" accessibilityDescription:@"Geometry"];
        geometryIcon.contentTintColor = [NSColor systemGrayColor];
        [rowContainer addSubview:geometryIcon];
        
        // Adding a visibility Button
        NSButton* visibilityButton = [NSButton new];
        visibilityButton.translatesAutoresizingMaskIntoConstraints = false;
        visibilityButton.bordered = NO;
        visibilityButton.bezelStyle = NSBezelStyleRegularSquare;
        visibilityButton.contentTintColor = [NSColor systemBlueColor];
        visibilityButton.title = @"";
        visibilityButton.image = [NSImage imageWithSystemSymbolName:@"eye"
                                                          accessibilityDescription:@"Eye"];
        visibilityButton.tag = i;
        [visibilityButton setTarget:self];
        [visibilityButton setAction:@selector(onVisibilityClicked:)];
        
        [rowContainer addSubview:visibilityButton];
        
        
        NSButton* expandButton = [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"chevron.right"
                                                                accessibilityDescription:@"Expand"]
                                          target:self
                                          action:nil];
        expandButton.translatesAutoresizingMaskIntoConstraints = false;
        expandButton.bordered = NO;
        expandButton.bezelStyle = NSBezelStyleRegularSquare;
        expandButton.contentTintColor = [NSColor systemGrayColor];
        expandButton.tag = i;
        [rowContainer addSubview:expandButton];
        

        
        
        if (i % 2 != 0) {
            [rowContainer setValue:darkStrip forKey:@"backgroundColor"];
        } else {
            [rowContainer setValue:lightStrip forKey:@"backgroundColor"];
        }
        [rowContainer addSubview:label];
        [_Vstack addArrangedSubview:rowContainer];
        
        
        [NSLayoutConstraint activateConstraints:@[
            // Row container height
//            [rowContainer.heightAnchor constraintEqualToConstant:30],
            [rowContainer.leadingAnchor constraintEqualToAnchor:_Vstack.leadingAnchor],
            [rowContainer.trailingAnchor constraintEqualToAnchor:_Vstack.trailingAnchor],
            
            // expand button
            [expandButton.leadingAnchor constraintEqualToAnchor:rowContainer.leadingAnchor constant:4],
            [expandButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
            [expandButton.widthAnchor constraintEqualToConstant:20],
            
            // Geometry Icon
            [geometryIcon.leadingAnchor constraintEqualToAnchor:expandButton.trailingAnchor constant:4],
            [geometryIcon.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
            [geometryIcon.widthAnchor constraintEqualToConstant:20],
            
            // Label (center)
            [label.leadingAnchor constraintEqualToAnchor:geometryIcon.trailingAnchor constant:8],
            [label.trailingAnchor constraintEqualToAnchor:visibilityButton.leadingAnchor constant:-4],
            [label.topAnchor constraintEqualToAnchor:rowContainer.topAnchor constant:8],
            [label.bottomAnchor constraintEqualToAnchor:rowContainer.bottomAnchor constant:-8],
            
            // Button Visibility
            [visibilityButton.widthAnchor constraintEqualToConstant:20],
            [visibilityButton.trailingAnchor constraintEqualToAnchor:rowContainer.trailingAnchor constant:-8],
            [visibilityButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor]
        ]];
        

        
//        NSView* host = [SwiftTextHost make:[NSString stringWithUTF8String:assets[i % 3].c_str()]];
        
    }

    

    
    
    NSScrollView* scrollView = [NSScrollView new];
    scrollView.translatesAutoresizingMaskIntoConstraints = false;
    scrollView.hasVerticalScroller = true;
    scrollView.hasHorizontalScroller = false;
    scrollView.documentView = _Vstack;
    [self.view addSubview:scrollView];
    
    [NSLayoutConstraint activateConstraints: @[
                                                [_Vstack.topAnchor constraintEqualToAnchor:scrollView.topAnchor],
                                                [_Vstack.leadingAnchor constraintEqualToAnchor:scrollView.leadingAnchor],
                                                [_Vstack.trailingAnchor constraintEqualToAnchor:scrollView.trailingAnchor],
                                                [scrollView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
                                                [scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
                                                [scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
                                                [scrollView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
                                              ]];

    
    std::cout << "Hello World FUCJ" << "\n";

}

-(void) updateAssetManager {
    
    NSColor* lightStrip = [NSColor colorWithName:nil dynamicProvider:^NSColor * _Nonnull(NSAppearance* appearance) {
        if ([appearance.name isEqualToString:NSAppearanceNameDarkAqua] ||
            [appearance.name isEqualToString:NSAppearanceNameVibrantDark]) {
            return [NSColor colorWithWhite:0.20 alpha:1.0]; // Dark mode - lighter gray
        } else {
            return [NSColor colorWithWhite:0.95 alpha:1.0]; // Light mode - light gray
        }
    }];
    
    // Dark strip
    NSColor* darkStrip = [NSColor colorWithName:nil
                             dynamicProvider:^NSColor * _Nonnull(NSAppearance * _Nonnull appearance) {
        if ([appearance.name isEqualToString:NSAppearanceNameDarkAqua] ||
            [appearance.name isEqualToString:NSAppearanceNameVibrantDark]) {
            return [NSColor colorWithWhite:0.15 alpha:1.0]; // Dark mode - darker gray
        } else {
            return [NSColor colorWithWhite:0.90 alpha:1.0]; // Light mode - darker gray
        }
    }];
    
    for (NSInteger i = self.Vstack.arrangedSubviews.count - 1; i >= 3; i--) {
        NSView *subview = self.Vstack.arrangedSubviews[i];
        [self.Vstack removeArrangedSubview:subview];
        [subview removeFromSuperview];
    }
    
    for (int i = 0; i <  (*Assets).size(); i ++) {
        NSTextField* label = [NSTextField wrappingLabelWithString:[NSString stringWithUTF8String:(*Assets)[i]->name.c_str()]];
        label.translatesAutoresizingMaskIntoConstraints = false;
        
        NSView* rowContainer = [[NSView alloc] init];
        rowContainer.translatesAutoresizingMaskIntoConstraints = false;
        rowContainer.wantsLayer = true;
        
        // adding the icon
        // Left icon (geometry indicator)
        NSImageView* geometryIcon = [[NSImageView alloc] init];
        geometryIcon.translatesAutoresizingMaskIntoConstraints = false;
        geometryIcon.image = [NSImage imageWithSystemSymbolName:@"cube.fill" accessibilityDescription:@"Geometry"];
        geometryIcon.contentTintColor = [NSColor systemGrayColor];
        [rowContainer addSubview:geometryIcon];
        
        // Adding a visibility Button
        NSButton* visibilityButton = [NSButton new];
        visibilityButton.translatesAutoresizingMaskIntoConstraints = false;
        visibilityButton.bordered = NO;
        visibilityButton.bezelStyle = NSBezelStyleRegularSquare;
        visibilityButton.contentTintColor = [NSColor systemBlueColor];
        visibilityButton.title = @"";
        visibilityButton.tag = i+3;
        if ((*Assets)[i]->visible) {
            visibilityButton.image = [NSImage imageWithSystemSymbolName:@"eye"
                                               accessibilityDescription:@"Eye"];
        } else {
            visibilityButton.image = [NSImage imageWithSystemSymbolName:@"eye.slash"
                                    accessibilityDescription:@"Eye Hidden"];
        }
        [visibilityButton setTarget:self];
        [visibilityButton setAction:@selector(onVisibilityClicked:)];
        [rowContainer addSubview:visibilityButton];
        
        // expand button
        NSButton* expandButton = [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"chevron.right"
                                                                accessibilityDescription:@"Expand"]
                                          target:self
                                          action:nil];
        expandButton.translatesAutoresizingMaskIntoConstraints = false;
        expandButton.bordered = NO;
        expandButton.bezelStyle = NSBezelStyleRegularSquare;
        expandButton.contentTintColor = [NSColor systemGrayColor];
        expandButton.tag = i;
        [rowContainer addSubview:expandButton];
        
        
        if (i % 2 != 0) {
            [rowContainer setValue:darkStrip forKey:@"backgroundColor"];
        } else {
            [rowContainer setValue:lightStrip forKey:@"backgroundColor"];
        }

        [rowContainer addSubview:label];
        [_Vstack addArrangedSubview:rowContainer];
        [NSLayoutConstraint activateConstraints:@[
            // Row container height
//            [rowContainer.heightAnchor constraintEqualToConstant:30],
            [rowContainer.leadingAnchor constraintEqualToAnchor:_Vstack.leadingAnchor],
            [rowContainer.trailingAnchor constraintEqualToAnchor:_Vstack.trailingAnchor],
            
            // expand button
            [expandButton.leadingAnchor constraintEqualToAnchor:rowContainer.leadingAnchor constant:4],
            [expandButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
            [expandButton.widthAnchor constraintEqualToConstant:20],
            
            // Geometry Icon
            [geometryIcon.leadingAnchor constraintEqualToAnchor:expandButton.trailingAnchor constant:4],
            [geometryIcon.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
            [geometryIcon.widthAnchor constraintEqualToConstant:20],
            
            // Label (center)
            [label.leadingAnchor constraintEqualToAnchor:geometryIcon.trailingAnchor constant:8],
            [label.trailingAnchor constraintEqualToAnchor:visibilityButton.leadingAnchor constant:-4],
            [label.topAnchor constraintEqualToAnchor:rowContainer.topAnchor constant:8],
            [label.bottomAnchor constraintEqualToAnchor:rowContainer.bottomAnchor constant:-8],
            
            // Button Visibility
            [visibilityButton.widthAnchor constraintEqualToConstant:20],
            [visibilityButton.trailingAnchor constraintEqualToAnchor:rowContainer.trailingAnchor constant:-8],
            [visibilityButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor]
        ]];
        

        
//        NSView* host = [SwiftTextHost make:[NSString stringWithUTF8String:assets[i % 3].c_str()]];
        
    }
}

@end
static const int kViewCount = 5000;
static const int kIterations = 50;
@implementation Benchmark {
    NSMutableArray<NSTextField *> *_labels;
}


- (void)viewDidLoad {
    [super viewDidLoad];
    

    _labels = [NSMutableArray arrayWithCapacity:kViewCount];
    FlippedStackView* Vstack = [[FlippedStackView alloc] init];
    Vstack.translatesAutoresizingMaskIntoConstraints = false;
    Vstack.orientation = NSUserInterfaceLayoutOrientationVertical;
    Vstack.spacing = 0;
    

    for (int i = 0; i < kViewCount; i++) {
        NSTextField *label = [NSTextField labelWithString:@"Test"];
        label.translatesAutoresizingMaskIntoConstraints = NO;
        [_labels addObject:label];
        [Vstack addArrangedSubview:label];


    }
    

    NSScrollView* scrollView = [NSScrollView new];
    scrollView.translatesAutoresizingMaskIntoConstraints = false;
    scrollView.hasVerticalScroller = true;
    scrollView.hasHorizontalScroller = false;
    scrollView.documentView = Vstack;
    [self.view addSubview:scrollView];
    
    
    [NSLayoutConstraint activateConstraints: @[
                                                [Vstack.topAnchor constraintEqualToAnchor:scrollView.topAnchor],
                                                [Vstack.leadingAnchor constraintEqualToAnchor:scrollView.leadingAnchor],
                                                [Vstack.trailingAnchor constraintEqualToAnchor:scrollView.trailingAnchor],
                                                [scrollView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
                                                [scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
                                                [scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
                                                [scrollView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor]
                                              ]];
    
    [self runBenchmark];
}

- (void)runBenchmark {
    double start = CFAbsoluteTimeGetCurrent();

    for (int iter = 0; iter < kIterations; iter++) {
        for (int j = 0; j < 500; j++) {
            int idx = arc4random_uniform(kViewCount);
            _labels[idx].stringValue = @"Updated";
        }
//        [self.view layoutSubtreeIfNeeded];
    }

    double end = CFAbsoluteTimeGetCurrent();
    NSLog(@"AutoLayout Benchmark: %.3f s", end - start);
}
@end
#endif

#if TARGET_OS_IPHONE
@import UIKit;

@interface SidePannel ()
// Keep track of the vector passed in init
@property (nonatomic, assign) std::vector<std::shared_ptr<GeometryNode<uint16_t>>>* Assets;
@end

@implementation SidePannel

- (instancetype)initWithVector:(std::vector<std::shared_ptr<GeometryNode<uint16_t>>>*)vec {
    self = [super initWithNibName:nil bundle:nil];
    if (self) {
        _Assets = vec;
    }
    return self;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];

    // 1. Setup ScrollView (Essential for iOS StackViews)
    UIScrollView *scrollView = [[UIScrollView alloc] init];
    scrollView.translatesAutoresizingMaskIntoConstraints = NO;
    scrollView.alwaysBounceVertical = YES; // Allows scrolling even if content is small
    [self.view addSubview:scrollView];
    
    // 2. Setup StackView
    self.Vstack = [[UIStackView alloc] init];
    self.Vstack.translatesAutoresizingMaskIntoConstraints = NO;
    self.Vstack.axis = UILayoutConstraintAxisVertical;
    self.Vstack.spacing = 0;
    self.Vstack.alignment = UIStackViewAlignmentFill;
    self.Vstack.distribution = UIStackViewDistributionEqualSpacing;
    [scrollView addSubview:self.Vstack];

    // 3. Constraints for ScrollView & StackView
    [NSLayoutConstraint activateConstraints:@[
        // ScrollView pins to View edges
        [scrollView.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor],
        [scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [scrollView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
        
        // StackView pins to ScrollView Content Layout Guide
        [self.Vstack.topAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.topAnchor],
        [self.Vstack.bottomAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.bottomAnchor],
        [self.Vstack.leadingAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.leadingAnchor],
        [self.Vstack.trailingAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.trailingAnchor],
        
        // CRITICAL: StackView width must match ScrollView Frame width (to prevent horizontal scrolling)
        [self.Vstack.widthAnchor constraintEqualToAnchor:scrollView.frameLayoutGuide.widthAnchor]
    ]];

    // 4. Dummy Data Loading (Mirroring your logic)
    std::vector<std::string> assets = {"Hello", "How are U"};
    
    // Define Colors (iOS Dynamic Provider)
    UIColor *lightStrip = [UIColor colorWithDynamicProvider:^UIColor * _Nonnull(UITraitCollection * _Nonnull traitCollection) {
        return (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
            ? [UIColor colorWithWhite:0.20 alpha:1.0]
            : [UIColor colorWithWhite:0.95 alpha:1.0];
    }];
    
    UIColor *darkStrip = [UIColor colorWithDynamicProvider:^UIColor * _Nonnull(UITraitCollection * _Nonnull traitCollection) {
        return (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
            ? [UIColor colorWithWhite:0.15 alpha:1.0]
            : [UIColor colorWithWhite:0.90 alpha:1.0];
    }];

    for (int i = 0; i < assets.size(); i++) {
        UIView *row = [self createRowViewWithTitle:[NSString stringWithUTF8String:assets[i].c_str()]
                                             index:i
                                           isDark:(i % 2 != 0)
                                       lightColor:lightStrip
                                        darkColor:darkStrip
                                          visible:YES];
        [self.Vstack addArrangedSubview:row];
    }
    
    std::cout << "Hello World iOS" << "\n";
}

- (void)updateAssetManager {
    if (!_Assets) return;

    // Define Colors again (or move to helper method)
    UIColor *lightStrip = [UIColor colorWithDynamicProvider:^UIColor * _Nonnull(UITraitCollection * _Nonnull traitCollection) {
        return (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
            ? [UIColor colorWithWhite:0.20 alpha:1.0]
            : [UIColor colorWithWhite:0.95 alpha:1.0];
    }];
    
    UIColor *darkStrip = [UIColor colorWithDynamicProvider:^UIColor * _Nonnull(UITraitCollection * _Nonnull traitCollection) {
        return (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark)
            ? [UIColor colorWithWhite:0.15 alpha:1.0]
            : [UIColor colorWithWhite:0.90 alpha:1.0];
    }];

    // Clean up old views (preserving top 3 as per your logic)
    // Note: If Vstack count < 3, this loop won't run, which is safe.
    for (NSInteger i = self.Vstack.arrangedSubviews.count - 1; i >= 3; i--) {
        UIView *subview = self.Vstack.arrangedSubviews[i];
        [self.Vstack removeArrangedSubview:subview];
        [subview removeFromSuperview];
    }
    
    // Add new views from C++ Vector
    for (int i = 0; i < (*_Assets).size(); i++) {
        std::string name = (*_Assets)[i]->name;
        bool isVisible = (*_Assets)[i]->visible;
        
        UIView *row = [self createRowViewWithTitle:[NSString stringWithUTF8String:name.c_str()]
                                             index:i
                                           isDark:(i % 2 != 0)
                                       lightColor:lightStrip
                                        darkColor:darkStrip
                                          visible:isVisible];
        
        [self.Vstack addArrangedSubview:row];
    }
}

// Helper to create rows (avoids code duplication between viewDidLoad and updateAssetManager)
- (UIView *)createRowViewWithTitle:(NSString *)title
                             index:(int)i
                            isDark:(BOOL)isDark
                        lightColor:(UIColor *)lightColor
                         darkColor:(UIColor *)darkColor
                           visible:(BOOL)isVisible {
    
    UIView *rowContainer = [[UIView alloc] init];
    rowContainer.translatesAutoresizingMaskIntoConstraints = NO;
    rowContainer.backgroundColor = isDark ? darkColor : lightColor;
    
    // 1. Expand Button (Chevron)
    UIButton *expandButton = [UIButton buttonWithType:UIButtonTypeSystem];
    expandButton.translatesAutoresizingMaskIntoConstraints = NO;
    [expandButton setImage:[UIImage systemImageNamed:@"chevron.right"] forState:UIControlStateNormal];
    expandButton.tintColor = [UIColor systemGrayColor];
    expandButton.tag = i;
    // [expandButton addTarget:self action:@selector(onExpandClicked:) forControlEvents:UIControlEventTouchUpInside];
    [rowContainer addSubview:expandButton];
    
    // 2. Geometry Icon (Cube)
    UIImageView *geometryIcon = [[UIImageView alloc] init];
    geometryIcon.translatesAutoresizingMaskIntoConstraints = NO;
    geometryIcon.image = [UIImage systemImageNamed:@"cube.fill"];
    geometryIcon.tintColor = [UIColor systemGrayColor];
    geometryIcon.contentMode = UIViewContentModeScaleAspectFit;
    [rowContainer addSubview:geometryIcon];
    
    // 3. Label
    UILabel *label = [[UILabel alloc] init];
    label.translatesAutoresizingMaskIntoConstraints = NO;
    label.text = title;
    label.font = [UIFont systemFontOfSize:14]; // Standard iOS body size
    label.textColor = [UIColor labelColor];
    label.numberOfLines = 0; // Wrapping
    [rowContainer addSubview:label];
    
    // 4. Visibility Button (Eye)
    UIButton *visibilityButton = [UIButton buttonWithType:UIButtonTypeSystem];
    visibilityButton.translatesAutoresizingMaskIntoConstraints = NO;
    
    NSString *iconName = isVisible ? @"eye" : @"eye.slash";
    [visibilityButton setImage:[UIImage systemImageNamed:iconName] forState:UIControlStateNormal];
    
    visibilityButton.tintColor = [UIColor systemBlueColor];
    visibilityButton.tag = i + 3; // Matching your logic
    [visibilityButton addTarget:self action:@selector(onVisibilityClicked:) forControlEvents:UIControlEventTouchUpInside];
    [rowContainer addSubview:visibilityButton];
    
    // 5. Constraints
    [NSLayoutConstraint activateConstraints:@[
        // Expand Button
        [expandButton.leadingAnchor constraintEqualToAnchor:rowContainer.leadingAnchor constant:4],
        [expandButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
        [expandButton.widthAnchor constraintEqualToConstant:20],
        [expandButton.heightAnchor constraintEqualToConstant:20],
        
        // Geometry Icon
        [geometryIcon.leadingAnchor constraintEqualToAnchor:expandButton.trailingAnchor constant:4],
        [geometryIcon.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor],
        [geometryIcon.widthAnchor constraintEqualToConstant:20],
        [geometryIcon.heightAnchor constraintEqualToConstant:20],
        
        // Label
        [label.leadingAnchor constraintEqualToAnchor:geometryIcon.trailingAnchor constant:8],
        [label.trailingAnchor constraintEqualToAnchor:visibilityButton.leadingAnchor constant:-4],
        [label.topAnchor constraintEqualToAnchor:rowContainer.topAnchor constant:12], // More padding for touch targets
        [label.bottomAnchor constraintEqualToAnchor:rowContainer.bottomAnchor constant:-12],
        
        // Visibility Button
        [visibilityButton.widthAnchor constraintEqualToConstant:24], // Slightly larger for touch
        [visibilityButton.heightAnchor constraintEqualToConstant:24],
        [visibilityButton.trailingAnchor constraintEqualToAnchor:rowContainer.trailingAnchor constant:-8],
        [visibilityButton.centerYAnchor constraintEqualToAnchor:rowContainer.centerYAnchor]
    ]];
    
    return rowContainer;
}

- (void)onVisibilityClicked:(UIButton *)sender {
    int index = (int)sender.tag - 3;
    NSLog(@"Visibility clicked for item index: %d", index);
    // Handle C++ logic here
}

@end

#endif

#endif /* MatrixH_h */


//
//  MatrixH.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 08/12/25.
//

#ifndef MatrixH_h
#define MatrixH_h
@import SwiftUI;
@import Cocoa;
@import AppKit;
@import GeometryNode;
#import <AdityaIntelligenceProMax-Swift.h>
#import <vector>
#import <string>
#import <iostream>
#import "SidePannel.h"



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

#endif /* MatrixH_h */

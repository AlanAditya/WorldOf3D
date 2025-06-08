//
//  EditorViewController.m
//  Video Editor
//
//  Created by Manoj Kumar on 28/01/25.
//

#import <Foundation/Foundation.h>

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#import "Header.h"



@implementation EditorViewController

- (void)viewDidAppear {
    // Create the main editor view
    NSRect windowSize = self.view.frame;
    NSView *mainView = [[NSView alloc] initWithFrame:windowSize];
    mainView.wantsLayer = YES;
    mainView.layer.backgroundColor = NSColor.darkGrayColor.CGColor;
    
    NSSplitView* verticalSplitView = [[NSSplitView alloc] initWithFrame:windowSize];
    verticalSplitView.dividerStyle = NSSplitViewDividerStyleThin;
    verticalSplitView.vertical = NO;
    
    NSRect upperPane = windowSize;
    upperPane.size.height = windowSize.size.height * 0.6;
    upperPane.origin.y = 0;
    
    NSRect lowerPane = windowSize;
    lowerPane.size.height = windowSize.size.height * 0.4;
    lowerPane.origin.y = windowSize.size.height * 0.6;
    
#pragma mark - Upper Pane Constraints
    
    NSSplitView* horizontalSplitView = [[NSSplitView alloc] initWithFrame:upperPane];
    horizontalSplitView.dividerStyle = NSSplitViewDividerStyleThin;
    horizontalSplitView.vertical = YES;
    
    auto mediaPaneRect = upperPane;
    mediaPaneRect.size.width = upperPane.size.width * 0.25;
    
    auto MetalHolderPaneRect = upperPane;
    MetalHolderPaneRect.size.width = upperPane.size.width * 0.5;
    
    auto inspectorPaneRect = upperPane;
    inspectorPaneRect.size.width = upperPane.size.width * 0.25;
    // Media Pane
    NSView *mediaPane = [[NSView alloc] initWithFrame:mediaPaneRect];
    mediaPane.wantsLayer = YES;
    mediaPane.layer.backgroundColor = NSColor.grayColor.CGColor;
    NSTextField *mediaLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, 160, 280, 30)];
    mediaLabel.stringValue = @"Media Pane";
    mediaLabel.editable = NO;
    mediaLabel.bezeled = NO;
    mediaLabel.drawsBackground = NO;
    mediaLabel.alignment = NSTextAlignmentCenter;
    mediaLabel.font = [NSFont boldSystemFontOfSize:16];
    [mediaPane addSubview:mediaLabel];
//    [horizontalSplitView addSubview:mediaPane];

    // Metal View for Rendering
    NSView *MetalViewHolder = [[NSView alloc] initWithFrame:MetalHolderPaneRect];
    mediaPane.wantsLayer = YES;
    mediaPane.layer.backgroundColor = NSColor.grayColor.CGColor;
    
    float aspectRatioX = 1920;
    float aspectRatioY = 1080;
    
    auto MetalPaneRect = MetalHolderPaneRect;
    if (MetalHolderPaneRect.size.width / aspectRatioX > MetalHolderPaneRect.size.height / aspectRatioY) {
        MetalPaneRect.size.width = (MetalHolderPaneRect.size.height / aspectRatioY) * aspectRatioX;
        MetalPaneRect.origin.x = (MetalHolderPaneRect.size.width / 2) - (MetalPaneRect.size.width / 2);
    } else {
        MetalPaneRect.size.height = (MetalHolderPaneRect.size.width / aspectRatioX) * aspectRatioY;
        MetalPaneRect.origin.y = (MetalHolderPaneRect.size.height / 2) - (MetalPaneRect.size.height / 2);
    }
    
    
//    MTKView *metalView = [[MTKView alloc] initWithFrame:MetalPaneRect];
//    metalView.wantsLayer = YES;
//    metalView.layer.backgroundColor = NSColor.blackColor.CGColor;
//    [MetalViewHolder addSubview:metalView];
//    [horizontalSplitView addSubview:MetalViewHolder];

    // Inspector Pane
    NSView *inspectorPane = [[NSView alloc] initWithFrame:inspectorPaneRect];
    inspectorPane.wantsLayer = YES;
    inspectorPane.layer.backgroundColor = NSColor.lightGrayColor.CGColor;
    NSTextField *inspectorLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, 360, 280, 30)];
    inspectorLabel.stringValue = @"Inspector Pane";
    inspectorLabel.editable = NO;
    inspectorLabel.bezeled = NO;
    inspectorLabel.drawsBackground = NO;
    inspectorLabel.alignment = NSTextAlignmentCenter;
    inspectorLabel.font = [NSFont boldSystemFontOfSize:16];
    [inspectorPane addSubview:inspectorLabel];
//    [horizontalSplitView addSubview:inspectorPane];


    
#pragma mark - Timeline Label Constraints
    NSView *timelinePane = [[NSView alloc] initWithFrame:lowerPane];
    timelinePane.wantsLayer = YES;
    timelinePane.layer.backgroundColor = NSColor.darkGrayColor.CGColor;
    NSTextField *timelineLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(lowerPane.size.width / 2, lowerPane.size.height / 2, 80, 30)];
    timelineLabel.stringValue = @"Timeline";
    timelineLabel.editable = NO;
    timelineLabel.bezeled = NO;
    timelineLabel.drawsBackground = NO;
    timelineLabel.alignment = NSTextAlignmentCenter;
    timelineLabel.font = [NSFont boldSystemFontOfSize:16];
    [timelinePane addSubview:timelineLabel];
    
    
    [verticalSplitView addSubview:horizontalSplitView];
    [verticalSplitView addSubview:timelinePane];

    
    self.view = verticalSplitView;


}
@end


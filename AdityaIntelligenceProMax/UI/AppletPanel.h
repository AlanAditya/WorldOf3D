//
//  AppletPanel.h
//  AdityaIntelligenceProMax
//
//  Native control panel for Intelligence applets: sections, sliders and buttons, built in
//  AppKit on macOS and UIKit on iOS. Swift shows `view` through a representable (ControlPanel
//  in 3D_Editor.swift), the same way SidePannel is shown.
//
//  Applets normally go through SliderGroup (UI/MatrixSliders.h), which creates the matrix leaves
//  and calls into this. Every method is safe to call from any thread: UI work hops to main, and
//  the callbacks are always called on the main thread.
//
//  Plain Objective-C on purpose: MatrixH.mm is the Swift bridging header, so this is imported
//  into Swift too.
//

#pragma once
#import <Foundation/Foundation.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif

NS_ASSUME_NONNULL_BEGIN

@interface AppletPanel : NSObject

#if TARGET_OS_IPHONE
@property (nonatomic, readonly) UIView* view;
// Clear background and see-through cards, for when the panel sits on a material (the floating
// sheet in 3D_Editor.swift). NO = opaque grouped background (the docked panel). Main thread only.
@property (nonatomic) BOOL translucent;
#else
@property (nonatomic, readonly) NSView* view;
#endif

// Starts a new titled group; the sliders and buttons after it go into that group's card.
- (void)addSectionWithTitle:(NSString*)title;

// `integer` snaps to whole numbers (and shows tick marks for small ranges on macOS).
// `onChange` is called on the main thread with every new value while dragging.
- (void)addSliderWithTitle:(NSString*)title
                       min:(float)min
                       max:(float)max
                     value:(float)value
                   integer:(BOOL)integer
                  onChange:(void (^)(float value))onChange;

// `onPress` is called on the main thread.
- (void)addButtonWithTitle:(NSString*)title onPress:(void (^)(void))onPress;

// Moves the slider with this title to `value` (clamped to its range) without calling its
// onChange: for code that has already set the value itself (e.g. an "Auto range" button).
- (void)setValue:(float)value forSliderTitled:(NSString*)title;

// An on/off switch. `onChange` is called on the main thread with the new state.
- (void)addToggleWithTitle:(NSString*)title value:(BOOL)value onChange:(void (^)(BOOL on))onChange;

// Removes every section, slider and button (e.g. before starting another applet).
- (void)removeAll;

#if TARGET_OS_IPHONE
// Pages, like the Home Screen: a page indicator with the current page's title at the top, and a
// horizontal swipe anywhere on the panel except on a control (sliders keep their drags) moves
// between `titles.count` pages; tapping the dots works too. Vertical drags still scroll.
// `onScroll` is called on the main thread with the position on every frame of the swipe and of
// the settle animation after it: 0 = first page, 1 = second, 0.4 = 40% of the way there
// (slightly outside [0, count - 1] while rubber-banding past the ends).
// nil, or fewer than 2 titles, removes the indicator and the swipe.
// Used through -[Intelligence showPages:titles:], which moves the renderer's pages with it.
- (void)setPageTitles:(nullable NSArray<NSString*>*)titles onScroll:(nullable void (^)(CGFloat position))onScroll;
#endif

@end

NS_ASSUME_NONNULL_END

//
//  AppletPanel.mm
//  AdityaIntelligenceProMax
//
//  AppKit (macOS) and UIKit (iOS) implementations of AppletPanel. Layout follows the platform's
//  settings style: small section headers over rounded cards, one row per control, hairline
//  separators between rows.
//

#import "AppletPanel.h"
#import <QuartzCore/QuartzCore.h>   // kCACornerCurveContinuous
#include <cmath>

static void APOnMain(dispatch_block_t block) {
    if (NSThread.isMainThread) block();
    else dispatch_async(dispatch_get_main_queue(), block);
}

// Fewer decimals for wider ranges, so the readout doesn't jitter in digits that don't matter.
static NSString* APFormat(float v, float min, float max, BOOL integer) {
    if (integer) return [NSString stringWithFormat:@"%ld", lroundf(v)];
    float range = fabsf(max - min);
    int decimals = range <= 1.0f ? 3 : range <= 10.0f ? 2 : range <= 100.0f ? 1 : 0;
    return [NSString stringWithFormat:@"%.*f", decimals, v];
}

static float APClamp(float v, float min, float max, BOOL integer) {
    v = fminf(fmaxf(v, fminf(min, max)), fmaxf(min, max));
    return integer ? roundf(v) : v;
}

#if !TARGET_OS_IPHONE
// ============================================================================================
// macOS (AppKit)
// ============================================================================================

@interface APFlippedView : NSView
@end
@implementation APFlippedView
- (BOOL)isFlipped { return YES; }
@end

// Rounded card behind a group of rows. Colours are resolved in updateLayer so they follow
// light/dark mode changes.
@interface APCardView : NSView
@end
@implementation APCardView
- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    self.wantsLayer = YES;
    self.layer.cornerRadius = 8;
    self.layer.cornerCurve = kCACornerCurveContinuous;
    return self;
}
- (BOOL)wantsUpdateLayer { return YES; }
- (void)updateLayer {
    self.layer.backgroundColor = [NSColor.controlBackgroundColor colorWithAlphaComponent:0.6].CGColor;
    self.layer.borderColor = NSColor.separatorColor.CGColor;
    self.layer.borderWidth = 1.0 / MAX(self.window.backingScaleFactor, 1.0);
}
@end

@interface APSliderRow : NSObject
@property (nonatomic, copy) NSString* title;
@property (nonatomic, strong) NSSlider* slider;
@property (nonatomic, strong) NSTextField* field;
@property (nonatomic) float min, max, initial, last;
@property (nonatomic) BOOL integer;
@property (nonatomic, copy) void (^onChange)(float);
@end

@implementation APSliderRow
- (void)apply:(float)v fromSlider:(BOOL)fromSlider {
    v = APClamp(v, self.min, self.max, self.integer);
    if (!fromSlider || self.integer) self.slider.floatValue = v;
    self.field.stringValue = APFormat(v, self.min, self.max, self.integer);
    if (v == self.last) return;   // integer sliders report the same value many times per drag
    self.last = v;
    if (self.onChange) self.onChange(v);
}
- (void)sliderMoved:(NSSlider*)sender { [self apply:sender.floatValue fromSlider:YES]; }
- (void)fieldCommitted:(NSTextField*)sender {
    NSScanner* scanner = [NSScanner scannerWithString:sender.stringValue];
    float v;
    if ([scanner scanFloat:&v]) [self apply:v fromSlider:NO];
    else sender.stringValue = APFormat(self.last, self.min, self.max, self.integer);
}
- (void)reset:(id)sender { [self apply:self.initial fromSlider:NO]; }
- (void)show:(float)v {
    v = APClamp(v, self.min, self.max, self.integer);
    self.slider.floatValue = v;
    self.field.stringValue = APFormat(v, self.min, self.max, self.integer);
    self.last = v;
}
@end

@interface APButtonRow : NSObject
@property (nonatomic, copy) void (^onPress)(void);
@end
@implementation APButtonRow
- (void)pressed:(id)sender { if (self.onPress) self.onPress(); }
@end

@interface APToggleRow : NSObject
@property (nonatomic, copy) void (^onChange)(BOOL);
@end
@implementation APToggleRow
- (void)toggled:(NSSwitch*)sender { if (self.onChange) self.onChange(sender.state == NSControlStateValueOn); }
@end

@interface APReadoutRow : NSObject
@property (nonatomic, copy) NSString* title;
@property (nonatomic, strong) NSTextField* valueField;
@end
@implementation APReadoutRow
@end

@implementation AppletPanel {
    NSVisualEffectView* _root;
    NSStackView* _stack;
    NSTextField* _placeholder;
    NSStackView* _currentCard;         // rows of the section being filled
    NSMutableArray* _rows;             // keeps the row objects (control targets are weak) alive
}

- (NSView*)view { return _root; }

- (instancetype)init {
    self = [super init];
    _rows = [NSMutableArray array];

    _root = [[NSVisualEffectView alloc] init];
    _root.material = NSVisualEffectMaterialSidebar;
    _root.blendingMode = NSVisualEffectBlendingModeBehindWindow;
    _root.state = NSVisualEffectStateFollowsWindowActiveState;

    NSScrollView* scroll = [[NSScrollView alloc] init];
    scroll.translatesAutoresizingMaskIntoConstraints = NO;
    scroll.drawsBackground = NO;
    scroll.hasVerticalScroller = YES;
    scroll.autohidesScrollers = YES;
    scroll.borderType = NSNoBorder;
    [_root addSubview:scroll];

    APFlippedView* doc = [[APFlippedView alloc] init];
    doc.translatesAutoresizingMaskIntoConstraints = NO;
    scroll.documentView = doc;

    _stack = [[NSStackView alloc] init];
    _stack.translatesAutoresizingMaskIntoConstraints = NO;
    _stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    _stack.alignment = NSLayoutAttributeLeading;
    _stack.spacing = 6;
    _stack.edgeInsets = NSEdgeInsetsMake(14, 14, 14, 14);
    [doc addSubview:_stack];

    _placeholder = [NSTextField labelWithString:@"No controls for this applet"];
    _placeholder.translatesAutoresizingMaskIntoConstraints = NO;
    _placeholder.textColor = NSColor.tertiaryLabelColor;
    _placeholder.font = [NSFont systemFontOfSize:12];
    [_root addSubview:_placeholder];

    NSClipView* clip = scroll.contentView;
    [NSLayoutConstraint activateConstraints:@[
        [scroll.leadingAnchor constraintEqualToAnchor:_root.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:_root.trailingAnchor],
        [scroll.topAnchor constraintEqualToAnchor:_root.topAnchor],
        [scroll.bottomAnchor constraintEqualToAnchor:_root.bottomAnchor],
        [doc.leadingAnchor constraintEqualToAnchor:clip.leadingAnchor],
        [doc.trailingAnchor constraintEqualToAnchor:clip.trailingAnchor],
        [doc.topAnchor constraintEqualToAnchor:clip.topAnchor],
        [_stack.leadingAnchor constraintEqualToAnchor:doc.leadingAnchor],
        [_stack.trailingAnchor constraintEqualToAnchor:doc.trailingAnchor],
        [_stack.topAnchor constraintEqualToAnchor:doc.topAnchor],
        [_stack.bottomAnchor constraintEqualToAnchor:doc.bottomAnchor],
        [_placeholder.centerXAnchor constraintEqualToAnchor:_root.centerXAnchor],
        [_placeholder.centerYAnchor constraintEqualToAnchor:_root.centerYAnchor],
        [_root.widthAnchor constraintGreaterThanOrEqualToConstant:220],
    ]];
    return self;
}

// Adds a view to the main stack at the full content width.
- (void)addFullWidth:(NSView*)view {
    view.translatesAutoresizingMaskIntoConstraints = NO;
    [_stack addArrangedSubview:view];
    [view.widthAnchor constraintEqualToAnchor:_stack.widthAnchor constant:-28].active = YES;
    _placeholder.hidden = YES;
}

- (NSStackView*)cardForNewRow {
    if (_currentCard) {
        NSBox* separator = [[NSBox alloc] init];
        separator.boxType = NSBoxSeparator;
        separator.translatesAutoresizingMaskIntoConstraints = NO;
        NSView* inset = [[NSView alloc] init];
        inset.translatesAutoresizingMaskIntoConstraints = NO;
        [inset addSubview:separator];
        [NSLayoutConstraint activateConstraints:@[
            [separator.leadingAnchor constraintEqualToAnchor:inset.leadingAnchor constant:12],
            [separator.trailingAnchor constraintEqualToAnchor:inset.trailingAnchor],
            [separator.topAnchor constraintEqualToAnchor:inset.topAnchor],
            [separator.bottomAnchor constraintEqualToAnchor:inset.bottomAnchor],
        ]];
        [self addRow:inset toCard:_currentCard];
        return _currentCard;
    }
    APCardView* card = [[APCardView alloc] init];
    NSStackView* rows = [[NSStackView alloc] init];
    rows.translatesAutoresizingMaskIntoConstraints = NO;
    rows.orientation = NSUserInterfaceLayoutOrientationVertical;
    rows.alignment = NSLayoutAttributeLeading;
    rows.spacing = 0;
    [card addSubview:rows];
    [NSLayoutConstraint activateConstraints:@[
        [rows.leadingAnchor constraintEqualToAnchor:card.leadingAnchor],
        [rows.trailingAnchor constraintEqualToAnchor:card.trailingAnchor],
        [rows.topAnchor constraintEqualToAnchor:card.topAnchor],
        [rows.bottomAnchor constraintEqualToAnchor:card.bottomAnchor],
    ]];
    [self addFullWidth:card];
    [_stack setCustomSpacing:16 afterView:card];
    _currentCard = rows;
    return rows;
}

- (void)addRow:(NSView*)row toCard:(NSStackView*)card {
    row.translatesAutoresizingMaskIntoConstraints = NO;
    [card addArrangedSubview:row];
    [row.widthAnchor constraintEqualToAnchor:card.widthAnchor].active = YES;
}

- (void)addSectionWithTitle:(NSString*)title {
    APOnMain(^{
        NSTextField* header = [NSTextField labelWithString:title];
        header.font = [NSFont systemFontOfSize:11 weight:NSFontWeightSemibold];
        header.textColor = NSColor.secondaryLabelColor;
        [self addFullWidth:header];
        self->_currentCard = nil;
    });
}

- (void)addSliderWithTitle:(NSString*)title min:(float)min max:(float)max value:(float)value
                   integer:(BOOL)integer onChange:(void (^)(float))onChange {
    APOnMain(^{
        APSliderRow* model = [[APSliderRow alloc] init];
        model.title = title;
        model.min = min; model.max = max; model.integer = integer;
        model.initial = model.last = APClamp(value, min, max, integer);
        model.onChange = onChange;

        NSView* row = [[NSView alloc] init];

        NSTextField* label = [NSTextField labelWithString:title];
        label.font = [NSFont systemFontOfSize:13];
        label.lineBreakMode = NSLineBreakByTruncatingTail;
        [label setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                        forOrientation:NSLayoutConstraintOrientationHorizontal];

        // The readout doubles as an input: type a value and press Return.
        NSTextField* field = [NSTextField textFieldWithString:APFormat(model.initial, min, max, integer)];
        field.bordered = NO;
        field.bezeled = NO;
        field.drawsBackground = NO;
        field.focusRingType = NSFocusRingTypeNone;
        field.alignment = NSTextAlignmentRight;
        field.font = [NSFont monospacedDigitSystemFontOfSize:12 weight:NSFontWeightRegular];
        field.textColor = NSColor.secondaryLabelColor;
        field.target = model;
        field.action = @selector(fieldCommitted:);

        NSSlider* slider = [NSSlider sliderWithValue:model.initial minValue:min maxValue:max
                                              target:model action:@selector(sliderMoved:)];
        slider.continuous = YES;
        slider.controlSize = NSControlSizeSmall;
        if (integer && max - min >= 1 && max - min <= 20) {
            slider.numberOfTickMarks = (NSInteger)(max - min) + 1;
            slider.allowsTickMarkValuesOnly = YES;
        }
        NSMenu* menu = [[NSMenu alloc] init];
        NSMenuItem* reset = [menu addItemWithTitle:@"Reset to Default" action:@selector(reset:) keyEquivalent:@""];
        reset.target = model;
        slider.menu = menu;
        field.menu = menu;

        model.slider = slider;
        model.field = field;
        [self->_rows addObject:model];

        for (NSView* v in @[label, field, slider]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:12],
            [label.topAnchor constraintEqualToAnchor:row.topAnchor constant:9],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:field.leadingAnchor constant:-8],
            [field.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-12],
            [field.firstBaselineAnchor constraintEqualToAnchor:label.firstBaselineAnchor],
            [field.widthAnchor constraintEqualToConstant:72],
            [slider.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:12],
            [slider.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-12],
            [slider.topAnchor constraintEqualToAnchor:label.bottomAnchor constant:6],
            [slider.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-10],
        ]];
        [self addRow:row toCard:[self cardForNewRow]];
    });
}

- (void)addButtonWithTitle:(NSString*)title onPress:(void (^)(void))onPress {
    APOnMain(^{
        APButtonRow* model = [[APButtonRow alloc] init];
        model.onPress = onPress;
        [self->_rows addObject:model];

        NSButton* button = [NSButton buttonWithTitle:title target:model action:@selector(pressed:)];
        button.bezelStyle = NSBezelStylePush;
        button.controlSize = NSControlSizeLarge;
        button.translatesAutoresizingMaskIntoConstraints = NO;

        NSView* row = [[NSView alloc] init];
        [row addSubview:button];
        [NSLayoutConstraint activateConstraints:@[
            [button.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:10],
            [button.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-10],
            [button.topAnchor constraintEqualToAnchor:row.topAnchor constant:8],
            [button.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-8],
        ]];
        [self addRow:row toCard:[self cardForNewRow]];
    });
}

- (void)addToggleWithTitle:(NSString*)title value:(BOOL)value onChange:(void (^)(BOOL))onChange {
    APOnMain(^{
        APToggleRow* model = [[APToggleRow alloc] init];
        model.onChange = onChange;
        [self->_rows addObject:model];

        NSTextField* label = [NSTextField labelWithString:title];
        label.font = [NSFont systemFontOfSize:13];
        label.lineBreakMode = NSLineBreakByTruncatingTail;
        NSSwitch* toggle = [[NSSwitch alloc] init];
        toggle.controlSize = NSControlSizeSmall;
        toggle.state = value ? NSControlStateValueOn : NSControlStateValueOff;
        toggle.target = model;
        toggle.action = @selector(toggled:);

        NSView* row = [[NSView alloc] init];
        for (NSView* v in @[label, toggle]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:12],
            [label.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:toggle.leadingAnchor constant:-8],
            [toggle.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-12],
            [toggle.topAnchor constraintEqualToAnchor:row.topAnchor constant:8],
            [toggle.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-8],
        ]];
        [self addRow:row toCard:[self cardForNewRow]];
    });
}

- (void)setValue:(float)value forSliderTitled:(NSString*)title {
    APOnMain(^{
        for (id row in self->_rows) {
            if ([row isKindOfClass:[APSliderRow class]] && [((APSliderRow*)row).title isEqualToString:title]) {
                [(APSliderRow*)row show:value];
            }
        }
    });
}

- (void)addReadoutWithTitle:(NSString*)title value:(NSString*)value {
    APOnMain(^{
        APReadoutRow* model = [[APReadoutRow alloc] init];
        model.title = title;
        [self->_rows addObject:model];

        NSTextField* label = [NSTextField labelWithString:title];
        label.font = [NSFont systemFontOfSize:13];
        label.lineBreakMode = NSLineBreakByTruncatingTail;
        [label setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                        forOrientation:NSLayoutConstraintOrientationHorizontal];
        NSTextField* field = [NSTextField labelWithString:value];
        field.font = [NSFont monospacedDigitSystemFontOfSize:12 weight:NSFontWeightMedium];
        field.textColor = NSColor.secondaryLabelColor;
        field.alignment = NSTextAlignmentRight;
        model.valueField = field;

        NSView* row = [[NSView alloc] init];
        for (NSView* v in @[label, field]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:12],
            [label.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:field.leadingAnchor constant:-8],
            [field.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-12],
            [field.topAnchor constraintEqualToAnchor:row.topAnchor constant:8],
            [field.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-8],
        ]];
        [self addRow:row toCard:[self cardForNewRow]];
    });
}

- (void)setText:(NSString*)text forReadoutTitled:(NSString*)title {
    APOnMain(^{
        for (id row in self->_rows) {
            if ([row isKindOfClass:[APReadoutRow class]] && [((APReadoutRow*)row).title isEqualToString:title]) {
                ((APReadoutRow*)row).valueField.stringValue = text;
            }
        }
    });
}

- (void)removeAll {
    APOnMain(^{
        for (NSView* v in [self->_stack.arrangedSubviews copy]) {
            [self->_stack removeArrangedSubview:v];
            [v removeFromSuperview];
        }
        [self->_rows removeAllObjects];
        self->_currentCard = nil;
        self->_placeholder.hidden = NO;
    });
}

@end

#else
// ============================================================================================
// iOS (UIKit)
// ============================================================================================

@interface APSliderRow : NSObject
@property (nonatomic, copy) NSString* title;
@property (nonatomic, strong) UISlider* slider;
@property (nonatomic, strong) UILabel* valueLabel;
@property (nonatomic) float min, max, initial, last;
@property (nonatomic) BOOL integer;
@property (nonatomic, copy) void (^onChange)(float);
@end

@implementation APSliderRow
- (void)apply:(float)v {
    v = APClamp(v, self.min, self.max, self.integer);
    if (self.integer) self.slider.value = v;
    self.valueLabel.text = APFormat(v, self.min, self.max, self.integer);
    if (v == self.last) return;
    self.last = v;
    if (self.onChange) self.onChange(v);
}
- (void)sliderMoved:(UISlider*)sender { [self apply:sender.value]; }
- (void)reset {
    self.slider.value = self.initial;
    [self apply:self.initial];
}
- (void)show:(float)v {
    v = APClamp(v, self.min, self.max, self.integer);
    self.slider.value = v;
    self.valueLabel.text = APFormat(v, self.min, self.max, self.integer);
    self.last = v;
}
@end

@interface APReadoutRow : NSObject
@property (nonatomic, copy) NSString* title;
@property (nonatomic, strong) UILabel* valueLabel;
@end
@implementation APReadoutRow
@end

@interface AppletPanel () <UIGestureRecognizerDelegate>
@end

@implementation AppletPanel {
    UIView* _root;
    UIStackView* _stack;
    UILabel* _placeholder;
    UIStackView* _currentCard;
    NSMutableArray* _rows;
    NSMutableArray<UIView*>* _cards;   // recoloured when `translucent` changes

    // Pages (setPageTitles:onScroll:)
    UIView* _pageHeader;
    NSLayoutConstraint* _pageHeaderHeight;   // 0 without pages
    UILabel* _pageTitle;
    UIPageControl* _pageDots;
    UIPanGestureRecognizer* _pagePan;
    NSArray<NSString*>* _pageTitles;
    void (^_onPageScroll)(CGFloat);
    CGFloat _pagePosition;                   // where the pages are now
    CGFloat _pageStart;                      // _pagePosition when the swipe began
    CGFloat _pageTarget, _pageVelocity;      // settle animation: spring towards _pageTarget
    CADisplayLink* _pageLink;
    CFTimeInterval _pageLastTick;
}

- (UIView*)view { return _root; }

- (UIColor*)cardColor {
    // On a material, a light translucent fill reads as a card without hiding the blur.
    return _translucent ? [UIColor.tertiarySystemFillColor colorWithAlphaComponent:0.35]
                        : UIColor.secondarySystemGroupedBackgroundColor;
}

- (void)setTranslucent:(BOOL)translucent {
    _translucent = translucent;
    _root.backgroundColor = translucent ? UIColor.clearColor : UIColor.systemGroupedBackgroundColor;
    for (UIView* card in _cards) card.backgroundColor = [self cardColor];
}

- (instancetype)init {
    self = [super init];
    _rows = [NSMutableArray array];
    _cards = [NSMutableArray array];

    _root = [[UIView alloc] init];
    _root.backgroundColor = UIColor.systemGroupedBackgroundColor;

    UIScrollView* scroll = [[UIScrollView alloc] init];
    scroll.translatesAutoresizingMaskIntoConstraints = NO;
    scroll.alwaysBounceVertical = YES;
    [_root addSubview:scroll];

    // Page indicator: "<title>  ● ○", above the controls; collapsed until there are pages.
    _pageHeader = [[UIView alloc] init];
    _pageHeader.translatesAutoresizingMaskIntoConstraints = NO;
    _pageHeader.clipsToBounds = YES;
    _pageHeader.hidden = YES;
    [_root addSubview:_pageHeader];
    _pageTitle = [[UILabel alloc] init];
    _pageTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleFootnote];
    _pageTitle.adjustsFontForContentSizeCategory = YES;
    _pageTitle.textColor = UIColor.secondaryLabelColor;
    _pageDots = [[UIPageControl alloc] init];
    _pageDots.hidesForSinglePage = YES;
    _pageDots.pageIndicatorTintColor = UIColor.tertiaryLabelColor;
    _pageDots.currentPageIndicatorTintColor = UIColor.labelColor;
    [_pageDots addTarget:self action:@selector(pageDotsChanged:) forControlEvents:UIControlEventValueChanged];
    UIStackView* pageRow = [[UIStackView alloc] initWithArrangedSubviews:@[_pageTitle, _pageDots]];
    pageRow.translatesAutoresizingMaskIntoConstraints = NO;
    pageRow.axis = UILayoutConstraintAxisHorizontal;
    pageRow.alignment = UIStackViewAlignmentCenter;
    pageRow.spacing = 2;
    [_pageHeader addSubview:pageRow];
    _pageHeaderHeight = [_pageHeader.heightAnchor constraintEqualToConstant:0];

    // Horizontal swipes page; vertical ones fail it straight away and scroll the controls.
    _pagePan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(pagePanned:)];
    _pagePan.delegate = self;
    _pagePan.enabled = NO;
    [_root addGestureRecognizer:_pagePan];
    // The scroll view's pan sits deeper, so it would win a horizontal drag; make it wait for ours.
    [scroll.panGestureRecognizer requireGestureRecognizerToFail:_pagePan];

    _stack = [[UIStackView alloc] init];
    _stack.translatesAutoresizingMaskIntoConstraints = NO;
    _stack.axis = UILayoutConstraintAxisVertical;
    _stack.alignment = UIStackViewAlignmentFill;
    _stack.spacing = 6;
    _stack.layoutMarginsRelativeArrangement = YES;
    _stack.directionalLayoutMargins = NSDirectionalEdgeInsetsMake(16, 16, 16, 16);
    [scroll addSubview:_stack];

    _placeholder = [[UILabel alloc] init];
    _placeholder.translatesAutoresizingMaskIntoConstraints = NO;
    _placeholder.text = @"No controls for this applet";
    _placeholder.font = [UIFont preferredFontForTextStyle:UIFontTextStyleFootnote];
    _placeholder.textColor = UIColor.tertiaryLabelColor;
    [_root addSubview:_placeholder];

    [NSLayoutConstraint activateConstraints:@[
        [_pageHeader.leadingAnchor constraintEqualToAnchor:_root.leadingAnchor],
        [_pageHeader.trailingAnchor constraintEqualToAnchor:_root.trailingAnchor],
        [_pageHeader.topAnchor constraintEqualToAnchor:_root.topAnchor],
        _pageHeaderHeight,
        [pageRow.centerXAnchor constraintEqualToAnchor:_pageHeader.centerXAnchor],
        [pageRow.centerYAnchor constraintEqualToAnchor:_pageHeader.centerYAnchor],
        [scroll.leadingAnchor constraintEqualToAnchor:_root.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:_root.trailingAnchor],
        [scroll.topAnchor constraintEqualToAnchor:_pageHeader.bottomAnchor],
        [scroll.bottomAnchor constraintEqualToAnchor:_root.bottomAnchor],
        [_stack.leadingAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.leadingAnchor],
        [_stack.trailingAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.trailingAnchor],
        [_stack.topAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.topAnchor],
        [_stack.bottomAnchor constraintEqualToAnchor:scroll.contentLayoutGuide.bottomAnchor],
        [_stack.widthAnchor constraintEqualToAnchor:scroll.frameLayoutGuide.widthAnchor],
        [_placeholder.centerXAnchor constraintEqualToAnchor:_root.centerXAnchor],
        [_placeholder.centerYAnchor constraintEqualToAnchor:_root.centerYAnchor],
    ]];
    return self;
}

- (UIStackView*)cardForNewRow {
    if (_currentCard) {
        UIView* inset = [[UIView alloc] init];
        UIView* separator = [[UIView alloc] init];
        separator.translatesAutoresizingMaskIntoConstraints = NO;
        separator.backgroundColor = UIColor.separatorColor;
        [inset addSubview:separator];
        [NSLayoutConstraint activateConstraints:@[
            [separator.leadingAnchor constraintEqualToAnchor:inset.leadingAnchor constant:16],
            [separator.trailingAnchor constraintEqualToAnchor:inset.trailingAnchor],
            [separator.topAnchor constraintEqualToAnchor:inset.topAnchor],
            [separator.bottomAnchor constraintEqualToAnchor:inset.bottomAnchor],
            [inset.heightAnchor constraintEqualToConstant:1.0 / UIScreen.mainScreen.scale],
        ]];
        [_currentCard addArrangedSubview:inset];
        return _currentCard;
    }
    UIView* card = [[UIView alloc] init];
    card.backgroundColor = [self cardColor];
    [_cards addObject:card];
    card.layer.cornerRadius = 12;
    card.layer.cornerCurve = kCACornerCurveContinuous;
    card.clipsToBounds = YES;

    UIStackView* rows = [[UIStackView alloc] init];
    rows.translatesAutoresizingMaskIntoConstraints = NO;
    rows.axis = UILayoutConstraintAxisVertical;
    rows.alignment = UIStackViewAlignmentFill;
    [card addSubview:rows];
    [NSLayoutConstraint activateConstraints:@[
        [rows.leadingAnchor constraintEqualToAnchor:card.leadingAnchor],
        [rows.trailingAnchor constraintEqualToAnchor:card.trailingAnchor],
        [rows.topAnchor constraintEqualToAnchor:card.topAnchor],
        [rows.bottomAnchor constraintEqualToAnchor:card.bottomAnchor],
    ]];
    [_stack addArrangedSubview:card];
    [_stack setCustomSpacing:20 afterView:card];
    _placeholder.hidden = YES;
    _currentCard = rows;
    return rows;
}

- (void)addSectionWithTitle:(NSString*)title {
    APOnMain(^{
        UILabel* header = [[UILabel alloc] init];
        header.text = title.uppercaseString;
        header.font = [UIFont preferredFontForTextStyle:UIFontTextStyleFootnote];
        header.textColor = UIColor.secondaryLabelColor;

        UIView* wrap = [[UIView alloc] init];
        header.translatesAutoresizingMaskIntoConstraints = NO;
        [wrap addSubview:header];
        [NSLayoutConstraint activateConstraints:@[
            [header.leadingAnchor constraintEqualToAnchor:wrap.leadingAnchor constant:16],
            [header.trailingAnchor constraintEqualToAnchor:wrap.trailingAnchor constant:-16],
            [header.topAnchor constraintEqualToAnchor:wrap.topAnchor],
            [header.bottomAnchor constraintEqualToAnchor:wrap.bottomAnchor],
        ]];
        [self->_stack addArrangedSubview:wrap];
        self->_placeholder.hidden = YES;
        self->_currentCard = nil;
    });
}

- (void)addSliderWithTitle:(NSString*)title min:(float)min max:(float)max value:(float)value
                   integer:(BOOL)integer onChange:(void (^)(float))onChange {
    APOnMain(^{
        APSliderRow* model = [[APSliderRow alloc] init];
        model.title = title;
        model.min = min; model.max = max; model.integer = integer;
        model.initial = model.last = APClamp(value, min, max, integer);
        model.onChange = onChange;

        UILabel* label = [[UILabel alloc] init];
        label.text = title;
        label.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
        label.adjustsFontForContentSizeCategory = YES;

        UILabel* valueLabel = [[UILabel alloc] init];
        valueLabel.text = APFormat(model.initial, min, max, integer);
        valueLabel.font = [UIFont monospacedDigitSystemFontOfSize:15 weight:UIFontWeightRegular];
        valueLabel.textColor = UIColor.secondaryLabelColor;
        valueLabel.textAlignment = NSTextAlignmentRight;
        [valueLabel setContentHuggingPriority:UILayoutPriorityRequired forAxis:UILayoutConstraintAxisHorizontal];
        // Double-tap the readout to go back to the default value.
        valueLabel.userInteractionEnabled = YES;
        UITapGestureRecognizer* tap = [[UITapGestureRecognizer alloc] initWithTarget:model action:@selector(reset)];
        tap.numberOfTapsRequired = 2;
        [valueLabel addGestureRecognizer:tap];

        UISlider* slider = [[UISlider alloc] init];
        slider.minimumValue = min;
        slider.maximumValue = max;
        slider.value = model.initial;
        slider.continuous = YES;
        [slider addTarget:model action:@selector(sliderMoved:) forControlEvents:UIControlEventValueChanged];

        model.slider = slider;
        model.valueLabel = valueLabel;
        [self->_rows addObject:model];

        UIView* row = [[UIView alloc] init];
        for (UIView* v in @[label, valueLabel, slider]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:16],
            [label.topAnchor constraintEqualToAnchor:row.topAnchor constant:12],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:valueLabel.leadingAnchor constant:-8],
            [valueLabel.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-16],
            [valueLabel.firstBaselineAnchor constraintEqualToAnchor:label.firstBaselineAnchor],
            [slider.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:16],
            [slider.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-16],
            [slider.topAnchor constraintEqualToAnchor:label.bottomAnchor constant:8],
            [slider.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-12],
        ]];
        [[self cardForNewRow] addArrangedSubview:row];
    });
}

- (void)addButtonWithTitle:(NSString*)title onPress:(void (^)(void))onPress {
    APOnMain(^{
        void (^press)(void) = [onPress copy];
        UIButtonConfiguration* config;
        if (@available(iOS 26.0, *)) {
            config = [UIButtonConfiguration glassButtonConfiguration];   // Liquid Glass
            config.cornerStyle = UIButtonConfigurationCornerStyleCapsule;
        } else {
            config = [UIButtonConfiguration tintedButtonConfiguration];
            config.cornerStyle = UIButtonConfigurationCornerStyleMedium;
        }
        config.title = title;
        UIButton* button = [UIButton buttonWithConfiguration:config
                                               primaryAction:[UIAction actionWithHandler:^(__kindof UIAction* action) {
            if (press) press();
        }]];
        button.translatesAutoresizingMaskIntoConstraints = NO;

        UIView* row = [[UIView alloc] init];
        [row addSubview:button];
        [NSLayoutConstraint activateConstraints:@[
            [button.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:12],
            [button.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-12],
            [button.topAnchor constraintEqualToAnchor:row.topAnchor constant:10],
            [button.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-10],
        ]];
        [[self cardForNewRow] addArrangedSubview:row];
    });
}

- (void)addToggleWithTitle:(NSString*)title value:(BOOL)value onChange:(void (^)(BOOL))onChange {
    APOnMain(^{
        void (^changed)(BOOL) = [onChange copy];
        UILabel* label = [[UILabel alloc] init];
        label.text = title;
        label.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
        label.adjustsFontForContentSizeCategory = YES;

        UISwitch* toggle = [[UISwitch alloc] init];
        toggle.on = value;
        [toggle addAction:[UIAction actionWithHandler:^(__kindof UIAction* action) {
            if (changed) changed(((UISwitch*)action.sender).isOn);
        }] forControlEvents:UIControlEventValueChanged];

        UIView* row = [[UIView alloc] init];
        for (UIView* v in @[label, toggle]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:16],
            [label.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:toggle.leadingAnchor constant:-8],
            [toggle.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-16],
            [toggle.topAnchor constraintEqualToAnchor:row.topAnchor constant:10],
            [toggle.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-10],
        ]];
        [[self cardForNewRow] addArrangedSubview:row];
    });
}

- (void)setValue:(float)value forSliderTitled:(NSString*)title {
    APOnMain(^{
        for (id row in self->_rows) {
            if ([row isKindOfClass:[APSliderRow class]] && [((APSliderRow*)row).title isEqualToString:title]) {
                [(APSliderRow*)row show:value];
            }
        }
    });
}

- (void)addReadoutWithTitle:(NSString*)title value:(NSString*)value {
    APOnMain(^{
        APReadoutRow* model = [[APReadoutRow alloc] init];
        model.title = title;
        [self->_rows addObject:model];

        UILabel* label = [[UILabel alloc] init];
        label.text = title;
        label.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
        label.adjustsFontForContentSizeCategory = YES;
        UILabel* valueLabel = [[UILabel alloc] init];
        valueLabel.text = value;
        valueLabel.font = [UIFont monospacedDigitSystemFontOfSize:15 weight:UIFontWeightMedium];
        valueLabel.textColor = UIColor.secondaryLabelColor;
        valueLabel.textAlignment = NSTextAlignmentRight;
        model.valueLabel = valueLabel;

        UIView* row = [[UIView alloc] init];
        for (UIView* v in @[label, valueLabel]) {
            v.translatesAutoresizingMaskIntoConstraints = NO;
            [row addSubview:v];
        }
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:16],
            [label.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
            [label.trailingAnchor constraintLessThanOrEqualToAnchor:valueLabel.leadingAnchor constant:-8],
            [valueLabel.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-16],
            [valueLabel.topAnchor constraintEqualToAnchor:row.topAnchor constant:12],
            [valueLabel.bottomAnchor constraintEqualToAnchor:row.bottomAnchor constant:-12],
        ]];
        [[self cardForNewRow] addArrangedSubview:row];
    });
}

- (void)setText:(NSString*)text forReadoutTitled:(NSString*)title {
    APOnMain(^{
        for (id row in self->_rows) {
            if ([row isKindOfClass:[APReadoutRow class]] && [((APReadoutRow*)row).title isEqualToString:title]) {
                ((APReadoutRow*)row).valueLabel.text = text;
            }
        }
    });
}

- (void)removeAll {
    APOnMain(^{
        for (UIView* v in [self->_stack.arrangedSubviews copy]) {
            [v removeFromSuperview];
        }
        [self->_rows removeAllObjects];
        [self->_cards removeAllObjects];
        self->_currentCard = nil;
        self->_placeholder.hidden = NO;
    });
}

// ---- Pages ----------------------------------------------------------------------------------

- (void)setPageTitles:(NSArray<NSString*>*)titles onScroll:(void (^)(CGFloat))onScroll {
    NSArray<NSString*>* copied = titles.count >= 2 ? [titles copy] : nil;
    APOnMain(^{
        [self stopPageAnimation];
        self->_pageTitles = copied;
        self->_onPageScroll = copied ? [onScroll copy] : nil;
        self->_pagePan.enabled = copied != nil;
        self->_pageHeader.hidden = copied == nil;
        self->_pageHeaderHeight.constant = copied ? 28 : 0;
        self->_pageDots.numberOfPages = (NSInteger)copied.count;
        [self setPagePosition:0];
    });
}

- (NSInteger)lastPage { return MAX((NSInteger)_pageTitles.count - 1, 0); }

- (void)setPagePosition:(CGFloat)position {
    _pagePosition = position;
    if (_pageTitles.count) {
        NSInteger page = MIN(MAX(lround(position), 0), [self lastPage]);
        _pageDots.currentPage = page;
        _pageTitle.text = _pageTitles[page];
    }
    if (_onPageScroll) _onPageScroll(position);
}

// Sliders, switches and buttons keep their own drags (a slider is dragged sideways too).
- (BOOL)gestureRecognizer:(UIGestureRecognizer*)recognizer shouldReceiveTouch:(UITouch*)touch {
    if (recognizer != _pagePan) return YES;
    for (UIView* v = touch.view; v && v != _root; v = v.superview) {
        if ([v isKindOfClass:[UIControl class]]) return NO;
    }
    return YES;
}

- (BOOL)gestureRecognizerShouldBegin:(UIGestureRecognizer*)recognizer {
    if (recognizer != _pagePan) return YES;
    CGPoint v = [_pagePan velocityInView:_root];
    return fabs(v.x) > fabs(v.y);
}

- (void)pagePanned:(UIPanGestureRecognizer*)pan {
    CGFloat width = MAX(_root.bounds.size.width, 1);
    CGFloat last = [self lastPage];
    switch (pan.state) {
        case UIGestureRecognizerStateBegan:
            [self stopPageAnimation];
            _pageStart = _pagePosition;
            break;
        case UIGestureRecognizerStateChanged: {
            // Finger moving left = towards the next page. Past either end it moves at a third of
            // the finger's speed (rubber band).
            CGFloat p = _pageStart - [pan translationInView:_root].x / width;
            if (p < 0) p /= 3;
            else if (p > last) p = last + (p - last) / 3;
            [self setPagePosition:p];
            break;
        }
        case UIGestureRecognizerStateEnded:
        case UIGestureRecognizerStateCancelled: {
            CGFloat velocity = -[pan velocityInView:_root].x / width;   // pages per second
            // Where a flick would carry it, at most one page from where the swipe started.
            CGFloat start = round(_pageStart);
            CGFloat target = round(_pagePosition + 0.2 * velocity);
            target = MIN(MAX(target, start - 1), start + 1);
            target = MIN(MAX(target, 0), last);
            [self animatePageTo:target velocity:velocity];
            break;
        }
        default:
            break;
    }
}

- (void)pageDotsChanged:(UIPageControl*)dots {
    [self animatePageTo:dots.currentPage velocity:0];
}

- (void)animatePageTo:(CGFloat)target velocity:(CGFloat)velocity {
    _pageTarget = target;
    _pageVelocity = velocity;
    _pageLastTick = CACurrentMediaTime();
    if (!_pageLink) {
        _pageLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(pageTick:)];
        [_pageLink addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
    }
}

- (void)stopPageAnimation {
    [_pageLink invalidate];   // the display link retains us until invalidated
    _pageLink = nil;
}

// Critically damped spring: carries the swipe's velocity, settles without overshooting.
- (void)pageTick:(CADisplayLink*)link {
    CFTimeInterval now = CACurrentMediaTime();
    CGFloat dt = MIN(now - _pageLastTick, 1.0 / 30);
    _pageLastTick = now;
    const CGFloat omega = 14;
    CGFloat accel = omega * omega * (_pageTarget - _pagePosition) - 2 * omega * _pageVelocity;
    _pageVelocity += accel * dt;
    CGFloat p = _pagePosition + _pageVelocity * dt;
    if (fabs(_pageTarget - p) < 0.0005 && fabs(_pageVelocity) < 0.01) {
        [self stopPageAnimation];
        p = _pageTarget;
    }
    [self setPagePosition:p];
}

@end
#endif

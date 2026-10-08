//
//  AppletToolbar.h
//  AdityaIntelligenceProMax
//
//  A small floating Liquid Glass capsule over the renderer (bottom right) holding icon buttons,
//  toggles and pop-up menus (e.g. a "..." menu). This is the model; SwiftUI draws it
//  (AppletToolbarView in UI/AppletToolbarView.swift), so the capsule, the glass and the menus
//  are the system's own on both macOS and iOS.
//
//  Every add/remove method is safe from any thread (it hops to main); the callbacks are called
//  on the main thread, the same thread the renderer draws on, so they can touch the Renderer
//  (its camera, renderInfiniteGrid) directly.
//
//  Plain Objective-C on purpose: MatrixH.mm is the Swift bridging header.
//

#pragma once
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, AppletToolbarItemKind) {
    AppletToolbarItemKindButton,
    AppletToolbarItemKindToggle,
    AppletToolbarItemKindMenu,
};

// One control in the capsule. Read by the SwiftUI view; applets don't need it.
@interface AppletToolbarItem : NSObject
@property (nonatomic, readonly) AppletToolbarItemKind kind;
@property (nonatomic, readonly, copy) NSString* icon;    // SF Symbol name
@property (nonatomic, readonly, copy) NSString* title;   // tooltip / accessibility label / menu header
@property (nonatomic, readonly) BOOL on;                 // toggles only
@property (nonatomic, readonly, copy) NSArray<NSString*>* menuTitles;
@property (nonatomic, readonly, copy) NSArray<NSString*>* menuIcons;   // same count as menuTitles, "" = no icon
@end

// ---- Legend: a small translucent card just above the capsule saying what each colour means ----

typedef NS_ENUM(NSInteger, AppletLegendKind) {
    AppletLegendKindSection,   // a small header; only `label` is used
    AppletLegendKindFill,      // a filled swatch (a surface / point colour)
    AppletLegendKindOutline,   // an outlined swatch (a wireframe box colour)
    AppletLegendKindNote,      // a line of secondary text (e.g. "brighter = taller")
};

// One line of the legend. Colours are the renderer's own linear values (what a shader writes to
// the sRGB drawable), so the swatch shows the same colour as the scene.
@interface AppletLegendEntry : NSObject
@property (nonatomic, readonly) AppletLegendKind kind;
@property (nonatomic, readonly, copy) NSString* label;
@property (nonatomic, readonly) float red, green, blue;
+ (instancetype)sectionWithTitle:(NSString*)title;
+ (instancetype)fillWithLabel:(NSString*)label red:(float)red green:(float)green blue:(float)blue;
+ (instancetype)outlineWithLabel:(NSString*)label red:(float)red green:(float)green blue:(float)blue;
+ (instancetype)noteWithText:(NSString*)text;
@end

@interface AppletToolbar : NSObject

// Replaces the whole legend in one go (so it never shows half-built); empty or nil hides it.
// Safe from any thread.
- (void)setLegend:(nullable NSArray<AppletLegendEntry*>*)entries;

// A button showing an SF Symbol. `onPress` is called on the main thread.
- (void)addButtonWithIcon:(NSString*)icon title:(NSString*)title onPress:(void (^)(void))onPress;

// A button that stays highlighted while on. `onChange` is called on the main thread.
- (void)addToggleWithIcon:(NSString*)icon
                    title:(NSString*)title
                    value:(BOOL)value
                 onChange:(void (^)(BOOL on))onChange;

// A button that pops up a menu. `itemIcons` is optional (SF Symbol per item, "" for none).
// `onSelect` is called on the main thread with the chosen item's index.
- (void)addMenuWithIcon:(NSString*)icon
                  title:(NSString*)title
             itemTitles:(NSArray<NSString*>*)itemTitles
              itemIcons:(nullable NSArray<NSString*>*)itemIcons
               onSelect:(void (^)(NSInteger index))onSelect;

// Removes every item and the legend (e.g. before starting another applet).
- (void)removeAll;

// ---- For the SwiftUI view (main thread) ----
@property (nonatomic, readonly, copy) NSArray<AppletToolbarItem*>* items;
@property (nonatomic, readonly, copy) NSArray<AppletLegendEntry*>* legend;
// Called on main whenever `items`, a toggle's state or the legend changes.
@property (nonatomic, copy, nullable) void (^onItemsChanged)(void);
- (void)pressItem:(AppletToolbarItem*)item NS_SWIFT_NAME(press(_:));
- (void)setItem:(AppletToolbarItem*)item on:(BOOL)on NS_SWIFT_NAME(set(_:on:));
- (void)selectMenuItem:(AppletToolbarItem*)item index:(NSInteger)index NS_SWIFT_NAME(select(_:index:));

@end

NS_ASSUME_NONNULL_END

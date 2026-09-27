//
//  AppletToolbar.mm
//  AdityaIntelligenceProMax
//
//  Model behind the floating capsule: see AppletToolbar.h. The drawing is SwiftUI
//  (UI/AppletToolbarView.swift).
//

#import "AppletToolbar.h"

static void ATOnMain(dispatch_block_t block) {
    if (NSThread.isMainThread) block();
    else dispatch_async(dispatch_get_main_queue(), block);
}

@interface AppletToolbarItem ()
@property (nonatomic) AppletToolbarItemKind kind;
@property (nonatomic, copy) NSString* icon;
@property (nonatomic, copy) NSString* title;
@property (nonatomic) BOOL on;
@property (nonatomic, copy) NSArray<NSString*>* menuTitles;
@property (nonatomic, copy) NSArray<NSString*>* menuIcons;
@property (nonatomic, copy) void (^onPress)(void);
@property (nonatomic, copy) void (^onChange)(BOOL);
@property (nonatomic, copy) void (^onSelect)(NSInteger);
@end

@implementation AppletToolbarItem
@end

@implementation AppletToolbar {
    NSMutableArray<AppletToolbarItem*>* _list;   // main thread only
}

- (instancetype)init {
    self = [super init];
    _list = [NSMutableArray array];
    return self;
}

- (NSArray<AppletToolbarItem*>*)items { return [_list copy]; }

- (void)changed {
    if (self.onItemsChanged) self.onItemsChanged();
}

- (void)append:(AppletToolbarItem*)item {
    ATOnMain(^{
        [self->_list addObject:item];
        [self changed];
    });
}

- (void)addButtonWithIcon:(NSString*)icon title:(NSString*)title onPress:(void (^)(void))onPress {
    AppletToolbarItem* item = [AppletToolbarItem new];
    item.kind = AppletToolbarItemKindButton;
    item.icon = icon;
    item.title = title;
    item.onPress = onPress;
    [self append:item];
}

- (void)addToggleWithIcon:(NSString*)icon
                    title:(NSString*)title
                    value:(BOOL)value
                 onChange:(void (^)(BOOL on))onChange {
    AppletToolbarItem* item = [AppletToolbarItem new];
    item.kind = AppletToolbarItemKindToggle;
    item.icon = icon;
    item.title = title;
    item.on = value;
    item.onChange = onChange;
    [self append:item];
}

- (void)addMenuWithIcon:(NSString*)icon
                  title:(NSString*)title
             itemTitles:(NSArray<NSString*>*)itemTitles
              itemIcons:(NSArray<NSString*>*)itemIcons
               onSelect:(void (^)(NSInteger index))onSelect {
    NSMutableArray<NSString*>* icons = [NSMutableArray arrayWithCapacity:itemTitles.count];
    for (NSUInteger i = 0; i < itemTitles.count; i++)
        [icons addObject:i < itemIcons.count ? itemIcons[i] : @""];

    AppletToolbarItem* item = [AppletToolbarItem new];
    item.kind = AppletToolbarItemKindMenu;
    item.icon = icon;
    item.title = title;
    item.menuTitles = itemTitles;
    item.menuIcons = icons;
    item.onSelect = onSelect;
    [self append:item];
}

- (void)removeAll {
    ATOnMain(^{
        [self->_list removeAllObjects];
        [self changed];
    });
}

- (void)pressItem:(AppletToolbarItem*)item {
    if (item.onPress) item.onPress();
}

- (void)setItem:(AppletToolbarItem*)item on:(BOOL)on {
    if (item.on == on) return;
    item.on = on;
    [self changed];
    if (item.onChange) item.onChange(on);
}

- (void)selectMenuItem:(AppletToolbarItem*)item index:(NSInteger)index {
    if (index < 0 || index >= (NSInteger)item.menuTitles.count) return;
    if (item.onSelect) item.onSelect(index);
}

@end

#pragma once
// Matrix-backed UI sliders for Intelligence applets.
//
// A slider owns a 1-element leaf matrix (matrix::leaf_scalar). Moving the slider writes the new
// value into that leaf *in place* and calls mark_changed(), so any graph built on it (arithmetic,
// or a generator parameter like perlin()'s scale) re-runs only the dirty path on its next pass.
// The graph is built once; nothing is reallocated per change.
//
//     SliderGroup ui(self);                                   // owner thread = main
//     ui.section(@"Noise");
//     matrix scale = ui.slider(@"Scale", 0.5, 20, 5);
//     ui.button(@"Reseed", ^{ ... });
//     matrix noise = matrix::perlin({512, 512}, scale, 4, 0.5f, 2.0f);
//     ui.on_change(noise, ^(matrix& out) { ... display out ... });
//
// Threading: every graph has one owner thread, and only that thread writes its leaves and runs
// its passes. SwiftUI calls the slider callback on the main thread; if the group's owner queue
// isn't main (e.g. an applet's compute loop queue), the write + mark_changed() is sent to the
// owner queue with dispatch_async, so the value is written on the same thread that reads it.
// That avoids a torn or stale read without adding atomics to the graph.
//
// The controls are drawn by AppletPanel (UI/AppletPanel.h, AppKit/UIKit). Requires matrix.h and
// AppletPanel.h to be visible before inclusion.

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

class SliderGroup {
    struct Watcher {
        matrix out;
        void (^publish)(matrix&);
    };
    struct SliderInfo {
        const void* node;      // the leaf's tape: identifies the slider from its matrix
        NSString* title;
        float min, max;
        bool integer;
    };
    struct State {
        std::vector<SliderInfo> sliders;
        dispatch_queue_t queue;
        bool queue_is_main = false;
        dispatch_source_t trigger = nil;   // DATA_OR: many slider ticks -> one pass
        std::vector<Watcher> watchers;
    };
    std::shared_ptr<State> state;
    __weak id owner;                       // the Intelligence instance holding `appletPanel`

    // Runs one pass on every watched output and publishes the ones that changed.
    // Always called on the owner queue.
    static void run_watchers(const std::shared_ptr<State>& s) {
        for (auto& w : s->watchers) {
            if (run_graph_pass(w.out) && w.publish) w.publish(w.out);
        }
    }

    // Sends `block` to the owner queue (runs inline when we're already on it and it's main).
    static void on_owner(const std::shared_ptr<State>& s, dispatch_block_t block) {
        if (s->queue_is_main && [NSThread isMainThread]) block();
        else dispatch_async(s->queue, block);
    }

    // The panel lives on the Intelligence instance (created in its init).
    AppletPanel* panel() const {
        id intel = owner;
        return intel ? (AppletPanel*)[intel valueForKey:@"appletPanel"] : nil;
    }

    // Wakes the owner queue's watchers after a leaf was written (called on the owner queue).
    static void notify(const std::shared_ptr<State>& s) {
        if (s->trigger) dispatch_source_merge_data(s->trigger, 1);
    }

    // Writes a scalar leaf of any dtype (rounding for integer types) and marks it changed.
    static void write_leaf(const matrix& leaf, float v) {
        matrix l = leaf;                 // same node (shares the tape)
        dispatch_type(l.type, l.buffer, [&](auto* data) {
            using T = std::decay_t<decltype(*data)>;
            *data = std::is_integral_v<T> ? static_cast<T>(std::lround(v)) : static_cast<T>(v);
        });
        l.mark_changed();
    }

    matrix add_slider(NSString* title, float min, float max, float init, dtype type) {
        matrix leaf = matrix::leaf_scalar(init, type);
        std::shared_ptr<State> s = state;
        s->sliders.push_back({leaf.tape, title, min, max, type != dtype::Float});

        [panel() addSliderWithTitle:title min:min max:max value:init
                            integer:(type != dtype::Float) onChange:^(float v) {
            on_owner(s, ^{
                write_leaf(leaf, v);
                notify(s);
            });
        }];
        return leaf;
    }

public:
    // One pass over a graph: invalidate from `out` (pass id from the global epoch, so it can't
    // collide with any other pass), then evaluate whatever is dirty. Returns true if `out`
    // was recomputed. Call it on the thread that owns the graph; loop-driven applets call it
    // directly inside their own loop instead of using on_change.
    static bool run_graph_pass(matrix& out) {
        if (!out.tape) return false;
        bool dirty = out.tape->invalidate_pass(++global_epoch) || !out.tape->evaluated;
        if (dirty) out.eval();
        return dirty;
    }

    // `owner_queue` must be a serial queue: the thread that owns the graphs these sliders feed.
    SliderGroup(id intelligence, dispatch_queue_t owner_queue = dispatch_get_main_queue())
        : state(std::make_shared<State>()), owner(intelligence) {
        state->queue = owner_queue;
        state->queue_is_main = (owner_queue == dispatch_get_main_queue());
    }

    // A Float slider; returns its leaf. Use it anywhere a matrix goes (arithmetic, ScalarArg).
    matrix slider(NSString* title, float min, float max, float init) {
        return add_slider(title, min, max, init, dtype::Float);
    }

    // An Int32 slider (rounded), e.g. perlin()'s octaves.
    matrix slider_int(NSString* title, int min, int max, int init) {
        return add_slider(title, (float)min, (float)max, (float)init, dtype::Int32);
    }

    // An on/off switch; returns a Float leaf holding 1 (on) or 0 (off), so it can drive a graph
    // directly, e.g. blend two results with `on * a + (1 - on) * b`.
    matrix toggle(NSString* title, bool init) {
        matrix leaf = matrix::leaf_scalar(init ? 1.0f : 0.0f);
        std::shared_ptr<State> s = state;
        [panel() addToggleWithTitle:title value:init onChange:^(BOOL on) {
            on_owner(s, ^{
                write_leaf(leaf, on ? 1.0f : 0.0f);
                notify(s);
            });
        }];
        return leaf;
    }

    // Sets a slider's value from code (e.g. an "Auto range" button): clamps it to the slider's
    // range, writes the leaf, re-runs the watched outputs and moves the slider on screen. Call it
    // on the owner queue, like the leaf writes (button actions already run there).
    void set(const matrix& slider_leaf, float value) const {
        for (auto& info : state->sliders) {
            if (info.node != slider_leaf.tape) continue;
            float v = std::fmin(std::fmax(value, info.min), info.max);
            if (info.integer) v = std::round(v);
            write_leaf(slider_leaf, v);
            notify(state);
            [panel() setValue:v forSliderTitled:info.title];
            return;
        }
    }

    // Starts a titled group in the panel; the controls added after it go into that group.
    void section(NSString* title) {
        [panel() addSectionWithTitle:title];
    }

    // A button. `action` runs on the owner queue, so it may write this group's leaves (then
    // mark_changed() them); the watched outputs are re-run afterwards either way.
    void button(NSString* title, dispatch_block_t action) {
        std::shared_ptr<State> s = state;
        dispatch_block_t act = [action copy];
        [panel() addButtonWithTitle:title onPress:^{
            on_owner(s, ^{
                if (act) act();
                notify(s);
            });
        }];
    }

    // Re-run `out` on the owner queue whenever any slider in this group moves, then call
    // `publish(out)` (still on the owner queue) if it changed. Many ticks between two passes
    // are coalesced into one. Also runs once right away so the initial state is shown.
    //
    // `publish` receives the graph's live output: it's rewritten in place by the next pass, so
    // consume it before returning (updateBaseImageV2 copies it into a texture, which is fine),
    // or copy it if another thread will keep reading it.
    void on_change(matrix out, void (^publish)(matrix&)) {
        std::shared_ptr<State> s = state;
        on_owner(s, ^{
            s->watchers.push_back({out, publish});
            if (!s->trigger) {
                s->trigger = dispatch_source_create(DISPATCH_SOURCE_TYPE_DATA_OR, 0, 0, s->queue);
                std::weak_ptr<State> weak = s;
                dispatch_source_set_event_handler(s->trigger, ^{
                    if (auto st = weak.lock()) run_watchers(st);
                });
                dispatch_resume(s->trigger);
            }
            Watcher& w = s->watchers.back();
            run_graph_pass(w.out);
            if (w.publish) w.publish(w.out);
        });
    }
};

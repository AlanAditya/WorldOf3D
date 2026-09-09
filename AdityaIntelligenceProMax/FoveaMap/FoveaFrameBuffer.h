#pragma once
#include "../matrix.h"

#include <deque>
#include <utility>

namespace fovea {

// Rolling history of recent LiDAR point-cloud frames (default 6). Stores
// `matrix` copies directly rather than converting every frame to plain C++ -
// once a matrix has actually been evaluated (which a `depth_streamer`-style
// points frame always has been by the time it reaches this buffer), copying
// it is cheap: the buffer is refcount-shared, not deep-copied
// (.agents/MemoryManagement.md). This is just a history of handles.
class FrameHistory {
public:
    explicit FrameHistory(size_t capacity = 6) : capacity_(capacity) {}

    void push(matrix frame) {
        frames_.push_back(std::move(frame));
        while (frames_.size() > capacity_) {
            frames_.pop_front();
        }
    }

    bool empty() const { return frames_.empty(); }
    size_t size() const { return frames_.size(); }
    size_t capacity() const { return capacity_; }

    matrix& latest() { return frames_.back(); }

    // framesAgo = 0 -> latest(), 1 -> the frame before that, etc.
    // Caller must check size() > framesAgo first.
    matrix& at(size_t framesAgo) { return frames_[frames_.size() - 1 - framesAgo]; }

private:
    size_t capacity_;
    std::deque<matrix> frames_;
};

} // namespace fovea

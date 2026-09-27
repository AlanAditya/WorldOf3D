//
// QuantumStream.h
// A Sci-Fi inspired network highway for continuous DAG matrix streaming.
//

#pragma once
#include <cstdint>
#include <cstddef>
#include <Network/Network.h>
#include <memory>
#include <mutex>
#include <vector>
#include "matrix.h"

#pragma pack(push, 1)
// Leads every frame across the slipspace - single-matrix or batch alike.
// Says how many matrices follow and how large the trailing shared aux
// blob is (frame_id/aux_size used to live on QuantumMetadata itself, but
// that only made sense for exactly one matrix per frame).
struct QuantumFrameHeader {
    uint32_t frame_id;
    uint32_t count;     // number of matrices packed into this frame (>= 1)
    uint32_t aux_size;
};

// One of these precedes each matrix's raw payload inside a frame.
struct QuantumMetadata {
    uint32_t dims;
    dtype    dtype_code;
    uint8_t  flags;
    size_t   total_size;
    size_t   shape[SBO_MAX_DIMS];
    size_t   stride[SBO_MAX_DIMS];
};
#pragma pack(pop)

// ------------------------------------------------------------------------
// QuantumSender (The Firehose)
// Lives on the iPhone. Constantly blasts matrices across the network.
// ------------------------------------------------------------------------
class QuantumSender {
private:
    nw_connection_t connection = nullptr;
    uint32_t frame_counter = 0;
    std::atomic<bool> is_sending{false}; // Prevents TCP buffer bloat!

    // Shared implementation behind both stream_frame overloads below.
    void send_frame_internal(std::vector<const matrix*> mats, const void* aux_data, uint32_t aux_size);

public:
    // Initiates the slipspace link to the target Mac
    void start(const char* ip, const char* port);

    // Sends a single matrix over the network (Fire and Forget)
    void stream_frame(matrix& mat, const void* aux_data = nullptr, uint32_t aux_size = 0);

    // Sends multiple matrices as one frame (e.g. depth + RGB together) so
    // the receiver gets them atomically, sharing a single aux blob. Takes a
    // const& so an inline temporary (stream_frame(std::vector<matrix>{depth,
    // image}, ...)) binds without needing a named local first.
    void stream_frame(const std::vector<matrix>& mats, const void* aux_data = nullptr, uint32_t aux_size = 0);
};

// ------------------------------------------------------------------------
// QuantumReceiver (The Pull-Node)
// Lives on the Mac. Buffers the firehose in the background, allowing the
// DAG thread to dip into the stream at its own independent rate.
// ------------------------------------------------------------------------
class QuantumReceiver {
private:
    nw_listener_t listener = nullptr;
    nw_connection_t connection = nullptr;

    // The background bucket - one metadata/payload pair per matrix in the
    // latest frame (size 1 for a plain single-matrix stream).
    std::vector<QuantumMetadata> latest_metas;
    std::vector<std::vector<uint8_t>> staging_buffers;
    std::vector<uint8_t> aux_staging_buffer; // Bucket for camera intrinsics etc.
    std::mutex buffer_mutex;
    bool has_received_first_frame = false;

    // The infinite background read loop: reads the frame header, then hands
    // off to read_matrix_at_index to pull the `count` matrices it declares.
    void background_receive_loop();
    void read_matrix_at_index(std::shared_ptr<QuantumFrameHeader> header,
                               std::shared_ptr<std::vector<QuantumMetadata>> metas,
                               std::shared_ptr<std::vector<std::vector<uint8_t>>> buffers,
                               uint32_t index);
    // Once all `count` matrices are in: reads the shared trailing aux blob
    // (if any), stashes everything under the lock, and loops for the next frame.
    void finish_frame(QuantumFrameHeader header,
                       std::vector<QuantumMetadata> metas,
                       std::vector<std::vector<uint8_t>> buffers);

public:
    matrix* dag_leaf_matrix = nullptr; // The first/only DAG leaf - kept for single-matrix callers
    std::vector<matrix*> dag_leaf_matrices; // One DAG leaf per matrix in the stream
    bool is_allocated = false;

    // Opens the slipspace portal on a specific port
    void start(const char* port);

    // Called by the Engine DAG thread right before evaluation
    bool pull_latest_data(void* out_aux_buffer = nullptr, uint32_t max_aux_size = 0);
};

//
// QuantumStream.h
// A Sci-Fi inspired network highway for continuous DAG matrix streaming.
//

#pragma once
#include <cstdint>
#include <cstddef>
#include <Network/Network.h>
#include <mutex>
#include <vector>
#include "matrix.h"

#pragma pack(push, 1)
// The metadata payload that leads every frame across the slipspace
struct QuantumMetadata {
    uint32_t frame_id;
    uint32_t dims;
    dtype    dtype_code;
    uint8_t  flags;
    size_t   total_size;
    size_t   shape[SBO_MAX_DIMS];
    size_t   stride[SBO_MAX_DIMS];
    uint32_t aux_size; // NEW: Supports arbitrary extra data!
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

public:
    // Initiates the slipspace link to the target Mac
    void start(const char* ip, const char* port);
    
    // Sends the matrix over the network (Fire and Forget)
    void stream_frame(matrix& mat, const void* aux_data = nullptr, uint32_t aux_size = 0);
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
    
    // The background bucket
    std::vector<uint8_t> staging_buffer;
    std::vector<uint8_t> aux_staging_buffer; // Bucket for camera intrinsics etc.
    QuantumMetadata latest_meta;
    std::mutex buffer_mutex;
    bool has_received_first_frame = false;

    // The infinite background read loop
    void background_receive_loop();

public:
    matrix* dag_leaf_matrix = nullptr; // The root of your engine's DAG
    bool is_allocated = false;

    // Opens the slipspace portal on a specific port
    void start(const char* port);
    
    // Called by the Engine DAG thread right before evaluation
    bool pull_latest_data(void* out_aux_buffer = nullptr, uint32_t max_aux_size = 0);
};

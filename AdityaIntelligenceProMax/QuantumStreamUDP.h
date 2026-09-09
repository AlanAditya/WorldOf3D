//
// QuantumStreamUDP.h
// The UDP Slicer for Zero-Latency Wireless matrix streaming.
//

#pragma once
#include <cstdint>
#include <cstddef>
#include <Network/Network.h>
#include <mutex>
#include <vector>
#include "matrix.h"

#pragma pack(push, 1)
struct QuantumChunkMetadata {
    uint32_t frame_id;
    uint16_t chunk_index;
    uint16_t total_chunks;
    
    uint32_t dims;
    dtype    dtype_code;
    uint8_t  flags;
    size_t   total_size;
    size_t   shape[SBO_MAX_DIMS];
    size_t   stride[SBO_MAX_DIMS];
    
    size_t   chunk_offset_bytes;
    size_t   chunk_payload_bytes;
};
#pragma pack(pop)

class QuantumSenderUDP {
private:
    nw_connection_t connection = nullptr;
    uint32_t frame_counter = 0;

public:
    void start(const char* ip, const char* port);
    void stream_frame(matrix& mat);
};

class QuantumReceiverUDP {
private:
    nw_listener_t listener = nullptr;
    nw_connection_t connection = nullptr;
    
    // Assembly area for incoming chunks
    std::vector<uint8_t> assembly_buffer;
    uint32_t current_assembly_frame = 0;
    uint16_t chunks_received = 0;
    
    // Ready area for the DAG to pull from
    std::vector<uint8_t> ready_buffer;
    QuantumChunkMetadata latest_meta;
    std::mutex buffer_mutex;
    bool has_ready_frame = false;

    void background_receive_loop();

public:
    matrix* dag_leaf_matrix = nullptr;
    bool is_allocated = false;

    void start(const char* port);
    bool pull_latest_data();
};

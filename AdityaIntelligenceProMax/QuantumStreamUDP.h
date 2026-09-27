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
    uint32_t matrix_index;  // which matrix within this frame (0-based); == matrix_count marks this chunk as the shared aux blob instead of matrix data
    uint32_t matrix_count;  // how many real matrices are packed into this frame
    uint32_t aux_size;      // shared aux blob size for the whole frame (0 if none)
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

    // Shared implementation behind both stream_frame overloads below.
    void send_frame_internal(std::vector<const matrix*> mats, const void* aux_data, uint32_t aux_size);

public:
    void start(const char* ip, const char* port);

    // Sends a single matrix over the network (Fire and Forget)
    void stream_frame(matrix& mat, const void* aux_data = nullptr, uint32_t aux_size = 0);

    // Sends multiple matrices as one frame (e.g. depth + RGB together).
    // aux_data must fit in a single UDP payload (~1300 bytes) - it's sent as
    // one extra untagged chunk, not fragmented like the matrix payloads are.
    void stream_frame(const std::vector<matrix>& mats, const void* aux_data = nullptr, uint32_t aux_size = 0);
};

class QuantumReceiverUDP {
private:
    nw_listener_t listener = nullptr;
    nw_connection_t connection = nullptr;

    // Assembly area for the frame currently being reassembled - one buffer
    // per matrix, plus the shared aux blob.
    uint32_t current_assembly_frame = 0;
    std::vector<std::vector<uint8_t>> assembly_buffers;
    std::vector<uint16_t> chunks_received_per_matrix;
    std::vector<QuantumChunkMetadata> assembly_metas;
    std::vector<bool> matrix_complete;
    std::vector<uint8_t> aux_assembly_buffer;
    bool aux_complete = false;

    // Ready area for the DAG to pull from
    std::vector<std::vector<uint8_t>> ready_buffers;
    std::vector<QuantumChunkMetadata> latest_metas;
    std::vector<uint8_t> ready_aux_buffer;
    std::mutex buffer_mutex;
    bool has_ready_frame = false;

    void background_receive_loop();

public:
    matrix* dag_leaf_matrix = nullptr; // The first/only DAG leaf - kept for single-matrix callers
    std::vector<matrix*> dag_leaf_matrices;
    bool is_allocated = false;

    void start(const char* port);
    bool pull_latest_data(void* out_aux_buffer = nullptr, uint32_t max_aux_size = 0);
};

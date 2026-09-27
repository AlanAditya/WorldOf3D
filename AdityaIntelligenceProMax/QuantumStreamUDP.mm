//
// QuantumStreamUDP.mm
//

#include "QuantumStreamUDP.h"
#include <algorithm>
#include <iostream>

// 1300 bytes easily fits inside the standard 1500-byte WiFi MTU.
// This prevents the OS from attempting (and failing) IP-level fragmentation.
#define MAX_UDP_PAYLOAD 1300 

// ========================================================================
// QuantumSenderUDP
// ========================================================================

void QuantumSenderUDP::start(const char* ip, const char* port) {
    nw_endpoint_t endpoint = nw_endpoint_create_host(ip, port);
    
    // RAW UDP! Zero latency, drops packets if the network is busy
    nw_parameters_t params = nw_parameters_create_secure_udp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    
    connection = nw_connection_create(endpoint, params);
    nw_connection_set_queue(connection, dispatch_get_main_queue());
    
    nw_connection_set_state_changed_handler(connection, ^(nw_connection_state_t state, nw_error_t error) {
        if (state == nw_connection_state_ready) {
            printf("[QuantumSenderUDP] Wireless Slipspace link established to %s:%s\n", ip, port);
        }
    });
    
    nw_connection_start(connection);
}

void QuantumSenderUDP::stream_frame(matrix& mat, const void* aux_data, uint32_t aux_size) {
    std::vector<const matrix*> mats{ &mat };
    send_frame_internal(mats, aux_data, aux_size);
}

void QuantumSenderUDP::stream_frame(const std::vector<matrix>& mats, const void* aux_data, uint32_t aux_size) {
    std::vector<const matrix*> ptrs;
    ptrs.reserve(mats.size());
    for (const matrix& m : mats) ptrs.push_back(&m);
    send_frame_internal(ptrs, aux_data, aux_size);
}

void QuantumSenderUDP::send_frame_internal(std::vector<const matrix*> mats, const void* aux_data, uint32_t aux_size) {
    if (!connection || mats.empty()) return;

    if (aux_size > 0 && aux_size > MAX_UDP_PAYLOAD - sizeof(QuantumChunkMetadata)) {
        printf("[QuantumSenderUDP] aux_size %u doesn't fit in a single UDP payload (max %zu) - dropping aux for this frame!\n",
               aux_size, MAX_UDP_PAYLOAD - sizeof(QuantumChunkMetadata));
        aux_data = nullptr;
        aux_size = 0;
    }

    uint32_t current_frame = frame_counter++;
    uint32_t matrix_count = (uint32_t)mats.size();

    // Batching locks the network stack and fires all packets in one single burst.
    // This reduces the 2600+ system calls/sec down to practically 1, instantly killing the CPU overhead.
    nw_connection_batch(connection, ^{
        for (uint32_t m = 0; m < matrix_count; ++m) {
            const matrix& mat = *mats[m];
            size_t total_payload_bytes = mat.total_size * dtype_size(mat.type);
            uint16_t total_chunks = (uint16_t)((total_payload_bytes + MAX_UDP_PAYLOAD - 1) / MAX_UDP_PAYLOAD);
            if (total_chunks == 0) total_chunks = 1; // still send one (empty) chunk so the receiver learns about this matrix

            for (uint16_t i = 0; i < total_chunks; ++i) {
                QuantumChunkMetadata meta = {};
                meta.frame_id = current_frame;
                meta.matrix_index = m;
                meta.matrix_count = matrix_count;
                meta.aux_size = aux_size;
                meta.chunk_index = i;
                meta.total_chunks = total_chunks;

                meta.dims = mat.dims;
                meta.dtype_code = mat.type;
                meta.flags = mat.flags;
                meta.total_size = mat.total_size;

                for (uint32_t d = 0; d < mat.dims; ++d) {
                    meta.shape[d] = mat.shape()[d];
                    meta.stride[d] = mat.strides()[d];
                }

                meta.chunk_offset_bytes = (size_t)i * MAX_UDP_PAYLOAD;
                meta.chunk_payload_bytes = std::min((size_t)MAX_UDP_PAYLOAD, total_payload_bytes - meta.chunk_offset_bytes);

                // Map the struct
                dispatch_data_t meta_data = dispatch_data_create(&meta, sizeof(QuantumChunkMetadata),
                                                                 dispatch_get_main_queue(),
                                                                 DISPATCH_DATA_DESTRUCTOR_DEFAULT);

                // Map ONLY this chunk's slice of the matrix buffer
                const uint8_t* slice_ptr = static_cast<const uint8_t*>(mat.buffer) + meta.chunk_offset_bytes;
                dispatch_data_t payload_data = dispatch_data_create(slice_ptr, meta.chunk_payload_bytes,
                                                                    dispatch_get_main_queue(),
                                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);

                dispatch_data_t packet = dispatch_data_create_concat(meta_data, payload_data);

                // Blast the chunk (Fire and Forget)
                nw_connection_send(connection, packet, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t error) {});
            }
        }

        // Shared aux blob (camera intrinsics etc), sent once per frame as one
        // extra untagged chunk - matrix_index == matrix_count marks it as aux
        // rather than real matrix data.
        if (aux_size > 0 && aux_data != nullptr) {
            QuantumChunkMetadata meta = {};
            meta.frame_id = current_frame;
            meta.matrix_index = matrix_count;
            meta.matrix_count = matrix_count;
            meta.aux_size = aux_size;
            meta.chunk_index = 0;
            meta.total_chunks = 1;
            meta.chunk_offset_bytes = 0;
            meta.chunk_payload_bytes = aux_size;

            dispatch_data_t meta_data = dispatch_data_create(&meta, sizeof(QuantumChunkMetadata),
                                                             dispatch_get_main_queue(),
                                                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            dispatch_data_t payload_data = dispatch_data_create(aux_data, aux_size,
                                                                dispatch_get_main_queue(),
                                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            dispatch_data_t packet = dispatch_data_create_concat(meta_data, payload_data);
            nw_connection_send(connection, packet, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t error) {});
        }
    });
}


// ========================================================================
// QuantumReceiverUDP
// ========================================================================

void QuantumReceiverUDP::start(const char* port) {
    nw_parameters_t params = nw_parameters_create_secure_udp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    listener = nw_listener_create_with_port(port, params);
    
    nw_listener_set_queue(listener, dispatch_get_main_queue());
    
    nw_listener_set_new_connection_handler(listener, ^(nw_connection_t new_conn) {
        printf("[QuantumReceiverUDP] Incoming wireless connection!\n");
        this->connection = new_conn;
        nw_connection_set_queue(this->connection, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
        nw_connection_start(this->connection);
        
        this->background_receive_loop();
    });
    
    nw_listener_start(listener);
}

void QuantumReceiverUDP::background_receive_loop() {
    // In UDP, receive_message gives us exactly ONE complete datagram packet.
    nw_connection_receive_message(connection,
    ^(dispatch_data_t content, nw_content_context_t context, bool is_complete, nw_error_t error) {

        if (error || content == nullptr) {
            printf("[QuantumReceiverUDP] Datagram read failed (error domain=%d code=%d) - read loop stops here!\n",
                   error ? nw_error_get_error_domain(error) : -1, error ? nw_error_get_error_code(error) : -1);
            return;
        }

        const void* packet_ptr = nullptr;
        size_t packet_size = 0;
        dispatch_data_t map = dispatch_data_create_map(content, &packet_ptr, &packet_size);
        (void)map; // ARC must retain this, or packet_ptr is instantly deallocated

        if (packet_size < sizeof(QuantumChunkMetadata)) {
            printf("[QuantumReceiverUDP] Datagram too small (%zu bytes, need %zu) - dropping\n", packet_size, sizeof(QuantumChunkMetadata));
            this->background_receive_loop();
            return;
        }

        // 1. Read Metadata
        QuantumChunkMetadata meta;
        memcpy(&meta, packet_ptr, sizeof(QuantumChunkMetadata));
        const uint8_t* payload_ptr = static_cast<const uint8_t*>(packet_ptr) + sizeof(QuantumChunkMetadata);

        printf("[QuantumReceiverUDP] Packet: frame_id=%u matrix_index=%u/%u chunk=%u/%u payload=%zu current_assembly_frame=%u\n",
               meta.frame_id, meta.matrix_index, meta.matrix_count, meta.chunk_index, meta.total_chunks,
               meta.chunk_payload_bytes, this->current_assembly_frame);

        // 2. Are we on a new frame? (Or did we miss packets from an old one?)
        // Only reset forward - a late packet from an already-superseded
        // older frame must NOT rewind our progress on the current one.
        if (meta.frame_id > this->current_assembly_frame || this->assembly_metas.empty()) {
            this->current_assembly_frame = meta.frame_id;
            this->assembly_buffers.assign(meta.matrix_count, {});
            this->chunks_received_per_matrix.assign(meta.matrix_count, 0);
            this->assembly_metas.assign(meta.matrix_count, QuantumChunkMetadata{});
            this->matrix_complete.assign(meta.matrix_count, false);
            this->aux_assembly_buffer.clear();
            this->aux_complete = (meta.aux_size == 0);
        }

        if (meta.frame_id == this->current_assembly_frame) {
            if (meta.matrix_index == meta.matrix_count) {
                // This chunk is the frame's shared aux blob, not matrix data.
                if (!this->aux_complete) {
                    this->aux_assembly_buffer.assign(payload_ptr, payload_ptr + meta.chunk_payload_bytes);
                    this->aux_complete = true;
                }
            } else if (meta.matrix_index < this->assembly_buffers.size()) {
                uint32_t mi = meta.matrix_index;
                std::vector<uint8_t>& buf = this->assembly_buffers[mi];
                size_t total_payload_bytes = meta.total_size * dtype_size(meta.dtype_code);
                if (buf.size() != total_payload_bytes) {
                    buf.resize(total_payload_bytes);
                }
                this->assembly_metas[mi] = meta;

                // Bounds check to absolutely prevent EXC_BAD_ACCESS memory crashes
                if (!this->matrix_complete[mi] &&
                    meta.chunk_offset_bytes + meta.chunk_payload_bytes <= buf.size()) {

                    memcpy(buf.data() + meta.chunk_offset_bytes, payload_ptr, meta.chunk_payload_bytes);
                    this->chunks_received_per_matrix[mi]++;

                    if (this->chunks_received_per_matrix[mi] == meta.total_chunks) {
                        this->matrix_complete[mi] = true;
                    }
                }
            }

            // 3. Did every matrix AND the shared aux blob (if any) arrive?
            bool all_matrices_done = !this->matrix_complete.empty() &&
                std::all_of(this->matrix_complete.begin(), this->matrix_complete.end(), [](bool b) { return b; });

            if (all_matrices_done && this->aux_complete) {
                printf("[QuantumReceiverUDP] Frame %u complete (%zu matrices)\n", meta.frame_id, this->matrix_complete.size());
                // LOCK and hand off to the DAG's Ready Buffers
                std::lock_guard<std::mutex> lock(this->buffer_mutex);
                this->latest_metas = this->assembly_metas;
                this->ready_buffers = this->assembly_buffers;
                this->ready_aux_buffer = this->aux_assembly_buffer;
                this->has_ready_frame = true;
            }
        }

        // Instantly wait for next chunk
        this->background_receive_loop();
    });
}

bool QuantumReceiverUDP::pull_latest_data(void* out_aux_buffer, uint32_t max_aux_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex);

    if (!has_ready_frame) return false;

    if (!is_allocated) {
        for (const QuantumChunkMetadata& meta : latest_metas) {
            matrix* leaf = new matrix(meta.dims, meta.total_size, meta.dtype_code);

            for (uint32_t i = 0; i < meta.dims; ++i) {
                leaf->shape()[i] = meta.shape[i];
                leaf->strides()[i] = meta.stride[i];
            }

            dag_leaf_matrices.push_back(leaf);
        }

        dag_leaf_matrix = dag_leaf_matrices.empty() ? nullptr : dag_leaf_matrices[0];
        is_allocated = true;
        printf("[QuantumReceiverUDP] Pre-allocated %zu DAG leaf node(s)\n", dag_leaf_matrices.size());
    }

    for (size_t i = 0; i < dag_leaf_matrices.size() && i < ready_buffers.size(); ++i) {
        matrix* leaf = dag_leaf_matrices[i];
        if (leaf && leaf->buffer) {
            size_t payload_bytes = latest_metas[i].total_size * dtype_size(latest_metas[i].dtype_code);
            memcpy(leaf->buffer, ready_buffers[i].data(), payload_bytes);
        }
    }

    if (out_aux_buffer != nullptr && ready_aux_buffer.size() > 0) {
        size_t copy_size = std::min((size_t)max_aux_size, ready_aux_buffer.size());
        memcpy(out_aux_buffer, ready_aux_buffer.data(), copy_size);
    }

    // Consumed: the next call returns true only once another frame has been reassembled, so a
    // caller can tell a new frame from the one it already has.
    has_ready_frame = false;
    return true;
}

//
// QuantumStreamUDP.mm
//

#include "QuantumStreamUDP.h"
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

void QuantumSenderUDP::stream_frame(matrix& mat) {
    if (!connection) return;

    size_t total_payload_bytes = mat.total_size * dtype_size(mat.type);
    uint16_t total_chunks = (total_payload_bytes + MAX_UDP_PAYLOAD - 1) / MAX_UDP_PAYLOAD;
    
    uint32_t current_frame = frame_counter++;

    // Batching locks the network stack and fires all packets in one single burst.
    // This reduces the 2600+ system calls/sec down to practically 1, instantly killing the CPU overhead.
    nw_connection_batch(connection, ^{
        for (uint16_t i = 0; i < total_chunks; ++i) {
            QuantumChunkMetadata meta = {};
            meta.frame_id = current_frame;
            meta.chunk_index = i;
            meta.total_chunks = total_chunks;
            
            meta.dims = mat.dims;
            meta.dtype_code = mat.type;
            meta.flags = mat.flags;
            meta.total_size = mat.total_size;
            
            for (int d = 0; d < mat.dims; ++d) {
                meta.shape[d] = mat.shape()[d];
                meta.stride[d] = mat.strides()[d];
            }
            
            meta.chunk_offset_bytes = i * MAX_UDP_PAYLOAD;
            meta.chunk_payload_bytes = std::min((size_t)MAX_UDP_PAYLOAD, total_payload_bytes - meta.chunk_offset_bytes);
            
            // Map the struct
            dispatch_data_t meta_data = dispatch_data_create(&meta, sizeof(QuantumChunkMetadata), 
                                                             dispatch_get_main_queue(), 
                                                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            
            // Map ONLY this chunk's slice of the matrix buffer
            uint8_t* slice_ptr = static_cast<uint8_t*>(mat.buffer) + meta.chunk_offset_bytes;
            dispatch_data_t payload_data = dispatch_data_create(slice_ptr, meta.chunk_payload_bytes, 
                                                                dispatch_get_main_queue(), 
                                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            
            dispatch_data_t packet = dispatch_data_create_concat(meta_data, payload_data);

            // Blast the chunk (Fire and Forget)
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
        
        if (error || content == nullptr) return;

        const void* packet_ptr = nullptr;
        size_t packet_size = 0;
        dispatch_data_t map = dispatch_data_create_map(content, &packet_ptr, &packet_size);
        
        if (packet_size < sizeof(QuantumChunkMetadata)) {
            this->background_receive_loop();
            return;
        }

        // 1. Read Metadata
        QuantumChunkMetadata meta;
        memcpy(&meta, packet_ptr, sizeof(QuantumChunkMetadata));
        
        // 2. Are we on a new frame? (Or did we miss packets from an old one?)
        if (meta.frame_id > this->current_assembly_frame || this->assembly_buffer.empty()) {
            this->current_assembly_frame = meta.frame_id;
            this->chunks_received = 0;
            size_t total_payload_bytes = meta.total_size * dtype_size(meta.dtype_code);
            if (this->assembly_buffer.size() != total_payload_bytes) {
                this->assembly_buffer.resize(total_payload_bytes);
            }
        }
        
        // 3. If this chunk belongs to our current assembly frame, copy it in
        if (meta.frame_id == this->current_assembly_frame) {
            
            // Bounds check to absolutely prevent EXC_BAD_ACCESS memory crashes
            if (meta.chunk_offset_bytes + meta.chunk_payload_bytes <= this->assembly_buffer.size()) {
                
                const uint8_t* payload_ptr = static_cast<const uint8_t*>(packet_ptr) + sizeof(QuantumChunkMetadata);
                memcpy(this->assembly_buffer.data() + meta.chunk_offset_bytes, payload_ptr, meta.chunk_payload_bytes);
                this->chunks_received++;
                
                // 4. Did we get all chunks for this frame?
                if (this->chunks_received == meta.total_chunks) {
                    // LOCK and hand off to the DAG's Ready Buffer
                    std::lock_guard<std::mutex> lock(this->buffer_mutex);
                    this->latest_meta = meta;
                    
                    if (this->ready_buffer.size() != this->assembly_buffer.size()) {
                        this->ready_buffer.resize(this->assembly_buffer.size());
                    }
                    
                    // Fast copy to ready area
                    memcpy(this->ready_buffer.data(), this->assembly_buffer.data(), this->assembly_buffer.size());
                    this->has_ready_frame = true;
                }
            } // <-- Missing brace added here!
        }

        // Instantly wait for next chunk
        this->background_receive_loop();
    });
}

bool QuantumReceiverUDP::pull_latest_data() {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    if (!has_ready_frame) return false;
    
    if (!is_allocated) {
        dag_leaf_matrix = new matrix(latest_meta.dims, latest_meta.total_size, latest_meta.dtype_code);
        
        for (int i = 0; i < latest_meta.dims; ++i) {
            dag_leaf_matrix->shape()[i] = latest_meta.shape[i];
            dag_leaf_matrix->strides()[i] = latest_meta.stride[i];
        }
        
        is_allocated = true;
        printf("[QuantumReceiverUDP] Pre-allocated DAG leaf node with %zu elements\n", latest_meta.total_size);
    }

    if (dag_leaf_matrix && dag_leaf_matrix->buffer) {
        size_t payload_bytes = latest_meta.total_size * dtype_size(latest_meta.dtype_code);
        memcpy(dag_leaf_matrix->buffer, ready_buffer.data(), payload_bytes);
    }
    
    return true;
}

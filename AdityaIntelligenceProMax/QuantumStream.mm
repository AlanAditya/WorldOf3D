//
// QuantumStream.mm
//

#include "QuantumStream.h"
#include <iostream>

// ========================================================================
// QuantumSender Implementation
// ========================================================================

void QuantumSender::start(const char* ip, const char* port) {
    nw_endpoint_t endpoint = nw_endpoint_create_host(ip, port);
    
    // Using secure TCP to automatically handle fragmentation for matrices larger than 64KB
    nw_parameters_t params = nw_parameters_create_secure_tcp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    
    connection = nw_connection_create(endpoint, params);
    nw_connection_set_queue(connection, dispatch_get_main_queue());
    
    nw_connection_set_state_changed_handler(connection, ^(nw_connection_state_t state, nw_error_t error) {
        if (state == nw_connection_state_ready) {
            printf("[QuantumSender] Slipspace link established to %s:%s\n", ip, port);
        }
    });
    
    nw_connection_start(connection);
}

void QuantumSender::stream_frame(matrix& mat, const void* aux_data, uint32_t aux_size) {
    if (!connection) return;
    
    // DROP FRAME LOGIC:
    bool expected = false;
    if (!is_sending.compare_exchange_strong(expected, true)) {
        return; // Drop this frame!
    }

    // 1. Prepare Metadata
    QuantumMetadata meta = {};
    meta.frame_id = frame_counter++;
    meta.dims = mat.dims;
    meta.dtype_code = mat.type;
    meta.flags = mat.flags;
    meta.total_size = mat.total_size;
    meta.aux_size = aux_size; // Track how much extra data we are sending
    
    for (int i = 0; i < mat.dims; ++i) {
        meta.shape[i] = mat.shape()[i];
        meta.stride[i] = mat.strides()[i];
    }

    // 2. Safely Copy Stack Memory for Async Transmission
    // (dispatch_data_t is NOT toll-free bridged to NSData! So we malloc/copy it explicitly)
    void* safe_meta = malloc(sizeof(QuantumMetadata));
    memcpy(safe_meta, &meta, sizeof(QuantumMetadata));
    dispatch_data_t meta_data = dispatch_data_create(safe_meta, sizeof(QuantumMetadata), 
                                                     dispatch_get_main_queue(), 
                                                     ^{ free(safe_meta); });
    
    // Copy the buffer before handing it to the network layer: `mat` is a
    // reused __block matrix in the caller's GCD loop, so its buffer can be
    // overwritten by the next capture tick before this async send actually
    // completes. Wrapping the live pointer directly (as before) raced the
    // capture callback - this is very likely the source of the earlier
    // "Use of deallocated memory" crash. Mirrors how the aux data below is
    // already safely copied.
    size_t payload_bytes = mat.total_size * dtype_size(mat.type);
    void* safe_payload = malloc(payload_bytes);
    memcpy(safe_payload, mat.buffer, payload_bytes);
    dispatch_data_t payload_data = dispatch_data_create(safe_payload, payload_bytes,
                                                        dispatch_get_main_queue(),
                                                        ^{ free(safe_payload); });
    
    dispatch_data_t packet = dispatch_data_create_concat(meta_data, payload_data);

    // 3. Append the auxiliary data (intrinsics, custom structs, etc)
    if (aux_size > 0 && aux_data != nullptr) {
        // Safely copy the auxiliary stack data
        void* safe_aux = malloc(aux_size);
        memcpy(safe_aux, aux_data, aux_size);
        dispatch_data_t aux_dispatch = dispatch_data_create(safe_aux, aux_size, 
                                                            dispatch_get_main_queue(), 
                                                            ^{ free(safe_aux); });
        packet = dispatch_data_create_concat(packet, aux_dispatch);
    }

    // 4. Blast over the network
    nw_connection_send(connection, packet, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t error) {
        is_sending.store(false); 
        
        if (error) {
            printf("[QuantumSender] Transmission failure on frame %d\n", meta.frame_id);
        }
    });
}


// ========================================================================
// QuantumReceiver Implementation
// ========================================================================

void QuantumReceiver::start(const char* port) {
    nw_parameters_t params = nw_parameters_create_secure_tcp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    listener = nw_listener_create_with_port(port, params);
    
    nw_listener_set_queue(listener, dispatch_get_main_queue());
    
    nw_listener_set_new_connection_handler(listener, ^(nw_connection_t new_conn) {
        printf("[QuantumReceiver] Incoming slipspace connection!\n");
        this->connection = new_conn;
        
        // Put the receiver on a dedicated high-priority background queue
        nw_connection_set_queue(this->connection, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
        nw_connection_start(this->connection);
        
        this->background_receive_loop();
    });
    
    nw_listener_start(listener);
}

void QuantumReceiver::background_receive_loop() {
    // 1. Read Metadata first
    nw_connection_receive(connection, sizeof(QuantumMetadata), sizeof(QuantumMetadata),
    ^(dispatch_data_t meta_content, nw_content_context_t context, bool is_complete, nw_error_t error) {

        if (error || meta_content == nullptr) return;

        QuantumMetadata meta;
        const void* meta_ptr = nullptr;
        size_t meta_size = 0;
        dispatch_data_t meta_map = dispatch_data_create_map(meta_content, &meta_ptr, &meta_size);
        memcpy(&meta, meta_ptr, sizeof(QuantumMetadata));

        size_t payload_bytes = meta.total_size * dtype_size(meta.dtype_code);

        // 2. Read the matrix payload as its own independent, exactly-sized
        // read. Previously this read payload+aux together and spliced aux
        // out via `payload_ptr + payload_bytes` pointer arithmetic - that's
        // what was corrupting the aux data (camera intrinsics arrived as
        // near-zero denormal floats, the signature of reading the wrong
        // offset). Giving aux its own dedicated receive below removes any
        // manual offset math entirely.
        nw_connection_receive(this->connection, payload_bytes, payload_bytes,
        ^(dispatch_data_t payload_content, nw_content_context_t ctx, bool comp, nw_error_t err) {

            if (err || payload_content == nullptr) return;

            const void* payload_ptr = nullptr;
            size_t p_size = 0;
            // ARC MUST retain this object, or payload_ptr will be instantly deallocated!
            dispatch_data_t payload_map = dispatch_data_create_map(payload_content, &payload_ptr, &p_size);
            (void)payload_map; // Silence unused variable warning

            if (meta.aux_size == 0) {
                std::lock_guard<std::mutex> lock(this->buffer_mutex);
                this->latest_meta = meta;
                if (this->staging_buffer.size() != payload_bytes) {
                    this->staging_buffer.resize(payload_bytes);
                }
                memcpy(this->staging_buffer.data(), payload_ptr, payload_bytes);
                this->has_received_first_frame = true;
                this->background_receive_loop();
                return;
            }

            // payload_content's mapped memory is only valid for this
            // callback's lifetime, and the aux read below is a second,
            // separate async network round-trip - stash a copy now.
            std::vector<uint8_t> payload_copy(
                static_cast<const uint8_t*>(payload_ptr),
                static_cast<const uint8_t*>(payload_ptr) + payload_bytes
            );

            // 3. Read the aux data (camera intrinsics etc.) as its own
            // independent, exactly-sized read.
            nw_connection_receive(this->connection, meta.aux_size, meta.aux_size,
            ^(dispatch_data_t aux_content, nw_content_context_t actx, bool acomp, nw_error_t aerr) {

                if (aerr || aux_content == nullptr) return;

                const void* aux_ptr = nullptr;
                size_t a_size = 0;
                dispatch_data_t aux_map = dispatch_data_create_map(aux_content, &aux_ptr, &a_size);
                (void)aux_map;

                // 4. Thread-safe handoff to the Staging Buffers
                {
                    std::lock_guard<std::mutex> lock(this->buffer_mutex);
                    this->latest_meta = meta;

                    if (this->staging_buffer.size() != payload_bytes) {
                        this->staging_buffer.resize(payload_bytes);
                    }
                    memcpy(this->staging_buffer.data(), payload_copy.data(), payload_bytes);

                    if (this->aux_staging_buffer.size() != meta.aux_size) {
                        this->aux_staging_buffer.resize(meta.aux_size);
                    }
                    memcpy(this->aux_staging_buffer.data(), aux_ptr, meta.aux_size);

                    this->has_received_first_frame = true;
                }

                // 5. Immediately listen for the next frame
                this->background_receive_loop();
            });
        });
    });
}

bool QuantumReceiver::pull_latest_data(void* out_aux_buffer, uint32_t max_aux_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    if (!has_received_first_frame) return false;
    
    // First time we pull, we initialize the DAG leaf matrix
    if (!is_allocated) {
        
        // Construct using your engine's leaf node constructor:
        dag_leaf_matrix = new matrix(latest_meta.dims, latest_meta.total_size, latest_meta.dtype_code);
        
        // Restore the array dimensions and strides so the DAG knows how to read the buffer
        for (int i = 0; i < latest_meta.dims; ++i) {
            dag_leaf_matrix->shape()[i] = latest_meta.shape[i];
            dag_leaf_matrix->strides()[i] = latest_meta.stride[i];
        }
        
        is_allocated = true;
        printf("[QuantumReceiver] Pre-allocated DAG leaf node with %zu elements\n", latest_meta.total_size);
    }

    if (dag_leaf_matrix && dag_leaf_matrix->buffer) {
        size_t payload_bytes = latest_meta.total_size * dtype_size(latest_meta.dtype_code);
        // Instant memcpy into the DAG's unified memory buffer!
        memcpy(dag_leaf_matrix->buffer, staging_buffer.data(), payload_bytes);
    }
    
    // Output the auxiliary data if requested
    if (out_aux_buffer != nullptr && latest_meta.aux_size > 0) {
        size_t copy_size = std::min((size_t)max_aux_size, (size_t)latest_meta.aux_size);
        memcpy(out_aux_buffer, aux_staging_buffer.data(), copy_size);
        
        if (max_aux_size != latest_meta.aux_size) {
            printf("[QuantumReceiver] Warning: Aux size mismatch (Mac asked for %u, iPhone sent %u). Struct padding difference?\n", 
                   max_aux_size, latest_meta.aux_size);
        }
    }
    
    return true;
}

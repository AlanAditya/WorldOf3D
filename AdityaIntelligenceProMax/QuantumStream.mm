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

    // Using secure TCP to automatically handle fragmentation for matrices larger than 64KB.
    // TLS is disabled (NW_PARAMETERS_DISABLE_PROTOCOL) but the default TCP
    // options still have Nagle's algorithm on, which can sit on small,
    // frequent sends for up to ~200ms - exactly the wrong thing for a
    // real-time frame stream. Disable it (TCP_NODELAY) so every send goes
    // out immediately.
    nw_parameters_t params = nw_parameters_create_secure_tcp(NW_PARAMETERS_DISABLE_PROTOCOL,
    ^(nw_protocol_options_t tcp_options) {
        nw_tcp_options_set_no_delay(tcp_options, true);
    });

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
    std::vector<const matrix*> mats{ &mat };
    send_frame_internal(mats, aux_data, aux_size);
}

void QuantumSender::stream_frame(const std::vector<matrix>& mats, const void* aux_data, uint32_t aux_size) {
    std::vector<const matrix*> ptrs;
    ptrs.reserve(mats.size());
    for (const matrix& m : mats) ptrs.push_back(&m);
    send_frame_internal(ptrs, aux_data, aux_size);
}

void QuantumSender::send_frame_internal(std::vector<const matrix*> mats, const void* aux_data, uint32_t aux_size) {
    if (!connection || mats.empty()) return;

    // DROP FRAME LOGIC:
    bool expected = false;
    if (!is_sending.compare_exchange_strong(expected, true)) {
        return; // Drop this frame!
    }

    // 1. Prepare the frame header (how many matrices, how big the aux blob)
    QuantumFrameHeader header = {};
    header.frame_id = frame_counter++;
    header.count = (uint32_t)mats.size();
    header.aux_size = aux_size;

    // Safely Copy Stack Memory for Async Transmission
    // (dispatch_data_t is NOT toll-free bridged to NSData! So we malloc/copy it explicitly)
    void* safe_header = malloc(sizeof(QuantumFrameHeader));
    memcpy(safe_header, &header, sizeof(QuantumFrameHeader));
    dispatch_data_t packet = dispatch_data_create(safe_header, sizeof(QuantumFrameHeader),
                                                   dispatch_get_main_queue(),
                                                   ^{ free(safe_header); });

    // 2. Per-matrix metadata + payload, back to back.
    for (const matrix* mat_ptr : mats) {
        const matrix& mat = *mat_ptr;

        QuantumMetadata meta = {};
        meta.dims = mat.dims;
        meta.dtype_code = mat.type;
        meta.flags = mat.flags;
        meta.total_size = mat.total_size;

        for (uint32_t i = 0; i < mat.dims; ++i) {
            meta.shape[i] = mat.shape()[i];
            meta.stride[i] = mat.strides()[i];
        }

        void* safe_meta = malloc(sizeof(QuantumMetadata));
        memcpy(safe_meta, &meta, sizeof(QuantumMetadata));
        dispatch_data_t meta_data = dispatch_data_create(safe_meta, sizeof(QuantumMetadata),
                                                         dispatch_get_main_queue(),
                                                         ^{ free(safe_meta); });
        packet = dispatch_data_create_concat(packet, meta_data);

        // Copy the buffer before handing it to the network layer: `mat` is
        // often a reused __block matrix in the caller's GCD loop, so its
        // buffer can be overwritten by the next capture tick before this
        // async send actually completes. Wrapping the live pointer directly
        // raced the capture callback - this is very likely the source of the
        // earlier "Use of deallocated memory" crash. Mirrors how the aux
        // data below is already safely copied.
        size_t payload_bytes = mat.total_size * dtype_size(mat.type);
        void* safe_payload = malloc(payload_bytes);
        memcpy(safe_payload, mat.buffer, payload_bytes);
        dispatch_data_t payload_data = dispatch_data_create(safe_payload, payload_bytes,
                                                            dispatch_get_main_queue(),
                                                            ^{ free(safe_payload); });
        packet = dispatch_data_create_concat(packet, payload_data);
    }

    // 3. Append the shared auxiliary data (intrinsics, custom structs, etc),
    // once for the whole frame.
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
    uint32_t sent_frame_id = header.frame_id;
    nw_connection_send(connection, packet, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t error) {
        is_sending.store(false);

        if (error) {
            printf("[QuantumSender] Transmission failure on frame %d\n", sent_frame_id);
        }
    });
}


// ========================================================================
// QuantumReceiver Implementation
// ========================================================================

void QuantumReceiver::start(const char* port) {
    // Same TCP_NODELAY reasoning as QuantumSender::start - must match on both
    // ends of the connection for the no-buffering behavior to actually apply.
    nw_parameters_t params = nw_parameters_create_secure_tcp(NW_PARAMETERS_DISABLE_PROTOCOL,
    ^(nw_protocol_options_t tcp_options) {
        nw_tcp_options_set_no_delay(tcp_options, true);
    });
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
    // 1. Read the frame header first: how many matrices follow, how big the
    // trailing shared aux blob is.
    nw_connection_receive(connection, sizeof(QuantumFrameHeader), sizeof(QuantumFrameHeader),
    ^(dispatch_data_t header_content, nw_content_context_t context, bool is_complete, nw_error_t error) {

        if (error || header_content == nullptr) {
            printf("[QuantumReceiver] Frame header read failed (error domain=%d code=%d, is_complete=%d, content_null=%d) - the read loop stops here until start() is called again!\n",
                   error ? nw_error_get_error_domain(error) : -1,
                   error ? nw_error_get_error_code(error) : -1,
                   is_complete, header_content == nullptr);
            return;
        }

        auto header = std::make_shared<QuantumFrameHeader>();
        const void* header_ptr = nullptr;
        size_t header_size = 0;
        dispatch_data_t header_map = dispatch_data_create_map(header_content, &header_ptr, &header_size);
        (void)header_map;
        memcpy(header.get(), header_ptr, sizeof(QuantumFrameHeader));

        printf("[QuantumReceiver] Frame header: frame_id=%u count=%u aux_size=%u\n",
               header->frame_id, header->count, header->aux_size);

        if (header->count == 0 || header->count > 64) {
            // A sane frame never packs zero or dozens of matrices - this is
            // almost certainly a stale sender still speaking the OLD wire
            // format (before QuantumFrameHeader existed), where these 12
            // bytes land in the middle of an old-style QuantumMetadata
            // instead. Rebuild + redeploy BOTH ends from current source.
            printf("[QuantumReceiver] Rejecting frame: count=%u looks bogus - sender/receiver protocol mismatch? Dropping connection read.\n", header->count);
            return;
        }

        auto metas = std::make_shared<std::vector<QuantumMetadata>>();
        auto buffers = std::make_shared<std::vector<std::vector<uint8_t>>>();
        metas->reserve(header->count);
        buffers->reserve(header->count);

        // 2. Pull the `count` matrices this frame declares, one at a time.
        this->read_matrix_at_index(header, metas, buffers, 0);
    });
}

void QuantumReceiver::read_matrix_at_index(std::shared_ptr<QuantumFrameHeader> header,
                                            std::shared_ptr<std::vector<QuantumMetadata>> metas,
                                            std::shared_ptr<std::vector<std::vector<uint8_t>>> buffers,
                                            uint32_t index) {
    if (index >= header->count) {
        // All matrices for this frame are in - move on to the (optional)
        // trailing aux blob.
        this->finish_frame(*header, *metas, *buffers);
        return;
    }

    // Read this matrix's metadata first.
    nw_connection_receive(this->connection, sizeof(QuantumMetadata), sizeof(QuantumMetadata),
    ^(dispatch_data_t meta_content, nw_content_context_t ctx, bool comp, nw_error_t err) {

        if (err || meta_content == nullptr) {
            printf("[QuantumReceiver] Metadata read failed for matrix %u/%u (error domain=%d code=%d) - read loop stops here!\n",
                   index, header->count, err ? nw_error_get_error_domain(err) : -1, err ? nw_error_get_error_code(err) : -1);
            return;
        }

        auto meta = std::make_shared<QuantumMetadata>();
        const void* meta_ptr = nullptr;
        size_t meta_size = 0;
        dispatch_data_t meta_map = dispatch_data_create_map(meta_content, &meta_ptr, &meta_size);
        (void)meta_map;
        memcpy(meta.get(), meta_ptr, sizeof(QuantumMetadata));

        size_t payload_bytes = meta->total_size * dtype_size(meta->dtype_code);
        printf("[QuantumReceiver] Matrix %u/%u metadata: dims=%u dtype=%d total_size=%zu payload_bytes=%zu\n",
               index, header->count, meta->dims, (int)meta->dtype_code, meta->total_size, payload_bytes);

        // Then its raw payload, as its own independent, exactly-sized read
        // (matches the original single-matrix path's approach of never
        // splicing multiple logical pieces out of one read).
        nw_connection_receive(this->connection, payload_bytes, payload_bytes,
        ^(dispatch_data_t payload_content, nw_content_context_t pctx, bool pcomp, nw_error_t perr) {

            if (perr || payload_content == nullptr) {
                printf("[QuantumReceiver] Payload read failed for matrix %u/%u (%zu bytes expected, error domain=%d code=%d) - read loop stops here!\n",
                       index, header->count, payload_bytes, perr ? nw_error_get_error_domain(perr) : -1, perr ? nw_error_get_error_code(perr) : -1);
                return;
            }

            const void* payload_ptr = nullptr;
            size_t p_size = 0;
            // ARC MUST retain this object, or payload_ptr will be instantly deallocated!
            dispatch_data_t payload_map = dispatch_data_create_map(payload_content, &payload_ptr, &p_size);
            (void)payload_map;

            metas->push_back(*meta);
            buffers->emplace_back(static_cast<const uint8_t*>(payload_ptr),
                                   static_cast<const uint8_t*>(payload_ptr) + payload_bytes);

            this->read_matrix_at_index(header, metas, buffers, index + 1);
        });
    });
}

void QuantumReceiver::finish_frame(QuantumFrameHeader header,
                                   std::vector<QuantumMetadata> metas,
                                   std::vector<std::vector<uint8_t>> buffers) {
    if (header.aux_size == 0) {
        printf("[QuantumReceiver] Frame %u complete (%u matrices, no aux)\n", header.frame_id, header.count);
        std::lock_guard<std::mutex> lock(this->buffer_mutex);
        this->latest_metas = std::move(metas);
        this->staging_buffers = std::move(buffers);
        this->has_received_first_frame = true;
        this->background_receive_loop();
        return;
    }

    // Read the aux data (camera intrinsics etc.) as its own independent,
    // exactly-sized read, shared across every matrix in this frame.
    nw_connection_receive(this->connection, header.aux_size, header.aux_size,
    ^(dispatch_data_t aux_content, nw_content_context_t actx, bool acomp, nw_error_t aerr) {

        if (aerr || aux_content == nullptr) {
            printf("[QuantumReceiver] Aux read failed (%u bytes expected, error domain=%d code=%d) - read loop stops here!\n",
                   header.aux_size, aerr ? nw_error_get_error_domain(aerr) : -1, aerr ? nw_error_get_error_code(aerr) : -1);
            return;
        }

        const void* aux_ptr = nullptr;
        size_t a_size = 0;
        dispatch_data_t aux_map = dispatch_data_create_map(aux_content, &aux_ptr, &a_size);
        (void)aux_map;

        // Thread-safe handoff to the Staging Buffers
        {
            std::lock_guard<std::mutex> lock(this->buffer_mutex);
            this->latest_metas = metas;
            this->staging_buffers = buffers;

            if (this->aux_staging_buffer.size() != header.aux_size) {
                this->aux_staging_buffer.resize(header.aux_size);
            }
            memcpy(this->aux_staging_buffer.data(), aux_ptr, header.aux_size);

            this->has_received_first_frame = true;
        }

        printf("[QuantumReceiver] Frame %u complete (%u matrices + %u aux bytes)\n", header.frame_id, header.count, header.aux_size);

        // Immediately listen for the next frame
        this->background_receive_loop();
    });
}

bool QuantumReceiver::pull_latest_data(void* out_aux_buffer, uint32_t max_aux_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex);

    if (!has_received_first_frame) return false;

    // First time we pull, we initialize one DAG leaf matrix per matrix in
    // the stream (a plain single-matrix stream just gives you one).
    if (!is_allocated) {

        for (const QuantumMetadata& meta : latest_metas) {
            // Construct using your engine's leaf node constructor:
            matrix* leaf = new matrix(meta.dims, meta.total_size, meta.dtype_code);

            // Restore the array dimensions and strides so the DAG knows how to read the buffer
            for (uint32_t i = 0; i < meta.dims; ++i) {
                leaf->shape()[i] = meta.shape[i];
                leaf->strides()[i] = meta.stride[i];
            }

            dag_leaf_matrices.push_back(leaf);
        }

        dag_leaf_matrix = dag_leaf_matrices.empty() ? nullptr : dag_leaf_matrices[0];
        is_allocated = true;
        printf("[QuantumReceiver] Pre-allocated %zu DAG leaf node(s)\n", dag_leaf_matrices.size());
    }

    for (size_t i = 0; i < dag_leaf_matrices.size() && i < staging_buffers.size(); ++i) {
        matrix* leaf = dag_leaf_matrices[i];
        if (leaf && leaf->buffer) {
            size_t payload_bytes = latest_metas[i].total_size * dtype_size(latest_metas[i].dtype_code);
            // Instant memcpy into the DAG's unified memory buffer!
            memcpy(leaf->buffer, staging_buffers[i].data(), payload_bytes);
        }
    }

    // Output the auxiliary data if requested
    if (out_aux_buffer != nullptr && aux_staging_buffer.size() > 0) {
        size_t copy_size = std::min((size_t)max_aux_size, aux_staging_buffer.size());
        memcpy(out_aux_buffer, aux_staging_buffer.data(), copy_size);

        if ((size_t)max_aux_size != aux_staging_buffer.size()) {
            printf("[QuantumReceiver] Warning: Aux size mismatch (Mac asked for %u, iPhone sent %zu). Struct padding difference?\n",
                   max_aux_size, aux_staging_buffer.size());
        }
    }
    
    return true;
}

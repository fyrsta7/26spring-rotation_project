ExpansionWorker* expansion_worker_alloc(FuriHalSerialId serial_id) {
    ExpansionWorker* instance = malloc(sizeof(ExpansionWorker));

    instance->thread = furi_thread_alloc_ex(
        TAG "Worker", EXPANSION_WORKER_STACK_SZIE, expansion_worker, instance);
    instance->rx_buf = furi_stream_buffer_alloc(EXPANSION_WORKER_BUFFER_SIZE, 1);
    instance->serial_id = serial_id;

    return instance;
}

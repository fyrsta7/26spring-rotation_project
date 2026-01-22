
    /* Wait for read operation to complete if pending. */
    while (c->io_read_state == CLIENT_PENDING_IO) {
        atomic_thread_fence(memory_order_acquire);
    }

    /* Wait for write operation to complete if pending. */
    while (c->io_write_state == CLIENT_PENDING_IO) {
        atomic_thread_fence(memory_order_acquire);
    }

    /* Final memory barrier to ensure all changes are visible */
    atomic_thread_fence(memory_order_acquire);
}

/** Adjusts the number of active I/O threads based on the current event load.
 * If increase_only is non-zero, only allows increasing the number of threads.*/
void adjustIOThreadsByEventLoad(int numevents, int increase_only) {
    if (server.io_threads_num == 1) return; /* All I/O is being done by the main thread. */
    debugServerAssertWithInfo(NULL, NULL, server.io_threads_num > 1);
    /* When events_per_io_thread is set to 0, we offload all events to the IO threads.
     * This is used mainly for testing purposes. */
    int target_threads = server.events_per_io_thread == 0 ? (numevents + 1) : numevents / server.events_per_io_thread;

    target_threads = max(1, min(target_threads, server.io_threads_num));

    if (target_threads == server.active_io_threads_num) return;

    if (target_threads < server.active_io_threads_num) {
        if (increase_only) return;

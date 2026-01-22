    while (server.repl_backlog->histlen > server.repl_backlog_size &&
           trimmed_blocks < max_blocks)
    {
        /* We never trim backlog to less than one block. */
        if (listLength(server.repl_buffer_blocks) <= 1) break;

        /* Replicas increment the refcount of the first replication buffer block
         * they refer to, in that case, we don't trim the backlog even if
         * backlog_histlen exceeds backlog_size. This implicitly makes backlog
         * bigger than our setting, but makes the master accept partial resync as
         * much as possible. So that backlog must be the last reference of
         * replication buffer blocks. */
        listNode *first = listFirst(server.repl_buffer_blocks);
        serverAssert(first == server.repl_backlog->ref_repl_buf_node);
        replBufBlock *fo = listNodeValue(first);
        if (fo->refcount != 1) break;

        /* We don't try trim backlog if backlog valid size will be lessen than
         * setting backlog size once we release the first repl buffer block. */
        if (server.repl_backlog->histlen - (long long)fo->size <=
            server.repl_backlog_size) break;

        /* Decr refcount and release the first block later. */
        fo->refcount--;
        trimmed_blocks++;
        server.repl_backlog->histlen -= fo->size;

        /* Go to use next replication buffer block node. */
        listNode *next = listNextNode(first);
        server.repl_backlog->ref_repl_buf_node = next;
        serverAssert(server.repl_backlog->ref_repl_buf_node != NULL);
        /* Incr reference count to keep the new head node. */
        ((replBufBlock *)listNodeValue(next))->refcount++;

        /* Remove the node in recorded blocks. */
        uint64_t encoded_offset = htonu64(fo->repl_offset);
        raxRemove(server.repl_backlog->blocks_index,
            (unsigned char*)&encoded_offset, sizeof(uint64_t), NULL);

        /* Delete the first node from global replication buffer. */
        serverAssert(fo->refcount == 0 && fo->used == fo->size);
        server.repl_buffer_mem -= (fo->size +
            sizeof(listNode) + sizeof(replBufBlock));
        listDelNode(server.repl_buffer_blocks, first);
    }

    /* Set the offset of the first byte we have in the backlog. */
    server.repl_backlog->offset = server.master_repl_offset -
                              server.repl_backlog->histlen + 1;
}

/* Free replication buffer blocks that are referenced by this client. */
void freeReplicaReferencedReplBuffer(client *replica) {
    if (replica->ref_repl_buf_node != NULL) {
        /* Decrease the start buffer node reference count. */
        replBufBlock *o = listNodeValue(replica->ref_repl_buf_node);
        serverAssert(o->refcount > 0);
        o->refcount--;
        incrementalTrimReplicationBacklog(REPL_BACKLOG_TRIM_BLOCKS_PER_CALL);
    }
    replica->ref_repl_buf_node = NULL;
    replica->ref_block_pos = 0;
}

/* Append bytes into the global replication buffer list, replication backlog and
 * all replica clients use replication buffers collectively, this function replace
 * 'addReply*', 'feedReplicationBacklog' for replicas and replication backlog,
 * First we add buffer into global replication buffer block list, and then
 * update replica / replication-backlog referenced node and block position. */
void feedReplicationBuffer(char *s, size_t len) {
    static long long repl_block_id = 0;

    if (server.repl_backlog == NULL) return;

    while(len > 0) {
        size_t start_pos = 0; /* The position of referenced block to start sending. */
        listNode *start_node = NULL; /* Replica/backlog starts referenced node. */
        int add_new_block = 0; /* Create new block if current block is total used. */
        listNode *ln = listLast(server.repl_buffer_blocks);
        replBufBlock *tail = ln ? listNodeValue(ln) : NULL;

        /* Append to tail string when possible. */
        if (tail && tail->size > tail->used) {
            start_node = listLast(server.repl_buffer_blocks);
            start_pos = tail->used;
            /* Copy the part we can fit into the tail, and leave the rest for a
             * new node */
            size_t avail = tail->size - tail->used;
            size_t copy = (avail >= len) ? len : avail;
            memcpy(tail->buf + tail->used, s, copy);

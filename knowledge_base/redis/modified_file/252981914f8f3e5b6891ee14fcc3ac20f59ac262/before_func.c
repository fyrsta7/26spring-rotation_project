    argv[5] = id;
    argv[6] = shared.time;
    argv[7] = createStringObjectFromLongLong(nack->delivery_time);
    argv[8] = shared.retrycount;
    argv[9] = createStringObjectFromLongLong(nack->delivery_count);
    argv[10] = shared.force;
    argv[11] = shared.justid;
    argv[12] = shared.lastid;
    argv[13] = createObjectFromStreamID(&group->last_id);

    /* We use propagate() because this code path is not always called from
     * the command execution context. Moreover this will just alter the
     * consumer group state, and we don't need MULTI/EXEC wrapping because
     * there is no message state cross-message atomicity required. */
    propagate(c->db->id,argv,14,PROPAGATE_AOF|PROPAGATE_REPL);
    decrRefCount(argv[3]);
    decrRefCount(argv[7]);
    decrRefCount(argv[9]);
    decrRefCount(argv[13]);
}

/* We need this when we want to propagate the new last-id of a consumer group
 * that was consumed by XREADGROUP with the NOACK option: in that case we can't
 * propagate the last ID just using the XCLAIM LASTID option, so we emit
 *
 *  XGROUP SETID <key> <groupname> <id>
 */
void streamPropagateGroupID(client *c, robj *key, streamCG *group, robj *groupname) {
    robj *argv[5];
    argv[0] = shared.xgroup;
    argv[1] = shared.setid;
    argv[2] = key;
    argv[3] = groupname;
    argv[4] = createObjectFromStreamID(&group->last_id);

    /* We use propagate() because this code path is not always called from
     * the command execution context. Moreover this will just alter the
     * consumer group state, and we don't need MULTI/EXEC wrapping because
     * there is no message state cross-message atomicity required. */
    propagate(c->db->id,argv,5,PROPAGATE_AOF|PROPAGATE_REPL);
    decrRefCount(argv[4]);
}

/* We need this when we want to propagate creation of consumer that was created
 * by XREADGROUP with the NOACK option. In that case, the only way to create
 * the consumer at the replica is by using XGROUP CREATECONSUMER (see issue #7140)
 *
 *  XGROUP CREATECONSUMER <key> <groupname> <consumername>
 */
void streamPropagateConsumerCreation(client *c, robj *key, robj *groupname, sds consumername) {
    robj *argv[5];
    argv[0] = shared.xgroup;
    argv[1] = shared.createconsumer;
    argv[2] = key;
    argv[3] = groupname;
    argv[4] = createObject(OBJ_STRING,sdsdup(consumername));

    /* We use propagate() because this code path is not always called from
     * the command execution context. Moreover this will just alter the
     * consumer group state, and we don't need MULTI/EXEC wrapping because
     * there is no message state cross-message atomicity required. */
    propagate(c->db->id,argv,5,PROPAGATE_AOF|PROPAGATE_REPL);
    decrRefCount(argv[4]);
}

/* Send the stream items in the specified range to the client 'c'. The range
 * the client will receive is between start and end inclusive, if 'count' is
 * non zero, no more than 'count' elements are sent.
 *
 * The 'end' pointer can be NULL to mean that we want all the elements from
 * 'start' till the end of the stream. If 'rev' is non zero, elements are
 * produced in reversed order from end to start.
 *

     * immediately after the start event if the flush is asynchronous. */
    moduleFireServerEvent(REDISMODULE_EVENT_FLUSHDB,
                          REDISMODULE_SUBEVENT_FLUSHDB_END,
                          &fi);

    return removed;
}

/* Initialize temporary db on replica for use during diskless replication. */
redisDb *initTempDb(void) {
    redisDb *tempDb = zcalloc(sizeof(redisDb)*server.dbnum);
    for (int i=0; i<server.dbnum; i++) {
        tempDb[i].dict_count = (server.cluster_enabled) ? CLUSTER_SLOTS : 1;
        tempDb[i].dict = dictCreateMultiple(&dbDictType, tempDb[i].dict_count);
        tempDb[i].expires = dictCreateMultiple(&dbExpiresDictType, tempDb[i].dict_count);
        for (dbKeyType subdict = DB_MAIN; subdict <= DB_EXPIRES; subdict++) {
            tempDb[i].sub_dict[subdict].slot_size_index = server.cluster_enabled ? zcalloc(sizeof(unsigned long long) * (CLUSTER_SLOTS + 1)) : NULL;
        }
    }

    return tempDb;
}

/* Discard tempDb, this can be slow (similar to FLUSHALL), but it's always async. */
void discardTempDb(redisDb *tempDb, void(callback)(dict*)) {
    int async = 1;

    /* Release temp DBs. */
    emptyDbStructure(tempDb, -1, async, callback);
    for (int i=0; i<server.dbnum; i++) {
        for (int j=0; j<tempDb[i].dict_count; j++) {
            dictRelease(tempDb[i].dict[j]);
            dictRelease(tempDb[i].expires[j]);
        }
        zfree(tempDb[i].dict);
        zfree(tempDb[i].expires);
        if (server.cluster_enabled) {

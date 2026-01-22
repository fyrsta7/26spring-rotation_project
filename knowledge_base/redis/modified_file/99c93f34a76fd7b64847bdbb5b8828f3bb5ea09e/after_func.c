{
    sds key;
    sentinelRedisInstance *slave;
    char buf[REDIS_PEER_ID_LEN];

    redisAssert(ri->flags & SRI_MASTER);
    anetFormatAddr(buf,sizeof(buf),ip,port);
    key = sdsnew(buf);
    slave = dictFetchValue(ri->slaves,key);
    sdsfree(key);
    return slave;
}

/* Return the name of the type of the instance as a string. */
const char *sentinelRedisInstanceTypeStr(sentinelRedisInstance *ri) {
    if (ri->flags & SRI_MASTER) return "master";
    else if (ri->flags & SRI_SLAVE) return "slave";
    else if (ri->flags & SRI_SENTINEL) return "sentinel";
    else return "unknown";
}

/* This function removes all the instances found in the dictionary of
 * sentinels in the specified 'master', having either:
 *
 * 1) The same ip/port as specified.
 * 2) The same runid.
 *
 * "1" and "2" don't need to verify at the same time, just one is enough.
 * If "runid" is NULL it is not checked.
 * Similarly if "ip" is NULL it is not checked.
 *
 * This function is useful because every time we add a new Sentinel into
 * a master's Sentinels dictionary, we want to be very sure about not
 * having duplicated instances for any reason. This is important because
 * other sentinels are needed to reach ODOWN quorum, and later to get
 * voted for a given configuration epoch in order to perform the failover.
 *
 * The function returns the number of Sentinels removed. */
int removeMatchingSentinelsFromMaster(sentinelRedisInstance *master, char *ip, int port, char *runid) {
    dictIterator *di;
    dictEntry *de;
    int removed = 0;

    di = dictGetSafeIterator(master->sentinels);
    while((de = dictNext(di)) != NULL) {
        sentinelRedisInstance *ri = dictGetVal(de);

        if ((ri->runid && runid && strcmp(ri->runid,runid) == 0) ||
            (ip && strcmp(ri->addr->ip,ip) == 0 && port == ri->addr->port))
        {
            dictDelete(master->sentinels,ri->name);
            removed++;

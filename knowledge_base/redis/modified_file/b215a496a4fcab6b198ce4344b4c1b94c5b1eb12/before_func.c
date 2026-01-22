    msetGenericCommand(c,1);
}

void incrDecrCommand(redisClient *c, long long incr) {
    long long value;
    robj *o;

    o = lookupKeyWrite(c->db,c->argv[1]);
    if (o != NULL && checkType(c,o,REDIS_STRING)) return;
    if (getLongLongFromObjectOrReply(c,o,&value,NULL) != REDIS_OK) return;

    value += incr;
    o = createStringObjectFromLongLong(value);
    dbReplace(c->db,c->argv[1],o);
    touchWatchedKey(c->db,c->argv[1]);
    server.dirty++;
    addReply(c,shared.colon);

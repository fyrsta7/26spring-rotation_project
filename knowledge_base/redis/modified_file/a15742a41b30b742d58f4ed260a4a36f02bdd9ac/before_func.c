    msetGenericCommand(c,1);
}

void incrDecrCommand(redisClient *c, long long incr) {
    long long value;
    robj *o;

    o = lookupKeyWrite(c->db,c->argv[1]);
    if (o != NULL && checkType(c,o,REDIS_STRING)) return;

    /* Fast path if the object is integer encoded and is not shared. */
    if (o && o->refcount == 1 && o->encoding == REDIS_ENCODING_INT) {
        long long newval = ((long)o->ptr) + incr;

        if (newval >= LONG_MIN && newval <= LONG_MAX) {
            o->ptr = (void*) (long) newval;
            touchWatchedKey(c->db,c->argv[1]);
            server.dirty++;
            addReplyLongLong(c,newval);
            return;
        }
        /* ... else take the usual safe path */
    }

    /* Otherwise we create a new object and replace the old one. */
    if (getLongLongFromObjectOrReply(c,o,&value,NULL) != REDIS_OK) return;
    value += incr;
    o = createStringObjectFromLongLong(value);
    dbReplace(c->db,c->argv[1],o);
    touchWatchedKey(c->db,c->argv[1]);
    server.dirty++;
    addReply(c,shared.colon);

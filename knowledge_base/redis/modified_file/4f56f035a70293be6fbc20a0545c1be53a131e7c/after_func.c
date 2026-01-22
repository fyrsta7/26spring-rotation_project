     * is only entered if the length of the string is greater than
     * REDIS_ENCODING_EMBSTR_SIZE_LIMIT. */
    if (o->encoding == REDIS_ENCODING_RAW &&
        sdsavail(s) > len/10)
    {
        o->ptr = sdsRemoveFreeSpace(o->ptr);
    }


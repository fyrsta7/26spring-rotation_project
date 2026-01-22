    return REDIS_OK;
}

/* Count number of bits set in the binary array pointed by 's' and long
 * 'count' bytes. The implementation of this function is required to
 * work with a input string length up to 512 MB. */
long popcount(void *s, long count) {
    long bits = 0;
    unsigned char *p = s;

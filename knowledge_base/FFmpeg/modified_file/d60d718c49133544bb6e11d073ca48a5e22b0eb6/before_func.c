    uint8_t *dst = out;
    uint8_t *end = out + out_size;

    v = 0;
    for (i = 0; ; i++) {
        unsigned int index= in[i]-43;
        if (index>=FF_ARRAY_ELEMS(map2) || map2[index] == 0xff)
            return in[i] && in[i] != '=' ? -1 : dst - out;
        v = (v << 6) + map2[index];
        if (i & 3) {
            if (dst < end) {
                *dst++ = v >> (6 - 2 * (i & 3));
            }
        }
    }

    av_assert1(0);
    return 0;
}

/*****************************************************************************
* b64_encode: Stolen from VLC's http.c.
* Simplified by Michael.
* Fixed edge cases and made it work from data (vs. strings) by Ryan.
*****************************************************************************/

char *av_base64_encode(char *out, int out_size, const uint8_t *in, int in_size)
{
    static const char b64[] =

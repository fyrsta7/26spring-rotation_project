    buf->length = length;
}

static inline int append(struct sd_filter *sd, struct buffer *buf, char c)
{
    if (buf->pos >= 0 && buf->pos < buf->length) {
        buf->string[buf->pos++] = c;
    } else {
        // ensure that terminating \0 is always written
        if (c == '\0')
            buf->string[buf->length - 1] = c;
    }
    return c;
}

static int get_char_bytes(char *str)
{
    // In case the first character is non-ASCII.

 * Release a string buffer and the memory it used. You should not use the
 * string buffer after using this function, unless you initialize it again.
 */
extern void strbuf_release(struct strbuf *);

/**
 * Detach the string from the strbuf and returns it; you now own the

static inline void strbuf_addch(struct strbuf *sb, int c)
{
	if (!strbuf_avail(sb))
		strbuf_grow(sb, 1);
	sb->buf[sb->len++] = c;
	sb->buf[sb->len] = '\0';
}
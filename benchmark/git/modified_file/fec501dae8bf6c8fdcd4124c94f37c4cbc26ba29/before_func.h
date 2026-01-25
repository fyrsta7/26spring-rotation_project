static inline void strbuf_addch(struct strbuf *sb, int c)
{
	strbuf_grow(sb, 1);
	sb->buf[sb->len++] = c;
	sb->buf[sb->len] = '\0';
}
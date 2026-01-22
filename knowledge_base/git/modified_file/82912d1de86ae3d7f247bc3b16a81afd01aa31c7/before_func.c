		strbuf_reset(sb);
	return -1;
}

int strbuf_getwholeline(struct strbuf *sb, FILE *fp, int term)
{
	int ch;

	if (feof(fp))
		return EOF;

	strbuf_reset(sb);
	while ((ch = getc(fp)) != EOF) {
		strbuf_grow(sb, 1);
		sb->buf[sb->len++] = ch;
		if (ch == term)
			break;
	}
	if (ch == EOF && sb->len == 0)
		return EOF;

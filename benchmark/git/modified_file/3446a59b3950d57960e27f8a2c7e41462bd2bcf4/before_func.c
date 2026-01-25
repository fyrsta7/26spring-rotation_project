int strbuf_getwholeline(struct strbuf *sb, FILE *fp, int term)
{
	int ch;

	if (feof(fp))
		return EOF;

	strbuf_reset(sb);
	while ((ch = fgetc(fp)) != EOF) {
		strbuf_grow(sb, 1);
		sb->buf[sb->len++] = ch;
		if (ch == term)
			break;
	}
	if (ch == EOF && sb->len == 0)
		return EOF;

	sb->buf[sb->len] = '\0';
	return 0;
}
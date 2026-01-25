static void trace_encoding(const char *context, const char *path,
			   const char *encoding, const char *buf, size_t len)
{
	static struct trace_key coe = TRACE_KEY_INIT(WORKING_TREE_ENCODING);
	struct strbuf trace = STRBUF_INIT;
	int i;

	if (!trace_want(&coe))
		return;

	strbuf_addf(&trace, "%s (%s, considered %s):\n", context, path, encoding);
	for (i = 0; i < len && buf; ++i) {
		strbuf_addf(
			&trace, "| \033[2m%2i:\033[0m %2x \033[2m%c\033[0m%c",
			i,
			(unsigned char) buf[i],
			(buf[i] > 32 && buf[i] < 127 ? buf[i] : ' '),
			((i+1) % 8 && (i+1) < len ? ' ' : '\n')
		);
	}
	strbuf_addchars(&trace, '\n', 1);

	trace_strbuf(&coe, &trace);
	strbuf_release(&trace);
}
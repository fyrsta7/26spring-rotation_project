	}
	free(sbs);
}

int strbuf_cmp(const struct strbuf *a, const struct strbuf *b)
{
	int cmp;
	if (a->len < b->len) {
		cmp = memcmp(a->buf, b->buf, a->len);
		return cmp ? cmp : -1;
	} else {

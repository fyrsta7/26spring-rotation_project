struct option *parse_options_concat(struct option *a, struct option *b)
{
	struct option *ret;
	size_t i, a_len = 0, b_len = 0;

	for (i = 0; a[i].type != OPTION_END; i++)
		a_len++;
	for (i = 0; b[i].type != OPTION_END; i++)
		b_len++;

	ALLOC_ARRAY(ret, st_add3(a_len, b_len, 1));
	COPY_ARRAY(ret, a, a_len);
	COPY_ARRAY(ret + a_len, b, b_len + 1); /* + 1 for final OPTION_END */

	return ret;
}
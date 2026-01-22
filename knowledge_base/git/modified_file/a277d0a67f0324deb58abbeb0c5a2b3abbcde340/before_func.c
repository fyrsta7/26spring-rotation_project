struct option *parse_options_concat(struct option *a, struct option *b)
{
	struct option *ret;
	size_t i, a_len = 0, b_len = 0;

	for (i = 0; a[i].type != OPTION_END; i++)
		a_len++;
	for (i = 0; b[i].type != OPTION_END; i++)
		b_len++;

	ALLOC_ARRAY(ret, st_add3(a_len, b_len, 1));
	for (i = 0; i < a_len; i++)
		ret[i] = a[i];
	for (i = 0; i < b_len; i++)
		ret[a_len + i] = b[i];
	ret[a_len + b_len] = b[b_len]; /* final OPTION_END */

	return ret;
}

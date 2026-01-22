
// XXX: wtf must deprecate
R_API void *r_str_free(void *ptr) {
	free (ptr);
	return NULL;
}

R_API char* r_str_replace(char *str, const char *key, const char *val, int g) {
	int off, i, klen, vlen, slen;
	char *newstr, *scnd, *p = str;

	if (!str || !key || !val) {
		return NULL;
	}
	klen = strlen (key);
	vlen = strlen (val);
	if (klen == vlen && !strcmp (key, val)) {
		return str;
	}
	slen = strlen (str);
	for (i = 0; i < slen; ) {
		p = (char *)r_mem_mem (
			(const ut8*)str + i, slen - i,
			(const ut8*)key, klen);
		if (!p) {
			break;
		}
		off = (int)(size_t)(p - str);
		scnd = strdup (p + klen);
		slen += vlen - klen;
		// HACK: this 32 avoids overwrites wtf
		newstr = realloc (str, slen + klen + 1);
		if (!newstr) {
			eprintf ("realloc fail\n");
			free (str);
			free (scnd);
			str = NULL;
			break;
		}
		str = newstr;
		p = str + off;
		memcpy (p, val, vlen);
		memcpy (p + vlen, scnd, strlen (scnd) + 1);
		i = off + vlen;

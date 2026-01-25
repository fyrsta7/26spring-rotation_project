static unsigned int contains(struct diff_filespec *one,
			     const char *needle, unsigned long len,
			     regex_t *regexp)
{
	unsigned int cnt;
	unsigned long sz;
	const char *data;
	if (diff_populate_filespec(one, 0))
		return 0;
	if (!len)
		return 0;

	sz = one->size;
	data = one->data;
	cnt = 0;

	if (regexp) {
		regmatch_t regmatch;
		int flags = 0;

		while (*data && !regexec(regexp, data, 1, &regmatch, flags)) {
			flags |= REG_NOTBOL;
			data += regmatch.rm_so;
			if (*data) data++;
			cnt++;
		}

	} else { /* Classic exact string match */
		while (sz) {
			const char *found = memmem(data, sz, needle, len);
			if (!found)
				break;
			sz -= found - data + len;
			data = found + len;
			cnt++;
		}
	}
	diff_free_filespec_data(one);
	return cnt;
}
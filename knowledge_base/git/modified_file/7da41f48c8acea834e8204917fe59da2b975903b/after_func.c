}

static int is_exact_match(struct diff_filespec *src,
			  struct diff_filespec *dst,
			  int contents_too)
{
	if (src->sha1_valid && dst->sha1_valid &&
	    !hashcmp(src->sha1, dst->sha1))
		return 1;
	if (!contents_too)
		return 0;
	if (diff_populate_filespec(src, 1) || diff_populate_filespec(dst, 1))
		return 0;
	if (src->size != dst->size)
		return 0;
	if (src->sha1_valid && dst->sha1_valid)
	    return !hashcmp(src->sha1, dst->sha1);
	if (diff_populate_filespec(src, 0) || diff_populate_filespec(dst, 0))
		return 0;
	if (src->size == dst->size &&
	    !memcmp(src->data, dst->data, src->size))
		return 1;

{
	if (sane_istest(x, GIT_ALPHA))
		x = (x & ~0x20) | high;
	return x;
}

static inline int prefixcmp(const char *str, const char *prefix)
{

static int same_name(const struct cache_entry *ce, const char *name, int namelen, int icase)
{
	int len = ce_namelen(ce);

	/*
	 * Always do exact compare, even if we want a case-ignoring comparison;
	 * we do the quick exact one first, because it will be the common case.
	 */
	if (len == namelen && !memcmp(name, ce->name, len))
		return 1;

	if (!icase)
		return 0;

	return slow_same_name(name, namelen, ce->name, len);
}
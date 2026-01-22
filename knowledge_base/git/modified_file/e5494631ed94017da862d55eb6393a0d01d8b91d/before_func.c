/*
 * Do we have another file that has the beginning components being a
 * proper superset of the name we're trying to add?
 */
static int has_file_name(struct index_state *istate,
			 const struct cache_entry *ce, int pos, int ok_to_replace)
{
	int retval = 0;
	int len = ce_namelen(ce);
	int stage = ce_stage(ce);
	const char *name = ce->name;

	while (pos < istate->cache_nr) {
		struct cache_entry *p = istate->cache[pos++];

		if (len >= ce_namelen(p))
			break;
		if (memcmp(name, p->name, len))
			break;
		if (ce_stage(p) != stage)
			continue;
		if (p->name[len] != '/')
			continue;
		if (p->ce_flags & CE_REMOVE)
			continue;
		retval = -1;
		if (!ok_to_replace)
			break;
		remove_index_entry_at(istate, --pos);
	}
	return retval;
}


/*
 * Like strcmp(), but also return the offset of the first change.
 * If strings are equal, return the length.
 */
int strcmp_offset(const char *s1, const char *s2, size_t *first_change)
{
	size_t k;

	if (!first_change)
		return strcmp(s1, s2);

	for (k = 0; s1[k] == s2[k]; k++)
		if (s1[k] == '\0')

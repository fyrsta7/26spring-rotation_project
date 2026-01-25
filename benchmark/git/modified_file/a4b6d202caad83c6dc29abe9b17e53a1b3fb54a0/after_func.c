static int verify_cache(struct cache_entry **cache,
			int entries, int flags)
{
	int i, funny;
	int silent = flags & WRITE_TREE_SILENT;

	/* Verify that the tree is merged */
	funny = 0;
	for (i = 0; i < entries; i++) {
		const struct cache_entry *ce = cache[i];
		if (ce_stage(ce)) {
			if (silent)
				return -1;
			if (10 < ++funny) {
				fprintf(stderr, "...\n");
				break;
			}
			fprintf(stderr, "%s: unmerged (%s)\n",
				ce->name, oid_to_hex(&ce->oid));
		}
	}
	if (funny)
		return -1;

	/* Also verify that the cache does not have path and path/file
	 * at the same time.  At this point we know the cache has only
	 * stage 0 entries.
	 */
	funny = 0;
	for (i = 0; i < entries - 1; i++) {
		/* path/file always comes after path because of the way
		 * the cache is sorted.  Also path can appear only once,
		 * which means conflicting one would immediately follow.
		 */
		const struct cache_entry *this_ce = cache[i];
		const struct cache_entry *next_ce = cache[i + 1];
		const char *this_name = this_ce->name;
		const char *next_name = next_ce->name;
		int this_len = ce_namelen(this_ce);
		if (this_len < ce_namelen(next_ce) &&
		    next_name[this_len] == '/' &&
		    strncmp(this_name, next_name, this_len) == 0) {
			if (10 < ++funny) {
				fprintf(stderr, "...\n");
				break;
			}
			fprintf(stderr, "You have both %s and %s\n",
				this_name, next_name);
		}
	}
	if (funny)
		return -1;
	return 0;
}
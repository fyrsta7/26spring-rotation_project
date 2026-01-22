}

/*
 * Handle a path that couldn't be lstat'ed. It's either:
 *  - missing file (ENOENT or ENOTDIR). That's ok if we're
 *    supposed to be removing it and the removal actually
 *    succeeds.
 *  - permission error. That's never ok.
 */
static int process_lstat_error(const char *path, int err)
{
	if (err == ENOENT || err == ENOTDIR)
		return remove_one_path(path);
	return error("lstat(\"%s\"): %s", path, strerror(errno));
}

static int add_one_path(struct cache_entry *old, const char *path, int len, struct stat *st)
{
	int option, size;
	struct cache_entry *ce;

	/* Was the old index entry already up-to-date? */
	if (old && !ce_stage(old) && !ce_match_stat(old, st, 0))
		return 0;

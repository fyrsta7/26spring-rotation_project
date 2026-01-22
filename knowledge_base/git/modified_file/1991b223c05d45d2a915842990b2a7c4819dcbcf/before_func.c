	if (len != expected_size) {
		free(target);
		return -1;
	}
	buffer = read_sha1_file(ce->sha1, type, &size);
	if (!buffer) {
		free(target);
		return -1;
	}
	if (size == expected_size)
		match = memcmp(buffer, target, size);
	free(buffer);
	free(target);
	return match;
}

/*
 * "refresh" does not calculate a new sha1 file or bring the
 * cache up-to-date for mode/content changes. But what it
 * _does_ do is to "re-match" the stat information of a file
 * with the cache, so that you can refresh the cache for a
 * file that hasn't been changed but where the stat entry is
 * out of date.
 *
 * For example, you'd want to do this after doing a "git-read-tree",
 * to link up the stat cache details with the proper files.
 */
static struct cache_entry *refresh_entry(struct cache_entry *ce)
{
	struct stat st;
	struct cache_entry *updated;
	int changed, size;

	if (lstat(ce->name, &st) < 0)
		return ERR_PTR(-errno);


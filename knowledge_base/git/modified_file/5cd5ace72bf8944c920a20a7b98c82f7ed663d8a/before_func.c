	unsigned char sha1[20];
	struct cache_entry *ce;

	if (sscanf(arg1, "%o", &mode) != 1)
		return -1;
	if (get_sha1_hex(arg2, sha1))
		return -1;
	if (!verify_path(arg3))
		return -1;

	len = strlen(arg3);
	size = cache_entry_size(len);
	ce = xmalloc(size);
	memset(ce, 0, size);

	memcpy(ce->sha1, sha1, 20);
	memcpy(ce->name, arg3, len);
	ce->ce_flags = htons(len);
	ce->ce_mode = create_ce_mode(mode);
	option = allow_add ? ADD_CACHE_OK_TO_ADD : 0;
	option |= allow_replace ? ADD_CACHE_OK_TO_REPLACE : 0;
	return add_cache_entry(ce, option);
}

static struct cache_file cache_file;


static void update_one(const char *path, const char *prefix, int prefix_length)
{
	const char *p = prefix_path(prefix, prefix_length, path);
	if (!verify_path(p)) {
		fprintf(stderr, "Ignoring path %s\n", path);
		return;
	}
	if (force_remove) {
		if (remove_file_from_cache(p))
			die("git-update-index: unable to remove %s", path);
		return;
	}
	if (add_file_to_cache(p))
		die("Unable to process file %s", path);
}

int main(int argc, const char **argv)
{
	int i, newfd, entries, has_errors = 0, line_termination = '\n';
	int allow_options = 1;
	int read_from_stdin = 0;
	const char *prefix = setup_git_directory();
	int prefix_length = prefix ? strlen(prefix) : 0;

	newfd = hold_index_file_for_update(&cache_file, get_index_file());
	if (newfd < 0)
		die("unable to create new cachefile");

	entries = read_cache();
	if (entries < 0)
		die("cache corrupted");

	for (i = 1 ; i < argc; i++) {
		const char *path = argv[i];

		if (allow_options && *path == '-') {
			if (!strcmp(path, "--")) {
				allow_options = 0;
				continue;
			}
			if (!strcmp(path, "-q")) {
				quiet = 1;
				continue;
			}
			if (!strcmp(path, "--add")) {
				allow_add = 1;
				continue;
			}
			if (!strcmp(path, "--replace")) {
				allow_replace = 1;
				continue;
			}
			if (!strcmp(path, "--remove")) {
				allow_remove = 1;
				continue;
			}
			if (!strcmp(path, "--unmerged")) {
				allow_unmerged = 1;
				continue;
			}
			if (!strcmp(path, "--refresh")) {
				has_errors |= refresh_cache();
				continue;
			}
			if (!strcmp(path, "--cacheinfo")) {
				if (i+3 >= argc)
					die("git-update-index: --cacheinfo <mode> <sha1> <path>");
				if (add_cacheinfo(argv[i+1], argv[i+2], argv[i+3]))
					die("git-update-index: --cacheinfo cannot add %s", argv[i+3]);
				i += 3;
				continue;

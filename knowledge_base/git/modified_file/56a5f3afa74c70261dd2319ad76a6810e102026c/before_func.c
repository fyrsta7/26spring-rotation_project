		printf("%s", dirty);
	printf("\n");

	if (!last_one)
		clear_commit_marks(cmit, -1);
}

int cmd_describe(int argc, const char **argv, const char *prefix)
{
	int contains = 0;
	struct option options[] = {
		OPT_BOOLEAN(0, "contains",   &contains, "find the tag that comes after the commit"),
		OPT_BOOLEAN(0, "debug",      &debug, "debug search strategy on stderr"),
		OPT_BOOLEAN(0, "all",        &all, "use any ref in .git/refs"),
		OPT_BOOLEAN(0, "tags",       &tags, "use any tag in .git/refs/tags"),
		OPT_BOOLEAN(0, "long",       &longformat, "always use long format"),
		OPT__ABBREV(&abbrev),
		OPT_SET_INT(0, "exact-match", &max_candidates,
			    "only output exact matches", 0),
		OPT_INTEGER(0, "candidates", &max_candidates,
			    "consider <n> most recent tags (default: 10)"),
		OPT_STRING(0, "match",       &pattern, "pattern",
			   "only consider tags matching <pattern>"),
		OPT_BOOLEAN(0, "always",     &always,
			   "show abbreviated commit object as fallback"),
		{OPTION_STRING, 0, "dirty",  &dirty, "mark",
			   "append <mark> on dirty working tree (default: \"-dirty\")",
		 PARSE_OPT_OPTARG, NULL, (intptr_t) "-dirty"},
		OPT_END(),
	};

	argc = parse_options(argc, argv, prefix, options, describe_usage, 0);
	if (max_candidates < 0)
		max_candidates = 0;
	else if (max_candidates > MAX_TAGS)
		max_candidates = MAX_TAGS;

	save_commit_buffer = 0;

	if (longformat && abbrev == 0)
		die("--long is incompatible with --abbrev=0");

	if (contains) {
		const char **args = xmalloc((7 + argc) * sizeof(char *));
		int i = 0;
		args[i++] = "name-rev";
		args[i++] = "--name-only";
		args[i++] = "--no-undefined";
		if (always)
			args[i++] = "--always";
		if (!all) {
			args[i++] = "--tags";
			if (pattern) {
				char *s = xmalloc(strlen("--refs=refs/tags/") + strlen(pattern) + 1);
				sprintf(s, "--refs=refs/tags/%s", pattern);
				args[i++] = s;
			}
		}
		memcpy(args + i, argv, argc * sizeof(char *));
		args[i + argc] = NULL;
		return cmd_name_rev(i + argc, args, prefix);
	}

	for_each_ref(get_name, NULL);
	if (!found_names && !always)
		die("No names found, cannot describe anything.");

	if (argc == 0) {
		if (dirty && !cmd_diff_index(ARRAY_SIZE(diff_index_args) - 1, diff_index_args, prefix))
			dirty = NULL;
		describe("HEAD", 1);
	} else if (dirty) {
		die("--dirty is incompatible with committishes");

	if (*arg == '^') {
		local_flags = UNINTERESTING;
		arg++;
	}
	if (get_sha1(arg, sha1))
		return -1;
	if (!cant_be_filename)
		verify_non_filename(revs->prefix, arg);
	object = get_reference(revs, arg, sha1, flags ^ local_flags);
	add_pending_object(revs, object, arg);
	return 0;
}

static void add_grep(struct rev_info *revs, const char *ptn, enum grep_pat_token what)
{
	if (!revs->grep_filter) {
		struct grep_opt *opt = xcalloc(1, sizeof(*opt));
		opt->status_only = 1;
		opt->pattern_tail = &(opt->pattern_list);
		opt->regflags = REG_NEWLINE;
		revs->grep_filter = opt;
	}
	append_grep_pattern(revs->grep_filter, ptn,
			    "command line", 0, what);
}

static void add_header_grep(struct rev_info *revs, const char *field, const char *pattern)
{
	char *pat;
	const char *prefix;
	int patlen, fldlen;

	fldlen = strlen(field);
	patlen = strlen(pattern);
	pat = xmalloc(patlen + fldlen + 10);
	prefix = ".*";
	if (*pattern == '^') {
		prefix = "";
		pattern++;
	}
	sprintf(pat, "^%s %s%s", field, prefix, pattern);
	add_grep(revs, pat, GREP_PATTERN_HEAD);
}

static void add_message_grep(struct rev_info *revs, const char *pattern)
{
	add_grep(revs, pattern, GREP_PATTERN_BODY);
}

static void add_ignore_packed(struct rev_info *revs, const char *name)
{
	int num = ++revs->num_ignore_packed;

	revs->ignore_packed = xrealloc(revs->ignore_packed,
				       sizeof(const char **) * (num + 1));
	revs->ignore_packed[num-1] = name;
	revs->ignore_packed[num] = NULL;
}

/*
 * Parse revision information, filling in the "rev_info" structure,
 * and removing the used arguments from the argument list.
 *
 * Returns the number of arguments left that weren't recognized
 * (which are also moved to the head of the argument list)
 */
int setup_revisions(int argc, const char **argv, struct rev_info *revs, const char *def)
{
	int i, flags, seen_dashdash, show_merge;
	const char **unrecognized = argv + 1;
	int left = 1;

	/* First, search for "--" */
	seen_dashdash = 0;
	for (i = 1; i < argc; i++) {
		const char *arg = argv[i];
		if (strcmp(arg, "--"))
			continue;
		argv[i] = NULL;
		argc = i;
		revs->prune_data = get_pathspec(revs->prefix, argv + i + 1);
		seen_dashdash = 1;
		break;
	}

	flags = show_merge = 0;
	for (i = 1; i < argc; i++) {
		const char *arg = argv[i];
		if (*arg == '-') {
			int opts;
			if (!strncmp(arg, "--max-count=", 12)) {
				revs->max_count = atoi(arg + 12);
				continue;
			}
			/* accept -<digit>, like traditional "head" */
			if ((*arg == '-') && isdigit(arg[1])) {
				revs->max_count = atoi(arg + 1);
				continue;
			}
			if (!strcmp(arg, "-n")) {
				if (argc <= i + 1)
					die("-n requires an argument");
				revs->max_count = atoi(argv[++i]);
				continue;
			}
			if (!strncmp(arg,"-n",2)) {
				revs->max_count = atoi(arg + 2);
				continue;
			}
			if (!strncmp(arg, "--max-age=", 10)) {
				revs->max_age = atoi(arg + 10);
				continue;
			}
			if (!strncmp(arg, "--since=", 8)) {
				revs->max_age = approxidate(arg + 8);
				continue;
			}
			if (!strncmp(arg, "--after=", 8)) {
				revs->max_age = approxidate(arg + 8);
				continue;
			}
			if (!strncmp(arg, "--min-age=", 10)) {
				revs->min_age = atoi(arg + 10);
				continue;
			}
			if (!strncmp(arg, "--before=", 9)) {
				revs->min_age = approxidate(arg + 9);
				continue;
			}
			if (!strncmp(arg, "--until=", 8)) {
				revs->min_age = approxidate(arg + 8);
				continue;
			}
			if (!strcmp(arg, "--all")) {
				handle_all(revs, flags);
				continue;
			}
			if (!strcmp(arg, "--not")) {
				flags ^= UNINTERESTING;
				continue;
			}
			if (!strcmp(arg, "--default")) {
				if (++i >= argc)
					die("bad --default argument");
				def = argv[i];
				continue;
			}
			if (!strcmp(arg, "--merge")) {
				show_merge = 1;
				continue;
			}
			if (!strcmp(arg, "--topo-order")) {
				revs->topo_order = 1;
				continue;
			}
			if (!strcmp(arg, "--date-order")) {
				revs->lifo = 0;
				revs->topo_order = 1;
				continue;
			}
			if (!strcmp(arg, "--parents")) {
				revs->parents = 1;
				continue;
			}
			if (!strcmp(arg, "--dense")) {
				revs->dense = 1;
				continue;
			}
			if (!strcmp(arg, "--sparse")) {
				revs->dense = 0;
				continue;
			}
			if (!strcmp(arg, "--remove-empty")) {
				revs->remove_empty_trees = 1;
				continue;
			}
			if (!strcmp(arg, "--no-merges")) {
				revs->no_merges = 1;
				continue;
			}
			if (!strcmp(arg, "--boundary")) {
				revs->boundary = 1;
				continue;
			}
			if (!strcmp(arg, "--objects")) {
				revs->tag_objects = 1;
				revs->tree_objects = 1;
				revs->blob_objects = 1;
				continue;
			}
			if (!strcmp(arg, "--objects-edge")) {
				revs->tag_objects = 1;
				revs->tree_objects = 1;
				revs->blob_objects = 1;
				revs->edge_hint = 1;
				continue;
			}
			if (!strcmp(arg, "--unpacked")) {
				revs->unpacked = 1;
				free(revs->ignore_packed);
				revs->ignore_packed = NULL;
				revs->num_ignore_packed = 0;
				continue;
			}
			if (!strncmp(arg, "--unpacked=", 11)) {
				revs->unpacked = 1;
				add_ignore_packed(revs, arg+11);
				continue;
			}
			if (!strcmp(arg, "-r")) {
				revs->diff = 1;
				revs->diffopt.recursive = 1;
				continue;
			}
			if (!strcmp(arg, "-t")) {
				revs->diff = 1;
				revs->diffopt.recursive = 1;
				revs->diffopt.tree_in_recursive = 1;
				continue;
			}
			if (!strcmp(arg, "-m")) {
				revs->ignore_merges = 0;
				continue;
			}
			if (!strcmp(arg, "-c")) {
				revs->diff = 1;
				revs->dense_combined_merges = 0;
				revs->combine_merges = 1;
				continue;
			}
			if (!strcmp(arg, "--cc")) {
				revs->diff = 1;
				revs->dense_combined_merges = 1;
				revs->combine_merges = 1;
				continue;
			}
			if (!strcmp(arg, "-v")) {
				revs->verbose_header = 1;
				continue;
			}
			if (!strncmp(arg, "--pretty", 8)) {
				revs->verbose_header = 1;
				revs->commit_format = get_commit_format(arg+8);
				continue;
			}
			if (!strcmp(arg, "--root")) {
				revs->show_root_diff = 1;
				continue;
			}
			if (!strcmp(arg, "--no-commit-id")) {
				revs->no_commit_id = 1;
				continue;
			}
			if (!strcmp(arg, "--always")) {
				revs->always_show_header = 1;
				continue;
			}
			if (!strcmp(arg, "--no-abbrev")) {
				revs->abbrev = 0;
				continue;
			}
			if (!strcmp(arg, "--abbrev")) {
				revs->abbrev = DEFAULT_ABBREV;
				continue;
			}
			if (!strncmp(arg, "--abbrev=", 9)) {
				revs->abbrev = strtoul(arg + 9, NULL, 10);
				if (revs->abbrev < MINIMUM_ABBREV)
					revs->abbrev = MINIMUM_ABBREV;
				else if (revs->abbrev > 40)
					revs->abbrev = 40;
				continue;
			}
			if (!strcmp(arg, "--abbrev-commit")) {
				revs->abbrev_commit = 1;
				continue;
			}
			if (!strcmp(arg, "--full-diff")) {
				revs->diff = 1;
				revs->full_diff = 1;
				continue;
			}
			if (!strcmp(arg, "--full-history")) {
				revs->simplify_history = 0;
				continue;
			}
			if (!strcmp(arg, "--relative-date")) {
				revs->relative_date = 1;
				continue;
			}

			/*
			 * Grepping the commit log
			 */
			if (!strncmp(arg, "--author=", 9)) {
				add_header_grep(revs, "author", arg+9);
				continue;
			}
			if (!strncmp(arg, "--committer=", 12)) {

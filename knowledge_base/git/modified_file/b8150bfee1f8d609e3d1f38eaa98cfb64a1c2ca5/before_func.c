
	strbuf_release(&sb);
}

static int option_parse_exact_match(const struct option *opt, const char *arg,
				    int unset)
{
	int *val = opt->value;

	BUG_ON_OPT_ARG(arg);

	*val = unset ? DEFAULT_CANDIDATES : 0;
	return 0;
}

int cmd_describe(int argc,
		 const char **argv,
		 const char *prefix,
		 struct repository *repo UNUSED )
{
	int contains = 0;
	struct option options[] = {
		OPT_BOOL(0, "contains",   &contains, N_("find the tag that comes after the commit")),
		OPT_BOOL(0, "debug",      &debug, N_("debug search strategy on stderr")),
		OPT_BOOL(0, "all",        &all, N_("use any ref")),
		OPT_BOOL(0, "tags",       &tags, N_("use any tag, even unannotated")),
		OPT_BOOL(0, "long",       &longformat, N_("always use long format")),
		OPT_BOOL(0, "first-parent", &first_parent, N_("only follow first parent")),
		OPT__ABBREV(&abbrev),
		OPT_CALLBACK_F(0, "exact-match", &max_candidates, NULL,
			       N_("only output exact matches"),
			       PARSE_OPT_NOARG, option_parse_exact_match),
		OPT_INTEGER(0, "candidates", &max_candidates,
			    N_("consider <n> most recent tags (default: 10)")),
		OPT_STRING_LIST(0, "match", &patterns, N_("pattern"),
			   N_("only consider tags matching <pattern>")),
		OPT_STRING_LIST(0, "exclude", &exclude_patterns, N_("pattern"),
			   N_("do not consider tags matching <pattern>")),
		OPT_BOOL(0, "always",        &always,
			N_("show abbreviated commit object as fallback")),
		{OPTION_STRING, 0, "dirty",  &dirty, N_("mark"),
			N_("append <mark> on dirty working tree (default: \"-dirty\")"),
			PARSE_OPT_OPTARG, NULL, (intptr_t) "-dirty"},
		{OPTION_STRING, 0, "broken",  &broken, N_("mark"),
			N_("append <mark> on broken working tree (default: \"-broken\")"),
			PARSE_OPT_OPTARG, NULL, (intptr_t) "-broken"},
		OPT_END(),
	};

	git_config(git_default_config, NULL);
	argc = parse_options(argc, argv, prefix, options, describe_usage, 0);
	if (abbrev < 0)
		abbrev = DEFAULT_ABBREV;

	if (max_candidates < 0)
		max_candidates = 0;
	else if (max_candidates > MAX_TAGS)
		max_candidates = MAX_TAGS;

	save_commit_buffer = 0;

	if (longformat && abbrev == 0)
		die(_("options '%s' and '%s' cannot be used together"), "--long", "--abbrev=0");

	if (contains) {
		struct string_list_item *item;
		struct strvec args;
		const char **argv_copy;
		int ret;

		strvec_init(&args);
		strvec_pushl(&args, "name-rev",
			     "--peel-tag", "--name-only", "--no-undefined",
			     NULL);
		if (always)
			strvec_push(&args, "--always");
		if (!all) {
			strvec_push(&args, "--tags");
			for_each_string_list_item(item, &patterns)
				strvec_pushf(&args, "--refs=refs/tags/%s", item->string);
			for_each_string_list_item(item, &exclude_patterns)
				strvec_pushf(&args, "--exclude=refs/tags/%s", item->string);
		}
		if (argc)
			strvec_pushv(&args, argv);
		else
			strvec_push(&args, "HEAD");

		/*
		 * `cmd_name_rev()` modifies the array, so we'd leak its
		 * contained strings if we didn't do a copy here.
		 */
		ALLOC_ARRAY(argv_copy, args.nr + 1);
		for (size_t i = 0; i < args.nr; i++)
			argv_copy[i] = args.v[i];
		argv_copy[args.nr] = NULL;

		ret = cmd_name_rev(args.nr, argv_copy, prefix, the_repository);

		strvec_clear(&args);
		free(argv_copy);
		return ret;
	}

	hashmap_init(&names, commit_name_neq, NULL, 0);
	refs_for_each_rawref(get_main_ref_store(the_repository), get_name,
			     NULL);
	if (!hashmap_get_size(&names) && !always)
		die(_("No names found, cannot describe anything."));

	if (argc == 0) {
		if (broken) {
			struct child_process cp = CHILD_PROCESS_INIT;

			strvec_pushv(&cp.args, update_index_args);
			cp.git_cmd = 1;
			cp.no_stdin = 1;
			cp.no_stdout = 1;
			run_command(&cp);

			child_process_init(&cp);
			strvec_pushv(&cp.args, diff_index_args);
			cp.git_cmd = 1;
			cp.no_stdin = 1;
			cp.no_stdout = 1;

			if (!dirty)
				dirty = "-dirty";

			switch (run_command(&cp)) {
			case 0:
				suffix = NULL;
				break;
			case 1:
				suffix = dirty;
				break;
			default:
				/* diff-index aborted abnormally */
				suffix = broken;
			}
		} else if (dirty) {
			struct lock_file index_lock = LOCK_INIT;
			struct rev_info revs;
			int fd;

			setup_work_tree();
			prepare_repo_settings(the_repository);
			the_repository->settings.command_requires_full_index = 0;
			repo_read_index(the_repository);
			refresh_index(the_repository->index, REFRESH_QUIET|REFRESH_UNMERGED,
				      NULL, NULL, NULL);
			fd = repo_hold_locked_index(the_repository,
						    &index_lock, 0);
			if (0 <= fd)
				repo_update_index_if_able(the_repository, &index_lock);

			repo_init_revisions(the_repository, &revs, prefix);

			if (setup_revisions(ARRAY_SIZE(diff_index_args) - 1,
					    diff_index_args, &revs, NULL) != 1)
				BUG("malformed internal diff-index command line");
			run_diff_index(&revs, 0);

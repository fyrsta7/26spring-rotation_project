	} else if (!strcmp(arg, "--full-diff")) {
		revs->diff = 1;
		revs->full_diff = 1;
	} else if (!strcmp(arg, "--full-history")) {
		revs->simplify_history = 0;
	} else if (!strcmp(arg, "--relative-date")) {
		revs->date_mode.type = DATE_RELATIVE;
		revs->date_mode_explicit = 1;
	} else if ((argcount = parse_long_opt("date", argv, &optarg))) {
		parse_date_format(optarg, &revs->date_mode);
		revs->date_mode_explicit = 1;
		return argcount;
	} else if (!strcmp(arg, "--log-size")) {
		revs->show_log_size = 1;
	}
	/*
	 * Grepping the commit log
	 */
	else if ((argcount = parse_long_opt("author", argv, &optarg))) {
		add_header_grep(revs, GREP_HEADER_AUTHOR, optarg);
		return argcount;
	} else if ((argcount = parse_long_opt("committer", argv, &optarg))) {
		add_header_grep(revs, GREP_HEADER_COMMITTER, optarg);
		return argcount;
	} else if ((argcount = parse_long_opt("grep-reflog", argv, &optarg))) {
		add_header_grep(revs, GREP_HEADER_REFLOG, optarg);
		return argcount;
	} else if ((argcount = parse_long_opt("grep", argv, &optarg))) {
		add_message_grep(revs, optarg);
		return argcount;
	} else if (!strcmp(arg, "--grep-debug")) {
		revs->grep_filter.debug = 1;
	} else if (!strcmp(arg, "--basic-regexp")) {
		revs->grep_filter.pattern_type_option = GREP_PATTERN_TYPE_BRE;
	} else if (!strcmp(arg, "--extended-regexp") || !strcmp(arg, "-E")) {
		revs->grep_filter.pattern_type_option = GREP_PATTERN_TYPE_ERE;
	} else if (!strcmp(arg, "--regexp-ignore-case") || !strcmp(arg, "-i")) {
		revs->grep_filter.ignore_case = 1;
		revs->diffopt.pickaxe_opts |= DIFF_PICKAXE_IGNORE_CASE;
	} else if (!strcmp(arg, "--fixed-strings") || !strcmp(arg, "-F")) {
		revs->grep_filter.pattern_type_option = GREP_PATTERN_TYPE_FIXED;
	} else if (!strcmp(arg, "--perl-regexp") || !strcmp(arg, "-P")) {
		revs->grep_filter.pattern_type_option = GREP_PATTERN_TYPE_PCRE;
	} else if (!strcmp(arg, "--all-match")) {
		revs->grep_filter.all_match = 1;
	} else if (!strcmp(arg, "--invert-grep")) {
		revs->invert_grep = 1;
	} else if ((argcount = parse_long_opt("encoding", argv, &optarg))) {
		if (strcmp(optarg, "none"))
			git_log_output_encoding = xstrdup(optarg);
		else
			git_log_output_encoding = "";
		return argcount;
	} else if (!strcmp(arg, "--reverse")) {
		revs->reverse ^= 1;
	} else if (!strcmp(arg, "--children")) {
		revs->children.name = "children";
		revs->limited = 1;
	} else if (!strcmp(arg, "--ignore-missing")) {
		revs->ignore_missing = 1;
	} else if (opt && opt->allow_exclude_promisor_objects &&
		   !strcmp(arg, "--exclude-promisor-objects")) {
		if (fetch_if_missing)
			BUG("exclude_promisor_objects can only be used when fetch_if_missing is 0");
		revs->exclude_promisor_objects = 1;
	} else {
		int opts = diff_opt_parse(&revs->diffopt, argv, argc, revs->prefix);
		if (!opts)
			unkv[(*unkc)++] = arg;
		return opts;
	}
	if (revs->graph && revs->track_linear)
		die("--show-linear-break and --graph are incompatible");

	return 1;
}

void parse_revision_opt(struct rev_info *revs, struct parse_opt_ctx_t *ctx,
			const struct option *options,
			const char * const usagestr[])
{
	int n = handle_revision_opt(revs, ctx->argc, ctx->argv,
				    &ctx->cpidx, ctx->out, NULL);
	if (n <= 0) {
		error("unknown option `%s'", ctx->argv[0]);
		usage_with_options(usagestr, options);
	}
	ctx->argv += n;
	ctx->argc -= n;
}

static int for_each_bisect_ref(struct ref_store *refs, each_ref_fn fn,
			       void *cb_data, const char *term)
{
	struct strbuf bisect_refs = STRBUF_INIT;
	int status;
	strbuf_addf(&bisect_refs, "refs/bisect/%s", term);
	status = refs_for_each_fullref_in(refs, bisect_refs.buf, fn, cb_data, 0);
	strbuf_release(&bisect_refs);
	return status;
}

static int for_each_bad_bisect_ref(struct ref_store *refs, each_ref_fn fn, void *cb_data)
{
	return for_each_bisect_ref(refs, fn, cb_data, term_bad);
}

static int for_each_good_bisect_ref(struct ref_store *refs, each_ref_fn fn, void *cb_data)
{
	return for_each_bisect_ref(refs, fn, cb_data, term_good);
}

static int handle_revision_pseudo_opt(const char *submodule,
				struct rev_info *revs,
				int argc, const char **argv, int *flags)
{
	const char *arg = argv[0];
	const char *optarg;
	struct ref_store *refs;
	int argcount;

	if (submodule) {
		/*
		 * We need some something like get_submodule_worktrees()
		 * before we can go through all worktrees of a submodule,
		 * .e.g with adding all HEADs from --all, which is not
		 * supported right now, so stick to single worktree.
		 */
		if (!revs->single_worktree)
			BUG("--single-worktree cannot be used together with submodule");
		refs = get_submodule_ref_store(submodule);
	} else
		refs = get_main_ref_store(revs->repo);

	/*
	 * NOTE!
	 *
	 * Commands like "git shortlog" will not accept the options below
	 * unless parse_revision_opt queues them (as opposed to erroring
	 * out).
	 *
	 * When implementing your new pseudo-option, remember to
	 * register it in the list at the top of handle_revision_opt.
	 */
	if (!strcmp(arg, "--all")) {
		handle_refs(refs, revs, *flags, refs_for_each_ref);
		handle_refs(refs, revs, *flags, refs_head_ref);
		if (!revs->single_worktree) {
			struct all_refs_cb cb;

			init_all_refs_cb(&cb, revs, *flags);
			other_head_refs(handle_one_ref, &cb);
		}
		clear_ref_exclusion(&revs->ref_excludes);
	} else if (!strcmp(arg, "--branches")) {
		handle_refs(refs, revs, *flags, refs_for_each_branch_ref);
		clear_ref_exclusion(&revs->ref_excludes);
	} else if (!strcmp(arg, "--bisect")) {
		read_bisect_terms(&term_bad, &term_good);
		handle_refs(refs, revs, *flags, for_each_bad_bisect_ref);
		handle_refs(refs, revs, *flags ^ (UNINTERESTING | BOTTOM),
			    for_each_good_bisect_ref);
		revs->bisect = 1;
	} else if (!strcmp(arg, "--tags")) {
		handle_refs(refs, revs, *flags, refs_for_each_tag_ref);
		clear_ref_exclusion(&revs->ref_excludes);
	} else if (!strcmp(arg, "--remotes")) {
		handle_refs(refs, revs, *flags, refs_for_each_remote_ref);
		clear_ref_exclusion(&revs->ref_excludes);
	} else if ((argcount = parse_long_opt("glob", argv, &optarg))) {
		struct all_refs_cb cb;
		init_all_refs_cb(&cb, revs, *flags);
		for_each_glob_ref(handle_one_ref, optarg, &cb);
		clear_ref_exclusion(&revs->ref_excludes);
		return argcount;
	} else if ((argcount = parse_long_opt("exclude", argv, &optarg))) {
		add_ref_exclusion(&revs->ref_excludes, optarg);
		return argcount;
	} else if (skip_prefix(arg, "--branches=", &optarg)) {
		struct all_refs_cb cb;
		init_all_refs_cb(&cb, revs, *flags);
		for_each_glob_ref_in(handle_one_ref, optarg, "refs/heads/", &cb);
		clear_ref_exclusion(&revs->ref_excludes);
	} else if (skip_prefix(arg, "--tags=", &optarg)) {
		struct all_refs_cb cb;
		init_all_refs_cb(&cb, revs, *flags);
		for_each_glob_ref_in(handle_one_ref, optarg, "refs/tags/", &cb);
		clear_ref_exclusion(&revs->ref_excludes);

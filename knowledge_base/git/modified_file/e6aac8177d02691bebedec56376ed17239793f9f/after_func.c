	const char *env;
	struct strbuf buf = STRBUF_INIT;

	if (!is_interactive(options))
		return;

	env = getenv(GIT_REFLOG_ACTION_ENVIRONMENT);
	if (env && strcmp("rebase", env))
		return; /* only override it if it is "rebase" */

	strbuf_addf(&buf, "rebase -i (%s)", options->action);
	setenv(GIT_REFLOG_ACTION_ENVIRONMENT, buf.buf, 1);
	strbuf_release(&buf);
}

int cmd_rebase(int argc, const char **argv, const char *prefix)
{
	struct rebase_options options = {
		.type = REBASE_UNSPECIFIED,
		.flags = REBASE_NO_QUIET,
		.git_am_opts = ARGV_ARRAY_INIT,
		.allow_rerere_autoupdate  = -1,
		.allow_empty_message = 1,
		.git_format_patch_opt = STRBUF_INIT,
	};
	const char *branch_name;
	int ret, flags, total_argc, in_progress = 0;
	int ok_to_skip_pre_rebase = 0;
	struct strbuf msg = STRBUF_INIT;
	struct strbuf revisions = STRBUF_INIT;
	struct strbuf buf = STRBUF_INIT;
	struct object_id merge_base;
	enum {
		NO_ACTION,
		ACTION_CONTINUE,
		ACTION_SKIP,
		ACTION_ABORT,
		ACTION_QUIT,
		ACTION_EDIT_TODO,
		ACTION_SHOW_CURRENT_PATCH,
	} action = NO_ACTION;
	const char *gpg_sign = NULL;
	struct string_list exec = STRING_LIST_INIT_NODUP;
	const char *rebase_merges = NULL;
	int fork_point = -1;
	struct string_list strategy_options = STRING_LIST_INIT_NODUP;
	struct object_id squash_onto;
	char *squash_onto_name = NULL;
	struct option builtin_rebase_options[] = {
		OPT_STRING(0, "onto", &options.onto_name,
			   N_("revision"),
			   N_("rebase onto given branch instead of upstream")),
		OPT_BOOL(0, "no-verify", &ok_to_skip_pre_rebase,
			 N_("allow pre-rebase hook to run")),
		OPT_NEGBIT('q', "quiet", &options.flags,
			   N_("be quiet. implies --no-stat"),
			   REBASE_NO_QUIET| REBASE_VERBOSE | REBASE_DIFFSTAT),
		OPT_BIT('v', "verbose", &options.flags,
			N_("display a diffstat of what changed upstream"),
			REBASE_NO_QUIET | REBASE_VERBOSE | REBASE_DIFFSTAT),
		{OPTION_NEGBIT, 'n', "no-stat", &options.flags, NULL,
			N_("do not show diffstat of what changed upstream"),
			PARSE_OPT_NOARG, NULL, REBASE_DIFFSTAT },
		OPT_BOOL(0, "signoff", &options.signoff,
			 N_("add a Signed-off-by: line to each commit")),
		OPT_PASSTHRU_ARGV(0, "ignore-whitespace", &options.git_am_opts,
				  NULL, N_("passed to 'git am'"),
				  PARSE_OPT_NOARG),
		OPT_PASSTHRU_ARGV(0, "committer-date-is-author-date",
				  &options.git_am_opts, NULL,
				  N_("passed to 'git am'"), PARSE_OPT_NOARG),
		OPT_PASSTHRU_ARGV(0, "ignore-date", &options.git_am_opts, NULL,
				  N_("passed to 'git am'"), PARSE_OPT_NOARG),
		OPT_PASSTHRU_ARGV('C', NULL, &options.git_am_opts, N_("n"),
				  N_("passed to 'git apply'"), 0),
		OPT_PASSTHRU_ARGV(0, "whitespace", &options.git_am_opts,
				  N_("action"), N_("passed to 'git apply'"), 0),
		OPT_BIT('f', "force-rebase", &options.flags,
			N_("cherry-pick all commits, even if unchanged"),
			REBASE_FORCE),
		OPT_BIT(0, "no-ff", &options.flags,
			N_("cherry-pick all commits, even if unchanged"),
			REBASE_FORCE),
		OPT_CMDMODE(0, "continue", &action, N_("continue"),
			    ACTION_CONTINUE),
		OPT_CMDMODE(0, "skip", &action,
			    N_("skip current patch and continue"), ACTION_SKIP),
		OPT_CMDMODE(0, "abort", &action,
			    N_("abort and check out the original branch"),
			    ACTION_ABORT),
		OPT_CMDMODE(0, "quit", &action,
			    N_("abort but keep HEAD where it is"), ACTION_QUIT),
		OPT_CMDMODE(0, "edit-todo", &action, N_("edit the todo list "
			    "during an interactive rebase"), ACTION_EDIT_TODO),
		OPT_CMDMODE(0, "show-current-patch", &action,
			    N_("show the patch file being applied or merged"),
			    ACTION_SHOW_CURRENT_PATCH),
		{ OPTION_CALLBACK, 'm', "merge", &options, NULL,
			N_("use merging strategies to rebase"),
			PARSE_OPT_NOARG | PARSE_OPT_NONEG,
			parse_opt_merge },
		{ OPTION_CALLBACK, 'i', "interactive", &options, NULL,
			N_("let the user edit the list of commits to rebase"),
			PARSE_OPT_NOARG | PARSE_OPT_NONEG,
			parse_opt_interactive },
		OPT_SET_INT('p', "preserve-merges", &options.type,
			    N_("try to recreate merges instead of ignoring "
			       "them"), REBASE_PRESERVE_MERGES),
		OPT_BOOL(0, "rerere-autoupdate",
			 &options.allow_rerere_autoupdate,
			 N_("allow rerere to update index with resolved "
			    "conflict")),
		OPT_BOOL('k', "keep-empty", &options.keep_empty,
			 N_("preserve empty commits during rebase")),
		OPT_BOOL(0, "autosquash", &options.autosquash,
			 N_("move commits that begin with "
			    "squash!/fixup! under -i")),
		{ OPTION_STRING, 'S', "gpg-sign", &gpg_sign, N_("key-id"),
			N_("GPG-sign commits"),
			PARSE_OPT_OPTARG, NULL, (intptr_t) "" },
		OPT_BOOL(0, "autostash", &options.autostash,
			 N_("automatically stash/stash pop before and after")),
		OPT_STRING_LIST('x', "exec", &exec, N_("exec"),
				N_("add exec lines after each commit of the "
				   "editable list")),
		OPT_BOOL(0, "allow-empty-message",
			 &options.allow_empty_message,
			 N_("allow rebasing commits with empty messages")),
		{OPTION_STRING, 'r', "rebase-merges", &rebase_merges,
			N_("mode"),
			N_("try to rebase merges instead of skipping them"),
			PARSE_OPT_OPTARG, NULL, (intptr_t)""},
		OPT_BOOL(0, "fork-point", &fork_point,
			 N_("use 'merge-base --fork-point' to refine upstream")),
		OPT_STRING('s', "strategy", &options.strategy,
			   N_("strategy"), N_("use the given merge strategy")),
		OPT_STRING_LIST('X', "strategy-option", &strategy_options,
				N_("option"),
				N_("pass the argument through to the merge "
				   "strategy")),
		OPT_BOOL(0, "root", &options.root,
			 N_("rebase all reachable commits up to the root(s)")),
		OPT_END(),
	};
	int i;

	/*
	 * NEEDSWORK: Once the builtin rebase has been tested enough
	 * and git-legacy-rebase.sh is retired to contrib/, this preamble
	 * can be removed.
	 */

	if (!use_builtin_rebase()) {
		const char *path = mkpath("%s/git-legacy-rebase",
					  git_exec_path());

		if (sane_execvp(path, (char **)argv) < 0)
			die_errno(_("could not exec %s"), path);
		else
			BUG("sane_execvp() returned???");
	}

	if (argc == 2 && !strcmp(argv[1], "-h"))
		usage_with_options(builtin_rebase_usage,
				   builtin_rebase_options);

	prefix = setup_git_directory();
	trace_repo_setup(prefix);
	setup_work_tree();

	git_config(rebase_config, &options);

	strbuf_reset(&buf);
	strbuf_addf(&buf, "%s/applying", apply_dir());
	if(file_exists(buf.buf))
		die(_("It looks like 'git am' is in progress. Cannot rebase."));

	if (is_directory(apply_dir())) {
		options.type = REBASE_AM;
		options.state_dir = apply_dir();
	} else if (is_directory(merge_dir())) {
		strbuf_reset(&buf);
		strbuf_addf(&buf, "%s/rewritten", merge_dir());
		if (is_directory(buf.buf)) {
			options.type = REBASE_PRESERVE_MERGES;
			options.flags |= REBASE_INTERACTIVE_EXPLICIT;
		} else {
			strbuf_reset(&buf);
			strbuf_addf(&buf, "%s/interactive", merge_dir());
			if(file_exists(buf.buf)) {
				options.type = REBASE_INTERACTIVE;
				options.flags |= REBASE_INTERACTIVE_EXPLICIT;
			} else
				options.type = REBASE_MERGE;
		}
		options.state_dir = merge_dir();
	}

	if (options.type != REBASE_UNSPECIFIED)
		in_progress = 1;

	total_argc = argc;
	argc = parse_options(argc, argv, prefix,
			     builtin_rebase_options,
			     builtin_rebase_usage, 0);

	if (action != NO_ACTION && total_argc != 2) {
		usage_with_options(builtin_rebase_usage,
				   builtin_rebase_options);
	}

	if (argc > 2)
		usage_with_options(builtin_rebase_usage,
				   builtin_rebase_options);

	if (action != NO_ACTION && !in_progress)
		die(_("No rebase in progress?"));
	setenv(GIT_REFLOG_ACTION_ENVIRONMENT, "rebase", 0);

	if (action == ACTION_EDIT_TODO && !is_interactive(&options))
		die(_("The --edit-todo action can only be used during "
		      "interactive rebase."));

	switch (action) {
	case ACTION_CONTINUE: {
		struct object_id head;
		struct lock_file lock_file = LOCK_INIT;
		int fd;

		options.action = "continue";
		set_reflog_action(&options);

		/* Sanity check */
		if (get_oid("HEAD", &head))
			die(_("Cannot read HEAD"));

		fd = hold_locked_index(&lock_file, 0);
		if (read_index(the_repository->index) < 0)
			die(_("could not read index"));
		refresh_index(the_repository->index, REFRESH_QUIET, NULL, NULL,
			      NULL);
		if (0 <= fd)
			update_index_if_able(the_repository->index,
					     &lock_file);
		rollback_lock_file(&lock_file);

		if (has_unstaged_changes(1)) {
			puts(_("You must edit all merge conflicts and then\n"
			       "mark them as resolved using git add"));
			exit(1);
		}
		if (read_basic_state(&options))
			exit(1);
		goto run_rebase;
	}
	case ACTION_SKIP: {
		struct string_list merge_rr = STRING_LIST_INIT_DUP;

		options.action = "skip";
		set_reflog_action(&options);

		rerere_clear(&merge_rr);
		string_list_clear(&merge_rr, 1);

		if (reset_head(NULL, "reset", NULL, RESET_HEAD_HARD,
			       NULL, NULL) < 0)
			die(_("could not discard worktree changes"));
		remove_branch_state();
		if (read_basic_state(&options))
			exit(1);
		goto run_rebase;
	}
	case ACTION_ABORT: {
		struct string_list merge_rr = STRING_LIST_INIT_DUP;
		options.action = "abort";
		set_reflog_action(&options);

		rerere_clear(&merge_rr);
		string_list_clear(&merge_rr, 1);

		if (read_basic_state(&options))
			exit(1);
		if (reset_head(&options.orig_head, "reset",
			       options.head_name, RESET_HEAD_HARD,
			       NULL, NULL) < 0)
			die(_("could not move back to %s"),
			    oid_to_hex(&options.orig_head));
		remove_branch_state();
		ret = finish_rebase(&options);
		goto cleanup;
	}
	case ACTION_QUIT: {
		strbuf_reset(&buf);
		strbuf_addstr(&buf, options.state_dir);
		ret = !!remove_dir_recursively(&buf, 0);
		if (ret)
			die(_("could not remove '%s'"), options.state_dir);
		goto cleanup;
	}
	case ACTION_EDIT_TODO:
		options.action = "edit-todo";
		options.dont_finish_rebase = 1;
		goto run_rebase;
	case ACTION_SHOW_CURRENT_PATCH:
		options.action = "show-current-patch";
		options.dont_finish_rebase = 1;
		goto run_rebase;
	case NO_ACTION:
		break;
	default:
		BUG("action: %d", action);
	}

	/* Make sure no rebase is in progress */
	if (in_progress) {
		const char *last_slash = strrchr(options.state_dir, '/');
		const char *state_dir_base =
			last_slash ? last_slash + 1 : options.state_dir;
		const char *cmd_live_rebase =
			"git rebase (--continue | --abort | --skip)";
		strbuf_reset(&buf);
		strbuf_addf(&buf, "rm -fr \"%s\"", options.state_dir);
		die(_("It seems that there is already a %s directory, and\n"
		      "I wonder if you are in the middle of another rebase.  "
		      "If that is the\n"
		      "case, please try\n\t%s\n"
		      "If that is not the case, please\n\t%s\n"
		      "and run me again.  I am stopping in case you still "
		      "have something\n"
		      "valuable there.\n"),
		    state_dir_base, cmd_live_rebase, buf.buf);
	}

	for (i = 0; i < options.git_am_opts.argc; i++) {
		const char *option = options.git_am_opts.argv[i], *p;
		if (!strcmp(option, "--committer-date-is-author-date") ||
		    !strcmp(option, "--ignore-date") ||
		    !strcmp(option, "--whitespace=fix") ||
		    !strcmp(option, "--whitespace=strip"))
			options.flags |= REBASE_FORCE;
		else if (skip_prefix(option, "-C", &p)) {
			while (*p)
				if (!isdigit(*(p++)))
					die(_("switch `C' expects a "
					      "numerical value"));
		} else if (skip_prefix(option, "--whitespace=", &p)) {
			if (*p && strcmp(p, "warn") && strcmp(p, "nowarn") &&
			    strcmp(p, "error") && strcmp(p, "error-all"))
				die("Invalid whitespace option: '%s'", p);
		}
	}

	if (!(options.flags & REBASE_NO_QUIET))
		argv_array_push(&options.git_am_opts, "-q");

	if (options.keep_empty)
		imply_interactive(&options, "--keep-empty");

	if (gpg_sign) {
		free(options.gpg_sign_opt);
		options.gpg_sign_opt = xstrfmt("-S%s", gpg_sign);
	}

	if (exec.nr) {
		int i;

		imply_interactive(&options, "--exec");

		strbuf_reset(&buf);
		for (i = 0; i < exec.nr; i++)
			strbuf_addf(&buf, "exec %s\n", exec.items[i].string);
		options.cmd = xstrdup(buf.buf);
	}

	if (rebase_merges) {
		if (!*rebase_merges)
			; /* default mode; do nothing */
		else if (!strcmp("rebase-cousins", rebase_merges))
			options.rebase_cousins = 1;
		else if (strcmp("no-rebase-cousins", rebase_merges))
			die(_("Unknown mode: %s"), rebase_merges);
		options.rebase_merges = 1;
		imply_interactive(&options, "--rebase-merges");
	}

	if (strategy_options.nr) {
		int i;

		if (!options.strategy)
			options.strategy = "recursive";

		strbuf_reset(&buf);
		for (i = 0; i < strategy_options.nr; i++)
			strbuf_addf(&buf, " --%s",
				    strategy_options.items[i].string);
		options.strategy_opts = xstrdup(buf.buf);
	}

	if (options.strategy) {
		options.strategy = xstrdup(options.strategy);
		switch (options.type) {
		case REBASE_AM:
			die(_("--strategy requires --merge or --interactive"));
		case REBASE_MERGE:
		case REBASE_INTERACTIVE:
		case REBASE_PRESERVE_MERGES:
			/* compatible */
			break;
		case REBASE_UNSPECIFIED:
			options.type = REBASE_MERGE;
			break;
		default:
			BUG("unhandled rebase type (%d)", options.type);
		}
	}

	if (options.root && !options.onto_name)
		imply_interactive(&options, "--root without --onto");

	if (isatty(2) && options.flags & REBASE_NO_QUIET)
		strbuf_addstr(&options.git_format_patch_opt, " --progress");

	switch (options.type) {
	case REBASE_MERGE:
	case REBASE_INTERACTIVE:
	case REBASE_PRESERVE_MERGES:
		options.state_dir = merge_dir();
		break;
	case REBASE_AM:
		options.state_dir = apply_dir();
		break;
	default:
		/* the default rebase backend is `--am` */
		options.type = REBASE_AM;
		options.state_dir = apply_dir();
		break;
	}

	if (options.git_am_opts.argc) {
		/* all am options except -q are compatible only with --am */
		for (i = options.git_am_opts.argc - 1; i >= 0; i--)
			if (strcmp(options.git_am_opts.argv[i], "-q"))
				break;

		if (is_interactive(&options) && i >= 0)
			die(_("error: cannot combine interactive options "
			      "(--interactive, --exec, --rebase-merges, "
			      "--preserve-merges, --keep-empty, --root + "
			      "--onto) with am options (%s)"), buf.buf);
		if (options.type == REBASE_MERGE && i >= 0)
			die(_("error: cannot combine merge options (--merge, "
			      "--strategy, --strategy-option) with am options "
			      "(%s)"), buf.buf);
	}

	if (options.signoff) {
		if (options.type == REBASE_PRESERVE_MERGES)
			die("cannot combine '--signoff' with "
			    "'--preserve-merges'");
		argv_array_push(&options.git_am_opts, "--signoff");
		options.flags |= REBASE_FORCE;
	}

	if (options.type == REBASE_PRESERVE_MERGES)
		/*
		 * Note: incompatibility with --signoff handled in signoff block above
		 * Note: incompatibility with --interactive is just a strong warning;
		 *       git-rebase.txt caveats with "unless you know what you are doing"
		 */
		if (options.rebase_merges)
			die(_("error: cannot combine '--preserve-merges' with "
			      "'--rebase-merges'"));

	if (options.rebase_merges) {
		if (strategy_options.nr)
			die(_("error: cannot combine '--rebase-merges' with "
			      "'--strategy-option'"));
		if (options.strategy)
			die(_("error: cannot combine '--rebase-merges' with "
			      "'--strategy'"));
	}

	if (!options.root) {
		if (argc < 1) {
			struct branch *branch;

			branch = branch_get(NULL);
			options.upstream_name = branch_get_upstream(branch,
								    NULL);
			if (!options.upstream_name)
				error_on_missing_default_upstream();
			if (fork_point < 0)
				fork_point = 1;
		} else {
			options.upstream_name = argv[0];
			argc--;
			argv++;
			if (!strcmp(options.upstream_name, "-"))
				options.upstream_name = "@{-1}";
		}
		options.upstream = peel_committish(options.upstream_name);
		if (!options.upstream)
			die(_("invalid upstream '%s'"), options.upstream_name);
		options.upstream_arg = options.upstream_name;
	} else {
		if (!options.onto_name) {
			if (commit_tree("", 0, the_hash_algo->empty_tree, NULL,
					&squash_onto, NULL, NULL) < 0)
				die(_("Could not create new root commit"));
			options.squash_onto = &squash_onto;
			options.onto_name = squash_onto_name =
				xstrdup(oid_to_hex(&squash_onto));
		}
		options.upstream_name = NULL;
		options.upstream = NULL;
		if (argc > 1)
			usage_with_options(builtin_rebase_usage,
					   builtin_rebase_options);
		options.upstream_arg = "--root";
	}

	/* Make sure the branch to rebase onto is valid. */
	if (!options.onto_name)
		options.onto_name = options.upstream_name;
	if (strstr(options.onto_name, "...")) {
		if (get_oid_mb(options.onto_name, &merge_base) < 0)
			die(_("'%s': need exactly one merge base"),
			    options.onto_name);
		options.onto = lookup_commit_or_die(&merge_base,
						    options.onto_name);
	} else {
		options.onto = peel_committish(options.onto_name);
		if (!options.onto)
			die(_("Does not point to a valid commit '%s'"),
				options.onto_name);
	}

	/*
	 * If the branch to rebase is given, that is the branch we will rebase
	 * branch_name -- branch/commit being rebased, or
	 * 		  HEAD (already detached)
	 * orig_head -- commit object name of tip of the branch before rebasing
	 * head_name -- refs/heads/<that-branch> or NULL (detached HEAD)
	 */
	if (argc == 1) {
		/* Is it "rebase other branchname" or "rebase other commit"? */
		branch_name = argv[0];
		options.switch_to = argv[0];

		/* Is it a local branch? */
		strbuf_reset(&buf);
		strbuf_addf(&buf, "refs/heads/%s", branch_name);
		if (!read_ref(buf.buf, &options.orig_head))
			options.head_name = xstrdup(buf.buf);
		/* If not is it a valid ref (branch or commit)? */
		else if (!get_oid(branch_name, &options.orig_head))
			options.head_name = NULL;
		else
			die(_("fatal: no such branch/commit '%s'"),
			    branch_name);
	} else if (argc == 0) {
		/* Do not need to switch branches, we are already on it. */
		options.head_name =
			xstrdup_or_null(resolve_ref_unsafe("HEAD", 0, NULL,
					 &flags));
		if (!options.head_name)
			die(_("No such ref: %s"), "HEAD");
		if (flags & REF_ISSYMREF) {
			if (!skip_prefix(options.head_name,
					 "refs/heads/", &branch_name))
				branch_name = options.head_name;

		} else {
			free(options.head_name);
			options.head_name = NULL;
			branch_name = "HEAD";
		}
		if (get_oid("HEAD", &options.orig_head))
			die(_("Could not resolve HEAD to a revision"));
	} else
		BUG("unexpected number of arguments left to parse");

	if (fork_point > 0) {
		struct commit *head =
			lookup_commit_reference(the_repository,
						&options.orig_head);
		options.restrict_revision =
			get_fork_point(options.upstream_name, head);
	}

	if (read_index(the_repository->index) < 0)
		die(_("could not read index"));

	if (options.autostash) {
		struct lock_file lock_file = LOCK_INIT;
		int fd;

		fd = hold_locked_index(&lock_file, 0);
		refresh_cache(REFRESH_QUIET);
		if (0 <= fd)
			update_index_if_able(&the_index, &lock_file);
		rollback_lock_file(&lock_file);

		if (has_unstaged_changes(1) || has_uncommitted_changes(1)) {
			const char *autostash =
				state_dir_path("autostash", &options);
			struct child_process stash = CHILD_PROCESS_INIT;
			struct object_id oid;
			struct commit *head =
				lookup_commit_reference(the_repository,
							&options.orig_head);

			argv_array_pushl(&stash.args,
					 "stash", "create", "autostash", NULL);
			stash.git_cmd = 1;
			stash.no_stdin = 1;
			strbuf_reset(&buf);
			if (capture_command(&stash, &buf, GIT_MAX_HEXSZ))
				die(_("Cannot autostash"));
			strbuf_trim_trailing_newline(&buf);
			if (get_oid(buf.buf, &oid))
				die(_("Unexpected stash response: '%s'"),
				    buf.buf);
			strbuf_reset(&buf);
			strbuf_add_unique_abbrev(&buf, &oid, DEFAULT_ABBREV);

			if (safe_create_leading_directories_const(autostash))
				die(_("Could not create directory for '%s'"),
				    options.state_dir);
			write_file(autostash, "%s", oid_to_hex(&oid));
			printf(_("Created autostash: %s\n"), buf.buf);
			if (reset_head(&head->object.oid, "reset --hard",
				       NULL, RESET_HEAD_HARD, NULL, NULL) < 0)
				die(_("could not reset --hard"));
			printf(_("HEAD is now at %s"),
			       find_unique_abbrev(&head->object.oid,
						  DEFAULT_ABBREV));
			strbuf_reset(&buf);
			pp_commit_easy(CMIT_FMT_ONELINE, head, &buf);
			if (buf.len > 0)
				printf(" %s", buf.buf);
			putchar('\n');

			if (discard_index(the_repository->index) < 0 ||
				read_index(the_repository->index) < 0)
				die(_("could not read index"));
		}
	}

	if (require_clean_work_tree("rebase",
				    _("Please commit or stash them."), 1, 1)) {
		ret = 1;
		goto cleanup;
	}

	/*
	 * Now we are rebasing commits upstream..orig_head (or with --root,
	 * everything leading up to orig_head) on top of onto.
	 */

	/*
	 * Check if we are already based on onto with linear history,
	 * but this should be done only when upstream and onto are the same
	 * and if this is not an interactive rebase.
	 */
	if (can_fast_forward(options.onto, &options.orig_head, &merge_base) &&
	    !is_interactive(&options) && !options.restrict_revision &&
	    options.upstream &&
	    !oidcmp(&options.upstream->object.oid, &options.onto->object.oid)) {
		int flag;

		if (!(options.flags & REBASE_FORCE)) {
			/* Lazily switch to the target branch if needed... */
			if (options.switch_to) {
				struct object_id oid;

				if (get_oid(options.switch_to, &oid) < 0) {
					ret = !!error(_("could not parse '%s'"),
						      options.switch_to);
					goto cleanup;
				}

				strbuf_reset(&buf);
				strbuf_addf(&buf, "%s: checkout %s",
					    getenv(GIT_REFLOG_ACTION_ENVIRONMENT),
					    options.switch_to);
				if (reset_head(&oid, "checkout",
					       options.head_name, 0,
					       NULL, buf.buf) < 0) {
					ret = !!error(_("could not switch to "
							"%s"),
						      options.switch_to);
					goto cleanup;
				}
			}

			if (!(options.flags & REBASE_NO_QUIET))
				; /* be quiet */
			else if (!strcmp(branch_name, "HEAD") &&
				 resolve_ref_unsafe("HEAD", 0, NULL, &flag))
				puts(_("HEAD is up to date."));
			else
				printf(_("Current branch %s is up to date.\n"),
				       branch_name);
			ret = !!finish_rebase(&options);
			goto cleanup;
		} else if (!(options.flags & REBASE_NO_QUIET))
			; /* be quiet */
		else if (!strcmp(branch_name, "HEAD") &&
			 resolve_ref_unsafe("HEAD", 0, NULL, &flag))
			puts(_("HEAD is up to date, rebase forced."));
		else
			printf(_("Current branch %s is up to date, rebase "
				 "forced.\n"), branch_name);
	}

	/* If a hook exists, give it a chance to interrupt*/
	if (!ok_to_skip_pre_rebase &&
	    run_hook_le(NULL, "pre-rebase", options.upstream_arg,
			argc ? argv[0] : NULL, NULL))
		die(_("The pre-rebase hook refused to rebase."));

	if (options.flags & REBASE_DIFFSTAT) {
		struct diff_options opts;

		if (options.flags & REBASE_VERBOSE) {
			if (is_null_oid(&merge_base))
				printf(_("Changes to %s:\n"),
				       oid_to_hex(&options.onto->object.oid));
			else
				printf(_("Changes from %s to %s:\n"),
				       oid_to_hex(&merge_base),
				       oid_to_hex(&options.onto->object.oid));
		}

		/* We want color (if set), but no pager */
		diff_setup(&opts);
		opts.stat_width = -1; /* use full terminal width */
		opts.stat_graph_width = -1; /* respect statGraphWidth config */
		opts.output_format |=
			DIFF_FORMAT_SUMMARY | DIFF_FORMAT_DIFFSTAT;
		opts.detect_rename = DIFF_DETECT_RENAME;
		diff_setup_done(&opts);
		diff_tree_oid(is_null_oid(&merge_base) ?
			      the_hash_algo->empty_tree : &merge_base,
			      &options.onto->object.oid, "", &opts);
		diffcore_std(&opts);
		diff_flush(&opts);
	}

	if (is_interactive(&options))
		goto run_rebase;

	/* Detach HEAD and reset the tree */
	if (options.flags & REBASE_NO_QUIET)
		printf(_("First, rewinding head to replay your work on top of "
			 "it...\n"));

	strbuf_addf(&msg, "%s: checkout %s",
		    getenv(GIT_REFLOG_ACTION_ENVIRONMENT), options.onto_name);
	if (reset_head(&options.onto->object.oid, "checkout", NULL,
		       RESET_HEAD_DETACH, NULL, msg.buf))
		die(_("Could not detach HEAD"));
	strbuf_release(&msg);

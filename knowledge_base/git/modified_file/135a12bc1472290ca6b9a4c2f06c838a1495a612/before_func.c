	int i, result = 0;
	struct strvec argv = STRVEC_INIT;

	if (!append && write_fetch_head) {
		int errcode = truncate_fetch_head();
		if (errcode)
			return errcode;
	}

	strvec_pushl(&argv, "fetch", "--append", "--no-auto-gc",
		     "--no-write-commit-graph", NULL);
	add_options_to_argv(&argv);

	if (max_children != 1 && list->nr != 1) {
		struct parallel_fetch_state state = { argv.v, list, 0, 0 };

		strvec_push(&argv, "--end-of-options");
		result = run_processes_parallel_tr2(max_children,
						    &fetch_next_remote,
						    &fetch_failed_to_start,
						    &fetch_finished,
						    &state,
						    "fetch", "parallel/fetch");

		if (!result)
			result = state.result;
	} else
		for (i = 0; i < list->nr; i++) {
			const char *name = list->items[i].string;
			strvec_push(&argv, name);
			if (verbosity >= 0)
				printf(_("Fetching %s\n"), name);
			if (run_command_v_opt(argv.v, RUN_GIT_CMD)) {
				error(_("Could not fetch %s"), name);
				result = 1;
			}
			strvec_pop(&argv);
		}

	strvec_clear(&argv);
	return !!result;
}

/*
 * Fetching from the promisor remote should use the given filter-spec
 * or inherit the default filter-spec from the config.
 */
static inline void fetch_one_setup_partial(struct remote *remote)
{
	/*
	 * Explicit --no-filter argument overrides everything, regardless
	 * of any prior partial clones and fetches.
	 */
	if (filter_options.no_filter)
		return;

	/*
	 * If no prior partial clone/fetch and the current fetch DID NOT
	 * request a partial-fetch, do a normal fetch.
	 */
	if (!has_promisor_remote() && !filter_options.choice)
		return;

	/*
	 * If this is a partial-fetch request, we enable partial on
	 * this repo if not already enabled and remember the given
	 * filter-spec as the default for subsequent fetches to this
	 * remote if there is currently no default filter-spec.
	 */
	if (filter_options.choice) {
		partial_clone_register(remote->name, &filter_options);
		return;
	}

	/*
	 * Do a partial-fetch from the promisor remote using either the
	 * explicitly given filter-spec or inherit the filter-spec from
	 * the config.
	 */
	if (!filter_options.choice)
		partial_clone_get_default_filter_spec(&filter_options, remote->name);
	return;
}

static int fetch_one(struct remote *remote, int argc, const char **argv,
		     int prune_tags_ok, int use_stdin_refspecs)
{
	struct refspec rs = REFSPEC_INIT_FETCH;
	int i;
	int exit_code;
	int maybe_prune_tags;
	int remote_via_config = remote_is_configured(remote, 0);

	if (!remote)
		die(_("No remote repository specified.  Please, specify either a URL or a\n"
		    "remote name from which new revisions should be fetched."));

	gtransport = prepare_transport(remote, 1);

	if (prune < 0) {
		/* no command line request */
		if (0 <= remote->prune)
			prune = remote->prune;
		else if (0 <= fetch_prune_config)
			prune = fetch_prune_config;
		else
			prune = PRUNE_BY_DEFAULT;
	}

	if (prune_tags < 0) {
		/* no command line request */
		if (0 <= remote->prune_tags)
			prune_tags = remote->prune_tags;
		else if (0 <= fetch_prune_tags_config)
			prune_tags = fetch_prune_tags_config;
		else
			prune_tags = PRUNE_TAGS_BY_DEFAULT;
	}

	maybe_prune_tags = prune_tags_ok && prune_tags;
	if (maybe_prune_tags && remote_via_config)
		refspec_append(&remote->fetch, TAG_REFSPEC);

	if (maybe_prune_tags && (argc || !remote_via_config))
		refspec_append(&rs, TAG_REFSPEC);

	for (i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "tag")) {
			i++;
			if (i >= argc)
				die(_("You need to specify a tag name."));

			refspec_appendf(&rs, "refs/tags/%s:refs/tags/%s",
					argv[i], argv[i]);
		} else {
			refspec_append(&rs, argv[i]);
		}
	}

	if (use_stdin_refspecs) {
		struct strbuf line = STRBUF_INIT;
		while (strbuf_getline_lf(&line, stdin) != EOF)
			refspec_append(&rs, line.buf);
		strbuf_release(&line);
	}

	if (server_options.nr)
		gtransport->server_options = &server_options;

	sigchain_push_common(unlock_pack_on_signal);
	atexit(unlock_pack);
	sigchain_push(SIGPIPE, SIG_IGN);
	exit_code = do_fetch(gtransport, &rs);
	sigchain_pop(SIGPIPE);
	refspec_clear(&rs);
	transport_disconnect(gtransport);
	gtransport = NULL;
	return exit_code;
}

int cmd_fetch(int argc, const char **argv, const char *prefix)
{
	int i;
	struct string_list list = STRING_LIST_INIT_DUP;
	struct remote *remote = NULL;
	int result = 0;
	int prune_tags_ok = 1;

	packet_trace_identity("fetch");

	/* Record the command line for the reflog */
	strbuf_addstr(&default_rla, "fetch");
	for (i = 1; i < argc; i++) {
		/* This handles non-URLs gracefully */
		char *anon = transport_anonymize_url(argv[i]);

		strbuf_addf(&default_rla, " %s", anon);
		free(anon);
	}

	git_config(git_fetch_config, NULL);


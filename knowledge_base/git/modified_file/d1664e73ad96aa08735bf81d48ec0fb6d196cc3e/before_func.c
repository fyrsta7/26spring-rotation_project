
	for (i = 0; i < dir->nr; i++) {
		check_embedded_repo(dir->entries[i]->name);
		if (add_file_to_index(&the_index, dir->entries[i]->name, flags)) {
			if (!ignore_add_errors)
				die(_("adding files failed"));
			exit_status = 1;
		}
	}
	return exit_status;
}

int cmd_add(int argc, const char **argv, const char *prefix)
{
	int exit_status = 0;
	struct pathspec pathspec;
	struct dir_struct dir;
	int flags;
	int add_new_files;
	int require_pathspec;
	char *seen = NULL;
	struct lock_file lock_file = LOCK_INIT;

	git_config(add_config, NULL);

	argc = parse_options(argc, argv, prefix, builtin_add_options,
			  builtin_add_usage, PARSE_OPT_KEEP_ARGV0);
	if (patch_interactive)
		add_interactive = 1;
	if (add_interactive)
		exit(interactive_add(argc - 1, argv + 1, prefix, patch_interactive));

	if (edit_interactive)
		return(edit_patch(argc, argv, prefix));
	argc--;
	argv++;

	if (0 <= addremove_explicit)
		addremove = addremove_explicit;
	else if (take_worktree_changes && ADDREMOVE_DEFAULT)
		addremove = 0; /* "-u" was given but not "-A" */

	if (addremove && take_worktree_changes)
		die(_("-A and -u are mutually incompatible"));

	if (!take_worktree_changes && addremove_explicit < 0 && argc)
		/* Turn "git add pathspec..." to "git add -A pathspec..." */
		addremove = 1;

	if (!show_only && ignore_missing)
		die(_("Option --ignore-missing can only be used together with --dry-run"));

	if (chmod_arg && ((chmod_arg[0] != '-' && chmod_arg[0] != '+') ||
			  chmod_arg[1] != 'x' || chmod_arg[2]))
		die(_("--chmod param '%s' must be either -x or +x"), chmod_arg);

	add_new_files = !take_worktree_changes && !refresh_only && !add_renormalize;
	require_pathspec = !(take_worktree_changes || (0 < addremove_explicit));

	hold_locked_index(&lock_file, LOCK_DIE_ON_ERROR);

	flags = ((verbose ? ADD_CACHE_VERBOSE : 0) |
		 (show_only ? ADD_CACHE_PRETEND : 0) |
		 (intent_to_add ? ADD_CACHE_INTENT : 0) |
		 (ignore_add_errors ? ADD_CACHE_IGNORE_ERRORS : 0) |
		 (!(addremove || take_worktree_changes)
		  ? ADD_CACHE_IGNORE_REMOVAL : 0));

	if (require_pathspec && argc == 0) {
		fprintf(stderr, _("Nothing specified, nothing added.\n"));
		fprintf(stderr, _("Maybe you wanted to say 'git add .'?\n"));
		return 0;
	}

	if (read_cache() < 0)
		die(_("index file corrupt"));

	die_in_unpopulated_submodule(&the_index, prefix);

	/*
	 * Check the "pathspec '%s' did not match any files" block
	 * below before enabling new magic.
	 */
	parse_pathspec(&pathspec, PATHSPEC_ATTR,
		       PATHSPEC_PREFER_FULL |
		       PATHSPEC_SYMLINK_LEADING_PATH,
		       prefix, argv);

	die_path_inside_submodule(&the_index, &pathspec);

	if (add_new_files) {
		int baselen;

		/* Set up the default git porcelain excludes */
		memset(&dir, 0, sizeof(dir));
		if (!ignored_too) {
			dir.flags |= DIR_COLLECT_IGNORED;
			setup_standard_excludes(&dir);
		}

		/* This picks up the paths that are not tracked */
		baselen = fill_directory(&dir, &the_index, &pathspec);
		if (pathspec.nr)
			seen = prune_directory(&dir, &pathspec, baselen);
	}

	if (refresh_only) {
		refresh(verbose, &pathspec);
		goto finish;
	}

	if (pathspec.nr) {
		int i;

		if (!seen)
			seen = find_pathspecs_matching_against_index(&pathspec, &the_index);

		/*
		 * file_exists() assumes exact match
		 */
		GUARD_PATHSPEC(&pathspec,
			       PATHSPEC_FROMTOP |
			       PATHSPEC_LITERAL |
			       PATHSPEC_GLOB |
			       PATHSPEC_ICASE |
			       PATHSPEC_EXCLUDE);

		for (i = 0; i < pathspec.nr; i++) {
			const char *path = pathspec.items[i].match;
			if (pathspec.items[i].magic & PATHSPEC_EXCLUDE)
				continue;
			if (!seen[i] && path[0] &&
			    ((pathspec.items[i].magic &
			      (PATHSPEC_GLOB | PATHSPEC_ICASE)) ||
			     !file_exists(path))) {
				if (ignore_missing) {
					int dtype = DT_UNKNOWN;
					if (is_excluded(&dir, &the_index, path, &dtype))
						dir_add_ignored(&dir, &the_index,
								path, pathspec.items[i].len);
				} else
					die(_("pathspec '%s' did not match any files"),
					    pathspec.items[i].original);
			}
		}
		free(seen);
	}

	plug_bulk_checkin();

	if (add_renormalize)
		exit_status |= renormalize_tracked_files(&pathspec, flags);
	else
		exit_status |= add_files_to_cache(prefix, &pathspec, flags);

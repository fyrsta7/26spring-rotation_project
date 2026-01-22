	*rev_ret = rev;
	return argv[0] ? get_pathspec(prefix, argv) : NULL;
}

static int update_refs(const char *rev, const unsigned char *sha1)
{
	int update_ref_status;
	struct strbuf msg = STRBUF_INIT;
	unsigned char *orig = NULL, sha1_orig[20],
		*old_orig = NULL, sha1_old_orig[20];

	if (!get_sha1("ORIG_HEAD", sha1_old_orig))
		old_orig = sha1_old_orig;
	if (!get_sha1("HEAD", sha1_orig)) {
		orig = sha1_orig;
		set_reflog_message(&msg, "updating ORIG_HEAD", NULL);
		update_ref(msg.buf, "ORIG_HEAD", orig, old_orig, 0, MSG_ON_ERR);
	} else if (old_orig)
		delete_ref("ORIG_HEAD", old_orig, 0);
	set_reflog_message(&msg, "updating HEAD", rev);
	update_ref_status = update_ref(msg.buf, "HEAD", sha1, orig, 0, MSG_ON_ERR);
	strbuf_release(&msg);
	return update_ref_status;
}

int cmd_reset(int argc, const char **argv, const char *prefix)
{
	int reset_type = NONE, update_ref_status = 0, quiet = 0;
	int patch_mode = 0, unborn;
	const char *rev;
	unsigned char sha1[20];
	const char **pathspec = NULL;
	const struct option options[] = {
		OPT__QUIET(&quiet, N_("be quiet, only report errors")),
		OPT_SET_INT(0, "mixed", &reset_type,
						N_("reset HEAD and index"), MIXED),
		OPT_SET_INT(0, "soft", &reset_type, N_("reset only HEAD"), SOFT),
		OPT_SET_INT(0, "hard", &reset_type,
				N_("reset HEAD, index and working tree"), HARD),
		OPT_SET_INT(0, "merge", &reset_type,
				N_("reset HEAD, index and working tree"), MERGE),
		OPT_SET_INT(0, "keep", &reset_type,
				N_("reset HEAD but keep local changes"), KEEP),
		OPT_BOOLEAN('p', "patch", &patch_mode, N_("select hunks interactively")),
		OPT_END()
	};

	git_config(git_default_config, NULL);

	argc = parse_options(argc, argv, prefix, options, git_reset_usage,
						PARSE_OPT_KEEP_DASHDASH);
	pathspec = parse_args(argv, prefix, &rev);

	unborn = !strcmp(rev, "HEAD") && get_sha1("HEAD", sha1);
	if (unborn) {
		/* reset on unborn branch: treat as reset to empty tree */
		hashcpy(sha1, EMPTY_TREE_SHA1_BIN);
	} else if (!pathspec) {
		struct commit *commit;
		if (get_sha1_committish(rev, sha1))
			die(_("Failed to resolve '%s' as a valid revision."), rev);
		commit = lookup_commit_reference(sha1);
		if (!commit)
			die(_("Could not parse object '%s'."), rev);
		hashcpy(sha1, commit->object.sha1);
	} else {
		struct tree *tree;
		if (get_sha1_treeish(rev, sha1))
			die(_("Failed to resolve '%s' as a valid tree."), rev);
		tree = parse_tree_indirect(sha1);
		if (!tree)
			die(_("Could not parse object '%s'."), rev);
		hashcpy(sha1, tree->object.sha1);
	}

	if (patch_mode) {
		if (reset_type != NONE)
			die(_("--patch is incompatible with --{hard,mixed,soft}"));
		return run_add_interactive(sha1_to_hex(sha1), "--patch=reset", pathspec);
	}

	/* git reset tree [--] paths... can be used to
	 * load chosen paths from the tree into the index without
	 * affecting the working tree nor HEAD. */
	if (pathspec) {
		if (reset_type == MIXED)
			warning(_("--mixed with paths is deprecated; use 'git reset -- <paths>' instead."));
		else if (reset_type != NONE)
			die(_("Cannot do %s reset with paths."),
					_(reset_type_names[reset_type]));
	}
	if (reset_type == NONE)
		reset_type = MIXED; /* by default */

	if (reset_type != SOFT && reset_type != MIXED)
		setup_work_tree();

	if (reset_type == MIXED && is_bare_repository())
		die(_("%s reset is not allowed in a bare repository"),
		    _(reset_type_names[reset_type]));

	/* Soft reset does not touch the index file nor the working tree
	 * at all, but requires them in a good order.  Other resets reset
	 * the index file to the tree object we are switching to. */
	if (reset_type == SOFT || reset_type == KEEP)
		die_if_unmerged_cache(reset_type);

	if (reset_type != SOFT) {
		struct lock_file *lock = xcalloc(1, sizeof(struct lock_file));
		int newfd = hold_locked_index(lock, 1);
		if (pathspec) {
			if (read_from_tree(pathspec, sha1))
				return 1;
		} else {
			int err = reset_index(sha1, reset_type, quiet);

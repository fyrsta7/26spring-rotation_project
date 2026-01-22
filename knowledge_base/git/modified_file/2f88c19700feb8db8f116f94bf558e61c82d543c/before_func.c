	/* if the entry is not checked out, don't examine work tree */
	cached = o->index_only ||
		(idx && ((idx->ce_flags & CE_VALID) || ce_skip_worktree(idx)));
	/*
	 * Backward compatibility wart - "diff-index -m" does
	 * not mean "do not ignore merges", but "match_missing".
	 *
	 * But with the revision flag parsing, that's found in
	 * "!revs->ignore_merges".
	 */
	match_missing = !revs->ignore_merges;

	if (cached && idx && ce_stage(idx)) {
		struct diff_filepair *pair;
		pair = diff_unmerge(&revs->diffopt, idx->name);
		fill_filespec(pair->one, idx->sha1, idx->ce_mode);
		return;
	}

	/*
	 * Something added to the tree?
	 */
	if (!tree) {
		show_new_file(revs, idx, cached, match_missing);
		return;
	}

	/*
	 * Something removed from the tree?
	 */
	if (!idx) {
		diff_index_show_file(revs, "-", tree, tree->sha1, tree->ce_mode, 0);
		return;
	}


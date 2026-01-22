/*
 * Has a file changed or has a submodule new commits or a dirty work tree?
 *
 * Return 1 when changes are detected, 0 otherwise. If the DIRTY_SUBMODULES
 * option is set, the caller does not only want to know if a submodule is
 * modified at all but wants to know all the conditions that are met (new
 * commits, untracked content and/or modified content).
 */
static int match_stat_with_submodule(struct diff_options *diffopt,
				     const struct cache_entry *ce,
				     struct stat *st, unsigned ce_option,
				     unsigned *dirty_submodule)
{
	int changed = ie_match_stat(diffopt->repo->index, ce, st, ce_option);
	if (S_ISGITLINK(ce->ce_mode)) {
		struct diff_flags orig_flags = diffopt->flags;
		if (!diffopt->flags.override_submodule_config)
			set_diffopt_flags_from_submodule_config(diffopt, ce->name);
		if (diffopt->flags.ignore_submodules)
			changed = 0;
		else if (!diffopt->flags.ignore_dirty_submodules &&
			 (!changed || diffopt->flags.dirty_submodules))
			*dirty_submodule = is_submodule_modified(ce->name,
								 diffopt->flags.ignore_untracked_in_submodules);
		diffopt->flags = orig_flags;
	}
	return changed;
}

int run_diff_files(struct rev_info *revs, unsigned int option)
{
	int entries, i;
	int diff_unmerged_stage = revs->max_count;
	unsigned ce_option = ((option & DIFF_RACY_IS_MODIFIED)
			      ? CE_MATCH_RACY_IS_DIRTY : 0);
	uint64_t start = getnanotime();
	struct index_state *istate = revs->diffopt.repo->index;

	diff_set_mnemonic_prefix(&revs->diffopt, "i/", "w/");

	if (diff_unmerged_stage < 0)
		diff_unmerged_stage = 2;
	entries = istate->cache_nr;
	for (i = 0; i < entries; i++) {
		unsigned int oldmode, newmode;
		struct cache_entry *ce = istate->cache[i];
		int changed;
		unsigned dirty_submodule = 0;
		const struct object_id *old_oid, *new_oid;

		if (diff_can_quit_early(&revs->diffopt))
			break;

		if (!ce_path_match(istate, ce, &revs->prune_data, NULL))
			continue;

		if (ce_stage(ce)) {
			struct combine_diff_path *dpath;
			struct diff_filepair *pair;
			unsigned int wt_mode = 0;
			int num_compare_stages = 0;
			size_t path_len;
			struct stat st;

			path_len = ce_namelen(ce);

			dpath = xmalloc(combine_diff_path_size(5, path_len));
			dpath->path = (char *) &(dpath->parent[5]);

			dpath->next = NULL;
			memcpy(dpath->path, ce->name, path_len);
			dpath->path[path_len] = '\0';
			oidclr(&dpath->oid);
			memset(&(dpath->parent[0]), 0,
			       sizeof(struct combine_diff_parent)*5);

			changed = check_removed(ce, &st);
			if (!changed)
				wt_mode = ce_mode_from_stat(ce, st.st_mode);
			else {
				if (changed < 0) {
					perror(ce->name);
					continue;
				}
				wt_mode = 0;
			}
			dpath->mode = wt_mode;

			while (i < entries) {
				struct cache_entry *nce = istate->cache[i];
				int stage;

				if (strcmp(ce->name, nce->name))
					break;

				/* Stage #2 (ours) is the first parent,
				 * stage #3 (theirs) is the second.
				 */
				stage = ce_stage(nce);
				if (2 <= stage) {
					int mode = nce->ce_mode;
					num_compare_stages++;
					oidcpy(&dpath->parent[stage - 2].oid,
					       &nce->oid);
					dpath->parent[stage-2].mode = ce_mode_from_stat(nce, mode);
					dpath->parent[stage-2].status =
						DIFF_STATUS_MODIFIED;
				}

				/* diff against the proper unmerged stage */
				if (stage == diff_unmerged_stage)
					ce = nce;
				i++;
			}
			/*
			 * Compensate for loop update
			 */
			i--;

			if (revs->combine_merges && num_compare_stages == 2) {
				show_combined_diff(dpath, 2, revs);
				free(dpath);
				continue;
			}
			FREE_AND_NULL(dpath);

			/*
			 * Show the diff for the 'ce' if we found the one
			 * from the desired stage.
			 */
			pair = diff_unmerge(&revs->diffopt, ce->name);
			if (wt_mode)
				pair->two->mode = wt_mode;
			if (ce_stage(ce) != diff_unmerged_stage)
				continue;
		}

		if (ce_uptodate(ce) || ce_skip_worktree(ce))
			continue;

		/* If CE_VALID is set, don't look at workdir for file removal */
		if (ce->ce_flags & CE_VALID) {
			changed = 0;
			newmode = ce->ce_mode;
		} else {
			struct stat st;

			changed = check_removed(ce, &st);
			if (changed) {
				if (changed < 0) {
					perror(ce->name);
					continue;
				}
				diff_addremove(&revs->diffopt, '-', ce->ce_mode,
					       &ce->oid,
					       !is_null_oid(&ce->oid),

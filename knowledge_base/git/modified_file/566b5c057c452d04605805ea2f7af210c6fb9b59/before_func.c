		}

		/*
		 * The previous round may already have decided to
		 * delete this path, which is in a subdirectory that
		 * is being replaced with a blob.
		 */
		cnt = cache_name_pos(ce->name, strlen(ce->name));
		if (0 <= cnt) {
			struct cache_entry *ce = active_cache[cnt];
			if (!ce_stage(ce) && !ce->ce_mode)
				return;
		}

		die("Untracked working tree file '%s' "
		    "would be %s by merge.", ce->name, action);
	}
}

static int merged_entry(struct cache_entry *merge, struct cache_entry *old,
		struct unpack_trees_options *o)
{
	merge->ce_flags |= htons(CE_UPDATE);
	if (old) {
		/*
		 * See if we can re-use the old CE directly?
		 * That way we get the uptodate stat info.
		 *
		 * This also removes the UPDATE flag on
		 * a match.
		 */
		if (same(old, merge)) {
			*merge = *old;
		} else {
			verify_uptodate(old, o);
			invalidate_ce_path(old);
		}
	}
	else {
		verify_absent(merge, "overwritten", o);
		invalidate_ce_path(merge);
	}

	merge->ce_flags &= ~htons(CE_STAGEMASK);
	add_cache_entry(merge, ADD_CACHE_OK_TO_ADD|ADD_CACHE_OK_TO_REPLACE);
	return 1;
}

static int deleted_entry(struct cache_entry *ce, struct cache_entry *old,
		struct unpack_trees_options *o)
{
	if (old)
		verify_uptodate(old, o);
	else
		verify_absent(ce, "removed", o);
	ce->ce_mode = 0;
	add_cache_entry(ce, ADD_CACHE_OK_TO_ADD|ADD_CACHE_OK_TO_REPLACE);
	invalidate_ce_path(ce);
	return 1;
}

static int keep_entry(struct cache_entry *ce, struct unpack_trees_options *o)
{
	add_cache_entry(ce, ADD_CACHE_OK_TO_ADD);
	return 1;
}

#if DBRT_DEBUG
static void show_stage_entry(FILE *o,
			     const char *label, const struct cache_entry *ce)
{
	if (!ce)
		fprintf(o, "%s (missing)\n", label);
	else
		fprintf(o, "%s%06o %s %d\t%s\n",
			label,
			ntohl(ce->ce_mode),
			sha1_to_hex(ce->sha1),
			ce_stage(ce),
			ce->name);
}
#endif

int threeway_merge(struct cache_entry **stages,
		struct unpack_trees_options *o,
		int remove)
{
	struct cache_entry *index;
	struct cache_entry *head;
	struct cache_entry *remote = stages[o->head_idx + 1];
	int count;
	int head_match = 0;
	int remote_match = 0;

	int df_conflict_head = 0;
	int df_conflict_remote = 0;

	int any_anc_missing = 0;
	int no_anc_exists = 1;
	int i;

	remove_entry(remove);
	for (i = 1; i < o->head_idx; i++) {
		if (!stages[i] || stages[i] == o->df_conflict_entry)
			any_anc_missing = 1;
		else
			no_anc_exists = 0;
	}

	index = stages[0];
	head = stages[o->head_idx];

	if (head == o->df_conflict_entry) {
		df_conflict_head = 1;
		head = NULL;
	}

	if (remote == o->df_conflict_entry) {
		df_conflict_remote = 1;
		remote = NULL;
	}

	/* First, if there's a #16 situation, note that to prevent #13
	 * and #14.
	 */
	if (!same(remote, head)) {
		for (i = 1; i < o->head_idx; i++) {
			if (same(stages[i], head)) {
				head_match = i;
			}
			if (same(stages[i], remote)) {
				remote_match = i;
			}
		}
	}

	/* We start with cases where the index is allowed to match
	 * something other than the head: #14(ALT) and #2ALT, where it
	 * is permitted to match the result instead.
	 */
	/* #14, #14ALT, #2ALT */
	if (remote && !df_conflict_head && head_match && !remote_match) {

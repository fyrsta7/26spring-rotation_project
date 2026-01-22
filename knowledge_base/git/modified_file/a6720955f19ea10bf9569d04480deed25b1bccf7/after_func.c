				 const struct traverse_info *info)
{
	int i;
	struct unpack_trees_options *o = info->data;
	unsigned long conflicts = info->df_conflicts | dirmask;

	/* Do we have *only* directories? Nothing to do */
	if (mask == dirmask && !src[0])
		return 0;

	/*
	 * Ok, we've filled in up to any potential index entry in src[0],
	 * now do the rest.
	 */
	for (i = 0; i < n; i++) {
		int stage;
		unsigned int bit = 1ul << i;
		if (conflicts & bit) {
			src[i + o->merge] = o->df_conflict_entry;
			continue;
		}
		if (!(mask & bit))
			continue;
		if (!o->merge)
			stage = 0;
		else if (i + 1 < o->head_idx)
			stage = 1;
		else if (i + 1 > o->head_idx)
			stage = 3;
		else
			stage = 2;
		src[i + o->merge] = create_ce_entry(info, names + i, stage);
	}

	if (o->merge) {
		int rc = call_unpack_fn((const struct cache_entry * const *)src,
					o);
		for (i = 0; i < n; i++) {
			struct cache_entry *ce = src[i + o->merge];
			if (ce != o->df_conflict_entry)
				free(ce);
		}
		return rc;
	}

	for (i = 0; i < n; i++)
		if (src[i] && src[i] != o->df_conflict_entry)
			if (do_add_entry(o, src[i], 0, 0))
				return -1;

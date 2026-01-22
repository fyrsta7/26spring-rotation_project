	struct commit *commit = NULL;
	unsigned int kind;

	if (flag & REF_BAD_NAME) {
		warning(_("ignoring ref with broken name %s"), refname);
		return 0;
	}

	if (flag & REF_ISBROKEN) {
		warning(_("ignoring broken ref %s"), refname);
		return 0;
	}

	/* Obtain the current ref kind from filter_ref_kind() and ignore unwanted refs. */
	kind = filter_ref_kind(filter, refname);
	if (!(kind & filter->kind))
		return 0;

	if (!filter_pattern_match(filter, refname))
		return 0;

	if (filter->points_at.nr && !match_points_at(&filter->points_at, oid, refname))
		return 0;

	/*
	 * A merge filter is applied on refs pointing to commits. Hence
	 * obtain the commit using the 'oid' available and discard all
	 * non-commits early. The actual filtering is done later.
	 */
	if (filter->reachable_from || filter->unreachable_from ||
	    filter->with_commit || filter->no_commit || filter->verbose) {
		commit = lookup_commit_reference_gently(the_repository, oid, 1);
		if (!commit)
			return 0;
		/* We perform the filtering for the '--contains' option... */
		if (filter->with_commit &&
		    !commit_contains(filter, commit, filter->with_commit, &ref_cbdata->contains_cache))
			return 0;
		/* ...or for the `--no-contains' option */

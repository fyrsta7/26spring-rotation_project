		if (cmp)
			return cmp;
	}
	pathlen = info->pathlen;
	ce_len = ce_namelen(ce);

	/* If ce_len < pathlen then we must have previously hit "name == directory" entry */
	if (ce_len < pathlen)
		return -1;

	ce_len -= pathlen;
	ce_name = ce->name + pathlen;

	len = tree_entry_len(n);
	return df_name_compare(ce_name, ce_len, S_IFREG, n->path, len, n->mode);
}

static int compare_entry(const struct cache_entry *ce, const struct traverse_info *info, const struct name_entry *n)
{
	int cmp = do_compare_entry(ce, info, n);
	if (cmp)
		return cmp;

	/*
	 * Even if the beginning compared identically, the ce should
	 * compare as bigger than a directory leading up to it!
	 */
	return ce_namelen(ce) > traverse_path_len(info, n);
}

static int ce_in_traverse_path(const struct cache_entry *ce,
			       const struct traverse_info *info)
{
	if (!info->prev)
		return 1;
	if (do_compare_entry(ce, info->prev, &info->name))
		return 0;
	/*
	 * If ce (blob) is the same name as the path (which is a tree
	 * we will be descending into), it won't be inside it.
	 */
	return (info->pathlen < ce_namelen(ce));
}

static struct cache_entry *create_ce_entry(const struct traverse_info *info, const struct name_entry *n, int stage)
{
	int len = traverse_path_len(info, n);
	struct cache_entry *ce = xcalloc(1, cache_entry_size(len));

	ce->ce_mode = create_ce_mode(n->mode);
	ce->ce_flags = create_ce_flags(stage);
	ce->ce_namelen = len;
	hashcpy(ce->sha1, n->sha1);
	make_traverse_path(ce->name, info, n);

	return ce;

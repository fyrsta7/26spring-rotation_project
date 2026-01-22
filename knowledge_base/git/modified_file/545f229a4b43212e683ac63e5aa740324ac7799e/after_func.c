	c2 = b->name[len];
	if (!c1 && !c2)
		/*
		 * git-write-tree used to write out a nonsense tree that has
		 * entries with the same name, one blob and one tree.  Make
		 * sure we do not have duplicate entries.
		 */
		return TREE_HAS_DUPS;
	if (!c1 && a->directory)
		c1 = '/';
	if (!c2 && b->directory)
		c2 = '/';
	return c1 < c2 ? 0 : TREE_UNORDERED;
}

static int fsck_tree(struct tree *item)
{
	int retval;
	int has_full_path = 0;
	int has_zero_pad = 0;
	int has_bad_modes = 0;
	int has_dup_entries = 0;
	int not_properly_sorted = 0;
	struct tree_entry_list *entry, *last;

	last = NULL;
	for (entry = item->entries; entry; entry = entry->next) {
		if (strchr(entry->name, '/'))
			has_full_path = 1;
		has_zero_pad |= entry->zeropad;

		switch (entry->mode) {
		/*
		 * Standard modes.. 
		 */
		case S_IFREG | 0755:
		case S_IFREG | 0644:
		case S_IFLNK:
		case S_IFDIR:
			break;
		/*
		 * This is nonstandard, but we had a few of these
		 * early on when we honored the full set of mode
		 * bits..
		 */
		case S_IFREG | 0664:
			if (!check_strict)
				break;
		default:
			has_bad_modes = 1;
		}

		if (last) {
			switch (verify_ordered(last, entry)) {
			case TREE_UNORDERED:
				not_properly_sorted = 1;
				break;
			case TREE_HAS_DUPS:
				has_dup_entries = 1;
				break;
			default:
				break;
			}
			free(last->name);
			free(last);
		}

		last = entry;
	}
	if (last) {
		free(last->name);
		free(last);

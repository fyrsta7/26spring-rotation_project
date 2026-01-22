		if (!(dir->flags & DIR_NO_GITLINKS)) {
			struct object_id oid;
			if (resolve_gitlink_ref(dirname, "HEAD", &oid) == 0)
				return exclude ? path_excluded : path_untracked;
		}
		return path_recurse;
	}

	/* This is the "show_other_directories" case */

	if (!(dir->flags & DIR_HIDE_EMPTY_DIRECTORIES))
		return exclude ? path_excluded : path_untracked;

	untracked = lookup_untracked(dir->untracked, untracked,
				     dirname + baselen, len - baselen);

	/*
	 * If this is an excluded directory, then we only need to check if
	 * the directory contains any files.
	 */
	return read_directory_recursive(dir, istate, dirname, len,
					untracked, 1, exclude, pathspec);
}

/*
 * This is an inexact early pruning of any recursive directory
 * reading - if the path cannot possibly be in the pathspec,
 * return true, and we'll skip it early.
 */
static int simplify_away(const char *path, int pathlen,
			 const struct pathspec *pathspec)
{
	int i;

	if (!pathspec || !pathspec->nr)
		return 0;

	GUARD_PATHSPEC(pathspec,
		       PATHSPEC_FROMTOP |
		       PATHSPEC_MAXDEPTH |
		       PATHSPEC_LITERAL |
		       PATHSPEC_GLOB |
		       PATHSPEC_ICASE |
		       PATHSPEC_EXCLUDE |

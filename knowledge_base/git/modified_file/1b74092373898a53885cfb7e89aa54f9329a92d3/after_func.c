	int m = -1; /* signals that we haven't called strncmp() */

	if (*never_interesting) {
		/*
		 * We have not seen any match that sorts later
		 * than the current path.
		 */

		/*
		 * Does match sort strictly earlier than path
		 * with their common parts?
		 */
		m = strncmp(match, entry->path,
			    (matchlen < pathlen) ? matchlen : pathlen);
		if (m < 0)
			return 0;

		/*
		 * If we come here even once, that means there is at
		 * least one pathspec that would sort equal to or
		 * later than the path we are currently looking at.
		 * In other words, if we have never reached this point
		 * after iterating all pathspecs, it means all
		 * pathspecs are either outside of base, or inside the
		 * base but sorts strictly earlier than the current
		 * one.  In either case, they will never match the
		 * subsequent entries.  In such a case, we initialized
		 * the variable to -1 and that is what will be
		 * returned, allowing the caller to terminate early.
		 */
		*never_interesting = 0;
	}

	if (pathlen > matchlen)
		return 0;

	if (matchlen > pathlen) {
		if (match[pathlen] != '/')
			return 0;
		if (!S_ISDIR(entry->mode))
			return 0;
	}

	if (m == -1)
		/*
		 * we cheated and did not do strncmp(), so we do
		 * that here.
		 */
		m = strncmp(match, entry->path, pathlen);

	/*
	 * If common part matched earlier then it is a hit,
	 * because we rejected the case where path is not a
	 * leading directory and is shorter than match.
	 */
	if (!m)
		return 1;

	return 0;
}

static int match_dir_prefix(const char *base,
			    const char *match, int matchlen)
{
	if (strncmp(base, match, matchlen))
		return 0;

	/*
	 * If the base is a subdirectory of a path which
	 * was specified, all of them are interesting.
	 */
	if (!matchlen ||
	    base[matchlen] == '/' ||
	    match[matchlen - 1] == '/')
		return 1;

	/* Just a random prefix match */

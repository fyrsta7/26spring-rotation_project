};

static int estimate_similarity(struct diff_filespec *src,
			       struct diff_filespec *dst,
			       int minimum_score)
{
	/* src points at a file that existed in the original tree (or
	 * optionally a file in the destination tree) and dst points
	 * at a newly created file.  They may be quite similar, in which
	 * case we want to say src is renamed to dst or src is copied into
	 * dst, and then some edit has been applied to dst.
	 *
	 * Compare them and return how similar they are, representing
	 * the score as an integer between 0 and MAX_SCORE.
	 *
	 * When there is an exact match, it is considered a better
	 * match than anything else; the destination does not even
	 * call into this function in that case.
	 */
	void *delta;
	unsigned long delta_size, base_size, src_copied, literal_added;
	unsigned long delta_limit;
	int score;

	/* We deal only with regular files.  Symlink renames are handled
	 * only when they are exact matches --- in other words, no edits
	 * after renaming.
	 */
	if (!S_ISREG(src->mode) || !S_ISREG(dst->mode))
		return 0;

	delta_size = ((src->size < dst->size) ?
		      (dst->size - src->size) : (src->size - dst->size));
	base_size = ((src->size < dst->size) ? src->size : dst->size);

	/* We would not consider edits that change the file size so
	 * drastically.  delta_size must be smaller than
	 * (MAX_SCORE-minimum_score)/MAX_SCORE * min(src->size, dst->size).
	 *
	 * Note that base_size == 0 case is handled here already
	 * and the final score computation below would not have a
	 * divide-by-zero issue.
	 */
	if (base_size * (MAX_SCORE-minimum_score) < delta_size * MAX_SCORE)
		return 0;

	if (diff_populate_filespec(src, 0) || diff_populate_filespec(dst, 0))
		return 0; /* error but caught downstream */

	delta_limit = base_size * (MAX_SCORE-minimum_score) / MAX_SCORE;
	delta = diff_delta(src->data, src->size,
			   dst->data, dst->size,
			   &delta_size, delta_limit);
	if (!delta)
		/* If delta_limit is exceeded, we have too much differences */
		return 0;

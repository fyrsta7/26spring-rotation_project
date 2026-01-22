		return 0; /* we do not break too small filepair */

	if (diffcore_count_changes(src, dst,
				   NULL, NULL,
				   0,
				   &src_copied, &literal_added))
		return 0;

	/* sanity */
	if (src->size < src_copied)
		src_copied = src->size;
	if (dst->size < literal_added + src_copied) {
		if (src_copied < dst->size)
			literal_added = dst->size - src_copied;
		else
			literal_added = 0;
	}
	src_removed = src->size - src_copied;

	/* Compute merge-score, which is "how much is removed
	 * from the source material".  The clean-up stage will
	 * merge the surviving pair together if the score is
	 * less than the minimum, after rename/copy runs.
	 */
	*merge_score_p = (int)(src_removed * MAX_SCORE / src->size);
	if (*merge_score_p > break_score)
		return 1;

	/* Extent of damage, which counts both inserts and
	 * deletes.
	 */
	delta_size = src_removed + literal_added;
	if (delta_size * MAX_SCORE / max_size < break_score)
		return 0;

	/* If you removed a lot without adding new material, that is
	 * not really a rewrite.
	 */
	if ((src->size * break_score < src_removed * MAX_SCORE) &&
	    (literal_added * 20 < src_removed) &&
	    (literal_added * 20 < src_copied))
		return 0;

	return 1;
}

void diffcore_break(int break_score)
{
	struct diff_queue_struct *q = &diff_queued_diff;
	struct diff_queue_struct outq;

	/* When the filepair has this much edit (insert and delete),
	 * it is first considered to be a rewrite and broken into a
	 * create and delete filepair.  This is to help breaking a
	 * file that had too much new stuff added, possibly from
	 * moving contents from another file, so that rename/copy can
	 * match it with the other file.
	 *
	 * int break_score; we reuse incoming parameter for this.
	 */

	/* After a pair is broken according to break_score and
	 * subjected to rename/copy, both of them may survive intact,
	 * due to lack of suitable rename/copy peer.  Or, the caller
	 * may be calling us without using rename/copy.  When that

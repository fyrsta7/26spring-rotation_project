#include "count-delta.h"

static int should_break(struct diff_filespec *src,
			struct diff_filespec *dst,
			int break_score,
			int *merge_score_p)
{
	/* dst is recorded as a modification of src.  Are they so
	 * different that we are better off recording this as a pair
	 * of delete and create?
	 *
	 * There are two criteria used in this algorithm.  For the
	 * purposes of helping later rename/copy, we take both delete
	 * and insert into account and estimate the amount of "edit".
	 * If the edit is very large, we break this pair so that
	 * rename/copy can pick the pieces up to match with other
	 * files.
	 *
	 * On the other hand, we would want to ignore inserts for the
	 * pure "complete rewrite" detection.  As long as most of the
	 * existing contents were removed from the file, it is a
	 * complete rewrite, and if sizable chunk from the original
	 * still remains in the result, it is not a rewrite.  It does
	 * not matter how much or how little new material is added to
	 * the file.
	 *
	 * The score we leave for such a broken filepair uses the
	 * latter definition so that later clean-up stage can find the
	 * pieces that should not have been broken according to the
	 * latter definition after rename/copy runs, and merge the
	 * broken pair that have a score lower than given criteria
	 * back together.  The break operation itself happens
	 * according to the former definition.
	 *
	 * The minimum_edit parameter tells us when to break (the
	 * amount of "edit" required for us to consider breaking the
	 * pair).  We leave the amount of deletion in *merge_score_p
	 * when we return.
	 *
	 * The value we return is 1 if we want the pair to be broken,
	 * or 0 if we do not.
	 */
	void *delta;
	unsigned long delta_size, base_size, src_copied, literal_added;
	int to_break = 0;

	*merge_score_p = 0; /* assume no deletion --- "do not break"
			     * is the default.
			     */

	if (!S_ISREG(src->mode) || !S_ISREG(dst->mode))
		return 0; /* leave symlink rename alone */

	if (src->sha1_valid && dst->sha1_valid &&
	    !memcmp(src->sha1, dst->sha1, 20))
		return 0; /* they are the same */

	if (diff_populate_filespec(src, 0) || diff_populate_filespec(dst, 0))
		return 0; /* error but caught downstream */


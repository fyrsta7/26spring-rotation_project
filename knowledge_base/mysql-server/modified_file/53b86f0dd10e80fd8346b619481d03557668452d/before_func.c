	mtr_commit(&mtr);

	mtr_start(&mtr);

	cursor.path_arr = path2;

	if (dtuple_get_n_fields(tuple2) > 0) {

		btr_cur_search_to_nth_level(index, 0, tuple2, mode2,
					    BTR_SEARCH_LEAF | BTR_ESTIMATE,
					    &cursor, 0, &mtr);
	} else {
		btr_cur_open_at_index_side(FALSE, index,
					   BTR_SEARCH_LEAF | BTR_ESTIMATE,
					   &cursor, &mtr);
	}

	mtr_commit(&mtr);

	/* We have the path information for the range in path1 and path2 */

	n_rows = 1;
	diverged = FALSE;	    /* This becomes true when the path is not
				    the same any more */
	diverged_lot = FALSE;	    /* This becomes true when the paths are
				    not the same or adjacent any more */
	divergence_level = 1000000; /* This is the level where paths diverged
				    a lot */
	for (i = 0; ; i++) {
		ut_ad(i < BTR_PATH_ARRAY_N_SLOTS);

		slot1 = path1 + i;
		slot2 = path2 + i;

		if (slot1->nth_rec == ULINT_UNDEFINED
		    || slot2->nth_rec == ULINT_UNDEFINED) {

			if (i > divergence_level + 1) {
				/* In trees whose height is > 1 our algorithm
				tends to underestimate: multiply the estimate
				by 2: */

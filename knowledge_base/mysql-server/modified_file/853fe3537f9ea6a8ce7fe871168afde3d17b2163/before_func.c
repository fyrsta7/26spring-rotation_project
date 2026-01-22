		cursor->low_match = match;

	} else if (mode == PAGE_CUR_G) {
		if (cmp != -1) {
			goto exit_func;
		}
	} else if (mode == PAGE_CUR_L) {
		if (cmp != 1) {
			goto exit_func;
		}
	}

	if (can_only_compare_to_cursor_rec) {
		/* Since we could not determine if our guess is right just by
		looking at the record under the cursor, return FALSE */
		goto exit_func;
	}

	match = 0;
	bytes = 0;

	if ((mode == PAGE_CUR_G) || (mode == PAGE_CUR_GE)) {
		rec_t*	prev_rec;

		ut_ad(!page_rec_is_infimum(rec));

		prev_rec = page_rec_get_prev(rec);

		if (page_rec_is_infimum(prev_rec)) {
			success = btr_page_get_prev(page_align(prev_rec), mtr)
				== FIL_NULL;

			goto exit_func;
		}

		offsets = rec_get_offsets(prev_rec, cursor->index, offsets,
					  n_unique, &heap);
		cmp = page_cmp_dtuple_rec_with_match(tuple, prev_rec,
						     offsets, &match, &bytes);
		if (mode == PAGE_CUR_GE) {
			success = cmp == 1;
		} else {
			success = cmp != -1;
		}

		goto exit_func;
	} else {
		rec_t*	next_rec;

		ut_ad(!page_rec_is_supremum(rec));

		next_rec = page_rec_get_next(rec);

		if (page_rec_is_supremum(next_rec)) {
			if (btr_page_get_next(page_align(next_rec), mtr)
			    == FIL_NULL) {

				cursor->up_match = 0;
				success = TRUE;
			}

			goto exit_func;
		}

		offsets = rec_get_offsets(next_rec, cursor->index, offsets,
					  n_unique, &heap);
		cmp = page_cmp_dtuple_rec_with_match(tuple, next_rec,
						     offsets, &match, &bytes);
		if (mode == PAGE_CUR_LE) {
			success = cmp == -1;
			cursor->up_match = match;
		} else {
			success = cmp != 1;
		}
	}
exit_func:
	if (UNIV_LIKELY_NULL(heap)) {
		mem_heap_free(heap);
	}
	return(success);
}

/**********************************************************************
Tries to guess the right search position based on the hash search info
of the index. Note that if mode is PAGE_CUR_LE, which is used in inserts,
and the function returns TRUE, then cursor->up_match and cursor->low_match
both have sensible values. */

ibool
btr_search_guess_on_hash(
/*=====================*/
					/* out: TRUE if succeeded */
	dict_index_t*	index,		/* in: index */
	btr_search_t*	info,		/* in: index search info */
	const dtuple_t*	tuple,		/* in: logical record */
	ulint		mode,		/* in: PAGE_CUR_L, ... */
	ulint		latch_mode,	/* in: BTR_SEARCH_LEAF, ...;
					NOTE that only if has_search_latch
					is 0, we will have a latch set on
					the cursor page, otherwise we assume
					the caller uses his search latch
					to protect the record! */
	btr_cur_t*	cursor,		/* out: tree cursor */
	ulint		has_search_latch,/* in: latch mode the caller
					currently has on btr_search_latch:
					RW_S_LATCH, RW_X_LATCH, or 0 */
	mtr_t*		mtr)		/* in: mtr */
{
	buf_block_t*	block;
	rec_t*		rec;
	const page_t*	page;
	ulint		fold;
	ulint		tuple_n_fields;
	dulint		index_id;
	ibool		can_only_compare_to_cursor_rec = TRUE;
#ifdef notdefined
	btr_cur_t	cursor2;
	btr_pcur_t	pcur;
#endif
	ut_ad(index && info && tuple && cursor && mtr);
	ut_ad((latch_mode == BTR_SEARCH_LEAF)
	      || (latch_mode == BTR_MODIFY_LEAF));

	/* Note that, for efficiency, the struct info may not be protected by
	any latch here! */

	if (UNIV_UNLIKELY(info->n_hash_potential == 0)) {

		return(FALSE);
	}

	cursor->n_fields = info->n_fields;
	cursor->n_bytes = info->n_bytes;

	tuple_n_fields = dtuple_get_n_fields(tuple);

	if (UNIV_UNLIKELY(tuple_n_fields < cursor->n_fields)) {

		return(FALSE);
	}

	if (UNIV_UNLIKELY(tuple_n_fields == cursor->n_fields)
	    && (cursor->n_bytes > 0)) {

		return(FALSE);
	}

	index_id = index->id;

#ifdef UNIV_SEARCH_PERF_STAT
	info->n_hash_succ++;
#endif
	fold = dtuple_fold(tuple, cursor->n_fields, cursor->n_bytes, index_id);

	cursor->fold = fold;
	cursor->flag = BTR_CUR_HASH;

	if (UNIV_LIKELY(!has_search_latch)) {
		rw_lock_s_lock(&btr_search_latch);
	}

	ut_ad(btr_search_latch.writer != RW_LOCK_EX);
	ut_ad(btr_search_latch.reader_count > 0);

	rec = ha_search_and_get_data(btr_search_sys->hash_index, fold);

	if (UNIV_UNLIKELY(!rec)) {
		goto failure_unlock;
	}

	page = page_align((rec_t*) rec);
	{
		ulint	page_no		= page_get_page_no(page);
		ulint	space_id	= page_get_space_id(page);

		mutex_enter(&buf_pool->mutex);
		block = (buf_block_t*) buf_page_hash_get(space_id, page_no);
		mutex_exit(&buf_pool->mutex);
	}

	if (UNIV_UNLIKELY(!block)
	    || UNIV_UNLIKELY(buf_block_get_state(block)
			     != BUF_BLOCK_FILE_PAGE)) {

		/* The block is most probably being freed.
		The function buf_LRU_search_and_free_block()
		first removes the block from buf_pool->page_hash
		by calling buf_LRU_block_remove_hashed_page().
		After that, it invokes btr_search_drop_page_hash_index().
		Let us pretend that the block was also removed from
		the adaptive hash index. */
		goto failure_unlock;
	}

	if (UNIV_LIKELY(!has_search_latch)) {

		if (UNIV_UNLIKELY(
			    !buf_page_get_known_nowait(latch_mode, block,
						       BUF_MAKE_YOUNG,
						       __FILE__, __LINE__,
						       mtr))) {
			goto failure_unlock;
		}

		rw_lock_s_unlock(&btr_search_latch);
		can_only_compare_to_cursor_rec = FALSE;

#ifdef UNIV_SYNC_DEBUG
		buf_block_dbg_add_level(block, SYNC_TREE_NODE_FROM_HASH);

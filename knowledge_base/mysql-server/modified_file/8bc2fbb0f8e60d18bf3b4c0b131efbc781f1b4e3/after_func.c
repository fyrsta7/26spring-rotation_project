				/* out: DB_SUCCESS or error code */
	row_prebuilt_t*	prebuilt,/* in: prebuilt struct in the handle */
	dict_index_t*	sec_index,/* in: secondary index where rec resides */
	rec_t*		rec,	/* in: record in a non-clustered index; if
				this is a locking read, then rec is not
				allowed to be delete-marked, and that would
				not make sense either */
	que_thr_t*	thr,	/* in: query thread */
	rec_t**		out_rec,/* out: clustered record or an old version of
				it, NULL if the old version did not exist
				in the read view, i.e., it was a fresh
				inserted version */
	ulint**		offsets,/* out: offsets returned by
				rec_get_offsets(out_rec, clust_index) */
	mem_heap_t**	offset_heap,/* in/out: memory heap from which
				the offsets are allocated */
	mtr_t*		mtr)	/* in: mtr used to get access to the
				non-clustered record; the same mtr is used to
				access the clustered index */
{
	dict_index_t*	clust_index;
	rec_t*		clust_rec;
	rec_t*		old_vers;
	ulint		err;
	trx_t*		trx;

	*out_rec = NULL;
	trx = thr_get_trx(thr);
	
	row_build_row_ref_in_tuple(prebuilt->clust_ref, sec_index, rec, trx);

	clust_index = dict_table_get_first_index(sec_index->table);
	
	btr_pcur_open_with_no_init(clust_index, prebuilt->clust_ref,
			PAGE_CUR_LE, BTR_SEARCH_LEAF,
			prebuilt->clust_pcur, 0, mtr);

	clust_rec = btr_pcur_get_rec(prebuilt->clust_pcur);

	prebuilt->clust_pcur->trx_if_known = trx;

	/* Note: only if the search ends up on a non-infimum record is the
	low_match value the real match to the search tuple */

	if (!page_rec_is_user_rec(clust_rec)
	    || btr_pcur_get_low_match(prebuilt->clust_pcur)
	       < dict_index_get_n_unique(clust_index)) {
	
		/* In a rare case it is possible that no clust rec is found
		for a delete-marked secondary index record: if in row0umod.c
		in row_undo_mod_remove_clust_low() we have already removed
		the clust rec, while purge is still cleaning and removing
		secondary index records associated with earlier versions of
		the clustered index record. In that case we know that the
		clustered index record did not exist in the read view of
		trx. */

		if (!rec_get_deleted_flag(rec, sec_index->table->comp)
		    || prebuilt->select_lock_type != LOCK_NONE) {
		        ut_print_timestamp(stderr);
			fputs("  InnoDB: error clustered record"
				" for sec rec not found\n"
				"InnoDB: ", stderr);
			dict_index_name_print(stderr, trx, sec_index);
			fputs("\n"
				"InnoDB: sec index record ", stderr);
			rec_print(stderr, rec, sec_index);
			fputs("\n"
				"InnoDB: clust index record ", stderr);
			rec_print(stderr, clust_rec, clust_index);
			putc('\n', stderr);
			trx_print(stderr, trx);

			fputs("\n"
"InnoDB: Submit a detailed bug report to http://bugs.mysql.com\n", stderr);
		}

		clust_rec = NULL;

		goto func_exit;
	}

	*offsets = rec_get_offsets(clust_rec, clust_index, *offsets,
					ULINT_UNDEFINED, offset_heap);

	if (prebuilt->select_lock_type != LOCK_NONE) {
		/* Try to place a lock on the index record; we are searching
		the clust rec with a unique condition, hence
		we set a LOCK_REC_NOT_GAP type lock */
		
		err = lock_clust_rec_read_check_and_lock(0, clust_rec,
					clust_index, *offsets,
					prebuilt->select_lock_type,
					LOCK_REC_NOT_GAP, thr);
		if (err != DB_SUCCESS) {

			goto err_exit;
		}
	} else {
		/* This is a non-locking consistent read: if necessary, fetch
		a previous version of the record */

		old_vers = NULL;

		/* If the isolation level allows reading of uncommitted data,
		then we never look for an earlier version */

		if (trx->isolation_level > TRX_ISO_READ_UNCOMMITTED
		    && !lock_clust_rec_cons_read_sees(clust_rec, clust_index,
						*offsets, trx->read_view)) {

			err = row_sel_build_prev_vers_for_mysql(
					trx->read_view, clust_index,
					prebuilt, clust_rec,
					offsets, offset_heap,
					&old_vers, mtr);
						
			if (err != DB_SUCCESS) {

				goto err_exit;
			}

			clust_rec = old_vers;
		}

		/* If we had to go to an earlier version of row or the
		secondary index record is delete marked, then it may be that
		the secondary index record corresponding to clust_rec
		(or old_vers) is not rec; in that case we must ignore
		such row because in our snapshot rec would not have existed.
		Remember that from rec we cannot see directly which transaction
		id corresponds to it: we have to go to the clustered index
		record. A query where we want to fetch all rows where
		the secondary index value is in some interval would return
		a wrong result if we would not drop rows which we come to
		visit through secondary index records that would not really
		exist in our snapshot. */
		
		if (clust_rec && (old_vers
			|| rec_get_deleted_flag(rec, sec_index->table->comp))
		    && !row_sel_sec_rec_is_for_clust_rec(rec, sec_index,
						clust_rec, clust_index)) {
			clust_rec = NULL;
		} else {
#ifdef UNIV_SEARCH_DEBUG
			ut_a(clust_rec == NULL ||
			    row_sel_sec_rec_is_for_clust_rec(rec, sec_index,
						clust_rec, clust_index));
#endif		
		}
	}

func_exit:
	*out_rec = clust_rec;

	if (prebuilt->select_lock_type == LOCK_X) {
		/* We may use the cursor in update: store its position */
		
		btr_pcur_store_position(prebuilt->clust_pcur, mtr);
	}

	err = DB_SUCCESS;
err_exit:
	return(err);
}

/************************************************************************
Restores cursor position after it has been stored. We have to take into
account that the record cursor was positioned on may have been deleted.
Then we may have to move the cursor one step up or down. */
static
ibool
sel_restore_position_for_mysql(
/*===========================*/
					/* out: TRUE if we may need to
					process the record the cursor is
					now positioned on (i.e. we should
					not go to the next record yet) */
	ibool*		same_user_rec,	/* out: TRUE if we were able to restore
					the cursor on a user record with the
					same ordering prefix in in the
					B-tree index */
	ulint		latch_mode,	/* in: latch mode wished in
					restoration */
	btr_pcur_t*	pcur,		/* in: cursor whose position
					has been stored */
	ibool		moves_up,	/* in: TRUE if the cursor moves up
					in the index */
	mtr_t*		mtr)		/* in: mtr; CAUTION: may commit
					mtr temporarily! */
{
	ibool	success;
	ulint	relative_position;

	relative_position = pcur->rel_pos;
	
	success = btr_pcur_restore_position(latch_mode, pcur, mtr);

	*same_user_rec = success;

	if (relative_position == BTR_PCUR_ON) {
		if (success) {
			return(FALSE);
		}

		if (moves_up) {
			btr_pcur_move_to_next(pcur, mtr);
		}

		return(TRUE);
	}

	if (relative_position == BTR_PCUR_AFTER
	    || relative_position == BTR_PCUR_AFTER_LAST_IN_TREE) {

		if (moves_up) {
			return(TRUE);
		}
					
		if (btr_pcur_is_on_user_rec(pcur, mtr)) {
			btr_pcur_move_to_prev(pcur, mtr);
		}

		return(TRUE);
	}

	ut_ad(relative_position == BTR_PCUR_BEFORE
	     || relative_position == BTR_PCUR_BEFORE_FIRST_IN_TREE);
	
	if (moves_up && btr_pcur_is_on_user_rec(pcur, mtr)) {
		btr_pcur_move_to_next(pcur, mtr);
	}

	return(TRUE);
}

/************************************************************************
Pops a cached row for MySQL from the fetch cache. */
UNIV_INLINE
void
row_sel_pop_cached_row_for_mysql(
/*=============================*/
	byte*		buf,		/* in/out: buffer where to copy the
					row */
	row_prebuilt_t*	prebuilt)	/* in: prebuilt struct */
{
	ulint			i;
	mysql_row_templ_t*	templ;
	byte*			cached_rec;
        ut_ad(prebuilt->n_fetch_cached > 0);
	ut_ad(prebuilt->mysql_prefix_len <= prebuilt->mysql_row_len);
	
	if (UNIV_UNLIKELY(prebuilt->keep_other_fields_on_keyread))
	{
		/* Copy cache record field by field, don't touch fields that 
		are not covered by current key */
		cached_rec = 
			prebuilt->fetch_cache[prebuilt->fetch_cache_first];

		for (i = 0; i < prebuilt->n_template; i++) {
			templ = prebuilt->mysql_template + i;
			ut_memcpy(
				buf + templ->mysql_col_offset, 
				cached_rec + templ->mysql_col_offset,
				templ->mysql_col_len);
			/* Copy NULL bit of the current field from cached_rec 
			to buf */
			if (templ->mysql_null_bit_mask)
			{
				buf[templ->mysql_null_byte_offset] ^=
				  (buf[templ->mysql_null_byte_offset] ^
				   cached_rec[templ->mysql_null_byte_offset]) &
				  (byte)templ->mysql_null_bit_mask;
			}
		}
	}
	else
	{
		ut_memcpy(buf, prebuilt->fetch_cache[prebuilt->fetch_cache_first],
				prebuilt->mysql_prefix_len);
	}
	prebuilt->n_fetch_cached--;
	prebuilt->fetch_cache_first++;

	if (prebuilt->n_fetch_cached == 0) {
		prebuilt->fetch_cache_first = 0;
	}
}

/************************************************************************
Pushes a row for MySQL to the fetch cache. */
UNIV_INLINE
void
row_sel_push_cache_row_for_mysql(
/*=============================*/
	row_prebuilt_t*	prebuilt,	/* in: prebuilt struct */
	rec_t*		rec,		/* in: record to push */
	const ulint*	offsets)	/* in: rec_get_offsets() */
{
	byte*	buf;
	ulint	i;

	ut_ad(prebuilt->n_fetch_cached < MYSQL_FETCH_CACHE_SIZE);
	ut_ad(rec_offs_validate(rec, NULL, offsets));
	ut_a(!prebuilt->templ_contains_blob);

	if (prebuilt->fetch_cache[0] == NULL) {
		/* Allocate memory for the fetch cache */

		for (i = 0; i < MYSQL_FETCH_CACHE_SIZE; i++) {

			/* A user has reported memory corruption in these
			buffers in Linux. Put magic numbers there to help
			to track a possible bug. */
			
			buf = mem_alloc(prebuilt->mysql_row_len + 8);

			prebuilt->fetch_cache[i] = buf + 4;
				
			mach_write_to_4(buf, ROW_PREBUILT_FETCH_MAGIC_N);
			mach_write_to_4(buf + 4 + prebuilt->mysql_row_len,
					ROW_PREBUILT_FETCH_MAGIC_N);
		}
	}

	ut_ad(prebuilt->fetch_cache_first == 0);

	if (UNIV_UNLIKELY(!row_sel_store_mysql_rec(
			prebuilt->fetch_cache[prebuilt->n_fetch_cached],
			prebuilt, rec, offsets))) {
		ut_error;
	}

	prebuilt->n_fetch_cached++;
}

/*************************************************************************
Tries to do a shortcut to fetch a clustered index record with a unique key,
using the hash index if possible (not always). We assume that the search
mode is PAGE_CUR_GE, it is a consistent read, there is a read view in trx,
btr search latch has been locked in S-mode. */
static
ulint
row_sel_try_search_shortcut_for_mysql(
/*==================================*/
				/* out: SEL_FOUND, SEL_EXHAUSTED, SEL_RETRY */
	rec_t**		out_rec,/* out: record if found */
	row_prebuilt_t*	prebuilt,/* in: prebuilt struct */
	ulint**		offsets,/* in/out: for rec_get_offsets(*out_rec) */
	mem_heap_t**	heap,	/* in/out: heap for rec_get_offsets() */
	mtr_t*		mtr)	/* in: started mtr */
{
	dict_index_t*	index		= prebuilt->index;
	dtuple_t*	search_tuple	= prebuilt->search_tuple;
	btr_pcur_t*	pcur		= prebuilt->pcur;
	trx_t*		trx		= prebuilt->trx;
	rec_t*		rec;
	
	ut_ad(index->type & DICT_CLUSTERED);
	ut_ad(!prebuilt->templ_contains_blob);
	
	btr_pcur_open_with_no_init(index, search_tuple, PAGE_CUR_GE,
					BTR_SEARCH_LEAF, pcur,
#ifndef UNIV_SEARCH_DEBUG
					RW_S_LATCH,
#else
					0,
#endif
					mtr);
	rec = btr_pcur_get_rec(pcur);
	
	if (!page_rec_is_user_rec(rec)) {

		return(SEL_RETRY);
	}

	/* As the cursor is now placed on a user record after a search with
	the mode PAGE_CUR_GE, the up_match field in the cursor tells how many
	fields in the user record matched to the search tuple */ 

	if (btr_pcur_get_up_match(pcur) < dtuple_get_n_fields(search_tuple)) {

		return(SEL_EXHAUSTED);
	}

	/* This is a non-locking consistent read: if necessary, fetch
	a previous version of the record */

	*offsets = rec_get_offsets(rec, index, *offsets,
					ULINT_UNDEFINED, heap);

	if (!lock_clust_rec_cons_read_sees(rec, index,
				*offsets, trx->read_view)) {

		return(SEL_RETRY);
	}

	if (rec_get_deleted_flag(rec, index->table->comp)) {

		return(SEL_EXHAUSTED);
	}

	*out_rec = rec;
	
	return(SEL_FOUND);
}

/************************************************************************
Searches for rows in the database. This is used in the interface to
MySQL. This function opens a cursor, and also implements fetch next
and fetch prev. NOTE that if we do a search with a full key value
from a unique index (ROW_SEL_EXACT), then we will not store the cursor
position and fetch next or fetch prev must not be tried to the cursor! */

ulint
row_search_for_mysql(
/*=================*/
					/* out: DB_SUCCESS,
					DB_RECORD_NOT_FOUND, 
					DB_END_OF_INDEX, DB_DEADLOCK,
					DB_LOCK_TABLE_FULL, DB_CORRUPTION,
					or DB_TOO_BIG_RECORD */
	byte*		buf,		/* in/out: buffer for the fetched
					row in the MySQL format */
	ulint		mode,		/* in: search mode PAGE_CUR_L, ... */
	row_prebuilt_t*	prebuilt,	/* in: prebuilt struct for the
					table handle; this contains the info
					of search_tuple, index; if search
					tuple contains 0 fields then we
					position the cursor at the start or
					the end of the index, depending on
					'mode' */
	ulint		match_mode,	/* in: 0 or ROW_SEL_EXACT or
					ROW_SEL_EXACT_PREFIX */ 
	ulint		direction)	/* in: 0 or ROW_SEL_NEXT or
					ROW_SEL_PREV; NOTE: if this is != 0,
					then prebuilt must have a pcur
					with stored position! In opening of a
					cursor 'direction' should be 0. */
{
	dict_index_t*	index		= prebuilt->index;
	dtuple_t*	search_tuple	= prebuilt->search_tuple;
	btr_pcur_t*	pcur		= prebuilt->pcur;
	trx_t*		trx		= prebuilt->trx;
	dict_index_t*	clust_index;
	que_thr_t*	thr;
	rec_t*		rec;
	rec_t*		index_rec;
	rec_t*		clust_rec;
	rec_t*		old_vers;
	ulint		err				= DB_SUCCESS;
	ibool		unique_search			= FALSE;
	ibool		unique_search_from_clust_index	= FALSE;
	ibool		mtr_has_extra_clust_latch 	= FALSE;
	ibool		moves_up 			= FALSE;
	ibool		set_also_gap_locks		= TRUE;
					/* if the query is a plain
					locking SELECT, and the isolation
					level is <= TRX_ISO_READ_COMMITTED,
					then this is set to FALSE */
#ifdef UNIV_SEARCH_DEBUG
	ulint		cnt				= 0;
#endif /* UNIV_SEARCH_DEBUG */
	ulint		next_offs;
	ibool		same_user_rec;
	mtr_t		mtr;
	mem_heap_t*	heap				= NULL;
	ulint		offsets_[REC_OFFS_NORMAL_SIZE];
	ulint*		offsets				= offsets_;

	*offsets_ = (sizeof offsets_) / sizeof *offsets_;

	ut_ad(index && pcur && search_tuple);
	ut_ad(trx->mysql_thread_id == os_thread_get_curr_id());

	if (UNIV_UNLIKELY(prebuilt->table->ibd_file_missing)) {
	        ut_print_timestamp(stderr);
	        fprintf(stderr, "  InnoDB: Error:\n"
"InnoDB: MySQL is trying to use a table handle but the .ibd file for\n"
"InnoDB: table %s does not exist.\n"
"InnoDB: Have you deleted the .ibd file from the database directory under\n"
"InnoDB: the MySQL datadir, or have you used DISCARD TABLESPACE?\n"
"InnoDB: Look from\n"
"http://dev.mysql.com/doc/mysql/en/InnoDB_troubleshooting_datadict.html\n"
"InnoDB: how you can resolve the problem.\n",
				prebuilt->table->name);
		return(DB_ERROR);
	}

	if (UNIV_UNLIKELY(prebuilt->magic_n != ROW_PREBUILT_ALLOCATED)) {
		fprintf(stderr,
		"InnoDB: Error: trying to free a corrupt\n"
		"InnoDB: table handle. Magic n %lu, table name ",
		(ulong) prebuilt->magic_n);
		ut_print_name(stderr, trx, prebuilt->table->name);
		putc('\n', stderr);

		mem_analyze_corruption((byte*)prebuilt);

		ut_error;
	}

	if (trx->n_mysql_tables_in_use == 0
	    && UNIV_UNLIKELY(prebuilt->select_lock_type == LOCK_NONE)) {
		/* Note that if MySQL uses an InnoDB temp table that it
		created inside LOCK TABLES, then n_mysql_tables_in_use can
		be zero; in that case select_lock_type is set to LOCK_X in
		::start_stmt. */

		fputs(
"InnoDB: Error: MySQL is trying to perform a SELECT\n"
"InnoDB: but it has not locked any tables in ::external_lock()!\n",
                      stderr);
		trx_print(stderr, trx);
                fputc('\n', stderr);
	}

/*	fprintf(stderr, "Match mode %lu\n search tuple ", (ulong) match_mode);
	dtuple_print(search_tuple);
	
	fprintf(stderr, "N tables locked %lu\n", trx->mysql_n_tables_locked);
*/
	/*-------------------------------------------------------------*/
	/* PHASE 0: Release a possible s-latch we are holding on the
	adaptive hash index latch if there is someone waiting behind */

	if (UNIV_UNLIKELY(btr_search_latch.writer != RW_LOCK_NOT_LOCKED)
	    && trx->has_search_latch) {

		/* There is an x-latch request on the adaptive hash index:
		release the s-latch to reduce starvation and wait for
		BTR_SEA_TIMEOUT rounds before trying to keep it again over
		calls from MySQL */

		rw_lock_s_unlock(&btr_search_latch);
		trx->has_search_latch = FALSE;

		trx->search_latch_timeout = BTR_SEA_TIMEOUT;
	}
	
	/* Reset the new record lock info if we srv_locks_unsafe_for_binlog
	is set. Then we are able to remove the record locks set here on an
	individual row. */

	if (srv_locks_unsafe_for_binlog
	    && prebuilt->select_lock_type != LOCK_NONE) {

		trx_reset_new_rec_lock_info(trx);
	}

	/*-------------------------------------------------------------*/
	/* PHASE 1: Try to pop the row from the prefetch cache */

	if (UNIV_UNLIKELY(direction == 0)) {
		trx->op_info = "starting index read";
	
		prebuilt->n_rows_fetched = 0;
		prebuilt->n_fetch_cached = 0;
		prebuilt->fetch_cache_first = 0;

		if (prebuilt->sel_graph == NULL) {
			/* Build a dummy select query graph */
			row_prebuild_sel_graph(prebuilt);
		}
	} else {
		trx->op_info = "fetching rows";

		if (prebuilt->n_rows_fetched == 0) {
			prebuilt->fetch_direction = direction;
		}

		if (UNIV_UNLIKELY(direction != prebuilt->fetch_direction)) {
			if (UNIV_UNLIKELY(prebuilt->n_fetch_cached > 0)) {
				ut_error;
				/* TODO: scrollable cursor: restore cursor to
				the place of the latest returned row,
				or better: prevent caching for a scroll
				cursor! */
			}
		
			prebuilt->n_rows_fetched = 0;
			prebuilt->n_fetch_cached = 0;
			prebuilt->fetch_cache_first = 0;

		} else if (UNIV_LIKELY(prebuilt->n_fetch_cached > 0)) {
			row_sel_pop_cached_row_for_mysql(buf, prebuilt);

			prebuilt->n_rows_fetched++;

			srv_n_rows_read++;
			err = DB_SUCCESS;
			goto func_exit;
		}

		if (prebuilt->fetch_cache_first > 0
		    && prebuilt->fetch_cache_first < MYSQL_FETCH_CACHE_SIZE) {

		    	/* The previous returned row was popped from the fetch
		    	cache, but the cache was not full at the time of the
		    	popping: no more rows can exist in the result set */

			err = DB_RECORD_NOT_FOUND;
			goto func_exit;
		}
		
		prebuilt->n_rows_fetched++;

		if (prebuilt->n_rows_fetched > 1000000000) {
			/* Prevent wrap-over */
			prebuilt->n_rows_fetched = 500000000;
		}

		mode = pcur->search_mode;
	}

	/* In a search where at most one record in the index may match, we
	can use a LOCK_REC_NOT_GAP type record lock when locking a non-delete-
	marked matching record.

	Note that in a unique secondary index there may be different delete-
	marked versions of a record where only the primary key values differ:
	thus in a secondary index we must use next-key locks when locking
	delete-marked records. */
	
	if (match_mode == ROW_SEL_EXACT
	    && index->type & DICT_UNIQUE
	    && dtuple_get_n_fields(search_tuple)
					== dict_index_get_n_unique(index)
	    && (index->type & DICT_CLUSTERED
		 || !dtuple_contains_null(search_tuple))) {

		/* Note above that a UNIQUE secondary index can contain many
		rows with the same key value if one of the columns is the SQL
		null. A clustered index under MySQL can never contain null
		columns because we demand that all the columns in primary key
		are non-null. */

		unique_search = TRUE;

		/* Even if the condition is unique, MySQL seems to try to
		retrieve also a second row if a primary key contains more than
		1 column. Return immediately if this is not a HANDLER
		command. */

		if (UNIV_UNLIKELY(direction != 0 &&
				!prebuilt->used_in_HANDLER)) {
        
			err = DB_RECORD_NOT_FOUND;
			goto func_exit;
		}
	}

	mtr_start(&mtr);

	/*-------------------------------------------------------------*/
	/* PHASE 2: Try fast adaptive hash index search if possible */

	/* Next test if this is the special case where we can use the fast
	adaptive hash index to try the search. Since we must release the
	search system latch when we retrieve an externally stored field, we
	cannot use the adaptive hash index in a search in the case the row
	may be long and there may be externally stored fields */

	if (UNIV_UNLIKELY(direction == 0)
	    && unique_search
	    && index->type & DICT_CLUSTERED
	    && !prebuilt->templ_contains_blob
	    && !prebuilt->used_in_HANDLER
	    && (prebuilt->mysql_row_len < UNIV_PAGE_SIZE / 8)) {

		mode = PAGE_CUR_GE;

		unique_search_from_clust_index = TRUE;

		if (trx->mysql_n_tables_locked == 0
		    && prebuilt->select_lock_type == LOCK_NONE
		    && trx->isolation_level > TRX_ISO_READ_UNCOMMITTED
		    && trx->read_view) {

			/* This is a SELECT query done as a consistent read,
			and the read view has already been allocated:
			let us try a search shortcut through the hash
			index.
			NOTE that we must also test that
			mysql_n_tables_locked == 0, because this might
			also be INSERT INTO ... SELECT ... or
			CREATE TABLE ... SELECT ... . Our algorithm is
			NOT prepared to inserts interleaved with the SELECT,
			and if we try that, we can deadlock on the adaptive
			hash index semaphore! */

#ifndef UNIV_SEARCH_DEBUG			
			if (!trx->has_search_latch) {
				rw_lock_s_lock(&btr_search_latch);
				trx->has_search_latch = TRUE;
			}
#endif
			switch (row_sel_try_search_shortcut_for_mysql(&rec,
					prebuilt, &offsets, &heap, &mtr)) {
			case SEL_FOUND:
#ifdef UNIV_SEARCH_DEBUG
				ut_a(0 == cmp_dtuple_rec(search_tuple,
							rec, offsets));
#endif 
				if (!row_sel_store_mysql_rec(buf, prebuilt,
							rec, offsets)) {
 					err = DB_TOO_BIG_RECORD;

					/* We let the main loop to do the
					error handling */
 					goto shortcut_fails_too_big_rec;
				}
	
 				mtr_commit(&mtr);

				/* ut_print_name(stderr, index->name);
				fputs(" shortcut\n", stderr); */

				srv_n_rows_read++;
				
				if (trx->search_latch_timeout > 0
				    && trx->has_search_latch) {

					trx->search_latch_timeout--;

			        	rw_lock_s_unlock(&btr_search_latch);
					trx->has_search_latch = FALSE;
				}    	
				
				/* NOTE that we do NOT store the cursor
				position */
				err = DB_SUCCESS;
				goto func_exit;

			case SEL_EXHAUSTED:
 				mtr_commit(&mtr);

				/* ut_print_name(stderr, index->name);
				fputs(" record not found 2\n", stderr); */

				if (trx->search_latch_timeout > 0
				    && trx->has_search_latch) {

					trx->search_latch_timeout--;

			        	rw_lock_s_unlock(&btr_search_latch);
					trx->has_search_latch = FALSE;
				}

				/* NOTE that we do NOT store the cursor
				position */

				err = DB_RECORD_NOT_FOUND;
				goto func_exit;
			}
shortcut_fails_too_big_rec:
			mtr_commit(&mtr);
			mtr_start(&mtr);
		}
	}

	/*-------------------------------------------------------------*/
	/* PHASE 3: Open or restore index cursor position */

	if (trx->has_search_latch) {
		rw_lock_s_unlock(&btr_search_latch);
		trx->has_search_latch = FALSE;
	}			

	trx_start_if_not_started(trx);

	if (trx->isolation_level <= TRX_ISO_READ_COMMITTED
	    && prebuilt->select_lock_type != LOCK_NONE
	    && trx->mysql_query_str) {

		/* Scan the MySQL query string; check if SELECT is the first
	        word there */
		ibool	success;

		dict_accept(*trx->mysql_query_str, "SELECT", &success);

		if (success) {
			/* It is a plain locking SELECT and the isolation
			level is low: do not lock gaps */

			set_also_gap_locks = FALSE;
		}
	}
	
	/* Note that if the search mode was GE or G, then the cursor
	naturally moves upward (in fetch next) in alphabetical order,
	otherwise downward */
	
	if (UNIV_UNLIKELY(direction == 0)) {
		if (mode == PAGE_CUR_GE || mode == PAGE_CUR_G) {
			moves_up = TRUE;
		}
	} else if (direction == ROW_SEL_NEXT) {
		moves_up = TRUE;
	}

	thr = que_fork_get_first_thr(prebuilt->sel_graph);

	que_thr_move_to_run_state_for_mysql(thr, trx);

	clust_index = dict_table_get_first_index(index->table);

	if (UNIV_LIKELY(direction != 0)) {
		if (!sel_restore_position_for_mysql(&same_user_rec,
						BTR_SEARCH_LEAF,
						pcur, moves_up, &mtr)) {
			goto next_rec;
		}

	} else if (dtuple_get_n_fields(search_tuple) > 0) {

		btr_pcur_open_with_no_init(index, search_tuple, mode,
					BTR_SEARCH_LEAF,
					pcur, 0, &mtr);

		pcur->trx_if_known = trx;
	} else {
		if (mode == PAGE_CUR_G) {
			btr_pcur_open_at_index_side(TRUE, index,
					BTR_SEARCH_LEAF, pcur, FALSE, &mtr);
		} else if (mode == PAGE_CUR_L) {
			btr_pcur_open_at_index_side(FALSE, index,
					BTR_SEARCH_LEAF, pcur, FALSE, &mtr);
		}
	}

	if (!prebuilt->sql_stat_start) {
		/* No need to set an intention lock or assign a read view */

		if (trx->read_view == NULL
		    && prebuilt->select_lock_type == LOCK_NONE) {

			fputs(
"InnoDB: Error: MySQL is trying to perform a consistent read\n"
"InnoDB: but the read view is not assigned!\n", stderr);
			trx_print(stderr, trx);
                        fputc('\n', stderr);
			ut_a(0);
		}
	} else if (prebuilt->select_lock_type == LOCK_NONE) {
		/* This is a consistent read */	
		/* Assign a read view for the query */

		trx_assign_read_view(trx);
		prebuilt->sql_stat_start = FALSE;
	} else {
		ulint	lock_mode;
		if (prebuilt->select_lock_type == LOCK_S) {		
			lock_mode = LOCK_IS;
		} else {
			lock_mode = LOCK_IX;
		}
		err = lock_table(0, index->table, lock_mode, thr);

		if (err != DB_SUCCESS) {

			goto lock_wait_or_error;
		}
		prebuilt->sql_stat_start = FALSE;
	}

rec_loop:
	/*-------------------------------------------------------------*/
	/* PHASE 4: Look for matching records in a loop */
	
	rec = btr_pcur_get_rec(pcur);
	ut_ad(!!page_rec_is_comp(rec) == index->table->comp);
#ifdef UNIV_SEARCH_DEBUG
/*
	fputs("Using ", stderr);
	dict_index_name_print(stderr, index);
	fprintf(stderr, " cnt %lu ; Page no %lu\n", cnt,
			buf_frame_get_page_no(buf_frame_align(rec)));
	rec_print(rec);
*/
#endif /* UNIV_SEARCH_DEBUG */

	if (page_rec_is_infimum(rec)) {

		/* The infimum record on a page cannot be in the result set,
		and neither can a record lock be placed on it: we skip such
		a record. */

		goto next_rec;
	}

	if (page_rec_is_supremum(rec)) {

		if (set_also_gap_locks
		    && !srv_locks_unsafe_for_binlog
		    && prebuilt->select_lock_type != LOCK_NONE) {

			/* Try to place a lock on the index record */

			/* If innodb_locks_unsafe_for_binlog option is used,
			we do not lock gaps. Supremum record is really
			a gap and therefore we do not set locks there. */
			

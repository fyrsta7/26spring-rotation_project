	if (page) {
		rec = page + offset;

		/* We do not need to reserve btr_search_latch, as the page
		is only being recovered, and there cannot be a hash index to
		it. Besides, these fields are being updated in place
		and the adaptive hash index does not depend on them. */

		btr_rec_set_deleted_flag(rec, page_zip, val);

		if (!(flags & BTR_KEEP_SYS_FLAG)) {
			mem_heap_t*	heap		= NULL;
			ulint		offsets_[REC_OFFS_NORMAL_SIZE];
			rec_offs_init(offsets_);

			row_upd_rec_sys_fields_in_recovery(
				rec, page_zip,
				rec_get_offsets(rec, index, offsets_,
						ULINT_UNDEFINED, &heap),
				pos, trx_id, roll_ptr);
			if (UNIV_LIKELY_NULL(heap)) {
				mem_heap_free(heap);
			}
		}
	}

	return(ptr);
}

#ifndef UNIV_HOTBACKUP
/***********************************************************//**
Marks a clustered index record deleted. Writes an undo log record to
undo log on this delete marking. Writes in the trx id field the id
of the deleting transaction, and in the roll ptr field pointer to the
undo log record created.
@return	DB_SUCCESS, DB_LOCK_WAIT, or error number */
UNIV_INTERN
dberr_t
btr_cur_del_mark_set_clust_rec(
/*===========================*/
	buf_block_t*	block,	/*!< in/out: buffer block of the record */
	rec_t*		rec,	/*!< in/out: record */
	dict_index_t*	index,	/*!< in: clustered index of the record */
	const ulint*	offsets,/*!< in: rec_get_offsets(rec) */
	que_thr_t*	thr,	/*!< in: query thread */
	mtr_t*		mtr)	/*!< in: mtr */
{
	roll_ptr_t	roll_ptr;
	dberr_t		err;
	page_zip_des_t*	page_zip;
	trx_t*		trx;

	ut_ad(dict_index_is_clust(index));
	ut_ad(rec_offs_validate(rec, index, offsets));
	ut_ad(!!page_rec_is_comp(rec) == dict_table_is_comp(index->table));
	ut_ad(buf_block_get_frame(block) == page_align(rec));
	ut_ad(page_is_leaf(page_align(rec)));

#ifdef UNIV_DEBUG
	if (btr_cur_print_record_ops && thr) {
		btr_cur_trx_report(thr_get_trx(thr)->id, index, "del mark ");
		rec_print_new(stderr, rec, offsets);
	}
#endif /* UNIV_DEBUG */

	ut_ad(dict_index_is_clust(index));
	ut_ad(!rec_get_deleted_flag(rec, rec_offs_comp(offsets)));

	err = lock_clust_rec_modify_check_and_lock(BTR_NO_LOCKING_FLAG, block,
						   rec, index, offsets, thr);

	if (err != DB_SUCCESS) {

		return(err);
	}

	err = trx_undo_report_row_operation(0, TRX_UNDO_MODIFY_OP, thr,
					    index, NULL, NULL, 0, rec, offsets,
					    &roll_ptr);
	if (err != DB_SUCCESS) {

		return(err);
	}

	/* The btr_search_latch is not needed here, because
	the adaptive hash index does not depend on the delete-mark
	and the delete-mark is being updated in place. */

	page_zip = buf_block_get_page_zip(block);

	btr_blob_dbg_set_deleted_flag(rec, index, offsets, TRUE);
	btr_rec_set_deleted_flag(rec, page_zip, TRUE);

	trx = thr_get_trx(thr);

	row_upd_rec_sys_fields(rec, page_zip, index, offsets, trx, roll_ptr);

	btr_cur_del_mark_set_clust_rec_log(rec, index, trx->id,
					   roll_ptr, mtr);

	return(err);
}

/****************************************************************//**
Writes the redo log record for a delete mark setting of a secondary
index record. */
UNIV_INLINE
void
btr_cur_del_mark_set_sec_rec_log(
/*=============================*/
	rec_t*		rec,	/*!< in: record */
	ibool		val,	/*!< in: value to set */
	mtr_t*		mtr)	/*!< in: mtr */
{
	byte*	log_ptr;
	ut_ad(val <= 1);

	log_ptr = mlog_open(mtr, 11 + 1 + 2);

	if (!log_ptr) {
		/* Logging in mtr is switched off during crash recovery:
		in that case mlog_open returns NULL */
		return;
	}

	log_ptr = mlog_write_initial_log_record_fast(
		rec, MLOG_REC_SEC_DELETE_MARK, log_ptr, mtr);
	mach_write_to_1(log_ptr, val);
	log_ptr++;

	mach_write_to_2(log_ptr, page_offset(rec));
	log_ptr += 2;

	mlog_close(mtr, log_ptr);
}
#endif /* !UNIV_HOTBACKUP */

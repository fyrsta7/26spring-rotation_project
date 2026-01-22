		if (undo_ptr->update_undo != NULL) {
			trx_undo_set_state_at_prepare(
				trx, undo_ptr->update_undo, false, &mtr);
		}

		mutex_exit(&rseg->mutex);

		/*--------------*/
		/* This mtr commit makes the transaction prepared in
		file-based world. */
		mtr_commit(&mtr);
		/*--------------*/

		lsn = mtr.commit_lsn();
		ut_ad(noredo_logging || lsn > 0);
	} else {
		lsn = 0;
	}

	return(lsn);
}

/****************************************************************//**
Prepares a transaction. */
static
void
trx_prepare(
/*========*/
	trx_t*	trx)	/*!< in/out: transaction */
{
	/* This transaction has crossed the point of no return and cannot
	be rolled back asynchronously now. It must commit or rollback
	synhronously. */

	lsn_t	lsn = 0;

	/* Only fresh user transactions can be prepared.
	Recovered transactions cannot. */
	ut_a(!trx->is_recovered);

	if (trx->rsegs.m_redo.rseg != NULL && trx_is_redo_rseg_updated(trx)) {

		lsn = trx_prepare_low(trx, &trx->rsegs.m_redo, false);
	}

	DBUG_EXECUTE_IF("ib_trx_crash_during_xa_prepare_step", DBUG_SUICIDE(););

	if (trx->rsegs.m_noredo.rseg != NULL
	    && trx_is_noredo_rseg_updated(trx)) {


		/*
		 * Force cache invalidation to happen outside of a valid transaction
		 * to prevent catalog access as we just caught an error.
		 */
		AbortCurrentTransaction();

		/* make sure there's no cache pollution */
		ReorderBufferExecuteInvalidations(rb, txn);

		if (using_subtxn)
			RollbackAndReleaseCurrentSubTransaction();

		if (snapshot_now->copied)
			ReorderBufferFreeSnap(rb, snapshot_now);

		/* remove potential on-disk data, and deallocate */
		ReorderBufferCleanupTXN(rb, txn);

		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * Abort a transaction that possibly has previous changes. Needs to be first
 * called for subtransactions and then for the toplevel xid.
 *
 * NB: Transactions handled here have to have actively aborted (i.e. have
 * produced an abort record). Implicitly aborted transactions are handled via
 * ReorderBufferAbortOld(); transactions we're just not interesteded in, but
 * which have committed are handled in ReorderBufferForget().
 *
 * This function purges this transaction and its contents from memory and
 * disk.
 */
void
ReorderBufferAbort(ReorderBuffer *rb, TransactionId xid, XLogRecPtr lsn)
{
	ReorderBufferTXN *txn;

	txn = ReorderBufferTXNByXid(rb, xid, false, NULL, InvalidXLogRecPtr,
								false);

	/* unknown, nothing to remove */
	if (txn == NULL)
		return;

	/* cosmetic... */
	txn->final_lsn = lsn;

	/* remove potential on-disk data, and deallocate */
	ReorderBufferCleanupTXN(rb, txn);
}

/*
 * Abort all transactions that aren't actually running anymore because the
 * server restarted.
 *
 * NB: These really have to be transactions that have aborted due to a server
 * crash/immediate restart, as we don't deal with invalidations here.
 */
void
ReorderBufferAbortOld(ReorderBuffer *rb, TransactionId oldestRunningXid)
{
	dlist_mutable_iter it;

	/*
	 * Iterate through all (potential) toplevel TXNs and abort all that are
	 * older than what possibly can be running. Once we've found the first
	 * that is alive we stop, there might be some that acquired an xid earlier
	 * but started writing later, but it's unlikely and they will cleaned up

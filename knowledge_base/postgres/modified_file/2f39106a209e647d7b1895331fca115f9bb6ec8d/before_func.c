
	LWLockAcquire(ProcArrayLock, LW_EXCLUSIVE);

	if (TransactionIdIsValid(latestXid))
	{
		Assert(TransactionIdIsValid(allPgXact[proc->pgprocno].xid));

		/* Advance global latestCompletedXid while holding the lock */
		if (TransactionIdPrecedes(ShmemVariableCache->latestCompletedXid,
								  latestXid))
			ShmemVariableCache->latestCompletedXid = latestXid;
	}
	else
	{
		/* Shouldn't be trying to remove a live transaction here */
		Assert(!TransactionIdIsValid(allPgXact[proc->pgprocno].xid));
	}

	for (index = 0; index < arrayP->numProcs; index++)
	{
		if (arrayP->pgprocnos[index] == proc->pgprocno)
		{
			/* Keep the PGPROC array sorted. See notes above */
			memmove(&arrayP->pgprocnos[index], &arrayP->pgprocnos[index + 1],
					(arrayP->numProcs - index - 1) * sizeof(int));
			arrayP->pgprocnos[arrayP->numProcs - 1] = -1;	/* for debugging */
			arrayP->numProcs--;
			LWLockRelease(ProcArrayLock);
			return;
		}
	}

	/* Oops */
	LWLockRelease(ProcArrayLock);

	elog(LOG, "failed to find proc %p in ProcArray", proc);
}


/*
 * ProcArrayEndTransaction -- mark a transaction as no longer running
 *
 * This is used interchangeably for commit and abort cases.  The transaction
 * commit/abort must already be reported to WAL and pg_xact.
 *
 * proc is currently always MyProc, but we pass it explicitly for flexibility.
 * latestXid is the latest Xid among the transaction's main XID and
 * subtransactions, or InvalidTransactionId if it has no XID.  (We must ask
 * the caller to pass latestXid, instead of computing it from the PGPROC's
 * contents, because the subxid information in the PGPROC might be
 * incomplete.)
 */
void
ProcArrayEndTransaction(PGPROC *proc, TransactionId latestXid)
{
	PGXACT	   *pgxact = &allPgXact[proc->pgprocno];

	if (TransactionIdIsValid(latestXid))
	{
		/*
		 * We must lock ProcArrayLock while clearing our advertised XID, so
		 * that we do not exit the set of "running" transactions while someone
		 * else is taking a snapshot.  See discussion in
		 * src/backend/access/transam/README.
		 */
		Assert(TransactionIdIsValid(allPgXact[proc->pgprocno].xid));

		/*
		 * If we can immediately acquire ProcArrayLock, we clear our own XID
		 * and release the lock.  If not, use group XID clearing to improve
		 * efficiency.
		 */
		if (LWLockConditionalAcquire(ProcArrayLock, LW_EXCLUSIVE))
		{
			ProcArrayEndTransactionInternal(proc, pgxact, latestXid);
			LWLockRelease(ProcArrayLock);
		}
		else
			ProcArrayGroupClearXid(proc, latestXid);
	}
	else
	{
		/*
		 * If we have no XID, we don't need to lock, since we won't affect
		 * anyone else's calculation of a snapshot.  We might change their
		 * estimate of global xmin, but that's OK.
		 */
		Assert(!TransactionIdIsValid(allPgXact[proc->pgprocno].xid));

		proc->lxid = InvalidLocalTransactionId;
		pgxact->xmin = InvalidTransactionId;
		/* must be cleared with xid/xmin: */
		pgxact->vacuumFlags &= ~PROC_VACUUM_STATE_MASK;
		pgxact->delayChkpt = false; /* be sure this is cleared in abort */
		proc->recoveryConflictPending = false;

		Assert(pgxact->nxids == 0);

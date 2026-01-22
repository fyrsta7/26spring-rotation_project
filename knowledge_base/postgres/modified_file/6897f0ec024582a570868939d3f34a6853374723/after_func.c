 * do about that --- data is only protected if the walsender runs continuously
 * while queries are executed on the standby.  (The Hot Standby code deals
 * with such cases by failing standby queries that needed to access
 * already-removed data, so there's no integrity bug.)
 *
 * Note: the approximate horizons (see definition of GlobalVisState) are
 * updated by the computations done here. That's currently required for
 * correctness and a small optimization. Without doing so it's possible that
 * heap vacuum's call to heap_page_prune_and_freeze() uses a more conservative
 * horizon than later when deciding which tuples can be removed - which the
 * code doesn't expect (breaking HOT).
 */
static void
ComputeXidHorizons(ComputeXidHorizonsResult *h)
{
	ProcArrayStruct *arrayP = procArray;
	TransactionId kaxmin;
	bool		in_recovery = RecoveryInProgress();
	TransactionId *other_xids = ProcGlobal->xids;

	/* inferred after ProcArrayLock is released */
	h->catalog_oldest_nonremovable = InvalidTransactionId;

	LWLockAcquire(ProcArrayLock, LW_SHARED);

	h->latest_completed = TransamVariables->latestCompletedXid;

	/*
	 * We initialize the MIN() calculation with latestCompletedXid + 1. This
	 * is a lower bound for the XIDs that might appear in the ProcArray later,
	 * and so protects us against overestimating the result due to future
	 * additions.
	 */
	{
		TransactionId initial;

		initial = XidFromFullTransactionId(h->latest_completed);
		Assert(TransactionIdIsValid(initial));
		TransactionIdAdvance(initial);

		h->oldest_considered_running = initial;
		h->shared_oldest_nonremovable = initial;
		h->data_oldest_nonremovable = initial;

		/*
		 * Only modifications made by this backend affect the horizon for
		 * temporary relations. Instead of a check in each iteration of the
		 * loop over all PGPROCs it is cheaper to just initialize to the
		 * current top-level xid any.
		 *
		 * Without an assigned xid we could use a horizon as aggressive as
		 * GetNewTransactionId(), but we can get away with the much cheaper
		 * latestCompletedXid + 1: If this backend has no xid there, by
		 * definition, can't be any newer changes in the temp table than
		 * latestCompletedXid.
		 */
		if (TransactionIdIsValid(MyProc->xid))
			h->temp_oldest_nonremovable = MyProc->xid;
		else
			h->temp_oldest_nonremovable = initial;
	}

	/*
	 * Fetch slot horizons while ProcArrayLock is held - the
	 * LWLockAcquire/LWLockRelease are a barrier, ensuring this happens inside
	 * the lock.
	 */
	h->slot_xmin = procArray->replication_slot_xmin;
	h->slot_catalog_xmin = procArray->replication_slot_catalog_xmin;

	for (int index = 0; index < arrayP->numProcs; index++)
	{
		int			pgprocno = arrayP->pgprocnos[index];
		PGPROC	   *proc = &allProcs[pgprocno];
		int8		statusFlags = ProcGlobal->statusFlags[index];
		TransactionId xid;
		TransactionId xmin;

		/* Fetch xid just once - see GetNewTransactionId */
		xid = UINT32_ACCESS_ONCE(other_xids[index]);
		xmin = UINT32_ACCESS_ONCE(proc->xmin);

		/*
		 * Consider both the transaction's Xmin, and its Xid.
		 *
		 * We must check both because a transaction might have an Xmin but not
		 * (yet) an Xid; conversely, if it has an Xid, that could determine
		 * some not-yet-set Xmin.
		 */
		xmin = TransactionIdOlder(xmin, xid);

		/* if neither is set, this proc doesn't influence the horizon */
		if (!TransactionIdIsValid(xmin))
			continue;

		/*
		 * Don't ignore any procs when determining which transactions might be
		 * considered running.  While slots should ensure logical decoding
		 * backends are protected even without this check, it can't hurt to
		 * include them here as well..
		 */
		h->oldest_considered_running =
			TransactionIdOlder(h->oldest_considered_running, xmin);

		/*
		 * Skip over backends either vacuuming (which is ok with rows being
		 * removed, as long as pg_subtrans is not truncated) or doing logical
		 * decoding (which manages xmin separately, check below).
		 */
		if (statusFlags & (PROC_IN_VACUUM | PROC_IN_LOGICAL_DECODING))
			continue;

		/* shared tables need to take backends in all databases into account */
		h->shared_oldest_nonremovable =
			TransactionIdOlder(h->shared_oldest_nonremovable, xmin);

		/*
		 * Normally sessions in other databases are ignored for anything but
		 * the shared horizon.
		 *
		 * However, include them when MyDatabaseId is not (yet) set.  A
		 * backend in the process of starting up must not compute a "too
		 * aggressive" horizon, otherwise we could end up using it to prune
		 * still-needed data away.  If the current backend never connects to a
		 * database this is harmless, because data_oldest_nonremovable will
		 * never be utilized.

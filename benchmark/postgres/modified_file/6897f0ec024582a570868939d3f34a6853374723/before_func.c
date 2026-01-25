/*
 * GetRunningTransactionData -- returns information about running transactions.
 *
 * Similar to GetSnapshotData but returns more information. We include
 * all PGPROCs with an assigned TransactionId, even VACUUM processes and
 * prepared transactions.
 *
 * We acquire XidGenLock and ProcArrayLock, but the caller is responsible for
 * releasing them. Acquiring XidGenLock ensures that no new XIDs enter the proc
 * array until the caller has WAL-logged this snapshot, and releases the
 * lock. Acquiring ProcArrayLock ensures that no transactions commit until the
 * lock is released.
 *
 * The returned data structure is statically allocated; caller should not
 * modify it, and must not assume it is valid past the next call.
 *
 * This is never executed during recovery so there is no need to look at
 * KnownAssignedXids.
 *
 * Dummy PGPROCs from prepared transaction are included, meaning that this
 * may return entries with duplicated TransactionId values coming from
 * transaction finishing to prepare.  Nothing is done about duplicated
 * entries here to not hold on ProcArrayLock more than necessary.
 *
 * We don't worry about updating other counters, we want to keep this as
 * simple as possible and leave GetSnapshotData() as the primary code for
 * that bookkeeping.
 *
 * Note that if any transaction has overflowed its cached subtransactions
 * then there is no real need include any subtransactions.
 */
RunningTransactions
GetRunningTransactionData(void)
{
	/* result workspace */
	static RunningTransactionsData CurrentRunningXactsData;

	ProcArrayStruct *arrayP = procArray;
	TransactionId *other_xids = ProcGlobal->xids;
	RunningTransactions CurrentRunningXacts = &CurrentRunningXactsData;
	TransactionId latestCompletedXid;
	TransactionId oldestRunningXid;
	TransactionId oldestDatabaseRunningXid;
	TransactionId *xids;
	int			index;
	int			count;
	int			subcount;
	bool		suboverflowed;

	Assert(!RecoveryInProgress());

	/*
	 * Allocating space for maxProcs xids is usually overkill; numProcs would
	 * be sufficient.  But it seems better to do the malloc while not holding
	 * the lock, so we can't look at numProcs.  Likewise, we allocate much
	 * more subxip storage than is probably needed.
	 *
	 * Should only be allocated in bgwriter, since only ever executed during
	 * checkpoints.
	 */
	if (CurrentRunningXacts->xids == NULL)
	{
		/*
		 * First call
		 */
		CurrentRunningXacts->xids = (TransactionId *)
			malloc(TOTAL_MAX_CACHED_SUBXIDS * sizeof(TransactionId));
		if (CurrentRunningXacts->xids == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("out of memory")));
	}

	xids = CurrentRunningXacts->xids;

	count = subcount = 0;
	suboverflowed = false;

	/*
	 * Ensure that no xids enter or leave the procarray while we obtain
	 * snapshot.
	 */
	LWLockAcquire(ProcArrayLock, LW_SHARED);
	LWLockAcquire(XidGenLock, LW_SHARED);

	latestCompletedXid =
		XidFromFullTransactionId(TransamVariables->latestCompletedXid);
	oldestDatabaseRunningXid = oldestRunningXid =
		XidFromFullTransactionId(TransamVariables->nextXid);

	/*
	 * Spin over procArray collecting all xids
	 */
	for (index = 0; index < arrayP->numProcs; index++)
	{
		int			pgprocno = arrayP->pgprocnos[index];
		PGPROC	   *proc = &allProcs[pgprocno];
		TransactionId xid;

		/* Fetch xid just once - see GetNewTransactionId */
		xid = UINT32_ACCESS_ONCE(other_xids[index]);

		/*
		 * We don't need to store transactions that don't have a TransactionId
		 * yet because they will not show as running on a standby server.
		 */
		if (!TransactionIdIsValid(xid))
			continue;

		/*
		 * Be careful not to exclude any xids before calculating the values of
		 * oldestRunningXid and suboverflowed, since these are used to clean
		 * up transaction information held on standbys.
		 */
		if (TransactionIdPrecedes(xid, oldestRunningXid))
			oldestRunningXid = xid;

		/*
		 * Also, update the oldest running xid within the current database.
		 */
		if (proc->databaseId == MyDatabaseId &&
			TransactionIdPrecedes(xid, oldestDatabaseRunningXid))
			oldestDatabaseRunningXid = xid;

		if (ProcGlobal->subxidStates[index].overflowed)
			suboverflowed = true;

		/*
		 * If we wished to exclude xids this would be the right place for it.
		 * Procs with the PROC_IN_VACUUM flag set don't usually assign xids,
		 * but they do during truncation at the end when they get the lock and
		 * truncate, so it is not much of a problem to include them if they
		 * are seen and it is cleaner to include them.
		 */

		xids[count++] = xid;
	}

	/*
	 * Spin over procArray collecting all subxids, but only if there hasn't
	 * been a suboverflow.
	 */
	if (!suboverflowed)
	{
		XidCacheStatus *other_subxidstates = ProcGlobal->subxidStates;

		for (index = 0; index < arrayP->numProcs; index++)
		{
			int			pgprocno = arrayP->pgprocnos[index];
			PGPROC	   *proc = &allProcs[pgprocno];
			int			nsubxids;

			/*
			 * Save subtransaction XIDs. Other backends can't add or remove
			 * entries while we're holding XidGenLock.
			 */
			nsubxids = other_subxidstates[index].count;
			if (nsubxids > 0)
			{
				/* barrier not really required, as XidGenLock is held, but ... */
				pg_read_barrier();	/* pairs with GetNewTransactionId */

				memcpy(&xids[count], proc->subxids.xids,
					   nsubxids * sizeof(TransactionId));
				count += nsubxids;
				subcount += nsubxids;

				/*
				 * Top-level XID of a transaction is always less than any of
				 * its subxids, so we don't need to check if any of the
				 * subxids are smaller than oldestRunningXid
				 */
			}
		}
	}

	/*
	 * It's important *not* to include the limits set by slots here because
	 * snapbuild.c uses oldestRunningXid to manage its xmin horizon. If those
	 * were to be included here the initial value could never increase because
	 * of a circular dependency where slots only increase their limits when
	 * running xacts increases oldestRunningXid and running xacts only
	 * increases if slots do.
	 */

	CurrentRunningXacts->xcnt = count - subcount;
	CurrentRunningXacts->subxcnt = subcount;
	CurrentRunningXacts->subxid_status = suboverflowed ? SUBXIDS_IN_SUBTRANS : SUBXIDS_IN_ARRAY;
	CurrentRunningXacts->nextXid = XidFromFullTransactionId(TransamVariables->nextXid);
	CurrentRunningXacts->oldestRunningXid = oldestRunningXid;
	CurrentRunningXacts->oldestDatabaseRunningXid = oldestDatabaseRunningXid;
	CurrentRunningXacts->latestCompletedXid = latestCompletedXid;

	Assert(TransactionIdIsValid(CurrentRunningXacts->nextXid));
	Assert(TransactionIdIsValid(CurrentRunningXacts->oldestRunningXid));
	Assert(TransactionIdIsNormal(CurrentRunningXacts->latestCompletedXid));

	/* We don't release the locks here, the caller is responsible for that */

	return CurrentRunningXacts;
}
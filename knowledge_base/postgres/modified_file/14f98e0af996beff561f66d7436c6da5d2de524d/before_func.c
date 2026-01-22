static void
RelationAddExtraBlocks(Relation relation, BulkInsertState bistate)
{
	BlockNumber blockNum,
				firstBlock = InvalidBlockNumber;
	int			extraBlocks;
	int			lockWaiters;

	/* Use the length of the lock wait queue to judge how much to extend. */
	lockWaiters = RelationExtensionLockWaiterCount(relation);
	if (lockWaiters <= 0)
		return;

	/*
	 * It might seem like multiplying the number of lock waiters by as much as
	 * 20 is too aggressive, but benchmarking revealed that smaller numbers
	 * were insufficient.  512 is just an arbitrary cap to prevent
	 * pathological results.
	 */
	extraBlocks = Min(512, lockWaiters * 20);

	do
	{
		Buffer		buffer;
		Page		page;
		Size		freespace;

		/*
		 * Extend by one page.  This should generally match the main-line
		 * extension code in RelationGetBufferForTuple, except that we hold
		 * the relation extension lock throughout, and we don't immediately
		 * initialize the page (see below).
		 */
		buffer = ReadBufferBI(relation, P_NEW, RBM_ZERO_AND_LOCK, bistate);
		page = BufferGetPage(buffer);

		if (!PageIsNew(page))
			elog(ERROR, "page %u of relation \"%s\" should be empty but is not",
				 BufferGetBlockNumber(buffer),
				 RelationGetRelationName(relation));

		/*
		 * Add the page to the FSM without initializing. If we were to
		 * initialize here, the page would potentially get flushed out to disk
		 * before we add any useful content. There's no guarantee that that'd
		 * happen before a potential crash, so we need to deal with
		 * uninitialized pages anyway, thus avoid the potential for
		 * unnecessary writes.
		 */

		/* we'll need this info below */
		blockNum = BufferGetBlockNumber(buffer);
		freespace = BufferGetPageSize(buffer) - SizeOfPageHeaderData;

		UnlockReleaseBuffer(buffer);

		/* Remember first block number thus added. */
		if (firstBlock == InvalidBlockNumber)
			firstBlock = blockNum;

		/*
		 * Immediately update the bottom level of the FSM.  This has a good
		 * chance of making this page visible to other concurrently inserting
		 * backends, and we want that to happen without delay.
		 */
		RecordPageWithFreeSpace(relation, blockNum, freespace);
	}
	while (--extraBlocks > 0);

	/*
	 * Updating the upper levels of the free space map is too expensive to do
	 * for every block, but it's worth doing once at the end to make sure that
	 * subsequent insertion activity sees all of those nifty free pages we
	 * just inserted.
	 */
	FreeSpaceMapVacuumRange(relation, firstBlock, blockNum + 1);
}

/*
 * RelationGetBufferForTuple
 *
 *	Returns pinned and exclusive-locked buffer of a page in given relation
 *	with free space >= given len.
 *
 *	If otherBuffer is not InvalidBuffer, then it references a previously
 *	pinned buffer of another page in the same relation; on return, this
 *	buffer will also be exclusive-locked.  (This case is used by heap_update;
 *	the otherBuffer contains the tuple being updated.)
 *
 *	The reason for passing otherBuffer is that if two backends are doing
 *	concurrent heap_update operations, a deadlock could occur if they try
 *	to lock the same two buffers in opposite orders.  To ensure that this
 *	can't happen, we impose the rule that buffers of a relation must be
 *	locked in increasing page number order.  This is most conveniently done
 *	by having RelationGetBufferForTuple lock them both, with suitable care
 *	for ordering.
 *
 *	NOTE: it is unlikely, but not quite impossible, for otherBuffer to be the
 *	same buffer we select for insertion of the new tuple (this could only
 *	happen if space is freed in that page after heap_update finds there's not
 *	enough there).  In that case, the page will be pinned and locked only once.
 *
 *	We also handle the possibility that the all-visible flag will need to be
 *	cleared on one or both pages.  If so, pin on the associated visibility map
 *	page must be acquired before acquiring buffer lock(s), to avoid possibly
 *	doing I/O while holding buffer locks.  The pins are passed back to the
 *	caller using the input-output arguments vmbuffer and vmbuffer_other.
 *	Note that in some cases the caller might have already acquired such pins,
 *	which is indicated by these arguments not being InvalidBuffer on entry.
 *
 *	We normally use FSM to help us find free space.  However,
 *	if HEAP_INSERT_SKIP_FSM is specified, we just append a new empty page to
 *	the end of the relation if the tuple won't fit on the current target page.
 *	This can save some cycles when we know the relation is new and doesn't
 *	contain useful amounts of free space.
 *
 *	HEAP_INSERT_SKIP_FSM is also useful for non-WAL-logged additions to a
 *	relation, if the caller holds exclusive lock and is careful to invalidate
 *	relation's smgr_targblock before the first insertion --- that ensures that
 *	all insertions will occur into newly added pages and not be intermixed
 *	with tuples from other transactions.  That way, a crash can't risk losing
 *	any committed data of other transactions.  (See heap_insert's comments
 *	for additional constraints needed for safe usage of this behavior.)
 *
 *	The caller can also provide a BulkInsertState object to optimize many
 *	insertions into the same relation.  This keeps a pin on the current
 *	insertion target page (to save pin/unpin cycles) and also passes a
 *	BULKWRITE buffer selection strategy object to the buffer manager.
 *	Passing NULL for bistate selects the default behavior.
 *
 *	We don't fill existing pages further than the fillfactor, except for large
 *	tuples in nearly-empty pages.  This is OK since this routine is not
 *	consulted when updating a tuple and keeping it on the same page, which is
 *	the scenario fillfactor is meant to reserve space for.
 *
 *	ereport(ERROR) is allowed here, so this routine *must* be called
 *	before any (unlogged) changes are made in buffer pool.
 */
Buffer
RelationGetBufferForTuple(Relation relation, Size len,
						  Buffer otherBuffer, int options,
						  BulkInsertState bistate,
						  Buffer *vmbuffer, Buffer *vmbuffer_other)
{
	bool		use_fsm = !(options & HEAP_INSERT_SKIP_FSM);
	Buffer		buffer = InvalidBuffer;
	Page		page;
	Size		nearlyEmptyFreeSpace,
				pageFreeSpace = 0,
				saveFreeSpace = 0,
				targetFreeSpace = 0;
	BlockNumber targetBlock,
				otherBlock;
	bool		needLock;

	len = MAXALIGN(len);		/* be conservative */

	/* Bulk insert is not supported for updates, only inserts. */
	Assert(otherBuffer == InvalidBuffer || !bistate);

	/*
	 * If we're gonna fail for oversize tuple, do it right away
	 */
	if (len > MaxHeapTupleSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("row is too big: size %zu, maximum size %zu",
						len, MaxHeapTupleSize)));

	/* Compute desired extra freespace due to fillfactor option */
	saveFreeSpace = RelationGetTargetPageFreeSpace(relation,
												   HEAP_DEFAULT_FILLFACTOR);

	/*
	 * Since pages without tuples can still have line pointers, we consider
	 * pages "empty" when the unavailable space is slight.  This threshold is
	 * somewhat arbitrary, but it should prevent most unnecessary relation
	 * extensions while inserting large tuples into low-fillfactor tables.
	 */
	nearlyEmptyFreeSpace = MaxHeapTupleSize -
		(MaxHeapTuplesPerPage / 8 * sizeof(ItemIdData));
	if (len + saveFreeSpace > nearlyEmptyFreeSpace)
		targetFreeSpace = Max(len, nearlyEmptyFreeSpace);
	else
		targetFreeSpace = len + saveFreeSpace;

	if (otherBuffer != InvalidBuffer)
		otherBlock = BufferGetBlockNumber(otherBuffer);
	else
		otherBlock = InvalidBlockNumber;	/* just to keep compiler quiet */

	/*
	 * We first try to put the tuple on the same page we last inserted a tuple
	 * on, as cached in the BulkInsertState or relcache entry.  If that
	 * doesn't work, we ask the Free Space Map to locate a suitable page.
	 * Since the FSM's info might be out of date, we have to be prepared to
	 * loop around and retry multiple times. (To insure this isn't an infinite
	 * loop, we must update the FSM with the correct amount of free space on
	 * each page that proves not to be suitable.)  If the FSM has no record of
	 * a page with enough free space, we give up and extend the relation.
	 *
	 * When use_fsm is false, we either put the tuple onto the existing target
	 * page or extend the relation.
	 */
	if (bistate && bistate->current_buf != InvalidBuffer)
		targetBlock = BufferGetBlockNumber(bistate->current_buf);
	else
		targetBlock = RelationGetTargetBlock(relation);

	if (targetBlock == InvalidBlockNumber && use_fsm)
	{
		/*
		 * We have no cached target page, so ask the FSM for an initial
		 * target.
		 */
		targetBlock = GetPageWithFreeSpace(relation, targetFreeSpace);
	}

	/*
	 * If the FSM knows nothing of the rel, try the last page before we give
	 * up and extend.  This avoids one-tuple-per-page syndrome during
	 * bootstrapping or in a recently-started system.
	 */
	if (targetBlock == InvalidBlockNumber)
	{
		BlockNumber nblocks = RelationGetNumberOfBlocks(relation);

		if (nblocks > 0)
			targetBlock = nblocks - 1;
	}

loop:
	while (targetBlock != InvalidBlockNumber)
	{
		/*
		 * Read and exclusive-lock the target block, as well as the other
		 * block if one was given, taking suitable care with lock ordering and
		 * the possibility they are the same block.
		 *
		 * If the page-level all-visible flag is set, caller will need to
		 * clear both that and the corresponding visibility map bit.  However,
		 * by the time we return, we'll have x-locked the buffer, and we don't
		 * want to do any I/O while in that state.  So we check the bit here
		 * before taking the lock, and pin the page if it appears necessary.
		 * Checking without the lock creates a risk of getting the wrong
		 * answer, so we'll have to recheck after acquiring the lock.
		 */
		if (otherBuffer == InvalidBuffer)
		{
			/* easy case */
			buffer = ReadBufferBI(relation, targetBlock, RBM_NORMAL, bistate);
			if (PageIsAllVisible(BufferGetPage(buffer)))
				visibilitymap_pin(relation, targetBlock, vmbuffer);

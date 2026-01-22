
	/* ----------------
	 *	increment access statistics
	 * ----------------
	 */
	IncrHeapAccessStat(local_RelationPutHeapTuple);
	IncrHeapAccessStat(global_RelationPutHeapTuple);

	pageHeader = (Page) BufferGetPage(buffer);
	len = (unsigned) DOUBLEALIGN(tuple->t_len); /* be conservative */
	Assert((int) len <= PageGetFreeSpace(pageHeader));

	offnum = PageAddItem((Page) pageHeader, (Item) tuple->t_data,
						 tuple->t_len, InvalidOffsetNumber, LP_USED);

	itemId = PageGetItemId((Page) pageHeader, offnum);
	item = PageGetItem((Page) pageHeader, itemId);

	ItemPointerSet(&((HeapTupleHeader) item)->t_ctid, 
					BufferGetBlockNumber(buffer), offnum);

	/*
	 * Let the caller do this!
	 *
	WriteBuffer(buffer);
	 */

	/* return an accurate tuple */
	ItemPointerSet(&tuple->t_self, BufferGetBlockNumber(buffer), offnum);
}

/*
 * This routine is another in the series of attempts to reduce the number
 * of I/O's and system calls executed in the various benchmarks.  In
 * particular, this routine is used to append data to the end of a relation
 * file without excessive lseeks.  This code should do no more than 2 semops
 * in the ideal case.
 *
 * Eventually, we should cache the number of blocks in a relation somewhere.
 * Until that time, this code will have to do an lseek to determine the number
 * of blocks in a relation.
 *
 * This code should ideally do at most 4 semops, 1 lseek, and possibly 1 write
 * to do an append; it's possible to eliminate 2 of the semops if we do direct
 * buffer stuff (!); the lseek and the write can go if we get
 * RelationGetNumberOfBlocks to be useful.
 *
 * NOTE: This code presumes that we have a write lock on the relation.
 * Not now - we use extend locking...
 *
 * Also note that this routine probably shouldn't have to exist, and does
 * screw up the call graph rather badly, but we are wasting so much time and
 * system resources being massively general that we are losing badly in our
 * performance benchmarks.
 */
void
RelationPutHeapTupleAtEnd(Relation relation, HeapTuple tuple)
{
	Buffer		buffer;
	Page		pageHeader;
	BlockNumber lastblock;
	OffsetNumber offnum;
	unsigned int len;
	ItemId		itemId;
	Item		item;

	if (!relation->rd_myxactonly)
		LockRelation(relation, ExtendLock);


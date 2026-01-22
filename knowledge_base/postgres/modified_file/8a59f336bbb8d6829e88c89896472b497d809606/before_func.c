	/* FSMChunk objects */
	nchunks = (MaxFSMPages - 1) / CHUNKPAGES + 1;

	size += MAXALIGN(nchunks * sizeof(FSMChunk));

	return size;
}

/*
 * GetPageWithFreeSpace - try to find a page in the given relation with
 *		at least the specified amount of free space.
 *
 * If successful, return the block number; if not, return InvalidBlockNumber.
 *
 * The caller must be prepared for the possibility that the returned page
 * will turn out to have too little space available by the time the caller
 * gets a lock on it.  In that case, the caller should report the actual
 * amount of free space available on that page (via RecordFreeSpace) and
 * then try again.  If InvalidBlockNumber is returned, extend the relation.
 */
BlockNumber
GetPageWithFreeSpace(RelFileNode *rel, Size spaceNeeded)
{
	FSMRelation *fsmrel;
	BlockNumber	freepage;

	SpinAcquire(FreeSpaceLock);
	/*
	 * We always add a rel to the hashtable when it is inquired about.
	 */
	fsmrel = create_fsm_rel(rel);
	/*
	 * Adjust the threshold towards the space request.  This essentially
	 * implements an exponential moving average with an equivalent period
	 * of about 63 requests.  Ignore silly requests, however, to ensure
	 * that the average stays in bounds.
	 *
	 * In theory, if the threshold increases here we should immediately
	 * delete any pages that fall below the new threshold.  In practice
	 * it seems OK to wait until we have a need to compact space.
	 */
	if (spaceNeeded > 0 && spaceNeeded < BLCKSZ)
	{
		int		cur_avg = (int) fsmrel->threshold;

		cur_avg += ((int) spaceNeeded - cur_avg) / 32;
		fsmrel->threshold = (Size) cur_avg;
	}
	freepage = find_free_space(fsmrel, spaceNeeded);
	SpinRelease(FreeSpaceLock);
	return freepage;

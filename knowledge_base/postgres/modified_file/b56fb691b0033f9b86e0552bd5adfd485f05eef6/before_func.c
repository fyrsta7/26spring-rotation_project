 * Obtain next data block number in the forward direction, or -1L if no more.
 *
 * Unless 'frozen' is true, release indirect blocks to the free pool after
 * reading them.
 */
static long
ltsRecallNextBlockNum(LogicalTapeSet *lts,
					  IndirectBlock *indirect,
					  bool frozen)
{
	/* Handle case of never-written-to tape */
	if (indirect == NULL)
		return -1L;

	if (indirect->nextSlot >= BLOCKS_PER_INDIR_BLOCK ||
		indirect->ptrs[indirect->nextSlot] == -1L)
	{
		long		indirblock;

		if (indirect->nextup == NULL)
			return -1L;			/* nothing left at this level */
		indirblock = ltsRecallNextBlockNum(lts, indirect->nextup, frozen);
		if (indirblock == -1L)
			return -1L;			/* nothing left at this level */
		ltsReadBlock(lts, indirblock, (void *) indirect->ptrs);
		if (!frozen)
			ltsReleaseBlock(lts, indirblock);
		indirect->nextSlot = 0;
	}
	if (indirect->ptrs[indirect->nextSlot] == -1L)
		return -1L;
	return indirect->ptrs[indirect->nextSlot++];
}

/*
 * Obtain next data block number in the reverse direction, or -1L if no more.
 *
 * Note this fetches the block# before the one last returned, no matter which
 * direction of call returned that one.  If we fail, no change in state.
 *
 * This routine can only be used in 'frozen' state, so there's no need to
 * pass a parameter telling whether to release blocks ... we never do.
 */
static long
ltsRecallPrevBlockNum(LogicalTapeSet *lts,
					  IndirectBlock *indirect)
{
	/* Handle case of never-written-to tape */
	if (indirect == NULL)
		return -1L;

	if (indirect->nextSlot <= 1)
	{
		long		indirblock;

		if (indirect->nextup == NULL)
			return -1L;			/* nothing left at this level */
		indirblock = ltsRecallPrevBlockNum(lts, indirect->nextup);
		if (indirblock == -1L)
			return -1L;			/* nothing left at this level */
		ltsReadBlock(lts, indirblock, (void *) indirect->ptrs);

		/*
		 * The previous block would only have been written out if full, so we
		 * need not search it for a -1 sentinel.
		 */
		indirect->nextSlot = BLOCKS_PER_INDIR_BLOCK + 1;
	}
	indirect->nextSlot--;
	return indirect->ptrs[indirect->nextSlot - 1];
}


/*

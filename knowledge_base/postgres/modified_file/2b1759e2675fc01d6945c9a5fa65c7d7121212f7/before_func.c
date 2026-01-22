 * true, the FSM file is extended.
 */
static Buffer
fsm_readbuf(Relation rel, FSMAddress addr, bool extend)
{
	BlockNumber blkno = fsm_logical_to_physical(addr);
	Buffer		buf;

	RelationOpenSmgr(rel);

	/*
	 * If we haven't cached the size of the FSM yet, check it first.  Also
	 * recheck if the requested block seems to be past end, since our cached
	 * value might be stale.  (We send smgr inval messages on truncation, but
	 * not on extension.)
	 */
	if (rel->rd_smgr->smgr_fsm_nblocks == InvalidBlockNumber ||
		blkno >= rel->rd_smgr->smgr_fsm_nblocks)
	{
		if (smgrexists(rel->rd_smgr, FSM_FORKNUM))
			rel->rd_smgr->smgr_fsm_nblocks = smgrnblocks(rel->rd_smgr,
														 FSM_FORKNUM);
		else
			rel->rd_smgr->smgr_fsm_nblocks = 0;
	}

	/* Handle requests beyond EOF */
	if (blkno >= rel->rd_smgr->smgr_fsm_nblocks)
	{
		if (extend)
			fsm_extend(rel, blkno + 1);
		else
			return InvalidBuffer;
	}

	/*
	 * Use ZERO_ON_ERROR mode, and initialize the page if necessary. The FSM
	 * information is not accurate anyway, so it's better to clear corrupt
	 * pages than error out. Since the FSM changes are not WAL-logged, the
	 * so-called torn page problem on crash can lead to pages with corrupt
	 * headers, for example.
	 */
	buf = ReadBufferExtended(rel, FSM_FORKNUM, blkno, RBM_ZERO_ON_ERROR, NULL);
	if (PageIsNew(BufferGetPage(buf)))
		PageInit(BufferGetPage(buf), BLCKSZ, 0);
	return buf;
}

/*
 * Ensure that the FSM fork is at least fsm_nblocks long, extending
 * it if necessary with empty pages. And by empty, I mean pages filled
 * with zeros, meaning there's no free space.
 */
static void
fsm_extend(Relation rel, BlockNumber fsm_nblocks)
{
	BlockNumber fsm_nblocks_now;
	Page		pg;

	pg = (Page) palloc(BLCKSZ);
	PageInit(pg, BLCKSZ, 0);

	/*
	 * We use the relation extension lock to lock out other backends trying to
	 * extend the FSM at the same time. It also locks out extension of the
	 * main fork, unnecessarily, but extending the FSM happens seldom enough
	 * that it doesn't seem worthwhile to have a separate lock tag type for
	 * it.
	 *
	 * Note that another backend might have extended or created the relation
	 * by the time we get the lock.
	 */
	LockRelationForExtension(rel, ExclusiveLock);

	/* Might have to re-open if a cache flush happened */
	RelationOpenSmgr(rel);

	/*
	 * Create the FSM file first if it doesn't exist.  If smgr_fsm_nblocks is
	 * positive then it must exist, no need for an smgrexists call.
	 */
	if ((rel->rd_smgr->smgr_fsm_nblocks == 0 ||
		 rel->rd_smgr->smgr_fsm_nblocks == InvalidBlockNumber) &&
		!smgrexists(rel->rd_smgr, FSM_FORKNUM))
		smgrcreate(rel->rd_smgr, FSM_FORKNUM, false);

	fsm_nblocks_now = smgrnblocks(rel->rd_smgr, FSM_FORKNUM);

	while (fsm_nblocks_now < fsm_nblocks)
	{
		PageSetChecksumInplace(pg, fsm_nblocks_now);

		smgrextend(rel->rd_smgr, FSM_FORKNUM, fsm_nblocks_now,

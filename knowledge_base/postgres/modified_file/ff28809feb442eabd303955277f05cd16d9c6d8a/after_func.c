	/* NB: this had better match element_alloc() */
	elementSize = MAXALIGN(sizeof(HASHELEMENT)) + MAXALIGN(entrysize);

	/*
	 * The idea here is to choose nelem_alloc at least 32, but round up so
	 * that the allocation request will be a power of 2 or just less. This
	 * makes little difference for hash tables in shared memory, but for hash
	 * tables managed by palloc, the allocation request will be rounded up to
	 * a power of 2 anyway.  If we fail to take this into account, we'll waste
	 * as much as half the allocated space.
	 */
	allocSize = 32 * 4;			/* assume elementSize at least 8 */
	do
	{
		allocSize <<= 1;
		nelem_alloc = allocSize / elementSize;
	} while (nelem_alloc < 32);

	return nelem_alloc;
}

/*
 * Compute derived fields of hctl and build the initial directory/segment
 * arrays
 */
static bool
init_htab(HTAB *hashp, long nelem)
{
	HASHHDR    *hctl = hashp->hctl;
	HASHSEGMENT *segp;
	int			nbuckets;
	int			nsegs;
	int			i;

	/*
	 * initialize mutexes if it's a partitioned table
	 */
	if (IS_PARTITIONED(hctl))
		for (i = 0; i < NUM_FREELISTS; i++)
			SpinLockInit(&(hctl->freeList[i].mutex));

	/*
	 * Allocate space for the next greater power of two number of buckets,
	 * assuming a desired maximum load factor of 1.
	 */
	nbuckets = next_pow2_int(nelem);

	/*
	 * In a partitioned table, nbuckets must be at least equal to
	 * num_partitions; were it less, keys with apparently different partition
	 * numbers would map to the same bucket, breaking partition independence.
	 * (Normally nbuckets will be much bigger; this is just a safety check.)
	 */
	while (nbuckets < hctl->num_partitions)
		nbuckets <<= 1;

	hctl->max_bucket = hctl->low_mask = nbuckets - 1;
	hctl->high_mask = (nbuckets << 1) - 1;

	/*
	 * Figure number of directory segments needed, round up to a power of 2
	 */
	nsegs = (nbuckets - 1) / hctl->ssize + 1;
	nsegs = next_pow2_int(nsegs);

	/*
	 * Make sure directory is big enough. If pre-allocated directory is too
	 * small, choke (caller screwed up).
	 */
	if (nsegs > hctl->dsize)
	{
		if (!(hashp->dir))
			hctl->dsize = nsegs;
		else
			return false;
	}

	/* Allocate a directory */
	if (!(hashp->dir))
	{
		CurrentDynaHashCxt = hashp->hcxt;
		hashp->dir = (HASHSEGMENT *)
			hashp->alloc(hctl->dsize * sizeof(HASHSEGMENT));
		if (!hashp->dir)
			return false;
	}

	/* Allocate initial segments */
	for (segp = hashp->dir; hctl->nsegs < nsegs; hctl->nsegs++, segp++)
	{
		*segp = seg_alloc(hashp);
		if (*segp == NULL)
			return false;
	}

	/* Choose number of entries to allocate at a time */
	hctl->nelem_alloc = choose_nelem_alloc(hctl->entrysize);

#ifdef HASH_DEBUG
	fprintf(stderr, "init_htab:\n%s%p\n%s%ld\n%s%ld\n%s%d\n%s%ld\n%s%u\n%s%x\n%s%x\n%s%ld\n",
			"TABLE POINTER   ", hashp,
			"DIRECTORY SIZE  ", hctl->dsize,
			"SEGMENT SIZE    ", hctl->ssize,
			"SEGMENT SHIFT   ", hctl->sshift,
			"MAX BUCKET      ", hctl->max_bucket,
			"HIGH MASK       ", hctl->high_mask,
			"LOW  MASK       ", hctl->low_mask,
			"NSEGS           ", hctl->nsegs);
#endif
	return true;
}

/*
 * Estimate the space needed for a hashtable containing the given number
 * of entries of given size.
 * NOTE: this is used to estimate the footprint of hashtables in shared
 * memory; therefore it does not count HTAB which is in local memory.
 * NB: assumes that all hash structure parameters have default values!
 */
Size
hash_estimate_size(long num_entries, Size entrysize)
{
	Size		size;
	long		nBuckets,
				nSegments,
				nDirEntries,
				nElementAllocs,
				elementSize,
				elementAllocCnt;

	/* estimate number of buckets wanted */
	nBuckets = next_pow2_long(num_entries);
	/* # of segments needed for nBuckets */
	nSegments = next_pow2_long((nBuckets - 1) / DEF_SEGSIZE + 1);
	/* directory entries */
	nDirEntries = DEF_DIRSIZE;
	while (nDirEntries < nSegments)
		nDirEntries <<= 1;		/* dir_alloc doubles dsize at each call */

	/* fixed control info */
	size = MAXALIGN(sizeof(HASHHDR));	/* but not HTAB, per above */
	/* directory */
	size = add_size(size, mul_size(nDirEntries, sizeof(HASHSEGMENT)));
	/* segments */
	size = add_size(size, mul_size(nSegments,
								   MAXALIGN(DEF_SEGSIZE * sizeof(HASHBUCKET))));
	/* elements --- allocated in groups of choose_nelem_alloc() entries */
	elementAllocCnt = choose_nelem_alloc(entrysize);
	nElementAllocs = (num_entries - 1) / elementAllocCnt + 1;
	elementSize = MAXALIGN(sizeof(HASHELEMENT)) + MAXALIGN(entrysize);

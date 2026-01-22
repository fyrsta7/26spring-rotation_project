#define PROCLOCK_PRINT(where, proclockP)
#endif   /* not LOCK_DEBUG */


static void RemoveLocalLock(LOCALLOCK *locallock);
static void GrantLockLocal(LOCALLOCK *locallock, ResourceOwner owner);
static int WaitOnLock(LOCKMETHODID lockmethodid, LOCALLOCK *locallock,
		   ResourceOwner owner);
static void LockCountMyLocks(SHMEM_OFFSET lockOffset, PGPROC *proc,
				 int *myHolding);


/*
 * InitLocks -- Init the lock module.  Create a private data
 *		structure for constructing conflict masks.
 */
void
InitLocks(void)
{
	/* NOP */
}


/*
 * Fetch the lock method table associated with a given lock
 */
LockMethod
GetLocksMethodTable(LOCK *lock)
{
	LOCKMETHODID lockmethodid = LOCK_LOCKMETHOD(*lock);

	Assert(0 < lockmethodid && lockmethodid < NumLockMethods);
	return LockMethods[lockmethodid];
}


/*
 * LockMethodInit -- initialize the lock table's lock type
 *		structures
 *
 * Notes: just copying.  Should only be called once.
 */
static void
LockMethodInit(LockMethod lockMethodTable,
			   const LOCKMASK *conflictsP,
			   int numModes)
{
	int			i;

	lockMethodTable->numLockModes = numModes;
	/* copies useless zero element as well as the N lockmodes */
	for (i = 0; i <= numModes; i++)
		lockMethodTable->conflictTab[i] = conflictsP[i];
}

/*
 * LockMethodTableInit -- initialize a lock table structure
 *
 * NOTE: data structures allocated here are allocated permanently, using
 * TopMemoryContext and shared memory.	We don't ever release them anyway,
 * and in normal multi-backend operation the lock table structures set up
 * by the postmaster are inherited by each backend, so they must be in
 * TopMemoryContext.
 */
LOCKMETHODID
LockMethodTableInit(const char *tabName,
					const LOCKMASK *conflictsP,
					int numModes,
					int maxBackends)
{
	LockMethod	newLockMethod;
	LOCKMETHODID lockmethodid;
	char	   *shmemName;
	HASHCTL		info;
	int			hash_flags;
	bool		found;
	long		init_table_size,
				max_table_size;

	if (numModes >= MAX_LOCKMODES)
		elog(ERROR, "too many lock types %d (limit is %d)",
			 numModes, MAX_LOCKMODES - 1);

	/* Compute init/max size to request for lock hashtables */
	max_table_size = NLOCKENTS(maxBackends);
	init_table_size = max_table_size / 2;

	/* Allocate a string for the shmem index table lookups. */
	/* This is just temp space in this routine, so palloc is OK. */
	shmemName = (char *) palloc(strlen(tabName) + 32);

	/* each lock table has a header in shared memory */
	sprintf(shmemName, "%s (lock method table)", tabName);
	newLockMethod = (LockMethod)
		ShmemInitStruct(shmemName, sizeof(LockMethodData), &found);

	if (!newLockMethod)
		elog(FATAL, "could not initialize lock table \"%s\"", tabName);

	/*

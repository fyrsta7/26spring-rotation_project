		 * because the caller, WaitOnLock(), has already reported that.
		 */
		ResolveRecoveryConflictWithVirtualXIDs(backends,
											   PROCSIG_RECOVERY_CONFLICT_LOCK,
											   PG_WAIT_LOCK | locktag.locktag_type,
											   false);
	}
	else
	{
		/*
		 * Wait (or wait again) until ltime, and check for deadlocks as well
		 * if we will be waiting longer than deadlock_timeout
		 */
		EnableTimeoutParams timeouts[2];
		int			cnt = 0;

		if (ltime != 0)
		{
			got_standby_lock_timeout = false;
			timeouts[cnt].id = STANDBY_LOCK_TIMEOUT;
			timeouts[cnt].type = TMPARAM_AT;
			timeouts[cnt].fin_time = ltime;
			cnt++;
		}


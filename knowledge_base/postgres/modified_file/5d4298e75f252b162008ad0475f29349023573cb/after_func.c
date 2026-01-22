		}

		/* Append new query text to file with only shared lock held */
		stored = qtext_store(norm_query ? norm_query : query, query_len,
							 &query_offset, &gc_count);

		/*
		 * Determine whether we need to garbage collect external query texts
		 * while the shared lock is still held.  This micro-optimization
		 * avoids taking the time to decide this while holding exclusive lock.
		 */
		do_gc = need_gc_qtexts();

		/* Need exclusive lock to make a new hashtable entry - promote */
		LWLockRelease(pgss->lock);
		LWLockAcquire(pgss->lock, LW_EXCLUSIVE);

		/*
		 * A garbage collection may have occurred while we weren't holding the
		 * lock.  In the unlikely event that this happens, the query text we
		 * stored above will have been garbage collected, so write it again.
		 * This should be infrequent enough that doing it while holding
		 * exclusive lock isn't a performance problem.
		 */
		if (!stored || pgss->gc_count != gc_count)
			stored = qtext_store(norm_query ? norm_query : query, query_len,
								 &query_offset, NULL);

		/* If we failed to write to the text file, give up */
		if (!stored)
			goto done;

		/* OK to create a new hashtable entry */
		entry = entry_alloc(&key, query_offset, query_len, encoding,
							jstate != NULL);

		/* If needed, perform garbage collection while exclusive lock held */
		if (do_gc)
			gc_qtexts();
	}

	/* Increment the counts, except when jstate is not NULL */
	if (!jstate)
	{
		Assert(kind == PGSS_PLAN || kind == PGSS_EXEC);

		/*
		 * Grab the spinlock while updating the counters (see comment about
		 * locking rules at the head of the file)
		 */
		SpinLockAcquire(&entry->mutex);

		/* "Unstick" entry if it was previously sticky */
		if (IS_STICKY(entry->counters))
			entry->counters.usage = USAGE_INIT;

		entry->counters.calls[kind] += 1;
		entry->counters.total_time[kind] += total_time;

		if (entry->counters.calls[kind] == 1)
		{
			entry->counters.min_time[kind] = total_time;
			entry->counters.max_time[kind] = total_time;
			entry->counters.mean_time[kind] = total_time;
		}
		else
		{
			/*
			 * Welford's method for accurately computing variance. See
			 * <http://www.johndcook.com/blog/standard_deviation/>
			 */
			double		old_mean = entry->counters.mean_time[kind];

			entry->counters.mean_time[kind] +=
				(total_time - old_mean) / entry->counters.calls[kind];
			entry->counters.sum_var_time[kind] +=
				(total_time - old_mean) * (total_time - entry->counters.mean_time[kind]);

			/*
			 * Calculate min and max time. min = 0 and max = 0 means that the
			 * min/max statistics were reset
			 */
			if (entry->counters.min_time[kind] == 0
				&& entry->counters.max_time[kind] == 0)
			{
				entry->counters.min_time[kind] = total_time;
				entry->counters.max_time[kind] = total_time;
			}
			else
			{
				if (entry->counters.min_time[kind] > total_time)
					entry->counters.min_time[kind] = total_time;
				if (entry->counters.max_time[kind] < total_time)
					entry->counters.max_time[kind] = total_time;
			}
		}
		entry->counters.rows += rows;
		entry->counters.shared_blks_hit += bufusage->shared_blks_hit;
		entry->counters.shared_blks_read += bufusage->shared_blks_read;
		entry->counters.shared_blks_dirtied += bufusage->shared_blks_dirtied;
		entry->counters.shared_blks_written += bufusage->shared_blks_written;
		entry->counters.local_blks_hit += bufusage->local_blks_hit;
		entry->counters.local_blks_read += bufusage->local_blks_read;
		entry->counters.local_blks_dirtied += bufusage->local_blks_dirtied;
		entry->counters.local_blks_written += bufusage->local_blks_written;
		entry->counters.temp_blks_read += bufusage->temp_blks_read;
		entry->counters.temp_blks_written += bufusage->temp_blks_written;
		entry->counters.shared_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_read_time);
		entry->counters.shared_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_write_time);
		entry->counters.local_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_read_time);
		entry->counters.local_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_write_time);
		entry->counters.temp_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_read_time);
		entry->counters.temp_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_write_time);
		entry->counters.usage += USAGE_EXEC(total_time);
		entry->counters.wal_records += walusage->wal_records;
		entry->counters.wal_fpi += walusage->wal_fpi;
		entry->counters.wal_bytes += walusage->wal_bytes;
		if (jitusage)
		{
			entry->counters.jit_functions += jitusage->created_functions;
			entry->counters.jit_generation_time += INSTR_TIME_GET_MILLISEC(jitusage->generation_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->deform_counter))
				entry->counters.jit_deform_count++;
			entry->counters.jit_deform_time += INSTR_TIME_GET_MILLISEC(jitusage->deform_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter))
				entry->counters.jit_inlining_count++;
			entry->counters.jit_inlining_time += INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter))
				entry->counters.jit_optimization_count++;
			entry->counters.jit_optimization_time += INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->emission_counter))
				entry->counters.jit_emission_count++;
			entry->counters.jit_emission_time += INSTR_TIME_GET_MILLISEC(jitusage->emission_counter);
		}

		/* parallel worker counters */
		entry->counters.parallel_workers_to_launch += parallel_workers_to_launch;
		entry->counters.parallel_workers_launched += parallel_workers_launched;

		SpinLockRelease(&entry->mutex);
	}

done:
	LWLockRelease(pgss->lock);

	/* We postpone this clean-up until we're out of the lock */
	if (norm_query)
		pfree(norm_query);
}

/*
 * Reset statement statistics corresponding to userid, dbid, and queryid.
 */
Datum
pg_stat_statements_reset_1_7(PG_FUNCTION_ARGS)
{
	Oid			userid;
	Oid			dbid;
	uint64		queryid;

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = (uint64) PG_GETARG_INT64(2);

	entry_reset(userid, dbid, queryid, false);

	PG_RETURN_VOID();
}

Datum
pg_stat_statements_reset_1_11(PG_FUNCTION_ARGS)
{
	Oid			userid;
	Oid			dbid;
	uint64		queryid;
	bool		minmax_only;

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = (uint64) PG_GETARG_INT64(2);
	minmax_only = PG_GETARG_BOOL(3);

	PG_RETURN_TIMESTAMPTZ(entry_reset(userid, dbid, queryid, minmax_only));
}

/*
 * Reset statement statistics.
 */
Datum
pg_stat_statements_reset(PG_FUNCTION_ARGS)
{
	entry_reset(0, 0, 0, false);

	PG_RETURN_VOID();
}

/* Number of output arguments (columns) for various API versions */
#define PG_STAT_STATEMENTS_COLS_V1_0	14
#define PG_STAT_STATEMENTS_COLS_V1_1	18
#define PG_STAT_STATEMENTS_COLS_V1_2	19
#define PG_STAT_STATEMENTS_COLS_V1_3	23
#define PG_STAT_STATEMENTS_COLS_V1_8	32
#define PG_STAT_STATEMENTS_COLS_V1_9	33
#define PG_STAT_STATEMENTS_COLS_V1_10	43
#define PG_STAT_STATEMENTS_COLS_V1_11	49
#define PG_STAT_STATEMENTS_COLS_V1_12	51
#define PG_STAT_STATEMENTS_COLS			51	/* maximum of above */

/*
 * Retrieve statement statistics.
 *
 * The SQL API of this function has changed multiple times, and will likely
 * do so again in future.  To support the case where a newer version of this
 * loadable module is being used with an old SQL declaration of the function,
 * we continue to support the older API versions.  For 1.2 and later, the
 * expected API version is identified by embedding it in the C name of the
 * function.  Unfortunately we weren't bright enough to do that for 1.1.
 */
Datum
pg_stat_statements_1_12(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_12, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_11(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_11, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_10(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_10, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_9(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_9, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_8(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_8, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_3(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_3, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_2(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_2, showtext);

	return (Datum) 0;
}

/*
 * Legacy entry point for pg_stat_statements() API versions 1.0 and 1.1.
 * This can be removed someday, perhaps.
 */
Datum
pg_stat_statements(PG_FUNCTION_ARGS)
{
	/* If it's really API 1.1, we'll figure that out below */

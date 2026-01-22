		{
			pgcform->relhaspkey = false;
			dirty = true;
		}
		if (pgcform->relhasexclusion && pgcform->relkind != RELKIND_INDEX)
		{
			pgcform->relhasexclusion = false;
			dirty = true;
		}
	}

	/* We also clear relhasrules and relhastriggers if needed */
	if (pgcform->relhasrules && relation->rd_rules == NULL)
	{
		pgcform->relhasrules = false;
		dirty = true;
	}
	if (pgcform->relhastriggers && relation->trigdesc == NULL)
	{
		pgcform->relhastriggers = false;
		dirty = true;
	}

	/*
	 * relfrozenxid should never go backward.  Caller can pass
	 * InvalidTransactionId if it has no new data.
	 */
	if (TransactionIdIsNormal(frozenxid) &&
		TransactionIdPrecedes(pgcform->relfrozenxid, frozenxid))
	{
		pgcform->relfrozenxid = frozenxid;
		dirty = true;
	}

	/* If anything changed, write out the tuple. */
	if (dirty)
		heap_inplace_update(rd, ctup);

	heap_close(rd, RowExclusiveLock);
}


/*
 *	vac_update_datfrozenxid() -- update pg_database.datfrozenxid for our DB
 *
 *		Update pg_database's datfrozenxid entry for our database to be the
 *		minimum of the pg_class.relfrozenxid values.  If we are able to
 *		advance pg_database.datfrozenxid, also try to truncate pg_clog.
 *
 *		We violate transaction semantics here by overwriting the database's
 *		existing pg_database tuple with the new value.	This is reasonably
 *		safe since the new value is correct whether or not this transaction
 *		commits.  As with vac_update_relstats, this avoids leaving dead tuples
 *		behind after a VACUUM.
 */
void
vac_update_datfrozenxid(void)
{
	HeapTuple	tuple;
	Form_pg_database dbform;
	Relation	relation;
	SysScanDesc scan;
	HeapTuple	classTup;
	TransactionId newFrozenXid;
	bool		dirty = false;

	/*
	 * Initialize the "min" calculation with GetOldestXmin, which is a
	 * reasonable approximation to the minimum relfrozenxid for not-yet-
	 * committed pg_class entries for new tables; see AddNewRelationTuple().
	 * Se we cannot produce a wrong minimum by starting with this.
	 */
	newFrozenXid = GetOldestXmin(true, true);

	/*
	 * We must seqscan pg_class to find the minimum Xid, because there is no
	 * index that can help us here.
	 */
	relation = heap_open(RelationRelationId, AccessShareLock);

	scan = systable_beginscan(relation, InvalidOid, false,
							  SnapshotNow, 0, NULL);

	while ((classTup = systable_getnext(scan)) != NULL)
	{
		Form_pg_class classForm = (Form_pg_class) GETSTRUCT(classTup);

		/*
		 * Only consider heap and TOAST tables (anything else should have
		 * InvalidTransactionId in relfrozenxid anyway.)
		 */
		if (classForm->relkind != RELKIND_RELATION &&
			classForm->relkind != RELKIND_TOASTVALUE)
			continue;

		Assert(TransactionIdIsNormal(classForm->relfrozenxid));

		if (TransactionIdPrecedes(classForm->relfrozenxid, newFrozenXid))
			newFrozenXid = classForm->relfrozenxid;
	}

	/* we're done with pg_class */
	systable_endscan(scan);
	heap_close(relation, AccessShareLock);

	Assert(TransactionIdIsNormal(newFrozenXid));

	/* Now fetch the pg_database tuple we need to update. */
	relation = heap_open(DatabaseRelationId, RowExclusiveLock);

	/* Fetch a copy of the tuple to scribble on */
	tuple = SearchSysCacheCopy(DATABASEOID,
							   ObjectIdGetDatum(MyDatabaseId),
							   0, 0, 0);
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "could not find tuple for database %u", MyDatabaseId);
	dbform = (Form_pg_database) GETSTRUCT(tuple);

	/*
	 * Don't allow datfrozenxid to go backward (probably can't happen anyway);
	 * and detect the common case where it doesn't go forward either.
	 */
	if (TransactionIdPrecedes(dbform->datfrozenxid, newFrozenXid))
	{
		dbform->datfrozenxid = newFrozenXid;
		dirty = true;
	}

	if (dirty)
		heap_inplace_update(relation, tuple);

	heap_freetuple(tuple);
	heap_close(relation, RowExclusiveLock);

	/*
	 * If we were able to advance datfrozenxid, see if we can truncate pg_clog.
	 * Also do it if the shared XID-wrap-limit info is stale, since this
	 * action will update that too.
	 */
	if (dirty || ForceTransactionIdLimitUpdate())
		vac_truncate_clog(newFrozenXid);
}


/*
 *	vac_truncate_clog() -- attempt to truncate the commit log
 *
 *		Scan pg_database to determine the system-wide oldest datfrozenxid,
 *		and use it to truncate the transaction commit log (pg_clog).

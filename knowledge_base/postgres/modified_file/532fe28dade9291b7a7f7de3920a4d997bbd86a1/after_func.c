	scan->xs_cbuf = InvalidBuffer;
	scan->xs_continue_hot = false;

	return scan;
}

/* ----------------
 *	IndexScanEnd -- End an index scan.
 *
 *		This routine just releases the storage acquired by
 *		RelationGetIndexScan().  Any AM-level resources are
 *		assumed to already have been released by the AM's
 *		endscan routine.
 *
 *	Returns:
 *		None.
 * ----------------
 */
void
IndexScanEnd(IndexScanDesc scan)
{
	if (scan->keyData != NULL)
		pfree(scan->keyData);
	if (scan->orderByData != NULL)
		pfree(scan->orderByData);

	pfree(scan);
}

/*
 * BuildIndexValueDescription
 *
 * Construct a string describing the contents of an index entry, in the
 * form "(key_name, ...)=(key_value, ...)".  This is currently used
 * for building unique-constraint and exclusion-constraint error messages.
 *
 * The passed-in values/nulls arrays are the "raw" input to the index AM,
 * e.g. results of FormIndexDatum --- this is not necessarily what is stored
 * in the index, but it's what the user perceives to be stored.
 */
char *
BuildIndexValueDescription(Relation indexRelation,
						   Datum *values, bool *isnull)
{
	StringInfoData buf;
	int			natts = indexRelation->rd_rel->relnatts;
	int			i;

	initStringInfo(&buf);
	appendStringInfo(&buf, "(%s)=(",
					 pg_get_indexdef_columns(RelationGetRelid(indexRelation),
											 true));

	for (i = 0; i < natts; i++)
	{
		char	   *val;

		if (isnull[i])
			val = "null";


	return rri;
}

/*
 * Infer a chunk's set of arbiter indexes. This is a subset of the chunk's
 * indexes that match the ON CONFLICT statement.
 */
static List *
chunk_infer_arbiter_indexes(int rindex, List *rtable, Query *parse)
{
	Query		query = *parse;
	PlannerInfo info = {
		.type = T_PlannerInfo,
		.parse = &query,
	};

	query.resultRelation = rindex;
	query.rtable = rtable;

	return infer_arbiter_indexes(&info);
}

/*
 * Check if tuple conversion is needed between a chunk and its parent table.
 *
 * Since a chunk should have the same attributes (columns) as its parent, the
 * only reason tuple conversion should be needed is if the parent has had one or
 * more columns removed, leading to a garbage attribute and inflated number of
 * attributes that aren't inherited by new children tables.
 */
static inline bool
tuple_conversion_needed(TupleDesc indesc,
						TupleDesc outdesc)
{
	return (indesc->natts != outdesc->natts ||
			indesc->tdhasoid != outdesc->tdhasoid);
}

/*
 * Create new insert chunk state.
 *
 * This is essentially a ResultRelInfo for a chunk. Initialization of the
 * ResultRelInfo should be similar to ExecInitModifyTable().
 */
extern ChunkInsertState *
chunk_insert_state_create(Chunk *chunk, ChunkDispatch *dispatch, CmdType operation)
{
	ChunkInsertState *state;
	Relation	rel,
				parent_rel;
	Index		rti;
	MemoryContext old_mcxt;
	MemoryContext cis_context = AllocSetContextCreate(dispatch->estate->es_query_cxt,
										 "chunk insert state memory context",
													  ALLOCSET_DEFAULT_SIZES);
	Query	   *parse = dispatch->parse;
	OnConflictAction onconflict = ONCONFLICT_NONE;
	ResultRelInfo *resrelinfo;

	if (parse && parse->onConflict)
		onconflict = parse->onConflict->action;

	/* permissions NOT checked here; were checked at hypertable level */
	if (check_enable_rls(chunk->table_id, InvalidOid, false) == RLS_ENABLED)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Hypertables don't support row-level security")));

	/*
	 * We must allocate the range table entry on the executor's per-query
	 * context
	 */
	old_mcxt = MemoryContextSwitchTo(dispatch->estate->es_query_cxt);

	rel = heap_open(chunk->table_id, RowExclusiveLock);


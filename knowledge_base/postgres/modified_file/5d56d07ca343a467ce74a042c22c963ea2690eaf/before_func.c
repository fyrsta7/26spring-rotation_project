	Assert(node->numGroups > 0);

	rustate->hashtable = BuildTupleHashTableExt(&rustate->ps,
												desc,
												node->numCols,
												node->dupColIdx,
												rustate->eqfuncoids,
												rustate->hashfunctions,
												node->dupCollations,
												node->numGroups,
												0,
												rustate->ps.state->es_query_cxt,
												rustate->tableContext,
												rustate->tempContext,
												false);
}


/* ----------------------------------------------------------------
 *		ExecRecursiveUnion(node)
 *
 *		Scans the recursive query sequentially and returns the next
 *		qualifying tuple.
 *
 * 1. evaluate non recursive term and assign the result to RT
 *
 * 2. execute recursive terms
 *
 * 2.1 WT := RT
 * 2.2 while WT is not empty repeat 2.3 to 2.6. if WT is empty returns RT
 * 2.3 replace the name of recursive term with WT
 * 2.4 evaluate the recursive term and store into WT
 * 2.5 append WT to RT
 * 2.6 go back to 2.2
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
ExecRecursiveUnion(PlanState *pstate)
{
	RecursiveUnionState *node = castNode(RecursiveUnionState, pstate);
	PlanState  *outerPlan = outerPlanState(node);
	PlanState  *innerPlan = innerPlanState(node);
	RecursiveUnion *plan = (RecursiveUnion *) node->ps.plan;
	TupleTableSlot *slot;
	bool		isnew;

	CHECK_FOR_INTERRUPTS();

	/* 1. Evaluate non-recursive term */
	if (!node->recursing)
	{
		for (;;)
		{
			slot = ExecProcNode(outerPlan);
			if (TupIsNull(slot))
				break;
			if (plan->numCols > 0)
			{
				/* Find or build hashtable entry for this tuple's group */
				LookupTupleHashEntry(node->hashtable, slot, &isnew, NULL);
				/* Must reset temp context after each hashtable lookup */
				MemoryContextReset(node->tempContext);
				/* Ignore tuple if already seen */
				if (!isnew)
					continue;
			}
			/* Each non-duplicate tuple goes to the working table ... */
			tuplestore_puttupleslot(node->working_table, slot);
			/* ... and to the caller */
			return slot;
		}
		node->recursing = true;
	}

	/* 2. Execute recursive term */
	for (;;)
	{
		slot = ExecProcNode(innerPlan);
		if (TupIsNull(slot))
		{
			/* Done if there's nothing in the intermediate table */
			if (node->intermediate_empty)
				break;

			/* done with old working table ... */
			tuplestore_end(node->working_table);


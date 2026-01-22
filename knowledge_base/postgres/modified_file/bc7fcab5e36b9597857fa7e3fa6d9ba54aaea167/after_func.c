	for (;;)
	{
		/*
		 * Get next tuple, either from one of our workers, or by running the
		 * plan ourselves.
		 */
		slot = gather_getnext(node);
		if (TupIsNull(slot))
			return NULL;

		/*
		 * form the result tuple using ExecProject(), and return it --- unless
		 * the projection produces an empty set, in which case we must loop
		 * back around for another tuple
		 */
		econtext->ecxt_outertuple = slot;
		resultSlot = ExecProject(node->ps.ps_ProjInfo, &isDone);

		if (isDone != ExprEndResult)
		{
			node->ps.ps_TupFromTlist = (isDone == ExprMultipleResult);
			return resultSlot;
		}
	}

	return slot;
}

/* ----------------------------------------------------------------
 *		ExecEndGather
 *
 *		frees any storage allocated through C routines.
 * ----------------------------------------------------------------
 */
void
ExecEndGather(GatherState *node)
{
	ExecShutdownGather(node);
	ExecFreeExprContext(&node->ps);
	ExecClearTuple(node->ps.ps_ResultTupleSlot);
	ExecEndNode(outerPlanState(node));
}

/*
 * Read the next tuple.  We might fetch a tuple from one of the tuple queues
 * using gather_readnext, or if no tuple queue contains a tuple and the
 * single_copy flag is not set, we might generate one locally instead.
 */
static TupleTableSlot *
gather_getnext(GatherState *gatherstate)
{
	PlanState  *outerPlan = outerPlanState(gatherstate);
	TupleTableSlot *outerTupleSlot;
	TupleTableSlot *fslot = gatherstate->funnel_slot;
	HeapTuple	tup;

	while (gatherstate->reader != NULL || gatherstate->need_to_scan_locally)
	{
		if (gatherstate->reader != NULL)
		{
			tup = gather_readnext(gatherstate);

			if (HeapTupleIsValid(tup))
			{

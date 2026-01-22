		tlist_cost.per_tuple * plan->plan_rows;

	tlist_rows = tlist_returns_set_rows(tlist);
	if (tlist_rows > 1)
	{
		/*
		 * We assume that execution costs of the tlist proper were all
		 * accounted for by cost_qual_eval.  However, it still seems
		 * appropriate to charge something more for the executor's general
		 * costs of processing the added tuples.  The cost is probably less
		 * than cpu_tuple_cost, though, so we arbitrarily use half of that.
		 */
		plan->total_cost += plan->plan_rows * (tlist_rows - 1) *
			cpu_tuple_cost / 2;

		plan->plan_rows *= tlist_rows;
	}
}

/*
 * Detect whether a plan node is a "dummy" plan created when a relation
 * is deemed not to need scanning due to constraint exclusion.
 *
 * Currently, such dummy plans are Result nodes with constant FALSE
 * filter quals (see set_dummy_rel_pathlist and create_append_plan).
 *
 * XXX this probably ought to be somewhere else, but not clear where.
 */
bool
is_dummy_plan(Plan *plan)
{
	if (IsA(plan, Result))
	{
		List	   *rcqual = (List *) ((Result *) plan)->resconstantqual;

		if (list_length(rcqual) == 1)
		{
			Const	   *constqual = (Const *) linitial(rcqual);

			if (constqual && IsA(constqual, Const))
			{
				if (!constqual->constisnull &&
					!DatumGetBool(constqual->constvalue))
					return true;
			}
		}
	}
	return false;
}

/*
 * Create a bitmapset of the RT indexes of live base relations
 *
 * Helper for preprocess_rowmarks ... at this point in the proceedings,
 * the only good way to distinguish baserels from appendrel children
 * is to see what is in the join tree.
 */
static Bitmapset *
get_base_rel_indexes(Node *jtnode)
{
	Bitmapset  *result;

	if (jtnode == NULL)
		return NULL;
	if (IsA(jtnode, RangeTblRef))
	{
		int			varno = ((RangeTblRef *) jtnode)->rtindex;

		result = bms_make_singleton(varno);
	}
	else if (IsA(jtnode, FromExpr))
	{
		FromExpr   *f = (FromExpr *) jtnode;
		ListCell   *l;

		result = NULL;
		foreach(l, f->fromlist)
			result = bms_join(result,
							  get_base_rel_indexes(lfirst(l)));
	}
	else if (IsA(jtnode, JoinExpr))
	{
		JoinExpr   *j = (JoinExpr *) jtnode;

		result = bms_join(get_base_rel_indexes(j->larg),
						  get_base_rel_indexes(j->rarg));
	}
	else
	{
		elog(ERROR, "unrecognized node type: %d",
			 (int) nodeTag(jtnode));
		result = NULL;			/* keep compiler quiet */
	}
	return result;
}

/*
 * preprocess_rowmarks - set up PlanRowMarks if needed
 */
static void
preprocess_rowmarks(PlannerInfo *root)
{
	Query	   *parse = root->parse;
	Bitmapset  *rels;
	List	   *prowmarks;

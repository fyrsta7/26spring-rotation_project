preprocess_function_rtes(PlannerInfo *root)
{
	ListCell   *rt;

	foreach(rt, root->parse->rtable)
	{
		RangeTblEntry *rte = (RangeTblEntry *) lfirst(rt);

		if (rte->rtekind == RTE_FUNCTION)
		{
			Query	   *funcquery;

			/* Apply const-simplification */
			rte->functions = (List *)
				eval_const_expressions(root, (Node *) rte->functions);

			/* Check safety of expansion, and expand if possible */
			funcquery = inline_set_returning_function(root, rte);
			if (funcquery)
			{
				/* Successful expansion, convert the RTE to a subquery */
				rte->rtekind = RTE_SUBQUERY;
				rte->subquery = funcquery;
				rte->security_barrier = false;
				/* Clear fields that should not be set in a subquery RTE */
				rte->functions = NIL;
				rte->funcordinality = false;
			}
		}
	}
}

/*
 * pull_up_subqueries
 *		Look for subqueries in the rangetable that can be pulled up into
 *		the parent query.  If the subquery has no special features like
 *		grouping/aggregation then we can merge it into the parent's jointree.
 *		Also, subqueries that are simple UNION ALL structures can be
 *		converted into "append relations".
 */
void
pull_up_subqueries(PlannerInfo *root)
{
	/* Top level of jointree must always be a FromExpr */
	Assert(IsA(root->parse->jointree, FromExpr));
	/* Recursion starts with no containing join nor appendrel */
	root->parse->jointree = (FromExpr *)
		pull_up_subqueries_recurse(root, (Node *) root->parse->jointree,
								   NULL, NULL, NULL);
	/* We should still have a FromExpr */
	Assert(IsA(root->parse->jointree, FromExpr));
}

/*
 * pull_up_subqueries_recurse
 *		Recursive guts of pull_up_subqueries.
 *
 * This recursively processes the jointree and returns a modified jointree.
 *
 * If this jointree node is within either side of an outer join, then
 * lowest_outer_join references the lowest such JoinExpr node; otherwise
 * it is NULL.  We use this to constrain the effects of LATERAL subqueries.
 *
 * If this jointree node is within the nullable side of an outer join, then
 * lowest_nulling_outer_join references the lowest such JoinExpr node;
 * otherwise it is NULL.  This forces use of the PlaceHolderVar mechanism for
 * references to non-nullable targetlist items, but only for references above
 * that join.
 *
 * If we are looking at a member subquery of an append relation,
 * containing_appendrel describes that relation; else it is NULL.
 * This forces use of the PlaceHolderVar mechanism for all non-Var targetlist
 * items, and puts some additional restrictions on what can be pulled up.
 *
 * A tricky aspect of this code is that if we pull up a subquery we have
 * to replace Vars that reference the subquery's outputs throughout the
 * parent query, including quals attached to jointree nodes above the one
 * we are currently processing!  We handle this by being careful to maintain
 * validity of the jointree structure while recursing, in the following sense:
 * whenever we recurse, all qual expressions in the tree must be reachable
 * from the top level, in case the recursive call needs to modify them.
 *
 * Notice also that we can't turn pullup_replace_vars loose on the whole
 * jointree, because it'd return a mutated copy of the tree; we have to
 * invoke it just on the quals, instead.  This behavior is what makes it
 * reasonable to pass lowest_outer_join and lowest_nulling_outer_join as
 * pointers rather than some more-indirect way of identifying the lowest
 * OJs.  Likewise, we don't replace append_rel_list members but only their
 * substructure, so the containing_appendrel reference is safe to use.
 */
static Node *
pull_up_subqueries_recurse(PlannerInfo *root, Node *jtnode,
						   JoinExpr *lowest_outer_join,
						   JoinExpr *lowest_nulling_outer_join,
						   AppendRelInfo *containing_appendrel)
{
	Assert(jtnode != NULL);
	if (IsA(jtnode, RangeTblRef))
	{
		int			varno = ((RangeTblRef *) jtnode)->rtindex;
		RangeTblEntry *rte = rt_fetch(varno, root->parse->rtable);

		/*
		 * Is this a subquery RTE, and if so, is the subquery simple enough to
		 * pull up?
		 *
		 * If we are looking at an append-relation member, we can't pull it up
		 * unless is_safe_append_member says so.
		 */
		if (rte->rtekind == RTE_SUBQUERY &&
			is_simple_subquery(root, rte->subquery, rte, lowest_outer_join) &&
			(containing_appendrel == NULL ||
			 is_safe_append_member(rte->subquery)))
			return pull_up_simple_subquery(root, jtnode, rte,
										   lowest_outer_join,
										   lowest_nulling_outer_join,
										   containing_appendrel);

		/*
		 * Alternatively, is it a simple UNION ALL subquery?  If so, flatten
		 * into an "append relation".
		 *
		 * It's safe to do this regardless of whether this query is itself an
		 * appendrel member.  (If you're thinking we should try to flatten the
		 * two levels of appendrel together, you're right; but we handle that
		 * in set_append_rel_pathlist, not here.)
		 */
		if (rte->rtekind == RTE_SUBQUERY &&
			is_simple_union_all(rte->subquery))
			return pull_up_simple_union_all(root, jtnode, rte);

		/*
		 * Or perhaps it's a simple VALUES RTE?
		 *
		 * We don't allow VALUES pullup below an outer join nor into an
		 * appendrel (such cases are impossible anyway at the moment).
		 */
		if (rte->rtekind == RTE_VALUES &&
			lowest_outer_join == NULL &&
			containing_appendrel == NULL &&
			is_simple_values(root, rte))
			return pull_up_simple_values(root, jtnode, rte);

		/*
		 * Or perhaps it's a FUNCTION RTE that we could inline?
		 */
		if (rte->rtekind == RTE_FUNCTION)
			return pull_up_constant_function(root, jtnode, rte,
											 lowest_nulling_outer_join,
											 containing_appendrel);

		/* Otherwise, do nothing at this node. */
	}
	else if (IsA(jtnode, FromExpr))
	{
		FromExpr   *f = (FromExpr *) jtnode;
		ListCell   *l;

		Assert(containing_appendrel == NULL);
		/* Recursively transform all the child nodes */
		foreach(l, f->fromlist)
		{
			lfirst(l) = pull_up_subqueries_recurse(root, lfirst(l),
												   lowest_outer_join,
												   lowest_nulling_outer_join,
												   NULL);
		}
	}
	else if (IsA(jtnode, JoinExpr))
	{
		JoinExpr   *j = (JoinExpr *) jtnode;

		Assert(containing_appendrel == NULL);
		/* Recurse, being careful to tell myself when inside outer join */
		switch (j->jointype)
		{
			case JOIN_INNER:
				j->larg = pull_up_subqueries_recurse(root, j->larg,
													 lowest_outer_join,
													 lowest_nulling_outer_join,
													 NULL);
				j->rarg = pull_up_subqueries_recurse(root, j->rarg,
													 lowest_outer_join,
													 lowest_nulling_outer_join,
													 NULL);
				break;
			case JOIN_LEFT:
			case JOIN_SEMI:
			case JOIN_ANTI:
				j->larg = pull_up_subqueries_recurse(root, j->larg,
													 j,
													 lowest_nulling_outer_join,
													 NULL);
				j->rarg = pull_up_subqueries_recurse(root, j->rarg,
													 j,
													 j,
													 NULL);
				break;
			case JOIN_FULL:
				j->larg = pull_up_subqueries_recurse(root, j->larg,
													 j,
													 j,
													 NULL);
				j->rarg = pull_up_subqueries_recurse(root, j->rarg,
													 j,

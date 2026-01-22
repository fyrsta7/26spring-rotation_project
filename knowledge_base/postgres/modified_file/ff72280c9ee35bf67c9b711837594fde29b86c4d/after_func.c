	 * should optimize for one-time evaluation.
	 */
	if (kind != EXPRKIND_VALUES &&
		(root->parse->jointree->fromlist != NIL ||
		 kind == EXPRKIND_QUAL ||
		 root->query_level > 1))
		expr = eval_const_expressions(expr);

	/*
	 * If it's a qual or havingQual, canonicalize it.
	 */
	if (kind == EXPRKIND_QUAL)
	{
		expr = (Node *) canonicalize_qual((Expr *) expr);

#ifdef OPTIMIZER_DEBUG
		printf("After canonicalize_qual()\n");
		pprint(expr);
#endif
	}

	/* Expand SubLinks to SubPlans */
	if (root->parse->hasSubLinks)
		expr = SS_process_sublinks(root, expr, (kind == EXPRKIND_QUAL));

	/*
	 * XXX do not insert anything here unless you have grokked the comments in
	 * SS_replace_correlation_vars ...
	 */

	/* Replace uplevel vars with Param nodes (this IS possible in VALUES) */
	if (root->query_level > 1)
		expr = SS_replace_correlation_vars(root, expr);

	/*
	 * If it's a qual or havingQual, convert it to implicit-AND format. (We
	 * don't want to do this before eval_const_expressions, since the latter
	 * would be unable to simplify a top-level AND correctly. Also,
	 * SS_process_sublinks expects explicit-AND format.)
	 */
	if (kind == EXPRKIND_QUAL)
		expr = (Node *) make_ands_implicit((Expr *) expr);

	return expr;
}

/*
 * preprocess_qual_conditions
 *		Recursively scan the query's jointree and do subquery_planner's
 *		preprocessing work on each qual condition found therein.
 */
static void
preprocess_qual_conditions(PlannerInfo *root, Node *jtnode)
{
	if (jtnode == NULL)
		return;
	if (IsA(jtnode, RangeTblRef))
	{
		/* nothing to do here */
	}
	else if (IsA(jtnode, FromExpr))
	{
		FromExpr   *f = (FromExpr *) jtnode;
		ListCell   *l;

		foreach(l, f->fromlist)
			preprocess_qual_conditions(root, lfirst(l));

		f->quals = preprocess_expression(root, f->quals, EXPRKIND_QUAL);
	}
	else if (IsA(jtnode, JoinExpr))
	{
		JoinExpr   *j = (JoinExpr *) jtnode;

		preprocess_qual_conditions(root, j->larg);
		preprocess_qual_conditions(root, j->rarg);

		j->quals = preprocess_expression(root, j->quals, EXPRKIND_QUAL);
	}
	else
		elog(ERROR, "unrecognized node type: %d",
			 (int) nodeTag(jtnode));
}

/*
 * inheritance_planner
 *	  Generate a plan in the case where the result relation is an
 *	  inheritance set.
 *
 * We have to handle this case differently from cases where a source relation
 * is an inheritance set. Source inheritance is expanded at the bottom of the
 * plan tree (see allpaths.c), but target inheritance has to be expanded at
 * the top.  The reason is that for UPDATE, each target relation needs a
 * different targetlist matching its own column set.  Also, for both UPDATE
 * and DELETE, the executor needs the Append plan node at the top, else it
 * can't keep track of which table is the current target table.  Fortunately,
 * the UPDATE/DELETE target can never be the nullable side of an outer join,
 * so it's OK to generate the plan this way.
 *
 * Returns a query plan.
 */
static Plan *
inheritance_planner(PlannerInfo *root)
{
	Query	   *parse = root->parse;
	int			parentRTindex = parse->resultRelation;
	List	   *subplans = NIL;
	List	   *resultRelations = NIL;
	List	   *returningLists = NIL;
	List	   *rtable = NIL;
	List	   *tlist = NIL;
	PlannerInfo subroot;
	ListCell   *l;

	foreach(l, root->append_rel_list)
	{
		AppendRelInfo *appinfo = (AppendRelInfo *) lfirst(l);
		Plan	   *subplan;

		/* append_rel_list contains all append rels; ignore others */
		if (appinfo->parent_relid != parentRTindex)
			continue;

		/*
		 * Generate modified query with this rel as target.  We have to be
		 * prepared to translate varnos in in_info_list as well as in the
		 * Query proper.
		 */
		memcpy(&subroot, root, sizeof(PlannerInfo));
		subroot.parse = (Query *)
			adjust_appendrel_attrs((Node *) parse,
								   appinfo);
		subroot.in_info_list = (List *)
			adjust_appendrel_attrs((Node *) root->in_info_list,
								   appinfo);
		subroot.init_plans = NIL;
		/* There shouldn't be any OJ info to translate, as yet */
		Assert(subroot.oj_info_list == NIL);

		/* Generate plan */
		subplan = grouping_planner(&subroot, 0.0 /* retrieve all tuples */ );

		/*
		 * If this child rel was excluded by constraint exclusion, exclude it
		 * from the plan.
		 */
		if (is_dummy_plan(subplan))
			continue;

		/* Save rtable and tlist from first rel for use below */
		if (subplans == NIL)
		{
			rtable = subroot.parse->rtable;
			tlist = subplan->targetlist;
		}

		subplans = lappend(subplans, subplan);

		/* Make sure any initplans from this rel get into the outer list */
		root->init_plans = list_concat(root->init_plans, subroot.init_plans);

		/* Build target-relations list for the executor */
		resultRelations = lappend_int(resultRelations, appinfo->child_relid);

		/* Build list of per-relation RETURNING targetlists */
		if (parse->returningList)
		{
			Assert(list_length(subroot.returningLists) == 1);
			returningLists = list_concat(returningLists,
										 subroot.returningLists);
		}
	}

	root->resultRelations = resultRelations;
	root->returningLists = returningLists;

	/* Mark result as unordered (probably unnecessary) */
	root->query_pathkeys = NIL;

	/*
	 * If we managed to exclude every child rel, return a dummy plan
	 */
	if (subplans == NIL)
	{
		root->resultRelations = list_make1_int(parentRTindex);
		/* although dummy, it must have a valid tlist for executor */
		tlist = preprocess_targetlist(root, parse->targetList);
		return (Plan *) make_result(root,
									tlist,
									(Node *) list_make1(makeBoolConst(false,
																	  false)),
									NULL);
	}

	/*
	 * Planning might have modified the rangetable, due to changes of the
	 * Query structures inside subquery RTEs.  We have to ensure that this
	 * gets propagated back to the master copy.  But can't do this until we
	 * are done planning, because all the calls to grouping_planner need
	 * virgin sub-Queries to work from.  (We are effectively assuming that
	 * sub-Queries will get planned identically each time, or at least that
	 * the impacts on their rangetables will be the same each time.)
	 *
	 * XXX should clean this up someday
	 */
	parse->rtable = rtable;

	/* Suppress Append if there's only one surviving child rel */
	if (list_length(subplans) == 1)
		return (Plan *) linitial(subplans);

	return (Plan *) make_append(subplans, true, tlist);
}

/*--------------------
 * grouping_planner
 *	  Perform planning steps related to grouping, aggregation, etc.
 *	  This primarily means adding top-level processing to the basic
 *	  query plan produced by query_planner.
 *
 * tuple_fraction is the fraction of tuples we expect will be retrieved
 *
 * tuple_fraction is interpreted as follows:
 *	  0: expect all tuples to be retrieved (normal case)
 *	  0 < tuple_fraction < 1: expect the given fraction of tuples available
 *		from the plan to be retrieved
 *	  tuple_fraction >= 1: tuple_fraction is the absolute number of tuples
 *		expected to be retrieved (ie, a LIMIT specification)
 *
 * Returns a query plan.  Also, root->query_pathkeys is returned as the
 * actual output ordering of the plan (in pathkey format).
 *--------------------
 */
static Plan *
grouping_planner(PlannerInfo *root, double tuple_fraction)
{
	Query	   *parse = root->parse;
	List	   *tlist = parse->targetList;
	int64		offset_est = 0;
	int64		count_est = 0;
	double		limit_tuples = -1.0;
	Plan	   *result_plan;
	List	   *current_pathkeys;
	List	   *sort_pathkeys;
	double		dNumGroups = 0;

	/* Tweak caller-supplied tuple_fraction if have LIMIT/OFFSET */
	if (parse->limitCount || parse->limitOffset)
	{
		tuple_fraction = preprocess_limit(root, tuple_fraction,
										  &offset_est, &count_est);

		/*
		 * If we have a known LIMIT, and don't have an unknown OFFSET, we can
		 * estimate the effects of using a bounded sort.
		 */
		if (count_est > 0 && offset_est >= 0)
			limit_tuples = (double) count_est + (double) offset_est;
	}

	if (parse->setOperations)
	{
		List	   *set_sortclauses;

		/*
		 * If there's a top-level ORDER BY, assume we have to fetch all the
		 * tuples.	This might seem too simplistic given all the hackery below
		 * to possibly avoid the sort ... but a nonzero tuple_fraction is only
		 * of use to plan_set_operations() when the setop is UNION ALL, and
		 * the result of UNION ALL is always unsorted.
		 */
		if (parse->sortClause)
			tuple_fraction = 0.0;

		/*
		 * Construct the plan for set operations.  The result will not need
		 * any work except perhaps a top-level sort and/or LIMIT.
		 */
		result_plan = plan_set_operations(root, tuple_fraction,
										  &set_sortclauses);

		/*
		 * Calculate pathkeys representing the sort order (if any) of the set
		 * operation's result.  We have to do this before overwriting the sort
		 * key information...
		 */
		current_pathkeys = make_pathkeys_for_sortclauses(root,
														 set_sortclauses,
													 result_plan->targetlist,
														 true);

		/*
		 * We should not need to call preprocess_targetlist, since we must be
		 * in a SELECT query node.	Instead, use the targetlist returned by
		 * plan_set_operations (since this tells whether it returned any
		 * resjunk columns!), and transfer any sort key information from the
		 * original tlist.
		 */
		Assert(parse->commandType == CMD_SELECT);

		tlist = postprocess_setop_tlist(result_plan->targetlist, tlist);

		/*
		 * Can't handle FOR UPDATE/SHARE here (parser should have checked
		 * already, but let's make sure).
		 */
		if (parse->rowMarks)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("SELECT FOR UPDATE/SHARE is not allowed with UNION/INTERSECT/EXCEPT")));

		/*
		 * Calculate pathkeys that represent result ordering requirements
		 */
		sort_pathkeys = make_pathkeys_for_sortclauses(root,
													  parse->sortClause,
													  tlist,
													  true);
	}
	else
	{
		/* No set operations, do regular planning */
		List	   *sub_tlist;
		List	   *group_pathkeys;
		AttrNumber *groupColIdx = NULL;
		Oid		   *groupOperators = NULL;
		bool		need_tlist_eval = true;
		QualCost	tlist_cost;
		Path	   *cheapest_path;
		Path	   *sorted_path;
		Path	   *best_path;
		long		numGroups = 0;
		AggClauseCounts agg_counts;
		int			numGroupCols = list_length(parse->groupClause);
		bool		use_hashed_grouping = false;

		MemSet(&agg_counts, 0, sizeof(AggClauseCounts));

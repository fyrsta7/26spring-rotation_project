} MinMaxAggInfo;

static bool find_minmax_aggs_walker(Node *node, List **context);
static bool build_minmax_path(PlannerInfo *root, RelOptInfo *rel,
				  MinMaxAggInfo *info);
static ScanDirection match_agg_to_index_col(MinMaxAggInfo *info,
					   IndexOptInfo *index, int indexcol);
static void make_agg_subplan(PlannerInfo *root, MinMaxAggInfo *info);
static Node *replace_aggs_with_params_mutator(Node *node, List **context);
static Oid	fetch_agg_sort_op(Oid aggfnoid);


/*
 * optimize_minmax_aggregates - check for optimizing MIN/MAX via indexes
 *
 * This checks to see if we can replace MIN/MAX aggregate functions by
 * subqueries of the form
 *		(SELECT col FROM tab WHERE ... ORDER BY col ASC/DESC LIMIT 1)
 * Given a suitable index on tab.col, this can be much faster than the
 * generic scan-all-the-rows plan.
 *
 * We are passed the preprocessed tlist, and the best path
 * devised for computing the input of a standard Agg node.	If we are able
 * to optimize all the aggregates, and the result is estimated to be cheaper
 * than the generic aggregate method, then generate and return a Plan that
 * does it that way.  Otherwise, return NULL.
 */
Plan *
optimize_minmax_aggregates(PlannerInfo *root, List *tlist, Path *best_path)
{
	Query	   *parse = root->parse;
	RangeTblRef *rtr;
	RangeTblEntry *rte;
	RelOptInfo *rel;
	List	   *aggs_list;
	ListCell   *l;
	Cost		total_cost;
	Path		agg_p;
	Plan	   *plan;
	Node	   *hqual;
	QualCost	tlist_cost;

	/* Nothing to do if query has no aggregates */
	if (!parse->hasAggs)
		return NULL;

	Assert(!parse->setOperations);		/* shouldn't get here if a setop */
	Assert(parse->rowMarks == NIL);		/* nor if FOR UPDATE */

	/*
	 * Reject unoptimizable cases.
	 *
	 * We don't handle GROUP BY, because our current implementations of
	 * grouping require looking at all the rows anyway, and so there's not
	 * much point in optimizing MIN/MAX.
	 */
	if (parse->groupClause)
		return NULL;

	/*
	 * We also restrict the query to reference exactly one table, since join
	 * conditions can't be handled reasonably.  (We could perhaps handle a
	 * query containing cartesian-product joins, but it hardly seems worth the
	 * trouble.)
	 */
	Assert(parse->jointree != NULL && IsA(parse->jointree, FromExpr));
	if (list_length(parse->jointree->fromlist) != 1)
		return NULL;
	rtr = (RangeTblRef *) linitial(parse->jointree->fromlist);
	if (!IsA(rtr, RangeTblRef))
		return NULL;
	rte = rt_fetch(rtr->rtindex, parse->rtable);
	if (rte->rtekind != RTE_RELATION || rte->inh)
		return NULL;
	rel = find_base_rel(root, rtr->rtindex);

	/*
	 * Since this optimization is not applicable all that often, we want to
	 * fall out before doing very much work if possible.  Therefore we do the
	 * work in several passes.	The first pass scans the tlist and HAVING qual
	 * to find all the aggregates and verify that each of them is a MIN/MAX
	 * aggregate.  If that succeeds, the second pass looks at each aggregate
	 * to see if it is optimizable; if so we make an IndexPath describing how
	 * we would scan it.  (We do not try to optimize if only some aggs are
	 * optimizable, since that means we'll have to scan all the rows anyway.)
	 * If that succeeds, we have enough info to compare costs against the
	 * generic implementation. Only if that test passes do we build a Plan.
	 */

	/* Pass 1: find all the aggregates */
	aggs_list = NIL;

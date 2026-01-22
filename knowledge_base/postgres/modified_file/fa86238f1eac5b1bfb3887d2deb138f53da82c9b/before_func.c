	}

	foreach(lc, ec->ec_derives)
	{
		rinfo = (RestrictInfo *) lfirst(lc);
		if (rinfo->left_em == leftem &&
			rinfo->right_em == rightem &&
			rinfo->parent_ec == parent_ec &&
			opno == ((OpExpr *) rinfo->clause)->opno)
			return rinfo;
	}

	/*
	 * Not there, so build it, in planner context so we can re-use it. (Not
	 * important in normal planning, but definitely so in GEQO.)
	 */
	oldcontext = MemoryContextSwitchTo(root->planner_cxt);

	rinfo = build_implied_join_equality(opno,
										ec->ec_collation,
										leftem->em_expr,
										rightem->em_expr,
										bms_union(leftem->em_relids,
												  rightem->em_relids),
										bms_union(leftem->em_nullable_relids,
												  rightem->em_nullable_relids),
										ec->ec_min_security);

	/* Mark the clause as redundant, or not */
	rinfo->parent_ec = parent_ec;

	/*
	 * We know the correct values for left_ec/right_ec, ie this particular EC,
	 * so we can just set them directly instead of forcing another lookup.
	 */
	rinfo->left_ec = ec;
	rinfo->right_ec = ec;

	/* Mark it as usable with these EMs */
	rinfo->left_em = leftem;
	rinfo->right_em = rightem;
	/* and save it for possible re-use */
	ec->ec_derives = lappend(ec->ec_derives, rinfo);

	MemoryContextSwitchTo(oldcontext);

	return rinfo;
}


/*
 * reconsider_outer_join_clauses
 *	  Re-examine any outer-join clauses that were set aside by
 *	  distribute_qual_to_rels(), and see if we can derive any
 *	  EquivalenceClasses from them.  Then, if they were not made
 *	  redundant, push them out into the regular join-clause lists.
 *
 * When we have mergejoinable clauses A = B that are outer-join clauses,
 * we can't blindly combine them with other clauses A = C to deduce B = C,
 * since in fact the "equality" A = B won't necessarily hold above the
 * outer join (one of the variables might be NULL instead).  Nonetheless

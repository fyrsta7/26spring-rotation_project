										  pathkey->pk_nulls_first);

		/* Add to list unless redundant */
		if (!pathkey_is_redundant(cpathkey, new_pathkeys))
			new_pathkeys = lappend(new_pathkeys, cpathkey);
	}
	return new_pathkeys;
}

/*
 * make_pathkey_from_sortinfo
 *	  Given an expression, a sortop, and a nulls-first flag, create
 *	  a PathKey.  If canonicalize = true, the result is a "canonical"
 *	  PathKey, otherwise not.  (But note it might be redundant anyway.)
 *
 * If the PathKey is being generated from a SortGroupClause, sortref should be
 * the SortGroupClause's SortGroupRef; otherwise zero.
 *
 * canonicalize should always be TRUE after EquivalenceClass merging has
 * been performed, but FALSE if we haven't done EquivalenceClass merging yet.
 */
static PathKey *
make_pathkey_from_sortinfo(PlannerInfo *root,
						   Expr *expr, Oid ordering_op,
						   bool nulls_first,
						   Index sortref,
						   bool canonicalize)
{
	Oid			opfamily,
				opcintype;
	int16		strategy;
	Oid			equality_op;

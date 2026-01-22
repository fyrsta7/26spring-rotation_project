 *	  Presently, the executor can only deal with indexquals that have the
 *	  indexkey on the left, so we can only use clauses that have the indexkey
 *	  on the right if we can commute the clause to put the key on the left.
 *	  We do not actually do the commuting here, but we check whether a
 *	  suitable commutator operator is available.
 *
 *	  It is also possible to match RowCompareExpr clauses to indexes (but
 *	  currently, only btree indexes handle this).  In this routine we will
 *	  report a match if the first column of the row comparison matches the
 *	  target index column.	This is sufficient to guarantee that some index
 *	  condition can be constructed from the RowCompareExpr --- whether the
 *	  remaining columns match the index too is considered in
 *	  expand_indexqual_rowcompare().
 *
 *	  It is also possible to match ScalarArrayOpExpr clauses to indexes, when
 *	  the clause is of the form "indexkey op ANY (arrayconst)".  Since the
 *	  executor can only handle these in the context of bitmap index scans,
 *	  our caller specifies whether to allow these or not.
 *
 *	  For boolean indexes, it is also possible to match the clause directly
 *	  to the indexkey; or perhaps the clause is (NOT indexkey).
 *
 * 'index' is the index of interest.
 * 'indexcol' is a column number of 'index' (counting from 0).
 * 'opfamily' is the corresponding operator family.
 * 'rinfo' is the clause to be tested (as a RestrictInfo node).
 * 'saop_control' indicates whether ScalarArrayOpExpr clauses can be used.

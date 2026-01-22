 * get_relattval
 *		Extract information from a restriction or join clause for
 *		selectivity estimation.  The inputs are an expression
 *		and a relation number (which can be 0 if we don't care which
 *		relation is used; that'd normally be the case for restriction
 *		clauses, where the caller already knows that only one relation
 *		is referenced in the clause).  The routine checks that the
 *		expression is of the form (var op something) or (something op var)
 *		where the var is an attribute of the specified relation, or
 *		a function of a var of the specified relation.  If so, it
 *		returns the following info:
 *			the found relation number (same as targetrelid unless that is 0)
 *			the found var number (or InvalidAttrNumber if a function)
 *			if the "something" is a constant, the value of the constant
 *			flags indicating whether a constant was found, and on which side.
 *		Default values are returned if the expression is too complicated,
 *		specifically 0 for the relid and attno, 0 for the constant value.
 *
 *		Note that negative attno values are *not* invalid, but represent
 *		system attributes such as OID.  It's sufficient to check for relid=0
 *		to determine whether the routine succeeded.
 */
void
get_relattval(Node *clause,
			  int targetrelid,
			  int *relid,
			  AttrNumber *attno,
			  Datum *constval,
			  int *flag)
{
	Var		   *left,
			   *right,
			   *other;
	int			funcvarno;

	/* Careful; the passed clause might not be a binary operator at all */

	if (!is_opclause(clause))
		goto default_results;

	left = get_leftop((Expr *) clause);
	right = get_rightop((Expr *) clause);

	if (!right)
		goto default_results;

	/* First look for the var or func */

	if (IsA(left, Var) &&
		(targetrelid == 0 || targetrelid == left->varno))
	{
		*relid = left->varno;
		*attno = left->varattno;
		*flag = SEL_RIGHT;
	}
	else if (IsA(right, Var) &&
			 (targetrelid == 0 || targetrelid == right->varno))
	{
		*relid = right->varno;
		*attno = right->varattno;
		*flag = 0;
	}
	else if ((funcvarno = is_single_func((Node *) left)) != 0 &&
			 (targetrelid == 0 || targetrelid == funcvarno))
	{
		*relid = funcvarno;
		*attno = InvalidAttrNumber;
		*flag = SEL_RIGHT;
	}
	else if ((funcvarno = is_single_func((Node *) right)) != 0 &&
			 (targetrelid == 0 || targetrelid == funcvarno))
	{
		*relid = funcvarno;
		*attno = InvalidAttrNumber;
		*flag = 0;
	}
	else
	{
		/* Duh, it's too complicated for me... */
default_results:
		*relid = 0;
		*attno = 0;
		*constval = 0;
		*flag = 0;
		return;
	}

	/* OK, we identified the var or func; now look at the other side */

	other = (*flag == 0) ? left : right;

	if (IsA(other, Const))
	{
		*constval = ((Const *) other)->constvalue;
		*flag |= SEL_CONSTANT;
	}
	else
	{
		*constval = 0;
	}
}

/*
 * is_single_func
 *   If the given expression is a function of a single relation,
 *   return the relation number; else return 0
 */
static int is_single_func(Node *node)
{
	if (is_funcclause(node))
	{
		List	   *varnos = pull_varnos(node);

		if (length(varnos) == 1)
		{
			int		funcvarno = lfirsti(varnos);

			freeList(varnos);
			return funcvarno;
		}
		freeList(varnos);
	}
	return 0;
}

/*
 * get_rels_atts
 *
 * Returns the info
 *				( relid1 attno1 relid2 attno2 )
 *		for a joinclause.
 *
 * If the clause is not of the form (var op var) or if any of the vars
 * refer to nested attributes, then zeroes are returned.
 *
 */
void
get_rels_atts(Node *clause,
			  int *relid1,
			  AttrNumber *attno1,
			  int *relid2,
			  AttrNumber *attno2)
{
	/* set default values */
	*relid1 = 0;
	*attno1 = 0;
	*relid2 = 0;
	*attno2 = 0;

	if (is_opclause(clause))
	{
		Var		   *left = get_leftop((Expr *) clause);
		Var		   *right = get_rightop((Expr *) clause);

		if (left && right)
		{
			int			funcvarno;

			if (IsA(left, Var))
			{
				*relid1 = left->varno;
				*attno1 = left->varattno;
			}
			else if ((funcvarno = is_single_func((Node *) left)) != 0)
			{
				*relid1 = funcvarno;
				*attno1 = InvalidAttrNumber;
			}

			if (IsA(right, Var))
			{
				*relid2 = right->varno;
				*attno2 = right->varattno;
			}
			else if ((funcvarno = is_single_func((Node *) right)) != 0)
			{
				*relid2 = funcvarno;
				*attno2 = InvalidAttrNumber;
			}
		}
	}
}

/*--------------------
 * CommuteClause: commute a binary operator clause
 *
 * XXX the clause is destructively modified!
 *--------------------
 */
void
CommuteClause(Expr *clause)
{
	HeapTuple	heapTup;
	Form_pg_operator commuTup;
	Oper	   *commu;
	Node	   *temp;

	if (!is_opclause((Node *) clause) ||
		length(clause->args) != 2)
		elog(ERROR, "CommuteClause: applied to non-binary-operator clause");

	heapTup = (HeapTuple)
		get_operator_tuple(get_commutator(((Oper *) clause->oper)->opno));

	if (heapTup == (HeapTuple) NULL)
		elog(ERROR, "CommuteClause: no commutator for operator %u",
			 ((Oper *) clause->oper)->opno);

	commuTup = (Form_pg_operator) GETSTRUCT(heapTup);

	commu = makeOper(heapTup->t_data->t_oid,
					 commuTup->oprcode,
					 commuTup->oprresult,
					 ((Oper *) clause->oper)->opsize,
					 NULL);

	/*
	 * re-form the clause in-place!
	 */
	clause->oper = (Node *) commu;
	temp = lfirst(clause->args);
	lfirst(clause->args) = lsecond(clause->args);
	lsecond(clause->args) = temp;
}


/*--------------------
 * eval_const_expressions
 *
 * Reduce any recognizably constant subexpressions of the given
 * expression tree, for example "2 + 2" => "4".  More interestingly,
 * we can reduce certain boolean expressions even when they contain
 * non-constant subexpressions: "x OR true" => "true" no matter what
 * the subexpression x is.  (XXX We assume that no such subexpression
 * will have important side-effects, which is not necessarily a good
 * assumption in the presence of user-defined functions; do we need a
 * pg_proc flag that prevents discarding the execution of a function?)
 *
 * We do understand that certain functions may deliver non-constant
 * results even with constant inputs, "nextval()" being the classic
 * example.  Functions that are not marked "proiscachable" in pg_proc
 * will not be pre-evaluated here, although we will reduce their
 * arguments as far as possible.  Functions that are the arguments
 * of Iter nodes are also not evaluated.
 *
 * We assume that the tree has already been type-checked and contains
 * only operators and functions that are reasonable to try to execute.
 *
 * This routine should be invoked before converting sublinks to subplans
 * (subselect.c's SS_process_sublinks()).  The converted form contains
 * bogus "Const" nodes that are actually placeholders where the executor
 * will insert values from the inner plan, and obviously we mustn't try
 * to reduce the expression as though these were really constants.
 * As a safeguard, if we happen to find an already-converted SubPlan node,
 * we will return it unchanged rather than recursing into it.
 *--------------------
 */
Node *
eval_const_expressions(Node *node)
{
	/* no context or special setup needed, so away we go... */
	return eval_const_expressions_mutator(node, NULL);
}

static Node *
eval_const_expressions_mutator (Node *node, void *context)
{
	if (node == NULL)
		return NULL;
	if (IsA(node, Expr))
	{
		Expr	   *expr = (Expr *) node;
		List	   *args;
		Const	   *const_input;
		Expr       *newexpr;

		/*
		 * Reduce constants in the Expr's arguments.  We know args is
		 * either NIL or a List node, so we can call expression_tree_mutator
		 * directly rather than recursing to self.
		 */
		args = (List *) expression_tree_mutator((Node *) expr->args,
												eval_const_expressions_mutator,
												(void *) context);

		switch (expr->opType)
		{
			case OP_EXPR:
			case FUNC_EXPR:
			{
				/*
				 * For an operator or function, we cannot simplify
				 * unless all the inputs are constants.  (XXX possible

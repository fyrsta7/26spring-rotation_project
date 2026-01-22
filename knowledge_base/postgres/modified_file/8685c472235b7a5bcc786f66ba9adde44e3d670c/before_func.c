 */
Oid
exprType(Node *expr)
{
	Oid			type;

	if (!expr)
		return InvalidOid;

	switch (nodeTag(expr))
	{
		case T_Var:
			type = ((Var *) expr)->vartype;
			break;
		case T_Const:
			type = ((Const *) expr)->consttype;
			break;
		case T_Param:
			type = ((Param *) expr)->paramtype;
			break;
		case T_Aggref:
			type = ((Aggref *) expr)->aggtype;
			break;
		case T_ArrayRef:
			type = ((ArrayRef *) expr)->refrestype;
			break;
		case T_FuncExpr:
			type = ((FuncExpr *) expr)->funcresulttype;
			break;
		case T_OpExpr:
			type = ((OpExpr *) expr)->opresulttype;
			break;
		case T_DistinctExpr:
			type = ((DistinctExpr *) expr)->opresulttype;
			break;
		case T_ScalarArrayOpExpr:
			type = BOOLOID;
			break;
		case T_BoolExpr:
			type = BOOLOID;
			break;
		case T_SubLink:
			{
				SubLink    *sublink = (SubLink *) expr;

				if (sublink->subLinkType == EXPR_SUBLINK ||
					sublink->subLinkType == ARRAY_SUBLINK)
				{
					/* get the type of the subselect's first target column */
					Query	   *qtree = (Query *) sublink->subselect;
					TargetEntry *tent;

					if (!qtree || !IsA(qtree, Query))
						elog(ERROR, "cannot get type for untransformed sublink");
					tent = (TargetEntry *) linitial(qtree->targetList);
					Assert(IsA(tent, TargetEntry));
					Assert(!tent->resjunk);
					type = exprType((Node *) tent->expr);
					if (sublink->subLinkType == ARRAY_SUBLINK)
					{
						type = get_array_type(type);
						if (!OidIsValid(type))
							ereport(ERROR,
									(errcode(ERRCODE_UNDEFINED_OBJECT),
									 errmsg("could not find array type for data type %s",
							format_type_be(exprType((Node *) tent->expr)))));
					}
				}
				else
				{
					/* for all other sublink types, result is boolean */
					type = BOOLOID;
				}
			}
			break;
		case T_SubPlan:
			{
				/*
				 * Although the parser does not ever deal with already-planned
				 * expression trees, we support SubPlan nodes in this routine
				 * for the convenience of ruleutils.c.
				 */
				SubPlan    *subplan = (SubPlan *) expr;

				if (subplan->subLinkType == EXPR_SUBLINK ||
					subplan->subLinkType == ARRAY_SUBLINK)
				{
					/* get the type of the subselect's first target column */
					TargetEntry *tent;

					tent = (TargetEntry *) linitial(subplan->plan->targetlist);
					Assert(IsA(tent, TargetEntry));
					Assert(!tent->resjunk);
					type = exprType((Node *) tent->expr);
					if (subplan->subLinkType == ARRAY_SUBLINK)
					{
						type = get_array_type(type);
						if (!OidIsValid(type))
							ereport(ERROR,
									(errcode(ERRCODE_UNDEFINED_OBJECT),
									 errmsg("could not find array type for data type %s",
							format_type_be(exprType((Node *) tent->expr)))));
					}
				}
				else
				{
					/* for all other subplan types, result is boolean */
					type = BOOLOID;
				}
			}
			break;
		case T_FieldSelect:
			type = ((FieldSelect *) expr)->resulttype;
			break;
		case T_FieldStore:
			type = ((FieldStore *) expr)->resulttype;
			break;
		case T_RelabelType:
			type = ((RelabelType *) expr)->resulttype;
			break;
		case T_ConvertRowtypeExpr:
			type = ((ConvertRowtypeExpr *) expr)->resulttype;
			break;
		case T_CaseExpr:
			type = ((CaseExpr *) expr)->casetype;
			break;
		case T_CaseWhen:
			type = exprType((Node *) ((CaseWhen *) expr)->result);
			break;
		case T_CaseTestExpr:
			type = ((CaseTestExpr *) expr)->typeId;
			break;
		case T_ArrayExpr:
			type = ((ArrayExpr *) expr)->array_typeid;
			break;
		case T_RowExpr:

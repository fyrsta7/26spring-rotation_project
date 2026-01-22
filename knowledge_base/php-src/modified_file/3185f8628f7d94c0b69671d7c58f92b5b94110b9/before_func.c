							break;
					}
				}
			} else if (block->successors_count == 1 && block->successors[0] == next_block_num) {
				op_num = block->start + block->len - 1;
				opline = op_array->opcodes + op_num;
				if (opline->opcode == ZEND_JMP) {
					MAKE_NOP(opline);
					removed_ops++;
				}
		    }
		}

		block_num = next_block_num;
	}

	return removed_ops;
}

void zend_dfa_optimize_op_array(zend_op_array *op_array, zend_optimizer_ctx *ctx, zend_ssa *ssa, zend_call_info **call_map)
{
	if (ctx->debug_level & ZEND_DUMP_BEFORE_DFA_PASS) {
		zend_dump_op_array(op_array, ZEND_DUMP_SSA, "before dfa pass", ssa);
	}

	if (ssa->var_info) {
		int op_1;
		int v;
		int remove_nops = 0;
		zend_op *opline;
		zval tmp;

		if (ZEND_OPTIMIZER_PASS_8 & ctx->optimization_level) {
			if (sccp_optimize_op_array(ctx, op_array, ssa, call_map)) {
				remove_nops = 1;
			}

			if (zend_dfa_optimize_jmps(op_array, ssa)) {
				remove_nops = 1;
			}

#if ZEND_DEBUG_DFA
			ssa_verify_integrity(op_array, ssa, "after sccp");
#endif
			if (ZEND_FUNC_INFO(op_array)) {
				if (zend_dfa_optimize_calls(op_array, ssa)) {
					remove_nops = 1;
				}
			}
			if (ctx->debug_level & ZEND_DUMP_AFTER_PASS_8) {
				zend_dump_op_array(op_array, ZEND_DUMP_SSA, "after sccp pass", ssa);
			}
#if ZEND_DEBUG_DFA
			ssa_verify_integrity(op_array, ssa, "after calls");
#endif
		}

		if (ZEND_OPTIMIZER_PASS_14 & ctx->optimization_level) {
			if (dce_optimize_op_array(op_array, ssa, 0)) {
				remove_nops = 1;
			}
			if (ctx->debug_level & ZEND_DUMP_AFTER_PASS_14) {
				zend_dump_op_array(op_array, ZEND_DUMP_SSA, "after dce pass", ssa);
			}
#if ZEND_DEBUG_DFA
			ssa_verify_integrity(op_array, ssa, "after dce");
#endif
		}

		for (v = op_array->last_var; v < ssa->vars_count; v++) {

			op_1 = ssa->vars[v].definition;

			if (op_1 < 0) {
				continue;
			}

			opline = op_array->opcodes + op_1;

			/* Convert LONG constants to DOUBLE */
			if (ssa->var_info[v].use_as_double) {
				if (opline->opcode == ZEND_ASSIGN
				 && opline->op2_type == IS_CONST
				 && ssa->ops[op_1].op1_def == v
				 && !RETURN_VALUE_USED(opline)
				) {

// op_1: ASSIGN ? -> #v [use_as_double], long(?) => ASSIGN ? -> #v, double(?)

					zval *zv = CT_CONSTANT_EX(op_array, opline->op2.constant);
					ZEND_ASSERT(Z_TYPE_INFO_P(zv) == IS_LONG);
					ZVAL_DOUBLE(&tmp, zval_get_double(zv));
					opline->op2.constant = zend_optimizer_add_literal(op_array, &tmp);

				} else if (opline->opcode == ZEND_QM_ASSIGN
				 && opline->op1_type == IS_CONST
				) {

// op_1: QM_ASSIGN #v [use_as_double], long(?) => QM_ASSIGN #v, double(?)

					zval *zv = CT_CONSTANT_EX(op_array, opline->op1.constant);
					ZEND_ASSERT(Z_TYPE_INFO_P(zv) == IS_LONG);
					ZVAL_DOUBLE(&tmp, zval_get_double(zv));
					opline->op1.constant = zend_optimizer_add_literal(op_array, &tmp);
				}

			} else {
				if (opline->opcode == ZEND_ADD
				 || opline->opcode == ZEND_SUB
				 || opline->opcode == ZEND_MUL
				 || opline->opcode == ZEND_IS_EQUAL
				 || opline->opcode == ZEND_IS_NOT_EQUAL
				 || opline->opcode == ZEND_IS_SMALLER
				 || opline->opcode == ZEND_IS_SMALLER_OR_EQUAL
				) {

					if (opline->op1_type == IS_CONST
					 && opline->op2_type != IS_CONST
					 && (OP2_INFO() & MAY_BE_ANY) == MAY_BE_DOUBLE
					 && Z_TYPE_INFO_P(CT_CONSTANT_EX(op_array, opline->op1.constant)) == IS_LONG
					) {

// op_1: #v.? = ADD long(?), #?.? [double] => #v.? = ADD double(?), #?.? [double]

						zval *zv = CT_CONSTANT_EX(op_array, opline->op1.constant);
						ZVAL_DOUBLE(&tmp, zval_get_double(zv));
						opline->op1.constant = zend_optimizer_add_literal(op_array, &tmp);

					} else if (opline->op1_type != IS_CONST
					 && opline->op2_type == IS_CONST
					 && (OP1_INFO() & MAY_BE_ANY) == MAY_BE_DOUBLE
					 && Z_TYPE_INFO_P(CT_CONSTANT_EX(op_array, opline->op2.constant)) == IS_LONG
					) {

// op_1: #v.? = ADD #?.? [double], long(?) => #v.? = ADD #?.? [double], double(?)

						zval *zv = CT_CONSTANT_EX(op_array, opline->op2.constant);
						ZVAL_DOUBLE(&tmp, zval_get_double(zv));
						opline->op2.constant = zend_optimizer_add_literal(op_array, &tmp);
					}
				}
			}

			if (ssa->vars[v].var >= op_array->last_var) {
				/* skip TMP and VAR */
				continue;
			}

			if (opline->opcode == ZEND_ASSIGN
			 && ssa->ops[op_1].op1_def == v
			 && !RETURN_VALUE_USED(opline)
			) {
				int orig_var = ssa->ops[op_1].op1_use;

				if (orig_var >= 0
				 && !(ssa->var_info[orig_var].type & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE|MAY_BE_REF))
				) {

					int src_var = ssa->ops[op_1].op2_use;

					if ((opline->op2_type & (IS_TMP_VAR|IS_VAR))
					 && src_var >= 0
					 && !(ssa->var_info[src_var].type & MAY_BE_REF)
					 && ssa->vars[src_var].definition >= 0
					 && ssa->ops[ssa->vars[src_var].definition].result_def == src_var
					 && ssa->ops[ssa->vars[src_var].definition].result_use < 0
					 && ssa->vars[src_var].use_chain == op_1
					 && ssa->ops[op_1].op2_use_chain < 0
					 && !ssa->vars[src_var].phi_use_chain
					 && !ssa->vars[src_var].sym_use_chain
					 && opline_supports_assign_contraction(
						 ssa, &op_array->opcodes[ssa->vars[src_var].definition],
						 src_var, opline->op1.var)
					) {

						int op_2 = ssa->vars[src_var].definition;

// op_2: #src_var.T = OP ...                                     => #v.CV = OP ...
// op_1: ASSIGN #orig_var.CV [undef,scalar] -> #v.CV, #src_var.T    NOP

						if (zend_ssa_unlink_use_chain(ssa, op_1, orig_var)) {
							/* Reconstruct SSA */
							ssa->vars[v].definition = op_2;
							ssa->ops[op_2].result_def = v;

							ssa->vars[src_var].definition = -1;
							ssa->vars[src_var].use_chain = -1;

							ssa->ops[op_1].op1_use = -1;
							ssa->ops[op_1].op2_use = -1;
							ssa->ops[op_1].op1_def = -1;
							ssa->ops[op_1].op1_use_chain = -1;

							/* Update opcodes */
							op_array->opcodes[op_2].result_type = opline->op1_type;
							op_array->opcodes[op_2].result.var = opline->op1.var;
							MAKE_NOP(opline);
							remove_nops = 1;
						}
					} else if (opline->op2_type == IS_CONST
					 || ((opline->op2_type & (IS_TMP_VAR|IS_VAR|IS_CV))
					     && ssa->ops[op_1].op2_use >= 0
					     && ssa->ops[op_1].op2_def < 0)
					) {

// op_1: ASSIGN #orig_var.CV [undef,scalar] -> #v.CV, CONST|TMPVAR => QM_ASSIGN v.CV, CONST|TMPVAR

						if (ssa->ops[op_1].op1_use != ssa->ops[op_1].op2_use) {
							zend_ssa_unlink_use_chain(ssa, op_1, orig_var);
						} else {
							ssa->ops[op_1].op2_use_chain = ssa->ops[op_1].op1_use_chain;
						}

						/* Reconstruct SSA */
						ssa->ops[op_1].result_def = v;
						ssa->ops[op_1].op1_def = -1;
						ssa->ops[op_1].op1_use = ssa->ops[op_1].op2_use;
						ssa->ops[op_1].op1_use_chain = ssa->ops[op_1].op2_use_chain;
						ssa->ops[op_1].op2_use = -1;
						ssa->ops[op_1].op2_use_chain = -1;

						/* Update opcode */
						opline->result_type = opline->op1_type;
						opline->result.var = opline->op1.var;
						opline->op1_type = opline->op2_type;
						opline->op1.var = opline->op2.var;
						opline->op2_type = IS_UNUSED;
						opline->op2.var = 0;
						opline->opcode = ZEND_QM_ASSIGN;
					}
				}

			} else if (opline->opcode == ZEND_ASSIGN_ADD
			 && opline->extended_value == 0
			 && ssa->ops[op_1].op1_def == v
			 && opline->op2_type == IS_CONST
			 && Z_TYPE_P(CT_CONSTANT_EX(op_array, opline->op2.constant)) == IS_LONG
			 && Z_LVAL_P(CT_CONSTANT_EX(op_array, opline->op2.constant)) == 1
			 && ssa->ops[op_1].op1_use >= 0
			 && !(ssa->var_info[ssa->ops[op_1].op1_use].type & (MAY_BE_FALSE|MAY_BE_TRUE|MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE|MAY_BE_REF))) {

// op_1: ASSIGN_ADD #?.CV [undef,null,int,foat] ->#v.CV, int(1) => PRE_INC #?.CV ->#v.CV

				opline->opcode = ZEND_PRE_INC;
				SET_UNUSED(opline->op2);

			} else if (opline->opcode == ZEND_ASSIGN_SUB
			 && opline->extended_value == 0
			 && ssa->ops[op_1].op1_def == v
			 && opline->op2_type == IS_CONST
			 && Z_TYPE_P(CT_CONSTANT_EX(op_array, opline->op2.constant)) == IS_LONG
			 && Z_LVAL_P(CT_CONSTANT_EX(op_array, opline->op2.constant)) == 1
			 && ssa->ops[op_1].op1_use >= 0
			 && !(ssa->var_info[ssa->ops[op_1].op1_use].type & (MAY_BE_FALSE|MAY_BE_TRUE|MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE|MAY_BE_REF))) {

// op_1: ASSIGN_SUB #?.CV [undef,null,int,foat] -> #v.CV, int(1) => PRE_DEC #?.CV ->#v.CV

				opline->opcode = ZEND_PRE_DEC;
				SET_UNUSED(opline->op2);

			} else if (opline->opcode == ZEND_VERIFY_RETURN_TYPE
			 && ssa->ops[op_1].op1_def == v
			 && ssa->ops[op_1].op1_use >= 0
			 && ssa->ops[op_1].op1_use_chain == -1
			 && ssa->vars[v].use_chain >= 0
			 && can_elide_return_type_check(op_array, ssa, &ssa->ops[op_1])) {

// op_1: VERIFY_RETURN_TYPE #orig_var.CV [T] -> #v.CV [T] => NOP

				int orig_var = ssa->ops[op_1].op1_use;
				if (zend_ssa_unlink_use_chain(ssa, op_1, orig_var)) {

					int ret = ssa->vars[v].use_chain;

					ssa->ops[ret].op1_use = orig_var;
					ssa->ops[ret].op1_use_chain = ssa->vars[orig_var].use_chain;
					ssa->vars[orig_var].use_chain = ret;

					ssa->vars[v].definition = -1;
					ssa->vars[v].use_chain = -1;

					ssa->ops[op_1].op1_def = -1;
					ssa->ops[op_1].op1_use = -1;

					MAKE_NOP(opline);
					remove_nops = 1;
				}

			} else if (ssa->ops[op_1].op1_def == v
			 && !RETURN_VALUE_USED(opline)
			 && ssa->ops[op_1].op1_use >= 0
			 && !(ssa->var_info[ssa->ops[op_1].op1_use].type & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE|MAY_BE_REF))
			 && (opline->opcode == ZEND_ASSIGN_ADD
			  || opline->opcode == ZEND_ASSIGN_SUB
			  || opline->opcode == ZEND_ASSIGN_MUL
			  || opline->opcode == ZEND_ASSIGN_DIV
			  || opline->opcode == ZEND_ASSIGN_MOD
			  || opline->opcode == ZEND_ASSIGN_SL
			  || opline->opcode == ZEND_ASSIGN_SR
			  || opline->opcode == ZEND_ASSIGN_BW_OR
			  || opline->opcode == ZEND_ASSIGN_BW_AND
			  || opline->opcode == ZEND_ASSIGN_BW_XOR)
			 && opline->extended_value == 0) {

// op_1: ASSIGN_ADD #orig_var.CV [undef,null,bool,int,double] -> #v.CV, ? => #v.CV = ADD #orig_var.CV, ?

				/* Reconstruct SSA */
				ssa->ops[op_1].result_def = ssa->ops[op_1].op1_def;
				ssa->ops[op_1].op1_def = -1;

				/* Update opcode */
				opline->opcode -= (ZEND_ASSIGN_ADD - ZEND_ADD);
				opline->result_type = opline->op1_type;
				opline->result.var = opline->op1.var;


					return 1;
				}
			}
		}
	}
	return 0;
}

typedef struct _zend_tssa {
	zend_ssa        ssa;
	const zend_op **tssa_opcodes;
	int             used_stack;
} zend_tssa;

static const zend_op _nop_opcode = {0};

static zend_ssa *zend_jit_trace_build_tssa(zend_jit_trace_rec *trace_buffer, uint32_t parent_trace, uint32_t exit_num, zend_script *script, const zend_op_array **op_arrays, int *num_op_arrays_ptr)
{
	zend_ssa *tssa;
	zend_ssa_op *ssa_ops, *op;
	zend_ssa_var *ssa_vars;
	zend_ssa_var_info *ssa_var_info;
	const zend_op_array *op_array;
	const zend_op *opline;
	const zend_op **ssa_opcodes;
	zend_jit_trace_rec *p;
	int i, v, idx, len, ssa_ops_count, vars_count, ssa_vars_count;
	zend_jit_trace_stack *stack;
	uint32_t build_flags = ZEND_SSA_RC_INFERENCE | ZEND_SSA_USE_CV_RESULTS;
	uint32_t optimization_level = 0;
	int call_level, level, num_op_arrays, used_stack, max_used_stack;
	size_t frame_size, stack_top, stack_size, stack_bottom;
	zend_jit_op_array_trace_extension *jit_extension;
	zend_ssa *ssa;
	zend_jit_trace_stack_frame *frame, *top, *call;
	zend_ssa_var_info return_value_info;

	/* 1. Count number of TSSA opcodes;
	 *    Count number of activation frames;
	 *    Calculate size of abstract stack;
	 *    Construct regular SSA for involved op_array */
	op_array = trace_buffer->op_array;
	stack_top = stack_size = zend_jit_trace_frame_size(op_array);
	stack_bottom = 0;
	p = trace_buffer + ZEND_JIT_TRACE_START_REC_SIZE;
	ssa_ops_count = 0;
	call_level = 0;
	level = 0;
	num_op_arrays = 0;
	/* Remember op_array to cleanup */
	op_arrays[num_op_arrays++] = op_array;
	/* Build SSA */
	ssa = zend_jit_trace_build_ssa(op_array, script);
	for (;;p++) {
		if (p->op == ZEND_JIT_TRACE_VM) {
			if (JIT_G(opt_level) < ZEND_JIT_LEVEL_OPT_FUNC) {
				const zend_op *opline = p->opline;

				switch (opline->opcode) {
					case ZEND_INCLUDE_OR_EVAL:
						ssa->cfg.flags |= ZEND_FUNC_INDIRECT_VAR_ACCESS;
						break;
					case ZEND_FETCH_R:
					case ZEND_FETCH_W:
					case ZEND_FETCH_RW:
					case ZEND_FETCH_FUNC_ARG:
					case ZEND_FETCH_IS:
					case ZEND_FETCH_UNSET:
					case ZEND_UNSET_VAR:
					case ZEND_ISSET_ISEMPTY_VAR:
						if (opline->extended_value & ZEND_FETCH_LOCAL) {
							ssa->cfg.flags |= ZEND_FUNC_INDIRECT_VAR_ACCESS;
						} else if ((opline->extended_value & (ZEND_FETCH_GLOBAL | ZEND_FETCH_GLOBAL_LOCK)) &&
						           !op_array->function_name) {
							ssa->cfg.flags |= ZEND_FUNC_INDIRECT_VAR_ACCESS;
						}
						break;
				}
			}
			ssa_ops_count += zend_jit_trace_op_len(p->opline);
		} else if (p->op == ZEND_JIT_TRACE_INIT_CALL) {
			call_level++;
			stack_top += zend_jit_trace_frame_size(p->op_array);
			if (stack_top > stack_size) {
				stack_size = stack_top;
			}
		} else if (p->op == ZEND_JIT_TRACE_DO_ICALL) {
			if (JIT_G(opt_level) < ZEND_JIT_LEVEL_OPT_FUNC) {
				if (p->func != (zend_function*)&zend_pass_function
				 && (zend_string_equals_literal(p->func->common.function_name, "extract")
				  || zend_string_equals_literal(p->func->common.function_name, "compact")
				  || zend_string_equals_literal(p->func->common.function_name, "get_defined_vars"))) {
					ssa->cfg.flags |= ZEND_FUNC_INDIRECT_VAR_ACCESS;
				}
			}
			frame_size = zend_jit_trace_frame_size(p->op_array);
			if (call_level == 0) {
				if (stack_top + frame_size > stack_size) {
					stack_size = stack_top + frame_size;
				}
			} else {
				call_level--;
				stack_top -= frame_size;
			}
		} else if (p->op == ZEND_JIT_TRACE_ENTER) {
			op_array = p->op_array;
			if (call_level == 0) {
				stack_top += zend_jit_trace_frame_size(op_array);
				if (stack_top > stack_size) {
					stack_size = stack_top;
				}
			} else {
				call_level--;
			}
			level++;
			jit_extension =
				(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
			ssa = &jit_extension->func_info.ssa;
			if (ssa->cfg.blocks_count) {
				/* pass */
			} else if (num_op_arrays == ZEND_JIT_TRACE_MAX_FUNCS) {
				/* Too many functions in single trace */
				*num_op_arrays_ptr = num_op_arrays;
				return NULL;
			} else {
				/* Remember op_array to cleanup */
				op_arrays[num_op_arrays++] = op_array;
				/* Build SSA */
				ssa = zend_jit_trace_build_ssa(op_array, script);
			}
		} else if (p->op == ZEND_JIT_TRACE_BACK) {
			if (level == 0) {
				stack_bottom += zend_jit_trace_frame_size(p->op_array);
				jit_extension =
					(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
				ssa = &jit_extension->func_info.ssa;
				if (ssa->cfg.blocks_count) {
					/* pass */
				} else if (num_op_arrays == ZEND_JIT_TRACE_MAX_FUNCS) {
					/* Too many functions in single trace */
					*num_op_arrays_ptr = num_op_arrays;
					return NULL;
				} else {
					/* Remember op_array to cleanup */
					op_arrays[num_op_arrays++] = op_array;
					/* Build SSA */
					ssa = zend_jit_trace_build_ssa(op_array, script);
				}
			} else {
				stack_top -= zend_jit_trace_frame_size(op_array);
				level--;
			}
			op_array = p->op_array;
		} else if (p->op == ZEND_JIT_TRACE_END) {
			break;
		}
	}
	*num_op_arrays_ptr = num_op_arrays;

	/* Allocate space for abstract stack */
	JIT_G(current_frame) = frame = (zend_jit_trace_stack_frame*)((char*)zend_arena_alloc(&CG(arena), stack_bottom + stack_size) + stack_bottom);

	/* 2. Construct TSSA */
	tssa = zend_arena_calloc(&CG(arena), 1, sizeof(zend_tssa));
	tssa->cfg.flags = ZEND_SSA_TSSA;
	tssa->cfg.blocks = zend_arena_calloc(&CG(arena), 2, sizeof(zend_basic_block));
	tssa->blocks = zend_arena_calloc(&CG(arena), 2, sizeof(zend_ssa_block));
	tssa->cfg.predecessors = zend_arena_calloc(&CG(arena), 2, sizeof(int));

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_CALL
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET) {
		tssa->cfg.blocks_count = 2;
		tssa->cfg.edges_count = 2;

		tssa->cfg.predecessors[0] = 0;
		tssa->cfg.predecessors[1] = 1;

		tssa->cfg.blocks[0].flags = ZEND_BB_START|ZEND_BB_REACHABLE;
		tssa->cfg.blocks[0].successors_count = 1;
		tssa->cfg.blocks[0].predecessors_count = 0;
		tssa->cfg.blocks[0].successors = tssa->cfg.blocks[0].successors_storage;
		tssa->cfg.blocks[0].successors[0] = 1;

		tssa->cfg.blocks[0].flags = ZEND_BB_FOLLOW|ZEND_BB_TARGET|ZEND_BB_LOOP_HEADER|ZEND_BB_REACHABLE;
		tssa->cfg.blocks[1].successors_count = 1;
		tssa->cfg.blocks[1].predecessors_count = 2;
		tssa->cfg.blocks[1].successors = tssa->cfg.blocks[1].successors_storage;
		tssa->cfg.blocks[1].successors[1] = 1;
	} else {
		tssa->cfg.blocks_count = 1;
		tssa->cfg.edges_count = 0;

		tssa->cfg.blocks[0].flags = ZEND_BB_START|ZEND_BB_EXIT|ZEND_BB_REACHABLE;
		tssa->cfg.blocks[0].successors_count = 0;
		tssa->cfg.blocks[0].predecessors_count = 0;
	}
	((zend_tssa*)tssa)->used_stack = -1;

	if (JIT_G(opt_level) < ZEND_JIT_LEVEL_INLINE) {
		return tssa;
	}

	tssa->ops = ssa_ops = zend_arena_alloc(&CG(arena), ssa_ops_count * sizeof(zend_ssa_op));
	memset(ssa_ops, -1, ssa_ops_count * sizeof(zend_ssa_op));
	ssa_opcodes = zend_arena_calloc(&CG(arena), ssa_ops_count + 1, sizeof(zend_op*));
	((zend_tssa*)tssa)->tssa_opcodes = ssa_opcodes;
	ssa_opcodes[ssa_ops_count] = &_nop_opcode;

	op_array = trace_buffer->op_array;
	if (trace_buffer->start == ZEND_JIT_TRACE_START_ENTER) {
		ssa_vars_count = op_array->last_var;
	} else {
		ssa_vars_count = op_array->last_var + op_array->T;
	}
	stack = frame->stack;
	for (i = 0; i < ssa_vars_count; i++) {
		SET_STACK_VAR(stack, i, i);
	}

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP) {
		// TODO: For tracing, it's possible, to create pseudo Phi functions
		//       at the end of loop, without this additional pass (like LuaJIT) ???
		ssa_vars_count = zend_jit_trace_add_phis(trace_buffer, ssa_vars_count, tssa, stack);
	} else if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_CALL) {
		ssa_vars_count = zend_jit_trace_add_call_phis(trace_buffer, ssa_vars_count, tssa, stack);
	} else if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET) {
		ssa_vars_count = zend_jit_trace_add_ret_phis(trace_buffer, ssa_vars_count, tssa, stack);
	}

	p = trace_buffer + ZEND_JIT_TRACE_START_REC_SIZE;
	idx = 0;
	level = 0;
	for (;;p++) {
		if (p->op == ZEND_JIT_TRACE_VM) {
			opline = p->opline;
			ssa_opcodes[idx] = opline;
			ssa_vars_count = zend_ssa_rename_op(op_array, opline, idx, build_flags, ssa_vars_count, ssa_ops, (int*)stack);
			idx++;
			len = zend_jit_trace_op_len(p->opline);
			while (len > 1) {
				opline++;
				ssa_opcodes[idx] = opline;
				if (opline->opcode != ZEND_OP_DATA) {
					ssa_vars_count = zend_ssa_rename_op(op_array, opline, idx, build_flags, ssa_vars_count, ssa_ops, (int*)stack);
				}
				idx++;
				len--;
			}
		} else if (p->op == ZEND_JIT_TRACE_ENTER) {
			frame = zend_jit_trace_call_frame(frame, op_array);
			stack = frame->stack;
			op_array = p->op_array;
			level++;
			if (ssa_vars_count >= ZEND_JIT_TRACE_MAX_SSA_VAR) {
				return NULL;
			}
			ZEND_JIT_TRACE_SET_FIRST_SSA_VAR(p->info, ssa_vars_count);
			for (i = 0; i < op_array->last_var; i++) {
				SET_STACK_VAR(stack, i, ssa_vars_count++);
			}
		} else if (p->op == ZEND_JIT_TRACE_BACK) {
			op_array = p->op_array;
			frame = zend_jit_trace_ret_frame(frame, op_array);
			stack = frame->stack;
			if (level == 0) {
				if (ssa_vars_count >= ZEND_JIT_TRACE_MAX_SSA_VAR) {
					return NULL;
				}
				ZEND_JIT_TRACE_SET_FIRST_SSA_VAR(p->info, ssa_vars_count);
				for (i = 0; i < op_array->last_var + op_array->T; i++) {
					SET_STACK_VAR(stack, i, ssa_vars_count++);
				}
			} else {
				level--;
			}
		} else if (p->op == ZEND_JIT_TRACE_END) {
			break;
		}
	}

	op_array = trace_buffer->op_array;
	tssa->vars_count = ssa_vars_count;
	tssa->vars = ssa_vars = zend_arena_calloc(&CG(arena), tssa->vars_count, sizeof(zend_ssa_var));
	if (trace_buffer->start == ZEND_JIT_TRACE_START_ENTER) {
		vars_count = op_array->last_var;
	} else {
		vars_count = op_array->last_var + op_array->T;
	}
	i = 0;
	while (i < vars_count) {
		ssa_vars[i].var = i;
		ssa_vars[i].scc = -1;
		ssa_vars[i].definition = -1;
		ssa_vars[i].use_chain = -1;
		i++;
	}
	while (i < tssa->vars_count) {
		ssa_vars[i].var = -1;
		ssa_vars[i].scc = -1;
		ssa_vars[i].definition = -1;
		ssa_vars[i].use_chain = -1;
		i++;
	}

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_CALL
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET) {
		/* Update Phi sources */
		zend_ssa_phi *phi = tssa->blocks[1].phis;

		while (phi) {
			phi->sources[1] = STACK_VAR(stack, phi->var);
			ssa_vars[phi->ssa_var].var = phi->var;
			ssa_vars[phi->ssa_var].definition_phi = phi;
			ssa_vars[phi->sources[0]].phi_use_chain = phi;
			ssa_vars[phi->sources[1]].phi_use_chain = phi;
			phi = phi->next;
		}
	}

	/* 3. Compute use-def chains */
	idx = (ssa_ops_count - 1);
	op = ssa_ops + idx;
	while (idx >= 0) {
		opline = ssa_opcodes[idx];
		if (op->op1_use >= 0) {
			op->op1_use_chain = ssa_vars[op->op1_use].use_chain;
			ssa_vars[op->op1_use].use_chain = idx;
		}
		if (op->op2_use >= 0 && op->op2_use != op->op1_use) {
			op->op2_use_chain = ssa_vars[op->op2_use].use_chain;
			ssa_vars[op->op2_use].use_chain = idx;
		}
		if (op->result_use >= 0 && op->result_use != op->op1_use && op->result_use != op->op2_use) {
			op->res_use_chain = ssa_vars[op->result_use].use_chain;
			ssa_vars[op->result_use].use_chain = idx;
		}
		if (op->op1_def >= 0) {
			ssa_vars[op->op1_def].var = EX_VAR_TO_NUM(opline->op1.var);
			ssa_vars[op->op1_def].definition = idx;
		}
		if (op->op2_def >= 0) {
			ssa_vars[op->op2_def].var = EX_VAR_TO_NUM(opline->op2.var);
			ssa_vars[op->op2_def].definition = idx;
		}
		if (op->result_def >= 0) {
			ssa_vars[op->result_def].var = EX_VAR_TO_NUM(opline->result.var);
			ssa_vars[op->result_def].definition = idx;
		}
		op--;
		idx--;
	}

	/* 4. Type inference */
	op_array = trace_buffer->op_array;
	jit_extension =
		(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
	ssa = &jit_extension->func_info.ssa;

	tssa->var_info = ssa_var_info = zend_arena_calloc(&CG(arena), tssa->vars_count, sizeof(zend_ssa_var_info));

	if (trace_buffer->start == ZEND_JIT_TRACE_START_ENTER) {
		i = 0;
		while (i < op_array->last_var) {
			if (i < op_array->num_args) {
				if (ssa->var_info
				 && zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, i)) {
					/* pass */
				} else {
					if (ssa->vars) {
						ssa_vars[i].no_val = ssa->vars[i].no_val;
						ssa_vars[i].alias = ssa->vars[i].alias;
					} else {
						ssa_vars[i].alias = zend_jit_var_may_alias(op_array, ssa, i);
					}
					if (op_array->arg_info) {
						zend_arg_info *arg_info = &op_array->arg_info[i];
						zend_class_entry *ce;
						uint32_t tmp = zend_fetch_arg_info_type(script, arg_info, &ce);

						if (ZEND_ARG_SEND_MODE(arg_info)) {
							tmp |= MAY_BE_REF;
						}
						ssa_var_info[i].type = tmp;
						ssa_var_info[i].ce = ce;
						ssa_var_info[i].is_instanceof = 1;
					} else {
						ssa_var_info[i].type = MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY  | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
					}
				}
			} else {
				if (ssa->vars) {
					ssa_vars[i].no_val = ssa->vars[i].no_val;
					ssa_vars[i].alias = ssa->vars[i].alias;
				} else {
					ssa_vars[i].alias = zend_jit_var_may_alias(op_array, ssa, i);
				}
				if (ssa_vars[i].alias == NO_ALIAS) {
					ssa_var_info[i].type = MAY_BE_UNDEF;
				} else {
					ssa_var_info[i].type = MAY_BE_UNDEF | MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
				}
			}
			i++;
		}
	} else {
		int parent_vars_count = 0;
		zend_jit_trace_stack *parent_stack = NULL;

		i = 0;
		if (parent_trace) {
			parent_vars_count = MIN(zend_jit_traces[parent_trace].exit_info[exit_num].stack_size,
				op_array->last_var + op_array->T);
			if (parent_vars_count) {
				parent_stack =
					zend_jit_traces[parent_trace].stack_map +
					zend_jit_traces[parent_trace].exit_info[exit_num].stack_offset;
			}
		}
		while (i < op_array->last_var + op_array->T) {
			if (!ssa->var_info
			 || !zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, i)) {
				if (ssa->vars) {
					ssa_vars[i].alias = ssa->vars[i].alias;
				} else {
					ssa_vars[i].alias = zend_jit_var_may_alias(op_array, ssa, i);
				}
				if (i < op_array->last_var) {
					ssa_var_info[i].type = MAY_BE_UNDEF | MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY  | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
				} else {
					ssa_var_info[i].type = MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
				}
			}
			if (i < parent_vars_count) {
				/* Initialize TSSA variable from parent trace */
				zend_uchar op_type = STACK_TYPE(parent_stack, i);

				if (op_type != IS_UNKNOWN) {
					ssa_var_info[i].type &= zend_jit_trace_type_to_info(op_type);
				}
			}
			i++;
		}
	}

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_CALL
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET) {
		/* Propagate initial value through Phi functions */
		zend_ssa_phi *phi = tssa->blocks[1].phis;

		while (phi) {
			if (!ssa->var_info
			 || !zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, phi->ssa_var)) {
				ssa_vars[phi->ssa_var].alias = ssa_vars[phi->sources[0]].alias;
				ssa_var_info[phi->ssa_var].type = ssa_var_info[phi->sources[0]].type;
			}
			phi = phi->next;
		}
	}

	frame = JIT_G(current_frame);
	top = zend_jit_trace_call_frame(frame, op_array);
	TRACE_FRAME_INIT(frame, op_array, 0, 0);
	TRACE_FRAME_SET_RETURN_SSA_VAR(frame, -1);
	frame->used_stack = 0;
	for (i = 0; i < op_array->last_var + op_array->T; i++) {
		SET_STACK_TYPE(frame->stack, i, IS_UNKNOWN, 1);
	}
	memset(&return_value_info, 0, sizeof(return_value_info));

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP) {
		max_used_stack = used_stack = 0;
	} else {
		max_used_stack = used_stack = -1;
	}

	p = trace_buffer + ZEND_JIT_TRACE_START_REC_SIZE;
	idx = 0;
	level = 0;
	opline = NULL;
	for (;;p++) {
		if (p->op == ZEND_JIT_TRACE_VM) {
			uint8_t orig_op1_type, orig_op2_type, op1_type, op2_type, op3_type;
			uint8_t val_type = IS_UNKNOWN;
//			zend_class_entry *op1_ce = NULL;
			zend_class_entry *op2_ce = NULL;

			opline = p->opline;

			op1_type = orig_op1_type = p->op1_type;
			op2_type = orig_op2_type = p->op2_type;
			op3_type = p->op3_type;
			if (op1_type & (IS_TRACE_REFERENCE|IS_TRACE_INDIRECT)) {
				op1_type = IS_UNKNOWN;
			}
			if (op1_type != IS_UNKNOWN) {
				op1_type &= ~IS_TRACE_PACKED;
			}
			if (op2_type & (IS_TRACE_REFERENCE|IS_TRACE_INDIRECT)) {
				op2_type = IS_UNKNOWN;
			}
			if (op3_type & (IS_TRACE_REFERENCE|IS_TRACE_INDIRECT)) {
				op3_type = IS_UNKNOWN;
			}

			if ((p+1)->op == ZEND_JIT_TRACE_OP1_TYPE) {
//				op1_ce = (zend_class_entry*)(p+1)->ce;
				p++;
			}
			if ((p+1)->op == ZEND_JIT_TRACE_OP2_TYPE) {
				op2_ce = (zend_class_entry*)(p+1)->ce;
				p++;
			}
			if ((p+1)->op == ZEND_JIT_TRACE_VAL_INFO) {
				val_type = (p+1)->op1_type;
				p++;
			}

			switch (opline->opcode) {
				case ZEND_ASSIGN_OP:
					if (opline->extended_value == ZEND_POW
					 || opline->extended_value == ZEND_DIV) {
						// TODO: check for division by zero ???
						break;
					}
					if (opline->op1_type != IS_CV || opline->result_type != IS_UNUSED) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					ADD_OP2_TRACE_GUARD();
					break;
				case ZEND_ASSIGN_DIM_OP:
					if (opline->extended_value == ZEND_POW
					 || opline->extended_value == ZEND_DIV) {
						// TODO: check for division by zero ???
						break;
					}
					if (opline->result_type != IS_UNUSED) {
						break;
					}
					ZEND_FALLTHROUGH;
				case ZEND_ASSIGN_DIM:
					if (opline->op1_type == IS_CV) {
						ADD_OP1_DATA_TRACE_GUARD();
						ADD_OP2_TRACE_GUARD();
						ADD_OP1_TRACE_GUARD();
					} else if (orig_op1_type != IS_UNKNOWN
					        && (orig_op1_type & IS_TRACE_INDIRECT)
					        && opline->result_type == IS_UNUSED) {
						if (opline->opcode == ZEND_ASSIGN_DIM_OP) {
							ADD_OP1_DATA_TRACE_GUARD();
						}
						ADD_OP2_TRACE_GUARD();
					}
					if (op1_type == IS_ARRAY
					 && ((opline->op2_type == IS_CONST
					   && Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) == IS_LONG)
					  || (opline->op2_type != IS_CONST
					   && op2_type == IS_LONG))) {

						if (!(orig_op1_type & IS_TRACE_PACKED)) {
							zend_ssa_var_info *info = &tssa->var_info[tssa->ops[idx].op1_use];

							if (MAY_BE_PACKED(info->type) && MAY_BE_HASH(info->type)) {
								info->type |= MAY_BE_PACKED_GUARD;
								info->type &= ~MAY_BE_ARRAY_PACKED;
							}
						} else if (opline->opcode == ZEND_ASSIGN_DIM_OP
								&& val_type != IS_UNKNOWN
								&& val_type != IS_UNDEF) {
							zend_ssa_var_info *info = &tssa->var_info[tssa->ops[idx].op1_use];

							if (MAY_BE_PACKED(info->type) && MAY_BE_HASH(info->type)) {
								info->type |= MAY_BE_PACKED_GUARD;
								info->type &= ~(MAY_BE_ARRAY_NUMERIC_HASH|MAY_BE_ARRAY_STRING_HASH);
							}
						}
					}
					break;
				case ZEND_ASSIGN_OBJ_OP:
					if (opline->extended_value == ZEND_POW
					 || opline->extended_value == ZEND_DIV) {
						// TODO: check for division by zero ???
						break;
					}
					if (opline->result_type != IS_UNUSED) {
						break;
					}
					ZEND_FALLTHROUGH;
				case ZEND_ASSIGN_OBJ:
				case ZEND_PRE_INC_OBJ:
				case ZEND_PRE_DEC_OBJ:
				case ZEND_POST_INC_OBJ:
				case ZEND_POST_DEC_OBJ:
					if (opline->op2_type != IS_CONST
					 || Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) != IS_STRING
					 || Z_STRVAL_P(RT_CONSTANT(opline, opline->op2))[0] == '\0') {
						break;
					}
					if (opline->opcode == ZEND_ASSIGN_OBJ_OP) {
						ADD_OP1_DATA_TRACE_GUARD();
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_IS_EQUAL:
				case ZEND_IS_NOT_EQUAL:
				case ZEND_IS_SMALLER:
				case ZEND_IS_SMALLER_OR_EQUAL:
				case ZEND_CASE:
				case ZEND_IS_IDENTICAL:
				case ZEND_IS_NOT_IDENTICAL:
				case ZEND_CASE_STRICT:
				case ZEND_BW_OR:
				case ZEND_BW_AND:
				case ZEND_BW_XOR:
				case ZEND_SL:
				case ZEND_SR:
				case ZEND_MOD:
				case ZEND_ADD:
				case ZEND_SUB:
				case ZEND_MUL:
//				case ZEND_DIV: // TODO: check for division by zero ???
				case ZEND_CONCAT:
				case ZEND_FAST_CONCAT:
					ADD_OP2_TRACE_GUARD();
					ZEND_FALLTHROUGH;
				case ZEND_ECHO:
				case ZEND_STRLEN:
				case ZEND_COUNT:
				case ZEND_QM_ASSIGN:
				case ZEND_FE_RESET_R:
				case ZEND_FE_FETCH_R:
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_VERIFY_RETURN_TYPE:
					if (opline->op1_type == IS_UNUSED) {
						/* Always throws */
						break;
					}
					if (opline->op1_type == IS_CONST) {
						/* TODO Different instruction format, has return value */
						break;
					}
					if (op_array->fn_flags & ZEND_ACC_RETURN_REFERENCE) {
						/* Not worth bothering with */
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_FETCH_DIM_FUNC_ARG:
					if (!frame
					 || !frame->call
					 || !frame->call->func
					 || !TRACE_FRAME_IS_LAST_SEND_BY_VAL(frame->call)) {
						break;
					}
					ADD_OP2_TRACE_GUARD();
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_PRE_INC:
				case ZEND_PRE_DEC:
				case ZEND_POST_INC:
				case ZEND_POST_DEC:
					if (opline->op1_type != IS_CV) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_ASSIGN:
					if (opline->op1_type != IS_CV) {
						break;
					}
					ADD_OP2_TRACE_GUARD();
					if (op1_type != IS_UNKNOWN
					 && (tssa->var_info[tssa->ops[idx].op1_use].type & MAY_BE_REF)) {
						ADD_OP1_TRACE_GUARD();
					}
					break;
				case ZEND_CAST:
					if (opline->extended_value != op1_type) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_JMPZ:
				case ZEND_JMPNZ:
				case ZEND_JMPZNZ:
				case ZEND_JMPZ_EX:
				case ZEND_JMPNZ_EX:
				case ZEND_BOOL:
				case ZEND_BOOL_NOT:
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_ISSET_ISEMPTY_CV:
					if ((opline->extended_value & ZEND_ISEMPTY)) {
						// TODO: support for empty() ???
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_IN_ARRAY:
					if (opline->op1_type == IS_VAR || opline->op1_type == IS_TMP_VAR) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_ISSET_ISEMPTY_DIM_OBJ:
					if ((opline->extended_value & ZEND_ISEMPTY)) {
						// TODO: support for empty() ???
						break;
					}
					ZEND_FALLTHROUGH;
				case ZEND_FETCH_DIM_R:
				case ZEND_FETCH_DIM_IS:
				case ZEND_FETCH_LIST_R:
					ADD_OP1_TRACE_GUARD();
					ADD_OP2_TRACE_GUARD();

					if (op1_type == IS_ARRAY
					 && opline->op1_type != IS_CONST
					 && ((opline->op2_type == IS_CONST
					   && Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) == IS_LONG)
					  || (opline->op2_type != IS_CONST
					   && op2_type == IS_LONG))) {

						zend_ssa_var_info *info = &tssa->var_info[tssa->ops[idx].op1_use];

						if (MAY_BE_PACKED(info->type) && MAY_BE_HASH(info->type)) {
							info->type |= MAY_BE_PACKED_GUARD;
							if (orig_op1_type & IS_TRACE_PACKED) {
								info->type &= ~(MAY_BE_ARRAY_NUMERIC_HASH|MAY_BE_ARRAY_STRING_HASH);
							} else {
								info->type &= ~MAY_BE_ARRAY_PACKED;
							}
						}
					}
					break;
				case ZEND_FETCH_DIM_W:
				case ZEND_FETCH_DIM_RW:
//				case ZEND_FETCH_DIM_UNSET:
				case ZEND_FETCH_LIST_W:
					if (opline->op1_type != IS_CV
					 && (orig_op1_type == IS_UNKNOWN
					  || !(orig_op1_type & IS_TRACE_INDIRECT))) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					ADD_OP2_TRACE_GUARD();
					if (op1_type == IS_ARRAY
					 && !(orig_op1_type & IS_TRACE_PACKED)
					 && ((opline->op2_type == IS_CONST
					   && Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) == IS_LONG)
					  || (opline->op2_type != IS_CONST
					   && op2_type == IS_LONG))) {

						zend_ssa_var_info *info = &tssa->var_info[tssa->ops[idx].op1_use];

						if (MAY_BE_PACKED(info->type) && MAY_BE_HASH(info->type)) {
							info->type |= MAY_BE_PACKED_GUARD;
							info->type &= ~MAY_BE_ARRAY_PACKED;
						}
					}
					break;
				case ZEND_SEND_VAL_EX:
				case ZEND_SEND_VAR_EX:
				case ZEND_SEND_VAR_NO_REF_EX:
					if (opline->op2_type == IS_CONST) {
						/* Named parameters not supported in JIT */
						break;
					}
					if (opline->op2.num > MAX_ARG_FLAG_NUM) {
						goto propagate_arg;
					}
					ZEND_FALLTHROUGH;
				case ZEND_SEND_VAL:
				case ZEND_SEND_VAR:
				case ZEND_SEND_VAR_NO_REF:
				case ZEND_SEND_FUNC_ARG:
					if (opline->op2_type == IS_CONST) {
						/* Named parameters not supported in JIT */
						break;
					}
					ADD_OP1_TRACE_GUARD();
propagate_arg:
					/* Propagate argument type */
					if (frame->call
					 && frame->call->func
					 && frame->call->func->type == ZEND_USER_FUNCTION
					 && opline->op2.num <= frame->call->func->op_array.num_args) {
						uint32_t info;

						if (opline->op1_type == IS_CONST) {
							info = _const_op_type(RT_CONSTANT(opline, opline->op1));
						} else {
							ZEND_ASSERT(ssa_ops[idx].op1_use >= 0);
							info = ssa_var_info[ssa_ops[idx].op1_use].type & ~MAY_BE_GUARD;
						}
						if (frame->call->func->op_array.fn_flags & ZEND_ACC_HAS_TYPE_HINTS) {
							zend_arg_info *arg_info;

							ZEND_ASSERT(frame->call->func->op_array.arg_info);
							arg_info = &frame->call->func->op_array.arg_info[opline->op2.num - 1];
							if (ZEND_TYPE_IS_SET(arg_info->type)) {
								zend_class_entry *ce;
								uint32_t tmp = zend_fetch_arg_info_type(script, arg_info, &ce);
								info &= tmp;
								if (!info) {
									break;
								}
							}
						}
						if (opline->op1_type == IS_CV && (info & MAY_BE_RC1)) {
							info |= MAY_BE_RCN;
						}
						if (info & MAY_BE_UNDEF) {
							info |= MAY_BE_NULL;
							info &= ~MAY_BE_UNDEF;
						}
						if (ARG_SHOULD_BE_SENT_BY_REF(frame->call->func, opline->op2.num)) {
							info |= MAY_BE_REF|MAY_BE_RC1|MAY_BE_RCN|MAY_BE_ANY|MAY_BE_ARRAY_OF_ANY|MAY_BE_ARRAY_KEY_ANY;
						}
						SET_STACK_INFO(frame->call->stack, opline->op2.num - 1, info);
					}
					break;
				case ZEND_RETURN:
					ADD_OP1_TRACE_GUARD();
					/* Propagate return value types */
					if (opline->op1_type == IS_UNUSED) {
						return_value_info.type = MAY_BE_NULL;
					} else if (opline->op1_type == IS_CONST) {
						return_value_info.type = _const_op_type(RT_CONSTANT(opline, opline->op1));
					} else {
						ZEND_ASSERT(ssa_ops[idx].op1_use >= 0);
						return_value_info = ssa_var_info[ssa_ops[idx].op1_use];
						if (return_value_info.type & MAY_BE_UNDEF) {
							return_value_info.type &= ~MAY_BE_UNDEF;
							return_value_info.type |= MAY_BE_NULL;
						}
						if (return_value_info.type & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE)) {
							/* CVs are going to be destructed and the reference-counter
							   of return value may be decremented to 1 */
							return_value_info.type |= MAY_BE_RC1;
						}
						return_value_info.type &= ~MAY_BE_GUARD;
					}
					break;
				case ZEND_CHECK_FUNC_ARG:
					if (!frame
					 || !frame->call
					 || !frame->call->func) {
						break;
					}
					if (opline->op2_type == IS_CONST
					 || opline->op2.num > MAX_ARG_FLAG_NUM) {
						/* Named parameters not supported in JIT */
						TRACE_FRAME_SET_LAST_SEND_UNKNOWN(frame->call);
						break;
					}
					if (ARG_SHOULD_BE_SENT_BY_REF(frame->call->func, opline->op2.num)) {
						TRACE_FRAME_SET_LAST_SEND_BY_REF(frame->call);
					} else {
						TRACE_FRAME_SET_LAST_SEND_BY_VAL(frame->call);
					}
					break;
				case ZEND_FETCH_OBJ_FUNC_ARG:
					if (!frame
					 || !frame->call
					 || !frame->call->func
					 || !TRACE_FRAME_IS_LAST_SEND_BY_VAL(frame->call)) {
						break;
					}
					ZEND_FALLTHROUGH;
				case ZEND_FETCH_OBJ_R:
				case ZEND_FETCH_OBJ_IS:
				case ZEND_FETCH_OBJ_W:
					if (opline->op2_type != IS_CONST
					 || Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) != IS_STRING
					 || Z_STRVAL_P(RT_CONSTANT(opline, opline->op2))[0] == '\0') {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_INIT_METHOD_CALL:
					if (opline->op2_type != IS_CONST
					 || Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) != IS_STRING) {
						break;
					}
					ADD_OP1_TRACE_GUARD();
					break;
				case ZEND_INIT_DYNAMIC_CALL:
					if (orig_op2_type == IS_OBJECT && op2_ce == zend_ce_closure) {
						ADD_OP2_TRACE_GUARD();
					}
					break;
				case ZEND_SEND_ARRAY:
				case ZEND_SEND_UNPACK:
				case ZEND_CHECK_UNDEF_ARGS:
				case ZEND_INCLUDE_OR_EVAL:
					max_used_stack = used_stack = -1;
					break;
				case ZEND_TYPE_CHECK:
					if (opline->extended_value == MAY_BE_RESOURCE) {
						// TODO: support for is_resource() ???
						break;
					}
					if (op1_type != IS_UNKNOWN
					 && (opline->extended_value == (1 << op1_type)
					  || opline->extended_value == MAY_BE_ANY - (1 << op1_type))) {
						/* add guards only for exact checks, to avoid code duplication */
						ADD_OP1_TRACE_GUARD();
					}
					break;
				case ZEND_ROPE_INIT:
				case ZEND_ROPE_ADD:
				case ZEND_ROPE_END:
					ADD_OP2_TRACE_GUARD();
					break;
				default:
					break;
			}
			len = zend_jit_trace_op_len(opline);
			if (ssa->var_info) {
				/* Add statically inferred ranges */
				if (ssa_ops[idx].op1_def >= 0) {
					zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
				}
				if (ssa_ops[idx].op2_def >= 0) {
					zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
				}
				if (ssa_ops[idx].result_def >= 0) {
					zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].result_def);
				}
				if (len == 2 && (opline+1)->opcode == ZEND_OP_DATA) {
					if (ssa_ops[idx+1].op1_def >= 0) {
						zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx+1].op1_def);
					}
					if (ssa_ops[idx+1].op2_def >= 0) {
						zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx+1].op2_def);
					}
					if (ssa_ops[idx+1].result_def >= 0) {
						zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx+1].result_def);
					}
				}
			} else {
				if (ssa_ops[idx].op1_def >= 0) {
					ssa_vars[ssa_ops[idx].op1_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->op1.var));
					if (ssa_ops[idx].op1_use < 0 || !(ssa_var_info[ssa_ops[idx].op1_use].type & MAY_BE_REF)) {
						zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
					}
				}
				if (ssa_ops[idx].op2_def >= 0) {
					ssa_vars[ssa_ops[idx].op2_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->op2.var));
					if (ssa_ops[idx].op2_use < 0 || !(ssa_var_info[ssa_ops[idx].op2_use].type & MAY_BE_REF)) {
						zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
					}
				}
				if (ssa_ops[idx].result_def >= 0) {
					ssa_vars[ssa_ops[idx].result_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->result.var));
					if (ssa_ops[idx].result_use < 0 || !(ssa_var_info[ssa_ops[idx].result_use].type & MAY_BE_REF)) {
						zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].result_def);
					}
				}
				if (len == 2 && (opline+1)->opcode == ZEND_OP_DATA) {
					if (ssa_ops[idx+1].op1_def >= 0) {
						ssa_vars[ssa_ops[idx+1].op1_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM((opline+1)->op1.var));
						if (ssa_ops[idx+1].op1_use < 0 || !(ssa_var_info[ssa_ops[idx+1].op1_use].type & MAY_BE_REF)) {
							zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx+1].op1_def);
						}
					}
					if (ssa_ops[idx+1].op2_def >= 0) {
						ssa_vars[ssa_ops[idx+1].op2_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM((opline+1)->op2.var));
						if (ssa_ops[idx+1].op2_use < 0 || !(ssa_var_info[ssa_ops[idx+1].op2_use].type & MAY_BE_REF)) {
							zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx+1].op2_def);
						}
					}
					if (ssa_ops[idx+1].result_def >= 0) {
						ssa_vars[ssa_ops[idx+1].result_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM((opline+1)->result.var));
						if (ssa_ops[idx+1].result_use < 0 || !(ssa_var_info[ssa_ops[idx+1].result_use].type & MAY_BE_REF)) {
							zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx+1].result_def);
						}
					}
				}
			}
			if (opline->opcode == ZEND_RECV_INIT
			 && !(op_array->fn_flags & ZEND_ACC_HAS_TYPE_HINTS)) {
				/* RECV_INIT always copy the constant */
				ssa_var_info[ssa_ops[idx].result_def].type = _const_op_type(RT_CONSTANT(opline, opline->op2));
			} else if ((opline->opcode == ZEND_FE_FETCH_R || opline->opcode == ZEND_FE_FETCH_RW)
			 && ssa_opcodes[idx + 1] == ZEND_OFFSET_TO_OPLINE(opline, opline->extended_value)) {
				if (ssa_ops[idx].op2_use >= 0 && ssa_ops[idx].op2_def >= 0) {
					ssa_var_info[ssa_ops[idx].op2_def] = ssa_var_info[ssa_ops[idx].op2_use];
				}
			} else {
				if (zend_update_type_info(op_array, tssa, script, (zend_op*)opline, ssa_ops + idx, ssa_opcodes, optimization_level) == FAILURE) {
					// TODO:
					assert(0);
				}
				if (opline->opcode == ZEND_ASSIGN_DIM_OP
				 && ssa_ops[idx].op1_def > 0
				 && op1_type == IS_ARRAY
				 && (orig_op1_type & IS_TRACE_PACKED)
				 && val_type != IS_UNKNOWN
				 && val_type != IS_UNDEF
				 && ((opline->op2_type == IS_CONST
				   && Z_TYPE_P(RT_CONSTANT(opline, opline->op2)) == IS_LONG)
				  || (opline->op2_type != IS_CONST
				   && op2_type == IS_LONG))) {
					zend_ssa_var_info *info = &ssa_var_info[ssa_ops[idx].op1_def];

					info->type &= ~(MAY_BE_ARRAY_NUMERIC_HASH|MAY_BE_ARRAY_STRING_HASH);
				}
			}
			if (ssa->var_info) {
				/* Add statically inferred restrictions */
				if (ssa_ops[idx].op1_def >= 0) {
					if (opline->opcode == ZEND_SEND_VAR_EX
					 && frame
					 && frame->call
					 && frame->call->func
					 && !ARG_SHOULD_BE_SENT_BY_REF(frame->call->func, opline->op2.num)) {
						ssa_var_info[ssa_ops[idx].op1_def] = ssa_var_info[ssa_ops[idx].op1_use];
						ssa_var_info[ssa_ops[idx].op1_def].type &= ~MAY_BE_GUARD;
						if (ssa_var_info[ssa_ops[idx].op1_def].type & MAY_BE_RC1) {
							ssa_var_info[ssa_ops[idx].op1_def].type |= MAY_BE_RCN;
						}
					} else {
						zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
					}
				}
				if (ssa_ops[idx].op2_def >= 0) {
					if ((opline->opcode != ZEND_FE_FETCH_R && opline->opcode != ZEND_FE_FETCH_RW)
					 || ssa_opcodes[idx + 1] != ZEND_OFFSET_TO_OPLINE(opline, opline->extended_value)) {
						zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
					}
				}
				if (ssa_ops[idx].result_def >= 0) {
					zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].result_def);
				}
			}
			idx++;
			while (len > 1) {
				opline++;
				if (opline->opcode != ZEND_OP_DATA) {
					if (ssa->var_info) {
						/* Add statically inferred ranges */
						if (ssa_ops[idx].op1_def >= 0) {
							zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
						}
						if (ssa_ops[idx].op2_def >= 0) {
							zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
						}
						if (ssa_ops[idx].result_def >= 0) {
							zend_jit_trace_copy_ssa_var_range(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].result_def);
						}
					} else {
						if (ssa_ops[idx].op1_def >= 0) {
							ssa_vars[ssa_ops[idx].op1_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->op1.var));
							if (ssa_ops[idx].op1_use < 0 || !(ssa_var_info[ssa_ops[idx].op1_use].type & MAY_BE_REF)) {
								zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
							}
						}
						if (ssa_ops[idx].op2_def >= 0) {
							ssa_vars[ssa_ops[idx].op2_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->op2.var));
							if (ssa_ops[idx].op2_use < 0 || !(ssa_var_info[ssa_ops[idx].op2_use].type & MAY_BE_REF)) {
								zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
							}
						}
						if (ssa_ops[idx].result_def >= 0) {
							ssa_vars[ssa_ops[idx].result_def].alias = zend_jit_var_may_alias(op_array, ssa, EX_VAR_TO_NUM(opline->result.var));
							if (ssa_ops[idx].result_use < 0 || !(ssa_var_info[ssa_ops[idx].result_use].type & MAY_BE_REF)) {
								zend_jit_trace_propagate_range(op_array, ssa_opcodes, tssa, ssa_ops[idx].result_def);
							}
						}
					}
					if (opline->opcode == ZEND_RECV_INIT
					 && !(op_array->fn_flags & ZEND_ACC_HAS_TYPE_HINTS)) {
						/* RECV_INIT always copy the constant */
						ssa_var_info[ssa_ops[idx].result_def].type = _const_op_type(RT_CONSTANT(opline, opline->op2));
					} else {
						if (zend_update_type_info(op_array, tssa, script, (zend_op*)opline, ssa_ops + idx, ssa_opcodes, optimization_level) == FAILURE) {
							// TODO:
							assert(0);
						}
					}
				}
				if (ssa->var_info) {
					/* Add statically inferred restrictions */
					if (ssa_ops[idx].op1_def >= 0) {
						zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op1_def);
					}
					if (ssa_ops[idx].op2_def >= 0) {
						zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].op2_def);
					}
					if (ssa_ops[idx].result_def >= 0) {
						zend_jit_trace_restrict_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, ssa_ops[idx].result_def);
					}
				}
				idx++;
				len--;
			}

		} else if (p->op == ZEND_JIT_TRACE_ENTER) {
			op_array = p->op_array;
			jit_extension =
				(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
			ssa = &jit_extension->func_info.ssa;

			call = frame->call;
			if (!call) {
				/* Trace missed INIT_FCALL opcode */
				call = top;
				TRACE_FRAME_INIT(call, op_array, 0, 0);
				call->used_stack = 0;
				top = zend_jit_trace_call_frame(top, op_array);
				for (i = 0; i < op_array->last_var + op_array->T; i++) {
					SET_STACK_TYPE(call->stack, i, IS_UNKNOWN, 1);
				}
			} else {
				ZEND_ASSERT(&call->func->op_array == op_array);
			}
			frame->call = call->prev;
			call->prev = frame;
			TRACE_FRAME_SET_RETURN_SSA_VAR(call, find_return_ssa_var(p - 1, ssa_ops + (idx - 1)));
			frame = call;

			level++;
			i = 0;
			v = ZEND_JIT_TRACE_GET_FIRST_SSA_VAR(p->info);
			while (i < op_array->last_var) {
				ssa_vars[v].var = i;
				if (i < op_array->num_args) {
					if (ssa->var_info
					 && zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, v)) {
						/* pass */
					} else {
						ssa_vars[v].alias = zend_jit_var_may_alias(op_array, ssa, i);
						if (op_array->arg_info) {
							zend_arg_info *arg_info = &op_array->arg_info[i];
							zend_class_entry *ce;
							uint32_t tmp = zend_fetch_arg_info_type(script, arg_info, &ce);

							if (ZEND_ARG_SEND_MODE(arg_info)) {
								tmp |= MAY_BE_REF;
							}
							ssa_var_info[v].type = tmp;
							ssa_var_info[v].ce = ce;
							ssa_var_info[v].is_instanceof = 1;
						} else {
							ssa_var_info[v].type = MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY  | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
						}
					}
				} else {
					if (ssa->vars) {
						ssa_vars[v].no_val = ssa->vars[i].no_val;
						ssa_vars[v].alias = ssa->vars[i].alias;
					} else {
						ssa_vars[v].alias = zend_jit_var_may_alias(op_array, ssa, i);
					}
					if (ssa_vars[v].alias == NO_ALIAS) {
						ssa_var_info[v].type = MAY_BE_UNDEF;
					} else {
						ssa_var_info[v].type = MAY_BE_UNDEF | MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
					}
				}
				if (!(op_array->fn_flags & ZEND_ACC_HAS_TYPE_HINTS)
				 && i < op_array->num_args) {
					/* Propagate argument type */
					ssa_var_info[v].type &= STACK_INFO(frame->stack, i);
				}
				i++;
				v++;
			}
		} else if (p->op == ZEND_JIT_TRACE_BACK) {
			op_array = p->op_array;
			jit_extension =
				(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
			ssa = &jit_extension->func_info.ssa;
			if (level == 0) {
				i = 0;
				v = ZEND_JIT_TRACE_GET_FIRST_SSA_VAR(p->info);
				while (i < op_array->last_var) {
					ssa_vars[v].var = i;
					if (!ssa->var_info
					 || !zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, v)) {
						ssa_var_info[v].type = MAY_BE_UNDEF | MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY  | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
					}
					i++;
					v++;
				}
				while (i < op_array->last_var + op_array->T) {
					ssa_vars[v].var = i;
					if (!ssa->var_info
					 || !zend_jit_trace_copy_ssa_var_info(op_array, ssa, ssa_opcodes, tssa, v)) {
						ssa_var_info[v].type = MAY_BE_RC1 | MAY_BE_RCN | MAY_BE_REF | MAY_BE_ANY  | MAY_BE_ARRAY_KEY_ANY | MAY_BE_ARRAY_OF_ANY | MAY_BE_ARRAY_OF_REF;
					}
					i++;
					v++;
				}
				if (return_value_info.type != 0) {
					zend_jit_trace_rec *q = p + 1;
					while (q->op == ZEND_JIT_TRACE_INIT_CALL) {
						q++;
					}
					if (q->op == ZEND_JIT_TRACE_VM
					 || (q->op == ZEND_JIT_TRACE_END
					  && q->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET)) {
						const zend_op *opline = q->opline - 1;
						if (opline->result_type != IS_UNUSED) {
							ssa_var_info[
								ZEND_JIT_TRACE_GET_FIRST_SSA_VAR(p->info) +
								EX_VAR_TO_NUM(opline->result.var)] = return_value_info;
						}
					}
					memset(&return_value_info, 0, sizeof(return_value_info));
				}
			} else {
				level--;
				if (return_value_info.type != 0) {
					if ((p+1)->op == ZEND_JIT_TRACE_VM) {
						const zend_op *opline = (p+1)->opline - 1;
						if (opline->result_type != IS_UNUSED) {
							if (TRACE_FRAME_RETURN_SSA_VAR(frame) >= 0) {
								ssa_var_info[TRACE_FRAME_RETURN_SSA_VAR(frame)] = return_value_info;
							}
						}
					}
					memset(&return_value_info, 0, sizeof(return_value_info));
				}
			}

			top = frame;
			if (frame->prev) {
				if (used_stack > 0) {
					used_stack -= frame->used_stack;
				}
				frame = frame->prev;
				ZEND_ASSERT(&frame->func->op_array == op_array);
			} else {
				max_used_stack = used_stack = -1;
				frame = zend_jit_trace_ret_frame(frame, op_array);
				TRACE_FRAME_INIT(frame, op_array, 0, 0);
				TRACE_FRAME_SET_RETURN_SSA_VAR(frame, -1);
				frame->used_stack = 0;
				for (i = 0; i < op_array->last_var + op_array->T; i++) {
					SET_STACK_TYPE(frame->stack, i, IS_UNKNOWN, 1);
				}
			}

		} else if (p->op == ZEND_JIT_TRACE_INIT_CALL) {
			call = top;
			TRACE_FRAME_INIT(call, p->func, 0, 0);
			call->prev = frame->call;
			call->used_stack = 0;
			frame->call = call;
			top = zend_jit_trace_call_frame(top, p->op_array);
			if (p->func && p->func->type == ZEND_USER_FUNCTION) {
				for (i = 0; i < p->op_array->last_var + p->op_array->T; i++) {
					SET_STACK_INFO(call->stack, i, -1);
				}
			}
			if (used_stack >= 0
			 && !(p->info & ZEND_JIT_TRACE_FAKE_INIT_CALL)) {
				if (p->func == NULL || (p-1)->op != ZEND_JIT_TRACE_VM) {
					max_used_stack = used_stack = -1;
				} else {
					const zend_op *opline = (p-1)->opline;

					switch (opline->opcode) {
						case ZEND_INIT_FCALL:
						case ZEND_INIT_FCALL_BY_NAME:
						case ZEND_INIT_NS_FCALL_BY_NAME:
						case ZEND_INIT_METHOD_CALL:
						case ZEND_INIT_DYNAMIC_CALL:
						//case ZEND_INIT_STATIC_METHOD_CALL:
						//case ZEND_INIT_USER_CALL:
						//case ZEND_NEW:
							frame->used_stack = zend_vm_calc_used_stack(opline->extended_value, (zend_function*)p->func);
							used_stack += frame->used_stack;
							if (used_stack > max_used_stack) {
								max_used_stack = used_stack;
							}
							break;
						default:
							max_used_stack = used_stack = -1;
					}
				}
			}
		} else if (p->op == ZEND_JIT_TRACE_DO_ICALL) {
			call = frame->call;
			if (call) {
				top = call;
				frame->call = call->prev;
			}

			if (idx > 0
			 && ssa_ops[idx-1].result_def >= 0
			 && (p->func->common.fn_flags & ZEND_ACC_HAS_RETURN_TYPE)
			 && !(p->func->common.fn_flags & ZEND_ACC_RETURN_REFERENCE)) {
				ZEND_ASSERT(ssa_opcodes[idx-1] == opline);
				ZEND_ASSERT(opline->opcode == ZEND_DO_ICALL ||
					opline->opcode == ZEND_DO_FCALL ||
					opline->opcode == ZEND_DO_FCALL_BY_NAME);

				if (opline->result_type != IS_UNDEF) {
					zend_class_entry *ce;
					const zend_function *func = p->func;
					zend_arg_info *ret_info = func->common.arg_info - 1;
					uint32_t ret_type = zend_fetch_arg_info_type(NULL, ret_info, &ce);

					ssa_var_info[ssa_ops[idx-1].result_def].type &= ret_type;
				}
			}
		} else if (p->op == ZEND_JIT_TRACE_END) {
			break;
		}
	}

	((zend_tssa*)tssa)->used_stack = max_used_stack;

	if (trace_buffer->stop == ZEND_JIT_TRACE_STOP_LOOP
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_CALL
	 || trace_buffer->stop == ZEND_JIT_TRACE_STOP_RECURSIVE_RET) {
		/* Propagate guards through Phi sources */
		zend_ssa_phi *phi = tssa->blocks[1].phis;

		op_array = trace_buffer->op_array;
		jit_extension =
			(zend_jit_op_array_trace_extension*)ZEND_FUNC_INFO(op_array);
		ssa = &jit_extension->func_info.ssa;

		while (phi) {
			uint32_t t = ssa_var_info[phi->ssa_var].type;

			if ((t & MAY_BE_GUARD) && tssa->vars[phi->ssa_var].alias == NO_ALIAS) {
				uint32_t t0 = ssa_var_info[phi->sources[0]].type;
				uint32_t t1 = ssa_var_info[phi->sources[1]].type;

				if (((t0 | t1) & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) == (t & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF))) {
					if (!((t0 | t1) & MAY_BE_GUARD)) {
						ssa_var_info[phi->ssa_var].type = t & ~MAY_BE_GUARD;
					}
				} else if ((t1 & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) == (t & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF))) {
					if (!(t1 & MAY_BE_GUARD)
					 || is_checked_guard(tssa, ssa_opcodes, phi->sources[1], phi->ssa_var)) {
						ssa_var_info[phi->ssa_var].type = t & ~MAY_BE_GUARD;
						t0 = (t & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
							(t0 & ~(MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
							MAY_BE_GUARD;
						if (!(t0 & MAY_BE_ARRAY)) {
							t0 &= ~(MAY_BE_ARRAY_OF_ANY|MAY_BE_ARRAY_OF_REF|MAY_BE_ARRAY_KEY_ANY);
						}
						if (!(t0 & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE))) {
							t0 &= ~(MAY_BE_RC1|MAY_BE_RCN);
						}
						ssa_var_info[phi->sources[0]].type = t0;
						ssa_var_info[phi->sources[0]].type = t0;
					}
				} else {
					if ((t0 & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) != (t & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF))) {
						t0 = (t & t0 & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
							(t0 & ~(MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
							MAY_BE_GUARD;
						if (!(t0 & MAY_BE_ARRAY)) {
							t0 &= ~(MAY_BE_ARRAY_OF_ANY|MAY_BE_ARRAY_OF_REF|MAY_BE_ARRAY_KEY_ANY);
						}
						if (!(t0 & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE))) {
							t0 &= ~(MAY_BE_RC1|MAY_BE_RCN);
						}
						ssa_var_info[phi->sources[0]].type = t0;
					}
					if ((t1 & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) != (t & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF))) {
						if (((t & t1) & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) != 0
						 && is_checked_guard(tssa, ssa_opcodes, phi->sources[1], phi->ssa_var)) {
							t1 = (t & t1 & (MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
								(t1 & ~(MAY_BE_ANY|MAY_BE_UNDEF|MAY_BE_REF)) |
								MAY_BE_GUARD;
							if (!(t1 & MAY_BE_ARRAY)) {
								t1 &= ~(MAY_BE_ARRAY_OF_ANY|MAY_BE_ARRAY_OF_REF|MAY_BE_ARRAY_KEY_ANY);
							}
							if (!(t1 & (MAY_BE_STRING|MAY_BE_ARRAY|MAY_BE_OBJECT|MAY_BE_RESOURCE))) {
								t1 &= ~(MAY_BE_RC1|MAY_BE_RCN);
							}
							ssa_var_info[phi->sources[1]].type = t1;
							ssa_var_info[phi->ssa_var].type = t & ~MAY_BE_GUARD;
						}
					}
				}
				t = ssa_var_info[phi->ssa_var].type;
			}

			if ((t & MAY_BE_PACKED_GUARD) && tssa->vars[phi->ssa_var].alias == NO_ALIAS) {
				uint32_t t0 = ssa_var_info[phi->sources[0]].type;
				uint32_t t1 = ssa_var_info[phi->sources[1]].type;

				if (((t0 | t1) & MAY_BE_ARRAY_KEY_ANY) == (t & MAY_BE_ARRAY_KEY_ANY)) {
					if (!((t0 | t1) & MAY_BE_PACKED_GUARD)) {
						ssa_var_info[phi->ssa_var].type = t & ~MAY_BE_PACKED_GUARD;
					}
				} else if ((t1 & MAY_BE_ARRAY_KEY_ANY) == (t & MAY_BE_ARRAY_KEY_ANY)) {
					if (!(t1 & MAY_BE_PACKED_GUARD)) {
						ssa_var_info[phi->ssa_var].type = t & ~MAY_BE_PACKED_GUARD;
						ssa_var_info[phi->sources[0]].type =
							(t0 & ~MAY_BE_ARRAY_KEY_ANY) | (t & MAY_BE_ARRAY_KEY_ANY) | MAY_BE_PACKED_GUARD;
					}
				}
			}
			phi = phi->next;
		}
	}

	if (UNEXPECTED(JIT_G(debug) & ZEND_JIT_DEBUG_TRACE_TSSA)) {
		if (parent_trace) {
			fprintf(stderr, "---- TRACE %d TSSA start (side trace %d/%d) %s%s%s() %s:%d\n",
				ZEND_JIT_TRACE_NUM,
				parent_trace,
				exit_num,
				trace_buffer->op_array->scope ? ZSTR_VAL(trace_buffer->op_array->scope->name) : "",
				trace_buffer->op_array->scope ? "::" : "",
				trace_buffer->op_array->function_name ?
					ZSTR_VAL(trace_buffer->op_array->function_name) : "$main",
				ZSTR_VAL(trace_buffer->op_array->filename),
				trace_buffer[1].opline->lineno);
		} else {
			fprintf(stderr, "---- TRACE %d TSSA start (%s) %s%s%s() %s:%d\n",
				ZEND_JIT_TRACE_NUM,
				zend_jit_trace_star_desc(trace_buffer->start),
				trace_buffer->op_array->scope ? ZSTR_VAL(trace_buffer->op_array->scope->name) : "",
				trace_buffer->op_array->scope ? "::" : "",
				trace_buffer->op_array->function_name ?

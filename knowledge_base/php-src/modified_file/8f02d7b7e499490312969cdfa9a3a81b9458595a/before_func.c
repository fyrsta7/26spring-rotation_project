			}

			zend_delete_call_instructions(op_array, opline-1);
		}
	}
}

/* arg_num is 1-based here, to match SEND encoding. */
static bool has_known_send_mode(const optimizer_call_info *info, uint32_t arg_num)
{
	if (!info->func) {
		return false;
	}

	/* For prototype functions we should not make assumptions about arguments that are not part of
	 * the signature: And inheriting method can add an optional by-ref argument. */
	return !info->is_prototype
		|| arg_num <= info->func->common.num_args
		|| (info->func->common.fn_flags & ZEND_ACC_VARIADIC);
}

void zend_optimize_func_calls(zend_op_array *op_array, zend_optimizer_ctx *ctx)
{
	zend_op *opline = op_array->opcodes;
	zend_op *end = opline + op_array->last;
	int call = 0;
	void *checkpoint;
	optimizer_call_info *call_stack;

	if (op_array->last < 2) {
		return;
	}

	checkpoint = zend_arena_checkpoint(ctx->arena);
	call_stack = zend_arena_calloc(&ctx->arena, op_array->last / 2, sizeof(optimizer_call_info));
	while (opline < end) {
		switch (opline->opcode) {
			case ZEND_INIT_FCALL_BY_NAME:
			case ZEND_INIT_NS_FCALL_BY_NAME:
			case ZEND_INIT_STATIC_METHOD_CALL:
			case ZEND_INIT_METHOD_CALL:
			case ZEND_INIT_FCALL:
			case ZEND_NEW:
				/* The argument passing optimizations are valid for prototypes as well,
				 * as inheritance cannot change between ref <-> non-ref arguments. */
				call_stack[call].func = zend_optimizer_get_called_func(
					ctx->script, op_array, opline, &call_stack[call].is_prototype);
				call_stack[call].try_inline =
					!call_stack[call].is_prototype && opline->opcode != ZEND_NEW;
				ZEND_FALLTHROUGH;
			case ZEND_INIT_DYNAMIC_CALL:
			case ZEND_INIT_USER_CALL:
				call_stack[call].opline = opline;
				call_stack[call].func_arg_num = (uint32_t)-1;
				call++;
				break;
			case ZEND_DO_FCALL:
			case ZEND_DO_ICALL:
			case ZEND_DO_UCALL:
			case ZEND_DO_FCALL_BY_NAME:
			case ZEND_CALLABLE_CONVERT:
				call--;
				if (call_stack[call].func && call_stack[call].opline) {
					zend_op *fcall = call_stack[call].opline;

					if (fcall->opcode == ZEND_INIT_FCALL) {
						/* nothing to do */
					} else if (fcall->opcode == ZEND_INIT_FCALL_BY_NAME) {
						fcall->opcode = ZEND_INIT_FCALL;
						fcall->op1.num = zend_vm_calc_used_stack(fcall->extended_value, call_stack[call].func);
						literal_dtor(&ZEND_OP2_LITERAL(fcall));
						fcall->op2.constant = fcall->op2.constant + 1;
						if (opline->opcode != ZEND_CALLABLE_CONVERT) {
							opline->opcode = zend_get_call_op(fcall, call_stack[call].func);
						}
					} else if (fcall->opcode == ZEND_INIT_NS_FCALL_BY_NAME) {
						fcall->opcode = ZEND_INIT_FCALL;
						fcall->op1.num = zend_vm_calc_used_stack(fcall->extended_value, call_stack[call].func);
						literal_dtor(&op_array->literals[fcall->op2.constant]);
						literal_dtor(&op_array->literals[fcall->op2.constant + 2]);
						fcall->op2.constant = fcall->op2.constant + 1;
						if (opline->opcode != ZEND_CALLABLE_CONVERT) {
							opline->opcode = zend_get_call_op(fcall, call_stack[call].func);
						}
					} else if (fcall->opcode == ZEND_INIT_STATIC_METHOD_CALL
							|| fcall->opcode == ZEND_INIT_METHOD_CALL
							|| fcall->opcode == ZEND_NEW) {
						/* We don't have specialized opcodes for this, do nothing */
					} else {
						ZEND_UNREACHABLE();
					}

					if ((ZEND_OPTIMIZER_PASS_16 & ctx->optimization_level)
							&& call_stack[call].try_inline
							&& opline->opcode != ZEND_CALLABLE_CONVERT) {
						zend_try_inline_call(op_array, fcall, opline, call_stack[call].func);
					}
				}
				call_stack[call].func = NULL;
				call_stack[call].opline = NULL;
				call_stack[call].try_inline = 0;
				call_stack[call].func_arg_num = (uint32_t)-1;
				break;
			case ZEND_FETCH_FUNC_ARG:
			case ZEND_FETCH_STATIC_PROP_FUNC_ARG:
			case ZEND_FETCH_OBJ_FUNC_ARG:
			case ZEND_FETCH_DIM_FUNC_ARG:
				if (call_stack[call - 1].func_arg_num != (uint32_t)-1
						&& has_known_send_mode(&call_stack[call - 1], call_stack[call - 1].func_arg_num)) {
					if (ARG_SHOULD_BE_SENT_BY_REF(call_stack[call - 1].func, call_stack[call - 1].func_arg_num)) {
						if (opline->opcode != ZEND_FETCH_STATIC_PROP_FUNC_ARG) {
							opline->opcode -= 9;
						} else {
							opline->opcode = ZEND_FETCH_STATIC_PROP_W;
						}
					} else {
						if (opline->opcode == ZEND_FETCH_DIM_FUNC_ARG
								&& opline->op2_type == IS_UNUSED) {
							/* FETCH_DIM_FUNC_ARG supports UNUSED op2, while FETCH_DIM_R does not.
							 * Performing the replacement would create an invalid opcode. */
							call_stack[call - 1].try_inline = 0;
							break;
						}

						if (opline->opcode != ZEND_FETCH_STATIC_PROP_FUNC_ARG) {
							opline->opcode -= 12;
						} else {
							opline->opcode = ZEND_FETCH_STATIC_PROP_R;
						}
					}
				}
				break;
			case ZEND_SEND_VAL_EX:
				if (opline->op2_type == IS_CONST) {
					call_stack[call - 1].try_inline = 0;
					break;
				}

				if (has_known_send_mode(&call_stack[call - 1], opline->op2.num)) {
					if (ARG_MUST_BE_SENT_BY_REF(call_stack[call - 1].func, opline->op2.num)) {
						/* We won't convert it into_DO_FCALL to emit error at run-time */
						call_stack[call - 1].opline = NULL;
					} else {
						opline->opcode = ZEND_SEND_VAL;
					}
				}
				break;
			case ZEND_CHECK_FUNC_ARG:
				if (opline->op2_type == IS_CONST) {
					call_stack[call - 1].try_inline = 0;
					call_stack[call - 1].func_arg_num = (uint32_t)-1;
					break;
				}

				if (has_known_send_mode(&call_stack[call - 1], opline->op2.num)) {
					call_stack[call - 1].func_arg_num = opline->op2.num;
					MAKE_NOP(opline);
				}
				break;
			case ZEND_SEND_VAR_EX:
			case ZEND_SEND_FUNC_ARG:
				if (opline->op2_type == IS_CONST) {
					call_stack[call - 1].try_inline = 0;
					break;
				}

				if (has_known_send_mode(&call_stack[call - 1], opline->op2.num)) {
					call_stack[call - 1].func_arg_num = (uint32_t)-1;
					if (ARG_SHOULD_BE_SENT_BY_REF(call_stack[call - 1].func, opline->op2.num)) {
						opline->opcode = ZEND_SEND_REF;
					} else {
						opline->opcode = ZEND_SEND_VAR;
					}
				}
				break;
			case ZEND_SEND_VAR_NO_REF_EX:
				if (opline->op2_type == IS_CONST) {
					call_stack[call - 1].try_inline = 0;
					break;
				}

				if (has_known_send_mode(&call_stack[call - 1], opline->op2.num)) {
					if (ARG_MUST_BE_SENT_BY_REF(call_stack[call - 1].func, opline->op2.num)) {
						opline->opcode = ZEND_SEND_VAR_NO_REF;
					} else if (ARG_MAY_BE_SENT_BY_REF(call_stack[call - 1].func, opline->op2.num)) {
						opline->opcode = ZEND_SEND_VAL;
					} else {
						opline->opcode = ZEND_SEND_VAR;
					}

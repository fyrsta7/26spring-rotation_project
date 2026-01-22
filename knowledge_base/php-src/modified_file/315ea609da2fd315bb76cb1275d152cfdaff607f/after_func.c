				if (call == 0) {
					if (opline->op1_type == IS_CONST) {
						MAKE_NOP(opline);
					} else if (opline->op1_type == IS_CV) {
						opline->opcode = ZEND_CHECK_VAR;
						opline->extended_value = 0;
						opline->result.var = 0;
					} else {
						opline->opcode = ZEND_FREE;
						opline->extended_value = 0;
						opline->result.var = 0;
					}
				}
				break;
		}
		opline--;
	}
}

static void zend_try_inline_call(zend_op_array *op_array, zend_op *fcall, zend_op *opline, zend_function *func)
{
	if (func->type == ZEND_USER_FUNCTION
	 && !(func->op_array.fn_flags & (ZEND_ACC_ABSTRACT|ZEND_ACC_HAS_TYPE_HINTS))
		/* TODO: function copied from trait may be inconsistent ??? */
	 && !(func->op_array.fn_flags & (ZEND_ACC_TRAIT_CLONE))
	 && fcall->extended_value >= func->op_array.required_num_args
	 && func->op_array.opcodes[func->op_array.num_args].opcode == ZEND_RETURN) {

		zend_op *ret_opline = func->op_array.opcodes + func->op_array.num_args;

		if (ret_opline->op1_type == IS_CONST) {
			uint32_t i, num_args = func->op_array.num_args;
			num_args += (func->op_array.fn_flags & ZEND_ACC_VARIADIC) != 0;

			if (fcall->opcode == ZEND_INIT_STATIC_METHOD_CALL
					&& !(func->op_array.fn_flags & ZEND_ACC_STATIC)) {
				/* Don't inline static call to instance method. */
				return;
			}

			for (i = 0; i < num_args; i++) {
				/* Don't inline functions with by-reference arguments. This would require
				 * correct handling of INDIRECT arguments. */
				if (ZEND_ARG_SEND_MODE(&func->op_array.arg_info[i])) {
					return;
				}
			}

			if (fcall->extended_value < func->op_array.num_args) {
				/* don't inline functions with named constants in default arguments */
				i = fcall->extended_value;

				do {
					if (Z_TYPE_P(CRT_CONSTANT_EX(&func->op_array, &func->op_array.opcodes[i], func->op_array.opcodes[i].op2)) == IS_CONSTANT_AST) {
						return;
					}

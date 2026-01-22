		zend_execute(EG(active_op_array) TSRMLS_CC);
		if (!fci->symbol_table) {
			zend_hash_destroy(EG(active_symbol_table));
			FREE_HASHTABLE(EG(active_symbol_table));
		}
		EG(active_symbol_table) = calling_symbol_table;
		EG(active_op_array) = original_op_array;
		EG(return_value_ptr_ptr)=original_return_value;
		EG(opline_ptr) = original_opline_ptr;
		EG(free_op1) = orig_free_op1;
		EG(free_op2) = orig_free_op2;
		EG(unary_op) = orig_unary_op;
		EG(binary_op) = orig_binary_op;
	} else {
		ALLOC_INIT_ZVAL(*fci->retval_ptr_ptr);
		if (EX(function_state).function->common.scope) {
			EG(scope) = EX(function_state).function->common.scope;
		}
		((zend_internal_function *) EX(function_state).function)->handler(fci->param_count, *fci->retval_ptr_ptr, (fci->object_pp?*fci->object_pp:NULL), 1 TSRMLS_CC);
		INIT_PZVAL(*fci->retval_ptr_ptr);
	}
	zend_ptr_stack_clear_multiple(TSRMLS_C);
	if (call_via_handler) {
		zval_ptr_dtor(&method_name);
		zval_ptr_dtor(&params_array);
	}
	EG(function_state_ptr) = original_function_state_ptr;

	if (EG(This)) {
		zval_ptr_dtor(&EG(This));
	}
	EG(scope) = current_scope;
	EG(This) = current_this;
	EG(current_execute_data) = EX(prev_execute_data);
	return SUCCESS;
}


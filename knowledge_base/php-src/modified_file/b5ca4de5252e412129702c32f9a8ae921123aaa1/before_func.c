static inline HashTable *zend_get_target_symbol_table(zend_op *opline, temp_variable *Ts, int type, zval *variable TSRMLS_DC)
{
	switch (opline->op2.u.EA.type) {
		case ZEND_FETCH_LOCAL:
			return EG(active_symbol_table);
			break;
		case ZEND_FETCH_GLOBAL:
			/* Don't think this is actually needed.
			   if (opline->op1.op_type == IS_VAR) {
				PZVAL_LOCK(varname);
			}
+ */
			return &EG(symbol_table);
			break;
		case ZEND_FETCH_STATIC:
			if (!EG(active_op_array)->static_variables) {
				ALLOC_HASHTABLE(EG(active_op_array)->static_variables);
				zend_hash_init(EG(active_op_array)->static_variables, 2, NULL, ZVAL_PTR_DTOR, 0);
			}
			return EG(active_op_array)->static_variables;
			break;
		EMPTY_SWITCH_DEFAULT_CASE()
	}
	return NULL;
}


static void zend_fetch_var_address(zend_op *opline, temp_variable *Ts, int type TSRMLS_DC)
{
	zval *free_op1;
	zval *varname = get_zval_ptr(&opline->op1, Ts, &free_op1, BP_VAR_R);
	zval **retval;
	zval tmp_varname;
	HashTable *target_symbol_table;

 	if (varname->type != IS_STRING) {
		tmp_varname = *varname;
		zval_copy_ctor(&tmp_varname);
		convert_to_string(&tmp_varname);
		varname = &tmp_varname;
	}

	if (opline->op2.u.EA.type == ZEND_FETCH_STATIC_MEMBER) {
		target_symbol_table = NULL;
		retval = zend_std_get_static_property(T(opline->op2.u.var).EA.class_entry, Z_STRVAL_P(varname), Z_STRLEN_P(varname), 0 TSRMLS_CC);
	} else {
		target_symbol_table = zend_get_target_symbol_table(opline, Ts, type, varname TSRMLS_CC);
		if (!target_symbol_table) {
			return;
		}
		if (zend_hash_find(target_symbol_table, varname->value.str.val, varname->value.str.len+1, (void **) &retval) == FAILURE) {
			switch (type) {
				case BP_VAR_R: 
					zend_error(E_NOTICE,"Undefined variable:  %s", varname->value.str.val);
					/* break missing intentionally */
				case BP_VAR_IS:
					retval = &EG(uninitialized_zval_ptr);
					break;
				case BP_VAR_RW:
					zend_error(E_NOTICE,"Undefined variable:  %s", varname->value.str.val);
					/* break missing intentionally */

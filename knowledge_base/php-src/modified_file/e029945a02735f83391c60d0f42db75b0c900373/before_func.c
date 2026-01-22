

static inline void zend_free_obj_get_result(zval *op)
{
	if (op) {
		if (op->refcount == 0) {
			zval_dtor(op);
			FREE_ZVAL(op);
		} else {
			zval_ptr_dtor(&op);
		}
	}
}

#define COMPARE_RETURN_AND_FREE(retval) \
			zend_free_obj_get_result(op1_free); \
			zend_free_obj_get_result(op2_free); \
			return retval;

ZEND_API int compare_function(zval *result, zval *op1, zval *op2 TSRMLS_DC)
{
	zval op1_copy, op2_copy;
	zval *op1_free, *op2_free;
	int op1_obj = Z_TYPE_P(op1) == IS_OBJECT;
	int op2_obj = Z_TYPE_P(op2) == IS_OBJECT;
	int eq_comp = op1_obj && op2_obj && (Z_OBJ_HANDLER_P(op1,compare_objects) 
	                                  == Z_OBJ_HANDLER_P(op2,compare_objects));

	if (op1_obj && !eq_comp) {
		if (Z_TYPE_P(op2) == IS_NULL) {
			ZVAL_LONG(result, 1);
			return SUCCESS;
		} else if (Z_OBJ_HT_P(op1)->get) {
			op1 = op1_free = Z_OBJ_HT_P(op1)->get(op1 TSRMLS_CC);
		} else if (!op2_obj && Z_OBJ_HT_P(op1)->cast_object) {
			ALLOC_INIT_ZVAL(op1_free);
			if (Z_OBJ_HT_P(op1)->cast_object(op1, op1_free, Z_TYPE_P(op2) TSRMLS_CC) == FAILURE) {
				op2_free = NULL;
				ZVAL_LONG(result, 1);
				COMPARE_RETURN_AND_FREE(SUCCESS);
			}
			op1 = op1_free;
		} else {
			op1_free = NULL;
		}
		op1_obj = Z_TYPE_P(op1) == IS_OBJECT;
		eq_comp = op1_obj && op2_obj && (Z_OBJ_HANDLER_P(op1,compare_objects) 
	                                  == Z_OBJ_HANDLER_P(op2,compare_objects));
	} else {
		op1_free = NULL;
	}
	if (op2_obj && !eq_comp) {
		if (Z_TYPE_P(op1) == IS_NULL) {
			op2_free = NULL;
			ZVAL_LONG(result, -1);
			COMPARE_RETURN_AND_FREE(SUCCESS);
		} else if (Z_OBJ_HT_P(op2)->get) {
			op2 = op2_free = Z_OBJ_HT_P(op2)->get(op2 TSRMLS_CC);
		} else if (!op1_obj && Z_OBJ_HT_P(op2)->cast_object) {
			ALLOC_INIT_ZVAL(op2_free);
			if (Z_OBJ_HT_P(op2)->cast_object(op2, op2_free, Z_TYPE_P(op1) TSRMLS_CC) == FAILURE) {
				ZVAL_LONG(result, -1);
				COMPARE_RETURN_AND_FREE(SUCCESS);
			}
			op2 = op2_free;
		} else {
			op2_free = NULL;
		}
		op2_obj = Z_TYPE_P(op2) == IS_OBJECT;
		eq_comp = op1_obj && op2_obj && (Z_OBJ_HANDLER_P(op1,compare_objects) 
	                                  == Z_OBJ_HANDLER_P(op2,compare_objects));
	} else {
		op2_free = NULL;
	}

	if ((Z_TYPE_P(op1) == IS_NULL && Z_TYPE_P(op2) == IS_STRING)
		|| (Z_TYPE_P(op2) == IS_NULL && Z_TYPE_P(op1) == IS_STRING)) {
		if (Z_TYPE_P(op1) == IS_NULL) {
			ZVAL_LONG(result, zend_binary_strcmp("", 0, Z_STRVAL_P(op2), Z_STRLEN_P(op2)));
			COMPARE_RETURN_AND_FREE(SUCCESS);
		} else {
			ZVAL_LONG(result, zend_binary_strcmp(Z_STRVAL_P(op1), Z_STRLEN_P(op1), "", 0));
			COMPARE_RETURN_AND_FREE(SUCCESS);
		}
	}
		
	if (op1->type == IS_STRING && op2->type == IS_STRING) {
		zendi_smart_strcmp(result, op1, op2);
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}

	if (Z_TYPE_P(op1) == IS_BOOL || Z_TYPE_P(op2) == IS_BOOL
		|| Z_TYPE_P(op1) == IS_NULL || Z_TYPE_P(op2) == IS_NULL) {
		zendi_convert_to_boolean(op1, op1_copy, result);
		zendi_convert_to_boolean(op2, op2_copy, result);
		ZVAL_LONG(result, ZEND_NORMALIZE_BOOL(Z_LVAL_P(op1) - Z_LVAL_P(op2)));
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}

	/* If both are objects sharing the same comparision handler then use is */
	if (eq_comp) {
		ZVAL_LONG(result, Z_OBJ_HT_P(op1)->compare_objects(op1, op2 TSRMLS_CC));
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}

	zendi_convert_scalar_to_number(op1, op1_copy, result);
	zendi_convert_scalar_to_number(op2, op2_copy, result);

	if (Z_TYPE_P(op1) == IS_LONG && Z_TYPE_P(op2) == IS_LONG) {
		ZVAL_LONG(result, Z_LVAL_P(op1)>Z_LVAL_P(op2)?1:(Z_LVAL_P(op1)<Z_LVAL_P(op2)?-1:0));
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}
	if ((Z_TYPE_P(op1) == IS_DOUBLE || Z_TYPE_P(op1) == IS_LONG)
		&& (Z_TYPE_P(op2) == IS_DOUBLE || Z_TYPE_P(op2) == IS_LONG)) {
		Z_DVAL_P(result) = (Z_TYPE_P(op1) == IS_LONG ? (double) Z_LVAL_P(op1) : Z_DVAL_P(op1)) - (Z_TYPE_P(op2) == IS_LONG ? (double) Z_LVAL_P(op2) : Z_DVAL_P(op2));
		ZVAL_LONG(result, ZEND_NORMALIZE_BOOL(Z_DVAL_P(result)));
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}
	if (Z_TYPE_P(op1)==IS_ARRAY && Z_TYPE_P(op2)==IS_ARRAY) {
		zend_compare_arrays(result, op1, op2 TSRMLS_CC);
		COMPARE_RETURN_AND_FREE(SUCCESS);
	}

	if (Z_TYPE_P(op1)==IS_ARRAY) {

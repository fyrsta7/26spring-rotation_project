
	if (ARRAYG(compare_func)(&result, &first, &second) == FAILURE) {
		return 0;
	}

	if (EXPECTED(Z_TYPE(result) == IS_LONG)) {
		return ZEND_NORMALIZE_BOOL(Z_LVAL(result));
	} else if (Z_TYPE(result) == IS_DOUBLE) {
		return ZEND_NORMALIZE_BOOL(Z_DVAL(result));
	}

	return ZEND_NORMALIZE_BOOL(zval_get_long(&result));
}
/* }}} */

static int php_array_reverse_key_compare(const void *a, const void *b) /* {{{ */

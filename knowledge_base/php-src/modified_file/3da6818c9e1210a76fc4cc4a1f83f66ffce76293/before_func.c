	}
	if (modifiers & ZEND_ACC_VIRTUAL) {
		add_next_index_stringl(return_value, "virtual", sizeof("virtual")-1);
	}

	/* These are mutually exclusive */
	switch (modifiers & ZEND_ACC_PPP_MASK) {
		case ZEND_ACC_PUBLIC:
			add_next_index_stringl(return_value, "public", sizeof("public")-1);
			break;
		case ZEND_ACC_PRIVATE:
			add_next_index_stringl(return_value, "private", sizeof("private")-1);
			break;
		case ZEND_ACC_PROTECTED:
			add_next_index_stringl(return_value, "protected", sizeof("protected")-1);
			break;
	}

	if (modifiers & ZEND_ACC_STATIC) {
		add_next_index_str(return_value, ZSTR_KNOWN(ZEND_STR_STATIC));
	}

	if (modifiers & (ZEND_ACC_READONLY | ZEND_ACC_READONLY_CLASS)) {
		add_next_index_stringl(return_value, "readonly", sizeof("readonly")-1);
	}
}
/* }}} */

/* {{{ Constructor. Throws an Exception in case the given function does not exist */
ZEND_METHOD(ReflectionFunction, __construct)
{
	zval *object;
	zend_object *closure_obj = NULL;
	reflection_object *intern;
	zend_function *fptr;
	zend_string *fname, *lcname;

	object = ZEND_THIS;
	intern = Z_REFLECTION_P(object);

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_OBJ_OF_CLASS_OR_STR(closure_obj, zend_ce_closure, fname)
	ZEND_PARSE_PARAMETERS_END();

	if (closure_obj) {
		fptr = (zend_function*)zend_get_closure_method_def(closure_obj);
	} else {
		if (UNEXPECTED(ZSTR_VAL(fname)[0] == '\\')) {
			/* Ignore leading "\" */
			ALLOCA_FLAG(use_heap)
			ZSTR_ALLOCA_ALLOC(lcname, ZSTR_LEN(fname) - 1, use_heap);
			zend_str_tolower_copy(ZSTR_VAL(lcname), ZSTR_VAL(fname) + 1, ZSTR_LEN(fname) - 1);
			fptr = zend_fetch_function(lcname);

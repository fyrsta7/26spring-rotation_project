	_class_const_string(&str, Z_STRVAL(name), ref, "");
	zval_ptr_dtor(&name);
	RETURN_STR(smart_str_extract(&str));
}
/* }}} */

/* {{{ proto public string ReflectionClassConstant::getName()
   Returns the constant' name */
ZEND_METHOD(reflection_class_constant, getName)
{
	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}
	_default_get_name(ZEND_THIS, return_value);
}
/* }}} */

static void _class_constant_check_flag(INTERNAL_FUNCTION_PARAMETERS, int mask) /* {{{ */
{
	reflection_object *intern;
	zend_class_constant *ref;

	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}
	GET_REFLECTION_OBJECT_PTR(ref);
	RETURN_BOOL(Z_ACCESS_FLAGS(ref->value) & mask);
}
/* }}} */

/* {{{ proto public bool ReflectionClassConstant::isPublic()
   Returns whether this constant is public */
ZEND_METHOD(reflection_class_constant, isPublic)
{
	_class_constant_check_flag(INTERNAL_FUNCTION_PARAM_PASSTHRU, ZEND_ACC_PUBLIC);
}
/* }}} */

/* {{{ proto public bool ReflectionClassConstant::isPrivate()
   Returns whether this constant is private */

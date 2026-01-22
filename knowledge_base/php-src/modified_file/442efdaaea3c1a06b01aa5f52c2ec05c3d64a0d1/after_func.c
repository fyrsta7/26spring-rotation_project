		zend_error(E_WARNING, "get_called_class() called from outside a class");
	}
	RETURN_FALSE;
}
/* }}} */

/* {{{ proto string get_parent_class([mixed object])
   Retrieves the parent class name for object or class or current scope. */
ZEND_FUNCTION(get_parent_class)
{
	zval *arg;
	zend_class_entry *ce = NULL;

	if (zend_parse_parameters(ZEND_NUM_ARGS(), "|z", &arg) == FAILURE) {
		return;
	}

	if (!ZEND_NUM_ARGS()) {
		ce = EG(scope);
		if (ce && ce->parent) {
			RETURN_STR(zend_string_copy(ce->parent->name));
		} else {
			RETURN_FALSE;
		}
	}

	if (Z_TYPE_P(arg) == IS_OBJECT) {
		ce = Z_OBJ_P(arg)->ce;
	} else if (Z_TYPE_P(arg) == IS_STRING) {
	    ce = zend_lookup_class(Z_STR_P(arg));
	}

	if (ce && ce->parent) {
		RETURN_STR(zend_string_copy(ce->parent->name));
	} else {
		RETURN_FALSE;
	}
}
/* }}} */

static void is_a_impl(INTERNAL_FUNCTION_PARAMETERS, zend_bool only_subclass) /* {{{ */
{
	zval *obj;
	zend_string *class_name;
	zend_class_entry *instance_ce;
	zend_class_entry *ce;
	zend_bool allow_string = only_subclass;
	zend_bool retval;

#ifndef FAST_ZPP
	if (zend_parse_parameters(ZEND_NUM_ARGS(), "zS|b", &obj, &class_name, &allow_string) == FAILURE) {

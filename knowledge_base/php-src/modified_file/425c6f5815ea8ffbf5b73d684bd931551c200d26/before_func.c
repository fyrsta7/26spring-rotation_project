
static zval *_default_load_name(zval *object) /* {{{ */
{
	return zend_hash_find_ex_ind(Z_OBJPROP_P(object), ZSTR_KNOWN(ZEND_STR_NAME), 1);
}
/* }}} */

static void _default_get_name(zval *object, zval *return_value) /* {{{ */
{
	zval *value;

	if ((value = _default_load_name(object)) == NULL) {
		RETURN_FALSE;
	}
	ZVAL_COPY(return_value, value);

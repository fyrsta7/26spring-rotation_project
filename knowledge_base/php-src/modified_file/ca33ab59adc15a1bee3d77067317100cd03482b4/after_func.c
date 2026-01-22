/* {{{ Sets the current language or Returns the current language as a string */
PHP_FUNCTION(mb_language)
{
	zend_string *name = NULL;

	ZEND_PARSE_PARAMETERS_START(0, 1)
		Z_PARAM_OPTIONAL
		Z_PARAM_STR_OR_NULL(name)
	ZEND_PARSE_PARAMETERS_END();

	if (name == NULL) {
		RETVAL_STRING((char *)mbfl_no_language2name(MBSTRG(language)));
	} else {
		zend_string *ini_name = zend_string_init("mbstring.language", sizeof("mbstring.language") - 1, 0);
		if (FAILURE == zend_alter_ini_entry(ini_name, name, PHP_INI_USER, PHP_INI_STAGE_RUNTIME)) {
			zend_argument_value_error(1, "must be a valid language, \"%s\" given", ZSTR_VAL(name));
			zend_string_release_ex(ini_name, 0);
			RETURN_THROWS();
		}
		// TODO Make return void
		RETVAL_TRUE;
		zend_string_release_ex(ini_name, 0);
	}
}
/* }}} */

/* {{{ Sets the current internal encoding or Returns the current internal encoding as a string */

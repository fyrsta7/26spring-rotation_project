					  (*BG(array_walk_func_name))->value.str.val);

		zend_hash_move_forward_ex(target_hash, &pos);
    }
	efree(key);
	
	return 0;
}

/* {{{ proto int array_walk(array input, string funcname [, mixed userdata])
   Apply a user function to every member of an array */
PHP_FUNCTION(array_walk)
{
	int    argc;
	zval **array,
		 **userdata = NULL,
		 **old_walk_func_name;
	HashTable *target_hash;

	argc = ZEND_NUM_ARGS();
	old_walk_func_name = BG(array_walk_func_name);
	if (argc < 2 || argc > 3 ||
		zend_get_parameters_ex(argc, &array, &BG(array_walk_func_name), &userdata) == FAILURE) {
		BG(array_walk_func_name) = old_walk_func_name;
		WRONG_PARAM_COUNT;
	}
	target_hash = HASH_OF(*array);
	if (!target_hash) {
		php_error(E_WARNING, "Wrong datatype in %s() call",
				  get_active_function_name(TSRMLS_C));
		BG(array_walk_func_name) = old_walk_func_name;
		RETURN_FALSE;
	}
	if (Z_TYPE_PP(BG(array_walk_func_name)) != IS_ARRAY && 
		Z_TYPE_PP(BG(array_walk_func_name)) != IS_STRING) {
		php_error(E_WARNING, "Wrong syntax for function name in %s() call",
				  get_active_function_name(TSRMLS_C));
		BG(array_walk_func_name) = old_walk_func_name;
		RETURN_FALSE;
	}
	php_array_walk(target_hash, userdata TSRMLS_CC);
	BG(array_walk_func_name) = old_walk_func_name;
	RETURN_TRUE;
}
/* }}} */

/* void php_search_array(INTERNAL_FUNCTION_PARAMETERS, int behavior)
 *      0 = return boolean
 *      1 = return key
 */
static void php_search_array(INTERNAL_FUNCTION_PARAMETERS, int behavior)
{
 	zval **value,				/* value to check for */
		 **array,				/* array to check in */
		 **strict,				/* strict comparison or not */
		 **entry,				/* pointer to array entry */
		  res;					/* comparison result */
	HashTable *target_hash;		/* array hashtable */
	HashPosition pos;			/* hash iterator */
   	ulong num_key;

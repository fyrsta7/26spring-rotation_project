   Deactivates the circular reference collector */
ZEND_FUNCTION(gc_disable)
{
	zend_alter_ini_entry("zend.enable_gc", sizeof("zend.enable_gc"), "0", sizeof("0")-1, ZEND_INI_USER, ZEND_INI_STAGE_RUNTIME);
}
/* }}} */

/* {{{ proto int func_num_args(void)
   Get the number of arguments that were passed to the function */
ZEND_FUNCTION(func_num_args)
{
	zend_execute_data *ex = EG(current_execute_data)->prev_execute_data;

	if (ex && ex->function_state.arguments) {
		RETURN_LONG((long)(zend_uintptr_t)*(ex->function_state.arguments));
	} else {
		zend_error(E_WARNING, "func_num_args():  Called from the global scope - no function context");
		RETURN_LONG(-1);
	}
}
/* }}} */


/* {{{ proto mixed func_get_arg(int arg_num)
   Get the $arg_num'th argument that was passed to the function */
ZEND_FUNCTION(func_get_arg)
{
	void **p;
	int arg_count;
	zval *arg;
	long requested_offset;
	zend_execute_data *ex = EG(current_execute_data)->prev_execute_data;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "l", &requested_offset) == FAILURE) {
		return;

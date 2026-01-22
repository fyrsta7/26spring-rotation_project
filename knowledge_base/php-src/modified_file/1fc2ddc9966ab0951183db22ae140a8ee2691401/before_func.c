		mname = ZSTR_INIT_LITERAL("generate", 0);
		generate_method = zend_hash_find_ptr(&engine_object->ce->function_table, mname);
		zend_string_release(mname);

		/* Create compatible state */
		state->object = engine_object;
		state->generate_method = generate_method;

		/* Mark self-allocated for memory management */
		randomizer->is_userland_algo = true;
	}
}

/* {{{ Random\Randomizer::__construct() */
PHP_METHOD(Random_Randomizer, __construct)
{
	php_random_randomizer *randomizer = Z_RANDOM_RANDOMIZER_P(ZEND_THIS);
	zval engine;
	zval *param_engine = NULL;

	ZEND_PARSE_PARAMETERS_START(0, 1)
		Z_PARAM_OPTIONAL
		Z_PARAM_OBJECT_OF_CLASS_OR_NULL(param_engine, random_ce_Random_Engine);
	ZEND_PARSE_PARAMETERS_END();

	if (param_engine != NULL) {
		ZVAL_COPY(&engine, param_engine);
	} else {

}
#endif // DEBUG_ENABLED

String GDScriptParser::get_error() const {

	return error;
}

int GDScriptParser::get_error_line() const {

	return error_line;
}
int GDScriptParser::get_error_column() const {

	return error_column;
}

Error GDScriptParser::_parse(const String &p_base_path) {

	base_path = p_base_path;

	//assume class
	ClassNode *main_class = alloc_node<ClassNode>();
	main_class->initializer = alloc_node<BlockNode>();
	main_class->initializer->parent_class = main_class;
	main_class->ready = alloc_node<BlockNode>();
	main_class->ready->parent_class = main_class;
	current_class = main_class;

	_parse_class(main_class);

	if (tokenizer->get_token() == GDScriptTokenizer::TK_ERROR) {
		error_set = false;
		_set_error("Parse Error: " + tokenizer->get_token_error());
	}

	if (error_set && !for_completion) {
		return ERR_PARSE_ERROR;
	}

	_determine_inheritance(main_class);

	if (error_set) {
		return ERR_PARSE_ERROR;
	}

	current_class = main_class;
	current_function = NULL;
	current_block = NULL;
#ifdef DEBUG_ENABLED
	if (for_completion) check_types = false;
#else
	check_types = false;
#endif

#ifdef DEBUG_ENABLED
	// Resolve all class-level stuff before getting into function blocks
	_check_class_level_types(main_class);

	if (error_set) {
		return ERR_PARSE_ERROR;
	}

	// Resolve the function blocks
	_check_class_blocks_types(main_class);

	if (error_set) {
		return ERR_PARSE_ERROR;
	}

	// Resolve warning ignores
	Vector<Pair<int, String> > warning_skips = tokenizer->get_warning_skips();
	bool warning_is_error = GLOBAL_GET("debug/gdscript/warnings/treat_warnings_as_errors").booleanize();
	for (List<GDScriptWarning>::Element *E = warnings.front(); E;) {
		GDScriptWarning &w = E->get();
		int skip_index = -1;
		for (int i = 0; i < warning_skips.size(); i++) {
			if (warning_skips[i].first >= w.line) {
				break;
			}
			skip_index = i;
		}
		List<GDScriptWarning>::Element *next = E->next();
		bool erase = false;
		if (skip_index != -1) {


	if (ns_support && ns_param == NULL){
		ns_param = ":";
	}

	object_init_ex(return_value, xml_parser_ce);
	parser = Z_XMLPARSER_P(return_value);
	parser->parser = XML_ParserCreate_MM((auto_detect ? NULL : encoding),
	                                     &php_xml_mem_hdlrs, (XML_Char*)ns_param);

	parser->target_encoding = encoding;
	parser->case_folding = 1;
	parser->isparsing = 0;
	parser->parsehuge = false; /* It's the default for BC & DoS protection */

	XML_SetUserData(parser->parser, parser);
	ZVAL_COPY_VALUE(&parser->index, return_value);
}
/* }}} */

/* {{{ Create an XML parser */
PHP_FUNCTION(xml_parser_create)
{
	php_xml_parser_create_impl(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0);
}
/* }}} */

/* {{{ Create an XML parser */
PHP_FUNCTION(xml_parser_create_ns)
{
	php_xml_parser_create_impl(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1);
}
/* }}} */

static bool php_xml_check_string_method_arg(

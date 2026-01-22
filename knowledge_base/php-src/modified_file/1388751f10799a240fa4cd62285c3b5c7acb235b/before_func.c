{
	php_info_print_table_start();
	php_info_print_table_row(2, "Multibyte Support", "enabled");
	php_info_print_table_row(2, "Multibyte string engine", "libmbfl");
	php_info_print_table_row(2, "HTTP input encoding translation", MBSTRG(encoding_translation) ? "enabled": "disabled");
	{
		char tmp[256];
		snprintf(tmp, sizeof(tmp), "%d.%d.%d", MBFL_VERSION_MAJOR, MBFL_VERSION_MINOR, MBFL_VERSION_TEENY);
		php_info_print_table_row(2, "libmbfl version", tmp);
	}
#if HAVE_ONIG
	{
		char tmp[256];
		snprintf(tmp, sizeof(tmp), "%d.%d.%d", ONIGURUMA_VERSION_MAJOR, ONIGURUMA_VERSION_MINOR, ONIGURUMA_VERSION_TEENY);
		php_info_print_table_row(2, "oniguruma version", tmp);
	}
#endif
	php_info_print_table_end();

	php_info_print_table_start();
	php_info_print_table_header(1, "mbstring extension makes use of \"streamable kanji code filter and converter\", which is distributed under the GNU Lesser General Public License version 2.1.");
	php_info_print_table_end();

#if HAVE_MBREGEX
	PHP_MINFO(mb_regex)(ZEND_MODULE_INFO_FUNC_ARGS_PASSTHRU);
#endif


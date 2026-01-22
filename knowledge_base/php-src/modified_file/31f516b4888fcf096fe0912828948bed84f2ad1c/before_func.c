		RETURN_FALSE; \
	}

#define PHP_GETTEXT_LENGTH_CHECK(check_name, check_len) \
	if (check_len > PHP_GETTEXT_MAX_MSGID_LENGTH) { \
		php_error_docref(NULL, E_WARNING, "%s passed too long", check_name); \
		RETURN_FALSE; \
	}

PHP_MINFO_FUNCTION(php_gettext)
{
	php_info_print_table_start();
	php_info_print_table_row(2, "GetText Support", "enabled");
	php_info_print_table_end();
}


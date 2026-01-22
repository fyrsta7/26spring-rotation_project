		char *value, *tmp = sapi_module.getenv(name, name_len TSRMLS_CC);
		if (tmp) {
			value = estrdup(tmp);
		} else {
			return NULL;
		}
		sapi_module.input_filter(PARSE_ENV, name, &value, strlen(value), NULL TSRMLS_CC);
		return value;
	}
	return NULL;
}

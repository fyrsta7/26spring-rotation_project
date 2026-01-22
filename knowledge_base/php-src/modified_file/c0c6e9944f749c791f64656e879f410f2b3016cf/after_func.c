			}
			c = data->children->content[j++];
			if (c >= '0' && c <= '9') {
				str[i] |= c - '0';
			} else if (c >= 'a' && c <= 'f') {
				str[i] |= c - 'a' + 10;
			} else if (c >= 'A' && c <= 'F') {
				str[i] |= c - 'A' + 10;
			}
		}
		str[str_len] = '\0';
		ZVAL_STRINGL(ret, (char*)str, str_len, 0);
	} else {
		ZVAL_EMPTY_STRING(ret);
	}
	return ret;
}

static xmlNodePtr to_xml_string(encodeTypePtr type, zval *data, int style, xmlNodePtr parent)
{
	xmlNodePtr ret, text;
	char *str;
	int new_len;
	TSRMLS_FETCH();

	ret = xmlNewNode(NULL,"BOGUS");
	xmlAddChild(parent, ret);
	FIND_ZVAL_NULL(data, ret, style);

	if (Z_TYPE_P(data) == IS_STRING) {
		str = estrndup(Z_STRVAL_P(data), Z_STRLEN_P(data));
		new_len = Z_STRLEN_P(data);
	} else {
		zval tmp = *data;

		zval_copy_ctor(&tmp);
		convert_to_string(&tmp);
		str = estrndup(Z_STRVAL(tmp), Z_STRLEN(tmp));
		new_len = Z_STRLEN(tmp);
		zval_dtor(&tmp);
	}

	if (SOAP_GLOBAL(encoding) != NULL) {
		xmlBufferPtr in  = xmlBufferCreateStatic(str, new_len);
		xmlBufferPtr out = xmlBufferCreate();
		int n = xmlCharEncInFunc(SOAP_GLOBAL(encoding), out, in);

		if (n >= 0) {
			efree(str);
			str = estrdup(xmlBufferContent(out));
			new_len = n;

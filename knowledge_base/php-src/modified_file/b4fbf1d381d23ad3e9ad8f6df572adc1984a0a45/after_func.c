	memcpy(res, s, l);
	res[l] = ';';
}
/* }}} */

static inline void php_var_serialize_string(smart_str *buf, char *str, size_t len) /* {{{ */
{
	char b[32];
	char *s = zend_print_long_to_buf(b + sizeof(b) - 1, len);
	size_t l = b + sizeof(b) - 1 - s;
	size_t new_len = smart_str_alloc(buf, 2 + l + 2 + len + 2, 0);
	char *res = ZSTR_VAL(buf->s) + ZSTR_LEN(buf->s);

	ZSTR_LEN(buf->s) = new_len;
	memcpy(res, "s:", 2);
	res += 2;
	memcpy(res, s, l);
	res += l;
	memcpy(res, ":\"", 2);
	res += 2;
	memcpy(res, str, len);
	res += len;
	memcpy(res, "\";", 2);
}
/* }}} */

		retval = 3;
	} else if (k < 0x200000) {
		buf[0] = 0xf0 | (k >> 18);
		buf[1] = 0x80 | ((k >> 12) & 0x3f);
		buf[2] = 0x80 | ((k >> 6) & 0x3f);
		buf[3] = 0x80 | (k & 0x3f);
		retval = 4;
	} else if (k < 0x4000000) {
		buf[0] = 0xf8 | (k >> 24);
		buf[1] = 0x80 | ((k >> 18) & 0x3f);
		buf[2] = 0x80 | ((k >> 12) & 0x3f);
		buf[3] = 0x80 | ((k >> 6) & 0x3f);
		buf[4] = 0x80 | (k & 0x3f);
		retval = 5;
	} else {
		buf[0] = 0xfc | (k >> 30);
		buf[1] = 0x80 | ((k >> 24) & 0x3f);
		buf[2] = 0x80 | ((k >> 18) & 0x3f);
		buf[3] = 0x80 | ((k >> 12) & 0x3f);
		buf[4] = 0x80 | ((k >> 6) & 0x3f);
		buf[5] = 0x80 | (k & 0x3f);
		retval = 6;
	}
	buf[retval] = '\0';

	return retval;
}
/* }}} */

/* {{{ php_unescape_html_entities
 */
PHPAPI char *php_unescape_html_entities(unsigned char *old, int oldlen, int *newlen, int all, int quote_style, char *hint_charset TSRMLS_DC)
{
	int retlen;
	int j, k;
	char *replaced, *ret, *p, *q, *lim, *next;
	enum entity_charset charset = determine_charset(hint_charset TSRMLS_CC);
	unsigned char replacement[15];
	int replacement_len;

	ret = estrdup(old);
	retlen = oldlen;
	if (!retlen) {
		goto empty_source;
	}
	
	if (all) {
		/* look for a match in the maps for this charset */
		for (j = 0; entity_map[j].charset != cs_terminator; j++) {
			if (entity_map[j].charset != charset)
				continue;

			for (k = entity_map[j].basechar; k <= entity_map[j].endchar; k++) {
				unsigned char entity[32];
				int entity_length = 0;

				if (entity_map[j].table[k - entity_map[j].basechar] == NULL)
					continue;

				entity[0] = '&';
				entity_length = strlen(entity_map[j].table[k - entity_map[j].basechar]);
				strncpy(&entity[1], entity_map[j].table[k - entity_map[j].basechar], sizeof(entity) - 2);
				entity[entity_length+1] = ';';
				entity[entity_length+2] = '\0';
				entity_length += 2;

				/* When we have MBCS entities in the tables above, this will need to handle it */
				replacement_len = 0;
				switch (charset) {
					case cs_8859_1:
					case cs_cp1252:
					case cs_8859_15:
					case cs_cp1251:
					case cs_8859_5:
					case cs_cp866:
						replacement[0] = k;
						replacement[1] = '\0';
						replacement_len = 1;
						break;

					case cs_big5:
					case cs_gb2312:
					case cs_big5hkscs:
					case cs_sjis:
					case cs_eucjp:
						/* we cannot properly handle those multibyte encodings
						 * with php_str_to_str. skip it. */ 
						continue;

					case cs_utf_8:
						replacement_len = php_utf32_utf8(replacement, k);
						break;

					default:
						php_error_docref(NULL TSRMLS_CC, E_WARNING, "cannot yet handle MBCS!");
						return 0;
				}

				if (php_memnstr(ret, entity, entity_length, ret+retlen)) {
					replaced = php_str_to_str(ret, retlen, entity, entity_length, replacement, replacement_len, &retlen);
					efree(ret);
					ret = replaced;
				}
			}
		}
	}

	for (j = 0; basic_entities[j].charcode != 0; j++) {

		if (basic_entities[j].flags && (quote_style & basic_entities[j].flags) == 0)
			continue;
		
		replacement[0] = (unsigned char)basic_entities[j].charcode;
		replacement[1] = '\0';

		if (php_memnstr(ret, basic_entities[j].entity, basic_entities[j].entitylen, ret+retlen)) {		
			replaced = php_str_to_str(ret, retlen, basic_entities[j].entity, basic_entities[j].entitylen, replacement, 1, &retlen);
			efree(ret);
			ret = replaced;
		}
	}

	/* replace numeric entities & "&amp;" */
	lim = ret + retlen;
	for (p = ret, q = ret; p < lim;) {
		int code;

		if (p[0] == '&') {
			if (p + 2 < lim) {
				if (p[1] == '#') {
					int invalid_code = 0;

					if (p[2] == 'x' || p[2] == 'X') {
						code = strtol(p + 3, &next, 16);
					} else {
						code = strtol(p + 2, &next, 10);
					}

					if (next != NULL && *next == ';') {
						switch (charset) {
							case cs_utf_8:
								q += php_utf32_utf8(q, code);
								break;

							case cs_8859_1:
							case cs_8859_5:
							case cs_8859_15:
								if ((code >= 0x80 && code < 0xa0) || code > 0xff) {
									invalid_code = 1;
								} else {
									*(q++) = code;
								}
								break;

							case cs_cp1252:
							case cs_cp1251:
							case cs_cp866:
								if (code > 0xff) {
									invalid_code = 1;
								} else {
									*(q++) = code;
								}
								break;

							case cs_big5:
							case cs_big5hkscs:
							case cs_sjis:
							case cs_eucjp:
								if (code >= 0x80) {
									invalid_code = 1;
								} else {
									*(q++) = code;
								}
								break;

							case cs_gb2312:
								if (code >= 0x81) {
									invalid_code = 1;
								} else {
									*(q++) = code;
								}
								break;

							default:
								/* for backwards compatilibity */
								invalid_code = 1;
								break;
						}

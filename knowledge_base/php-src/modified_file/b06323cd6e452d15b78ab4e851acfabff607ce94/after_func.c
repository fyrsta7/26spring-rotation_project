				break;
			case '.':
				t[j++] = '\\';
				t[j] = '.';
				break;
			default:
				t[j] = pattern->value.str.val[i];
				break;
		}
	}
	t[j]=0;
	pattern->value.str.val = t;
	pattern->value.str.len = j;
}


static void php_browscap_parser_cb(zval *arg1, zval *arg2, int callback_type, void *arg)
{
	switch (callback_type) {
		case ZEND_INI_PARSER_ENTRY:
			if (current_section) {
				zval *new_property;
				char *new_key;

				new_property = (zval *) malloc(sizeof(zval));
				INIT_PZVAL(new_property);
				new_property->value.str.val = Z_STRLEN_P(arg2)?zend_strndup(Z_STRVAL_P(arg2), Z_STRLEN_P(arg2)):"";
				new_property->value.str.len = Z_STRLEN_P(arg2);
				new_property->type = IS_STRING;
				
				new_key = zend_strndup(Z_STRVAL_P(arg1), Z_STRLEN_P(arg1));
				zend_str_tolower(new_key, Z_STRLEN_P(arg1));
				zend_hash_update(current_section->value.obj.properties, new_key, Z_STRLEN_P(arg1)+1, &new_property, sizeof(zval *), NULL);
				free(new_key);
			}
			break;
		case ZEND_INI_PARSER_SECTION: {
				zval *processed;

				/*printf("'%s' (%d)\n",$1.value.str.val,$1.value.str.len+1);*/
				current_section = (zval *) malloc(sizeof(zval));
				INIT_PZVAL(current_section);
				processed = (zval *) malloc(sizeof(zval));
				INIT_PZVAL(processed);

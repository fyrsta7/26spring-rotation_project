				case IS_DOUBLE:
					STR_FREE(op1->value.str.val);
					op1->value.dval = dval - 1;
					op1->type = IS_DOUBLE;
					break;
			}
			break;
		default:
			return FAILURE;
	}

	return SUCCESS;
}


ZEND_API int zval_is_true(zval *op)

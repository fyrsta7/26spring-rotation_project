

ZEND_API int zend_binary_zval_strcmp(zval *s1, zval *s2)
{
	return zend_binary_strcmp(s1->value.str.val, s1->value.str.len, s2->value.str.val, s2->value.str.len);
}

ZEND_API int zend_binary_zval_strncmp(zval *s1, zval *s2, zval *s3)
{
	return zend_binary_strncmp(s1->value.str.val, s1->value.str.len, s2->value.str.val, s2->value.str.len, s3->value.lval);
}


ZEND_API int zend_binary_zval_strcasecmp(zval *s1, zval *s2)
{
	return zend_binary_strcasecmp(s1->value.str.val, s1->value.str.len, s2->value.str.val, s2->value.str.len);
}


ZEND_API int zend_binary_zval_strncasecmp(zval *s1, zval *s2, zval *s3)
{
	return zend_binary_strncasecmp(s1->value.str.val, s1->value.str.len, s2->value.str.val, s2->value.str.len, s3->value.lval);
}


ZEND_API void zendi_smart_strcmp(zval *result, zval *s1, zval *s2)
{
	int ret1, ret2;
	long lval1, lval2;

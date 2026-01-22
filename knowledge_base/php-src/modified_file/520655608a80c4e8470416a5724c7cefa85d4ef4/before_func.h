ZEND_API int zval_string_to_unicode_ex(zval *string, UConverter *conv TSRMLS_DC);
ZEND_API int zval_string_to_unicode(zval *string TSRMLS_DC);
ZEND_API int zval_unicode_to_string_ex(zval *string, UConverter *conv TSRMLS_DC);
ZEND_API int zval_unicode_to_string(zval *string TSRMLS_DC);

ZEND_API int zend_cmp_unicode_and_string(UChar *ustr, char* str, uint len);
ZEND_API int zend_cmp_unicode_and_literal(UChar *ustr, int ulen, char* str, int slen);

ZEND_API void zend_case_fold_string(UChar **dest, int *dest_len, UChar *src, int src_len, uint32_t options, UErrorCode *status);

ZEND_API int zend_is_valid_identifier(UChar *ident, int ident_len);
ZEND_API int zend_normalize_identifier(UChar **dest, int *dest_len, UChar *ident, int ident_len, zend_bool fold_case);

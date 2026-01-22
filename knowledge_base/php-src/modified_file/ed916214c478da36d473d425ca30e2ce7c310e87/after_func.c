zend_result dom_document_substitue_entities_write(dom_object *obj, zval *newval)
{
	if (obj->document) {
		dom_doc_propsptr doc_prop = dom_get_doc_props(obj->document);
		doc_prop->substituteentities = zend_is_true(newval);
	}

	return SUCCESS;
}
/* }}} */

/* {{{ documentURI	string
readonly=no
URL: http://www.w3.org/TR/2003/WD-DOM-Level-3-Core-20030226/DOM3-Core.html#core-Document3-documentURI
Since: DOM Level 3
*/
zend_result dom_document_document_uri_read(dom_object *obj, zval *retval)
{
	DOM_PROP_NODE(xmlDocPtr, docp, obj);

	const char *url = (const char *) docp->URL;
	if (url != NULL) {
		ZVAL_STRING(retval, url);
	} else {
		if (php_dom_follow_spec_intern(obj)) {
			ZVAL_STRING(retval, "about:blank");
		} else {
			ZVAL_NULL(retval);
		}
	}

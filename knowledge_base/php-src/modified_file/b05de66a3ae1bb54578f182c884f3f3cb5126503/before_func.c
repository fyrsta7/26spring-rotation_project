		RETURN_THROWS();
	}

	nodep = xmlNewText(BAD_CAST value);

	if (!nodep) {
		php_dom_throw_error(INVALID_STATE_ERR, true);
		RETURN_THROWS();
	}

	intern = Z_DOMOBJ_P(ZEND_THIS);
	oldnode = dom_object_get_node(intern);
	if (oldnode != NULL) {
		php_libxml_node_decrement_resource((php_libxml_node_object *)intern);
	}
	php_libxml_increment_node_ptr((php_libxml_node_object *)intern, nodep, (void *)intern);
}
/* }}} end DOMText::__construct */

/* {{{ wholeText	string
readonly=yes
URL: http://www.w3.org/TR/2003/WD-DOM-Level-3-Core-20030226/DOM3-Core.html#core-Text3-wholeText
Since: DOM Level 3
*/
zend_result dom_text_whole_text_read(dom_object *obj, zval *retval)
{

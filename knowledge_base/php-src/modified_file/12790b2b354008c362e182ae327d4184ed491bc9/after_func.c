
/* {{{ URL: http://www.w3.org/TR/2003/WD-DOM-Level-3-Core-20030226/DOM3-Core.html#Node3-lookupNamespacePrefix
Since: DOM Level 3
*/
PHP_METHOD(DOMNode, lookupPrefix)
{
	zval *id;
	xmlNodePtr nodep, lookupp = NULL;
	dom_object *intern;
	xmlNsPtr nsptr;
	size_t uri_len = 0;
	char *uri;

	id = ZEND_THIS;
	if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &uri, &uri_len) == FAILURE) {
		RETURN_THROWS();
	}

	DOM_GET_OBJ(nodep, id, xmlNodePtr, intern);

	if (uri_len > 0) {
		switch (nodep->type) {
			case XML_ELEMENT_NODE:
				lookupp = nodep;
				break;
			case XML_DOCUMENT_NODE:
			case XML_HTML_DOCUMENT_NODE:
				lookupp = xmlDocGetRootElement((xmlDocPtr) nodep);
				break;
			case XML_ENTITY_NODE :
			case XML_NOTATION_NODE:
			case XML_DOCUMENT_FRAG_NODE:
			case XML_DOCUMENT_TYPE_NODE:
			case XML_DTD_NODE:
				RETURN_NULL();
				break;
			default:
				lookupp =  nodep->parent;
		}

		if (lookupp != NULL) {
			nsptr = xmlSearchNsByHref(lookupp->doc, lookupp, (xmlChar *) uri);
			if (nsptr && nsptr->prefix != NULL) {
				RETURN_STRING((char *) nsptr->prefix);
			}
		}
	}

	RETURN_NULL();
}
/* }}} end dom_node_lookup_prefix */

/* {{{ URL: http://www.w3.org/TR/DOM-Level-3-Core/core.html#Node3-isDefaultNamespace
Since: DOM Level 3
*/
PHP_METHOD(DOMNode, isDefaultNamespace)
{
	zval *id;
	xmlNodePtr nodep;
	dom_object *intern;
	xmlNsPtr nsptr;
	size_t uri_len = 0;
	char *uri;

	id = ZEND_THIS;
	if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &uri, &uri_len) == FAILURE) {
		RETURN_THROWS();
	}

	DOM_GET_OBJ(nodep, id, xmlNodePtr, intern);
	if (nodep->type == XML_DOCUMENT_NODE || nodep->type == XML_HTML_DOCUMENT_NODE) {
		nodep = xmlDocGetRootElement((xmlDocPtr) nodep);
	}

	if (nodep && uri_len > 0) {
		nsptr = xmlSearchNs(nodep->doc, nodep, NULL);
		if (nsptr && xmlStrEqual(nsptr->href, (xmlChar *) uri)) {
			RETURN_TRUE;
		}
	}

	RETURN_FALSE;
}
/* }}} end dom_node_is_default_namespace */

/* {{{ URL: http://www.w3.org/TR/DOM-Level-3-Core/core.html#Node3-lookupNamespaceURI
Since: DOM Level 3
*/
PHP_METHOD(DOMNode, lookupNamespaceURI)
{
	zval *id;
	xmlNodePtr nodep;
	dom_object *intern;
	xmlNsPtr nsptr;
	size_t prefix_len;
	char *prefix;

	id = ZEND_THIS;
	if (zend_parse_parameters(ZEND_NUM_ARGS(), "s!", &prefix, &prefix_len) == FAILURE) {
		RETURN_THROWS();
	}

	DOM_GET_OBJ(nodep, id, xmlNodePtr, intern);
	if (nodep->type == XML_DOCUMENT_NODE || nodep->type == XML_HTML_DOCUMENT_NODE) {
		nodep = xmlDocGetRootElement((xmlDocPtr) nodep);
		if (nodep == NULL) {
			RETURN_NULL();
		}
	}

	nsptr = xmlSearchNs(nodep->doc, nodep, (xmlChar *) prefix);
	if (nsptr && nsptr->href != NULL) {
		RETURN_STRING((char *) nsptr->href);
	}

	RETURN_NULL();
}
/* }}} end dom_node_lookup_namespace_uri */

static int dom_canonicalize_node_parent_lookup_cb(void *user_data, xmlNodePtr node, xmlNodePtr parent)
{
	xmlNodePtr root = user_data;
	/* We have to unroll the first iteration because node->parent
	 * is not necessarily equal to parent due to libxml2 tree rules (ns decls out of the tree for example). */
	if (node == root) {
		return 1;
	}
	node = parent;
	while (node != NULL) {
		if (node == root) {
			return 1;
		}
		node = node->parent;
	}

	return 0;
}

static void dom_canonicalization(INTERNAL_FUNCTION_PARAMETERS, int mode) /* {{{ */
{
	zval *id;
	zval *xpath_array=NULL, *ns_prefixes=NULL;
	xmlNodePtr nodep;
	xmlDocPtr docp;
	xmlNodeSetPtr nodeset = NULL;
	dom_object *intern;
	bool exclusive=0, with_comments=0;
	xmlChar **inclusive_ns_prefixes = NULL;
	char *file = NULL;
	int ret = -1;
	size_t file_len = 0;
	xmlOutputBufferPtr buf;
	xmlXPathContextPtr ctxp=NULL;
	xmlXPathObjectPtr xpathobjp=NULL;

	id = ZEND_THIS;
	if (mode == 0) {
		if (zend_parse_parameters(ZEND_NUM_ARGS(),
			"|bba!a!", &exclusive, &with_comments,
			&xpath_array, &ns_prefixes) == FAILURE) {
			RETURN_THROWS();
		}
	} else {

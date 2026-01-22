   |          Rob Richards <rrichards@php.net>                            |
   +----------------------------------------------------------------------+
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#if defined(HAVE_LIBXML) && defined(HAVE_DOM)
#include "php_dom.h"

/*
* class DOMElement extends DOMNode
*
* URL: https://www.w3.org/TR/2003/WD-DOM-Level-3-Core-20030226/DOM3-Core.html#core-ID-745549614
* Since:
*/

/* {{{ */
PHP_METHOD(DOMElement, __construct)
{
	xmlNodePtr nodep = NULL, oldnode = NULL;
	dom_object *intern;
	char *name, *value = NULL, *uri = NULL;
	char *localname = NULL, *prefix = NULL;
	int errorcode = 0;
	size_t name_len, value_len = 0, uri_len = 0;
	int name_valid;
	xmlNsPtr nsptr = NULL;

	if (zend_parse_parameters(ZEND_NUM_ARGS(), "s|s!s", &name, &name_len, &value, &value_len, &uri, &uri_len) == FAILURE) {
		RETURN_THROWS();
	}

	name_valid = xmlValidateName((xmlChar *) name, 0);
	if (name_valid != 0) {
		php_dom_throw_error(INVALID_CHARACTER_ERR, 1);
		RETURN_THROWS();
	}

	/* Namespace logic is separate and only when uri passed in to insure no BC breakage */
	if (uri_len > 0) {
		errorcode = dom_check_qname(name, &localname, &prefix, uri_len, name_len);
		if (errorcode == 0) {
			nodep = xmlNewNode (NULL, (xmlChar *)localname);
			if (nodep != NULL && uri != NULL) {
				nsptr = dom_get_ns(nodep, uri, &errorcode, prefix);
				xmlSetNs(nodep, nsptr);
			}
		}
		xmlFree(localname);
		if (prefix != NULL) {
			xmlFree(prefix);
		}
		if (errorcode != 0) {
			if (nodep != NULL) {
				xmlFreeNode(nodep);
			}
			php_dom_throw_error(errorcode, 1);
			RETURN_THROWS();
		}
	} else {
	    /* If you don't pass a namespace uri, then you can't set a prefix */
	    localname = (char *) xmlSplitQName2((xmlChar *) name, (xmlChar **) &prefix);
	    if (prefix != NULL) {
			xmlFree(localname);
			xmlFree(prefix);
	        php_dom_throw_error(NAMESPACE_ERR, 1);
	        RETURN_THROWS();

    return (unsigned short) line;
}

static const php_dom_ns_magic_token *get_libxml_namespace_href(uintptr_t lexbor_namespace)
{
    if (lexbor_namespace == LXB_NS_SVG) {
        return php_dom_ns_is_svg_magic_token;
    } else if (lexbor_namespace == LXB_NS_MATH) {
        return php_dom_ns_is_mathml_magic_token;
    } else {
        return php_dom_ns_is_html_magic_token;
    }
}

static zend_always_inline xmlNodePtr lexbor_libxml2_bridge_new_text_node_fast(xmlDocPtr lxml_doc, const lxb_char_t *data, size_t data_length, bool compact_text_nodes)
{
    if (compact_text_nodes && data_length < LXML_INTERNED_STRINGS_SIZE) {
        /* See xmlSAX2TextNode() in libxml2 */
        xmlNodePtr lxml_text = xmlMalloc(sizeof(*lxml_text));
        if (UNEXPECTED(lxml_text == NULL)) {
            return NULL;

    if (is<HTMLTemplateElement>(*adjusted_insertion_location.parent))
        return { verify_cast<HTMLTemplateElement>(*adjusted_insertion_location.parent).content().ptr(), nullptr };

    return adjusted_insertion_location;
}

JS::NonnullGCPtr<DOM::Element> HTMLParser::create_element_for(HTMLToken const& token, Optional<FlyString> const& namespace_, DOM::Node& intended_parent)
{
    // FIXME: 1. If the active speculative HTML parser is not null, then return the result of creating a speculative mock element given given namespace, the tag name of the given token, and the attributes of the given token.
    // FIXME: 2. Otherwise, optionally create a speculative mock element given given namespace, the tag name of the given token, and the attributes of the given token.

    // 3. Let document be intended parent's node document.
    JS::NonnullGCPtr<DOM::Document> document = intended_parent.document();

    // 4. Let local name be the tag name of the token.
    auto const& local_name = token.tag_name();

    // 5. Let is be the value of the "is" attribute in the given token, if such an attribute exists, or null otherwise.
    auto is_value_deprecated_string = token.attribute(AttributeNames::is);
    Optional<String> is_value;
    if (!is_value_deprecated_string.is_null())
        is_value = String::from_utf8(is_value_deprecated_string).release_value_but_fixme_should_propagate_errors();

    // 6. Let definition be the result of looking up a custom element definition given document, given namespace, local name, and is.
    auto definition = document->lookup_custom_element_definition(namespace_, local_name, is_value);

    // 7. If definition is non-null and the parser was not created as part of the HTML fragment parsing algorithm, then let will execute script be true. Otherwise, let it be false.
    bool will_execute_script = definition && !m_parsing_fragment;

    // 8. If will execute script is true, then:
    if (will_execute_script) {
        // 1. Increment document's throw-on-dynamic-markup-insertion counter.
        document->increment_throw_on_dynamic_markup_insertion_counter({});

        // 2. If the JavaScript execution context stack is empty, then perform a microtask checkpoint.
        auto& vm = main_thread_event_loop().vm();
        if (vm.execution_context_stack().is_empty())
            perform_a_microtask_checkpoint();

        // 3. Push a new element queue onto document's relevant agent's custom element reactions stack.
        auto& custom_data = verify_cast<Bindings::WebEngineCustomData>(*vm.custom_data());
        custom_data.custom_element_reactions_stack.element_queue_stack.append({});
    }

    // 9. Let element be the result of creating an element given document, localName, given namespace, null, and is.
    //    If will execute script is true, set the synchronous custom elements flag; otherwise, leave it unset.
    auto element = create_element(*document, local_name, namespace_, {}, is_value, will_execute_script).release_value_but_fixme_should_propagate_errors();

    // 10. Append each attribute in the given token to element.
    // FIXME: This isn't the exact `append` the spec is talking about.
    token.for_each_attribute([&](auto& attribute) {
        MUST(element->set_attribute(attribute.local_name, attribute.value));
        return IterationDecision::Continue;
    });

    // 11. If will execute script is true, then:
    if (will_execute_script) {
        // 1. Let queue be the result of popping from document's relevant agent's custom element reactions stack. (This will be the same element queue as was pushed above.)
        auto& vm = main_thread_event_loop().vm();
        auto& custom_data = verify_cast<Bindings::WebEngineCustomData>(*vm.custom_data());
        auto queue = custom_data.custom_element_reactions_stack.element_queue_stack.take_last();

        // 2. Invoke custom element reactions in queue.
        Bindings::invoke_custom_element_reactions(queue);

        // 3. Decrement document's throw-on-dynamic-markup-insertion counter.
        document->decrement_throw_on_dynamic_markup_insertion_counter({});
    }

    // FIXME: 12. If element has an xmlns attribute in the XMLNS namespace whose value is not exactly the same as the element's namespace, that is a parse error.
    //            Similarly, if element has an xmlns:xlink attribute in the XMLNS namespace whose value is not the XLink Namespace, that is a parse error.

    // FIXME: 13. If element is a resettable element, invoke its reset algorithm. (This initializes the element's value and checkedness based on the element's attributes.)

    // 14. If element is a form-associated element and not a form-associated custom element, the form element pointer is not null, there is no template element on the stack of open elements,
    //     element is either not listed or doesn't have a form attribute, and the intended parent is in the same tree as the element pointed to by the form element pointer,
    //     then associate element with the form element pointed to by the form element pointer and set element's parser inserted flag.
    // FIXME: Check if the element is not a form-associated custom element.
    if (is<FormAssociatedElement>(*element)) {
        auto* form_associated_element = dynamic_cast<FormAssociatedElement*>(element.ptr());
        VERIFY(form_associated_element);

        auto& html_element = form_associated_element->form_associated_element_to_html_element();

        if (m_form_element.ptr()
            && !m_stack_of_open_elements.contains(HTML::TagNames::template_)
            && (!form_associated_element->is_listed() || !html_element.has_attribute(HTML::AttributeNames::form))
            && &intended_parent.root() == &m_form_element->root()) {
            form_associated_element->set_form(m_form_element.ptr());
            form_associated_element->set_parser_inserted({});

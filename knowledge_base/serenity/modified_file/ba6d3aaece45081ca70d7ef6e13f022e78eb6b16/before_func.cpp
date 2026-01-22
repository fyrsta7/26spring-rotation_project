        invalidation |= CSS::compute_property_invalidation(property_id, old_value, new_value);
    }
    return invalidation;
}

CSS::RequiredInvalidationAfterStyleChange Element::recompute_style()
{
    set_needs_style_update(false);
    VERIFY(parent());

    auto& style_computer = document().style_computer();
    auto new_computed_css_values = style_computer.compute_style(*this);

    // Tables must not inherit -libweb-* values for text-align.
    // FIXME: Find the spec for this.
    if (is<HTML::HTMLTableElement>(*this)) {
        auto text_align = new_computed_css_values->text_align();
        if (text_align.has_value() && (text_align.value() == CSS::TextAlign::LibwebLeft || text_align.value() == CSS::TextAlign::LibwebCenter || text_align.value() == CSS::TextAlign::LibwebRight))
            new_computed_css_values->set_property(CSS::PropertyID::TextAlign, CSS::CSSKeywordValue::create(CSS::Keyword::Start));
    }

    CSS::RequiredInvalidationAfterStyleChange invalidation;
    if (m_computed_css_values)
        invalidation = compute_required_invalidation(*m_computed_css_values, *new_computed_css_values);
    else
        invalidation = CSS::RequiredInvalidationAfterStyleChange::full();

    if (!invalidation.is_none())
        set_computed_css_values(move(new_computed_css_values));

    // Any document change that can cause this element's style to change, could also affect its pseudo-elements.
    for (auto i = 0; i < to_underlying(CSS::Selector::PseudoElement::Type::KnownPseudoElementCount); i++) {
        style_computer.push_ancestor(*this);

        auto pseudo_element = static_cast<CSS::Selector::PseudoElement::Type>(i);
        auto pseudo_element_style = pseudo_element_computed_css_values(pseudo_element);
        auto new_pseudo_element_style = style_computer.compute_pseudo_element_style_if_needed(*this, pseudo_element);

        // TODO: Can we be smarter about invalidation?
        if (pseudo_element_style && new_pseudo_element_style) {
            invalidation |= compute_required_invalidation(*pseudo_element_style, *new_pseudo_element_style);
        } else if (pseudo_element_style || new_pseudo_element_style) {
            invalidation = CSS::RequiredInvalidationAfterStyleChange::full();
        }

        set_pseudo_element_computed_css_values(pseudo_element, move(new_pseudo_element_style));
        style_computer.pop_ancestor(*this);
    }

    if (invalidation.is_none())
        return invalidation;

    if (invalidation.repaint)
        document().set_needs_to_resolve_paint_only_properties();

    if (!invalidation.rebuild_layout_tree && layout_node()) {
        // If we're keeping the layout tree, we can just apply the new style to the existing layout tree.
        layout_node()->apply_style(*m_computed_css_values);
        if (invalidation.repaint && paintable())
            paintable()->set_needs_display();

        // Do the same for pseudo-elements.
        for (auto i = 0; i < to_underlying(CSS::Selector::PseudoElement::Type::KnownPseudoElementCount); i++) {
            auto pseudo_element_type = static_cast<CSS::Selector::PseudoElement::Type>(i);
            auto pseudo_element = get_pseudo_element(pseudo_element_type);
            if (!pseudo_element.has_value() || !pseudo_element->layout_node)
                continue;

            auto pseudo_element_style = pseudo_element_computed_css_values(pseudo_element_type);
            if (!pseudo_element_style)
                continue;

            if (auto* node_with_style = dynamic_cast<Layout::NodeWithStyle*>(pseudo_element->layout_node.ptr())) {
                node_with_style->apply_style(*pseudo_element_style);
                if (invalidation.repaint && node_with_style->paintable())
                    node_with_style->paintable()->set_needs_display();
            }

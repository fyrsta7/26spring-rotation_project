        verify_cast<CharacterData>(this)->set_data(value);
    }

    // Otherwise: Do nothing.
}

void Node::invalidate_style()
{
    for_each_in_inclusive_subtree([&](Node& node) {
        node.m_needs_style_update = true;
        if (node.has_children())
            node.m_child_needs_style_update = true;
        if (auto* shadow_root = node.is_element() ? static_cast<DOM::Element&>(node).shadow_root() : nullptr) {
            shadow_root->m_needs_style_update = true;
            if (shadow_root->has_children())
                shadow_root->m_child_needs_style_update = true;
        }
        return IterationDecision::Continue;
    });
    for (auto* ancestor = parent_or_shadow_host(); ancestor; ancestor = parent_or_shadow_host()) {

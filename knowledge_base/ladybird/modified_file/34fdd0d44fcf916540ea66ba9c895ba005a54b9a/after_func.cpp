    }

    collect_ancestor_hashes();
}

void Selector::collect_ancestor_hashes()
{
    size_t next_hash_index = 0;
    auto append_unique_hash = [&](u32 hash) -> bool {
        if (next_hash_index >= m_ancestor_hashes.size())
            return true;
        for (size_t i = 0; i < next_hash_index; ++i) {
            if (m_ancestor_hashes[i] == hash)
                return false;
        }
        m_ancestor_hashes[next_hash_index++] = hash;
        return false;
    };

    auto last_combinator = m_compound_selectors.last().combinator;
    for (ssize_t compound_selector_index = static_cast<ssize_t>(m_compound_selectors.size()) - 2; compound_selector_index >= 0; --compound_selector_index) {
        auto const& compound_selector = m_compound_selectors[compound_selector_index];
        if (last_combinator == Combinator::Descendant || last_combinator == Combinator::ImmediateChild) {
            for (auto const& simple_selector : compound_selector.simple_selectors) {
                switch (simple_selector.type) {
                case SimpleSelector::Type::Id:
                case SimpleSelector::Type::Class:
                    if (append_unique_hash(simple_selector.name().hash()))
                        return;
                    break;
                case SimpleSelector::Type::TagName:
                    if (append_unique_hash(simple_selector.qualified_name().name.name.hash()))
                        return;
                    break;
                case SimpleSelector::Type::Attribute:
                    if (append_unique_hash(simple_selector.attribute().qualified_name.name.name.hash()))
                        return;
                    break;
                default:
                    break;
                }
            }
        }
        last_combinator = compound_selector.combinator;

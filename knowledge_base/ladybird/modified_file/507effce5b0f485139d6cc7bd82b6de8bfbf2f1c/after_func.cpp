
    return outline_item;
}

NonnullRefPtrVector<OutlineItem> Document::build_outline_item_chain(Value const& first_ref, Value const& last_ref)
{
    VERIFY(first_ref.is_ref());
    VERIFY(last_ref.is_ref());

    NonnullRefPtrVector<OutlineItem> children;

    auto first_dict = object_cast<DictObject>(get_or_load_value(first_ref.as_ref_index()).as_object());
    auto first = build_outline_item(first_dict);
    children.append(first);

    auto current_child_dict = first_dict;
    u32 current_child_index = first_ref.as_ref_index();

    while (current_child_dict->contains(CommonNames::Next)) {
        auto next_child_dict_ref = current_child_dict->get_value(CommonNames::Next);
        current_child_index = next_child_dict_ref.as_ref_index();
        auto next_child_dict = object_cast<DictObject>(get_or_load_value(current_child_index).as_object());
        auto next_child = build_outline_item(next_child_dict);
        children.append(next_child);

        current_child_dict = move(next_child_dict);
    }


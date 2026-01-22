}

void Document::set_needs_to_refresh_scroll_state(bool b)
{
    if (auto* paintable = this->paintable())
        paintable->set_needs_to_refresh_scroll_state(b);
}

Vector<JS::Handle<DOM::Range>> Document::find_matching_text(String const& query, CaseSensitivity case_sensitivity)
{
    if (!layout_node())
        return {};

    // Ensure the layout tree exists before searching for text matches.
    update_layout();

    auto const& text_blocks = layout_node()->text_blocks();
    if (text_blocks.is_empty())
        return {};

    Vector<JS::Handle<DOM::Range>> matches;
    for (auto const& text_block : text_blocks) {
        size_t offset = 0;
        size_t i = 0;
        auto const& text = text_block.text;
        auto* match_start_position = &text_block.positions[0];
        while (true) {
            auto match_index = case_sensitivity == CaseSensitivity::CaseInsensitive
                ? text.find_byte_offset_ignoring_case(query, offset)
                : text.find_byte_offset(query, offset);
            if (!match_index.has_value())
                break;

            for (; i < text_block.positions.size() - 1 && match_index.value() > text_block.positions[i + 1].start_offset; ++i)
                match_start_position = &text_block.positions[i + 1];

            auto range = create_range();
            auto start_position = match_index.value() - match_start_position->start_offset;
            auto& start_dom_node = match_start_position->dom_node;
            (void)range->set_start(start_dom_node, start_position);

            auto* match_end_position = match_start_position;
            for (; i < text_block.positions.size() - 1 && (match_index.value() + query.bytes_as_string_view().length() > text_block.positions[i + 1].start_offset); ++i)
                match_end_position = &text_block.positions[i + 1];

            auto& end_dom_node = match_end_position->dom_node;
            auto end_position = match_index.value() + query.bytes_as_string_view().length() - match_end_position->start_offset;
            (void)range->set_end(end_dom_node, end_position);

            matches.append(range);
            match_start_position = match_end_position;

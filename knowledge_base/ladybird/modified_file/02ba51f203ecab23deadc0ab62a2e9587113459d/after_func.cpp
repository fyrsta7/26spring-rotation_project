    viewport_computed_values.set_overflow_x(overflow_origin_computed_values.overflow_x());
    viewport_computed_values.set_overflow_y(overflow_origin_computed_values.overflow_y());

    // The element from which the value is propagated must then have a used overflow value of visible.
    overflow_origin_computed_values.set_overflow_x(CSS::Overflow::Visible);
    overflow_origin_computed_values.set_overflow_y(CSS::Overflow::Visible);
}

void Document::update_layout()
{
    auto navigable = this->navigable();
    if (!navigable || navigable->active_document() != this)
        return;

    // NOTE: If our parent document needs a relayout, we must do that *first*.
    //       This is necessary as the parent layout may cause our viewport to change.
    if (navigable->container())
        navigable->container()->document().update_layout();

    update_style();

    if (!m_needs_layout && m_layout_root)
        return;

    // NOTE: If this is a document hosting <template> contents, layout is unnecessary.
    if (m_created_for_appropriate_template_contents)
        return;

    auto* document_element = this->document_element();
    auto viewport_rect = navigable->viewport_rect();

    if (!m_layout_root) {
        Layout::TreeBuilder tree_builder;
        m_layout_root = verify_cast<Layout::Viewport>(*tree_builder.build(*this));

        if (document_element && document_element->layout_node()) {
            propagate_overflow_to_viewport(*document_element, *m_layout_root);
            propagate_scrollbar_width_to_viewport(*document_element, *m_layout_root);
        }
    }

    Layout::LayoutState layout_state;

    {
        Layout::BlockFormattingContext root_formatting_context(layout_state, *m_layout_root, nullptr);

        auto& viewport = static_cast<Layout::Viewport&>(*m_layout_root);
        auto& viewport_state = layout_state.get_mutable(viewport);
        viewport_state.set_content_width(viewport_rect.width());
        viewport_state.set_content_height(viewport_rect.height());

        if (document_element && document_element->layout_node()) {
            auto& icb_state = layout_state.get_mutable(verify_cast<Layout::NodeWithStyleAndBoxModelMetrics>(*document_element->layout_node()));
            icb_state.set_content_width(viewport_rect.width());
        }

        root_formatting_context.run(
            *m_layout_root,
            Layout::LayoutMode::Normal,
            Layout::AvailableSpace(
                Layout::AvailableSize::make_definite(viewport_rect.width()),
                Layout::AvailableSize::make_definite(viewport_rect.height())));
    }

    layout_state.commit(*m_layout_root);

    // Broadcast the current viewport rect to any new paintables, so they know whether they're visible or not.
    inform_all_viewport_clients_about_the_current_viewport_rect();

    navigable->set_needs_display();
    set_needs_to_resolve_paint_only_properties();

    if (navigable->is_traversable()) {
        // NOTE: The assignment of scroll frames only needs to occur for traversables because they take care of all
        //       nested navigable documents.
        paintable()->assign_scroll_frames();
        paintable()->assign_clip_frames();

        page().client().page_did_layout();
    }

    paintable()->recompute_selection_states();

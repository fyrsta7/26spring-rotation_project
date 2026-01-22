        return max(viewport_rect.width(), viewport_rect.height()) * (m_value / 100);
    default:
        VERIFY_NOT_REACHED();
    }
}

float Length::to_px(Layout::Node const& layout_node) const
{
    if (is_calculated())
        return m_calculated_style->resolve_length(layout_node)->to_px(layout_node);

    if (is_absolute())
        return absolute_length_to_px();

    if (!layout_node.document().browsing_context())
        return 0;

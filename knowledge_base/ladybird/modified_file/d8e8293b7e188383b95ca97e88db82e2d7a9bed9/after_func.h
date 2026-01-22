        (void)reference_value;
        VERIFY_NOT_REACHED();
    }

    CSSPixels to_px(Layout::Node const& layout_node, CSSPixels reference_value) const
    {
        if constexpr (IsSame<T, Length>) {
            if (auto const* length = m_value.template get_pointer<Length>()) {
                if (length->is_absolute())
                    return length->absolute_length_to_px();

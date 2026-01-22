        s_icon = Gfx::Bitmap::try_load_from_file("/res/icons/16x16/move.png"sv).release_value_but_fixme_should_propagate_errors();
    return *s_icon;
}

Window::Window(Core::Object& parent, WindowType type)
    : Core::Object(&parent)
    , m_type(type)
    , m_icon(default_window_icon())
    , m_frame(*this)

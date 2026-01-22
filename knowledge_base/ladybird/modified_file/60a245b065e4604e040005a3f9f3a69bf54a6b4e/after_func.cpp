    m_front_store = nullptr;
    m_cursor = Gfx::StandardCursor::None;
}

void Window::hide()
{
    if (!is_visible())
        return;

    // NOTE: Don't bother asking WindowServer to destroy windows during application teardown.
    //       All our windows will be automatically garbage-collected by WindowServer anyway.
    if (GUI::Application::in_teardown())
        return;

    auto destroyed_window_ids = WindowServerConnection::the().destroy_window(m_window_id);
    server_did_destroy();

    for (auto child_window_id : destroyed_window_ids) {
        if (auto* window = Window::from_window_id(child_window_id)) {
            window->server_did_destroy();
        }
    }

    if (auto* app = Application::the()) {
        bool app_has_visible_windows = false;
        for (auto& window : *all_windows) {
            if (window->is_visible()) {
                app_has_visible_windows = true;
                break;
            }
        }

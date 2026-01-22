    update_cursor();
}

void Window::set_cursor(const Gfx::Bitmap& cursor)
{
    if (m_custom_cursor == &cursor)
        return;
    m_cursor = Gfx::StandardCursor::None;
    m_custom_cursor = &cursor;
    update_cursor();
}

void Window::handle_drop_event(DropEvent& event)
{
    if (!m_main_widget)
        return;
    auto result = m_main_widget->hit_test(event.position());
    auto local_event = make<DropEvent>(result.local_position, event.text(), event.mime_data());
    VERIFY(result.widget);
    result.widget->dispatch_event(*local_event, this);

    Application::the()->set_drag_hovered_widget({}, nullptr);
}

void Window::handle_mouse_event(MouseEvent& event)
{
    if (m_global_cursor_tracking_widget) {
        auto window_relative_rect = m_global_cursor_tracking_widget->window_relative_rect();
        Gfx::IntPoint local_point { event.x() - window_relative_rect.x(), event.y() - window_relative_rect.y() };
        auto local_event = make<MouseEvent>((Event::Type)event.type(), local_point, event.buttons(), event.button(), event.modifiers(), event.wheel_delta());

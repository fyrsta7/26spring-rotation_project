    }
    GUI::Frame::event(event);
}

void TimelineTrack::paint_event(GUI::PaintEvent& event)
{
    GUI::Frame::paint_event(event);

    GUI::Painter painter(*this);
    painter.add_clip_rect(event.rect());

    u64 const start_of_trace = m_profile.first_timestamp();
    u64 const end_of_trace = start_of_trace + m_profile.length_in_ms();

    auto const clamp_timestamp = [start_of_trace, end_of_trace](u64 timestamp) -> u64 {
        return min(end_of_trace, max(timestamp, start_of_trace));
    };

    recompute_histograms_if_needed({ start_of_trace, end_of_trace, (size_t)m_profile.length_in_ms() });

    float column_width = this->column_width();
    float frame_height = (float)frame_inner_rect().height() / (float)m_max_value;

    for (size_t bucket = 0; bucket < m_kernel_histogram->size(); bucket++) {
        auto kernel_value = m_kernel_histogram->at(bucket);
        auto user_value = m_user_histogram->at(bucket);
        if (kernel_value + user_value == 0)
            continue;

        auto t = bucket;

        int x = (int)((float)t * column_width);
        int cw = max(1, (int)column_width);

        int kernel_column_height = frame_inner_rect().height() - (int)((float)kernel_value * frame_height);
        int user_column_height = frame_inner_rect().height() - (int)((float)(kernel_value + user_value) * frame_height);

        constexpr auto kernel_color = Color::from_rgb(0xc25e5a);
        constexpr auto user_color = Color::from_rgb(0x5a65c2);
        painter.fill_rect({ x, frame_thickness() + user_column_height, cw, height() - frame_thickness() * 2 }, user_color);
        painter.fill_rect({ x, frame_thickness() + kernel_column_height, cw, height() - frame_thickness() * 2 }, kernel_color);
    }

    u64 normalized_start_time = clamp_timestamp(min(m_view.select_start_time(), m_view.select_end_time()));
    u64 normalized_end_time = clamp_timestamp(max(m_view.select_start_time(), m_view.select_end_time()));
    u64 normalized_hover_time = clamp_timestamp(m_view.hover_time());

    int select_start_x = (int)((float)(normalized_start_time - start_of_trace) * column_width);
    int select_end_x = (int)((float)(normalized_end_time - start_of_trace) * column_width);
    int select_hover_x = (int)((float)(normalized_hover_time - start_of_trace) * column_width);
    painter.fill_rect({ select_start_x, frame_thickness(), select_end_x - select_start_x, height() - frame_thickness() * 2 }, Color(0, 0, 0, 60));
    painter.fill_rect({ select_hover_x, frame_thickness(), 1, height() - frame_thickness() * 2 }, Color::NamedColor::Black);

    for_each_signpost([&](auto& signpost) {
        int x = (int)((float)(signpost.timestamp - start_of_trace) * column_width);
        int y1 = frame_thickness();
        int y2 = height() - frame_thickness() * 2;

        painter.draw_line({ x, y1 }, { x, y2 }, Color::Magenta);

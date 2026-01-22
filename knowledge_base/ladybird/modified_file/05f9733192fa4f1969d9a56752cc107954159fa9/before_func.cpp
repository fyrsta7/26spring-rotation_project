        return fill_internal(painter, path, move(sampler), winding_rule, offset);
    });
}

template<unsigned SamplesPerPixel>
void EdgeFlagPathRasterizer<SamplesPerPixel>::fill_internal(Painter& painter, Path const& path, auto color_or_function, Painter::WindingRule winding_rule, FloatPoint offset)
{
    // FIXME: Figure out how painter scaling works here...
    VERIFY(painter.scale() == 1);

    auto bounding_box = enclosing_int_rect(path.bounding_box().translated(offset));
    auto dest_rect = bounding_box.translated(painter.translation());
    auto origin = bounding_box.top_left().to_type<float>() - offset;
    m_blit_origin = dest_rect.top_left();
    m_clip = dest_rect.intersected(painter.clip_rect());

    // Only allocate enough to plot the parts of the scanline that could be visible.
    // Note: This can't clip the LHS.
    auto scanline_length = min(m_size.width(), m_clip.right() - m_blit_origin.x());
    if (scanline_length <= 0)
        return;

    m_scanline.resize(scanline_length);

    if (m_clip.is_empty())
        return;

    auto lines = path.split_lines();
    if (lines.is_empty())
        return;

    int min_edge_y = 0;
    int max_edge_y = 0;
    auto top_clip_scanline = m_clip.top() - m_blit_origin.y();
    auto bottom_clip_scanline = m_clip.bottom() - m_blit_origin.y() - 1;
    auto edges = prepare_edges(lines, SamplesPerPixel, origin, top_clip_scanline, bottom_clip_scanline, min_edge_y, max_edge_y);
    if (edges.is_empty())
        return;

    int min_scanline = min_edge_y / SamplesPerPixel;
    int max_scanline = max_edge_y / SamplesPerPixel;
    m_edge_table.set_scanline_range(min_scanline, max_scanline);
    for (auto& edge : edges) {
        // Create a linked-list of edges starting on this scanline:
        int start_scanline = edge.min_y / SamplesPerPixel;
        edge.next_edge = m_edge_table[start_scanline];
        m_edge_table[start_scanline] = &edge;
    }

    auto empty_edge_extent = [&] {
        return EdgeExtent { m_size.width() - 1, 0 };
    };

    auto for_each_sample = [&](Detail::Edge& edge, int start_subpixel_y, int end_subpixel_y, EdgeExtent& edge_extent, auto callback) {
        for (int y = start_subpixel_y; y < end_subpixel_y; y++) {
            auto xi = static_cast<int>(edge.x + SubpixelSample::nrooks_subpixel_offsets[y]);
            if (xi >= 0 && size_t(xi) < m_scanline.size()) [[likely]] {
                SampleType sample = 1 << y;
                callback(xi, y, sample);
            } else if (xi < 0) {
                if (edge.dxdy <= 0)
                    return;
            } else {
                xi = m_scanline.size() - 1;
            }
            edge.x += edge.dxdy;
            edge_extent.min_x = min(edge_extent.min_x, xi);
            edge_extent.max_x = max(edge_extent.max_x, xi);
        }
    };

    Detail::Edge* active_edges = nullptr;

    if (winding_rule == Painter::WindingRule::EvenOdd) {
        auto plot_edge = [&](Detail::Edge& edge, int start_subpixel_y, int end_subpixel_y, EdgeExtent& edge_extent) {
            for_each_sample(edge, start_subpixel_y, end_subpixel_y, edge_extent, [&](int xi, int, SampleType sample) {
                m_scanline[xi] ^= sample;
            });
        };
        for (int scanline = min_scanline; scanline <= max_scanline; scanline++) {
            auto edge_extent = empty_edge_extent();
            active_edges = plot_edges_for_scanline(scanline, plot_edge, edge_extent, active_edges);
            write_scanline<Painter::WindingRule::EvenOdd>(painter, scanline, edge_extent, color_or_function);
        }
    } else {
        VERIFY(winding_rule == Painter::WindingRule::Nonzero);
        // Only allocate the winding buffer if needed.
        // NOTE: non-zero fills are a fair bit less efficient. So if you can do an even-odd fill do that :^)
        if (m_windings.is_empty())
            m_windings.resize(m_scanline.size());

        auto plot_edge = [&](Detail::Edge& edge, int start_subpixel_y, int end_subpixel_y, EdgeExtent& edge_extent) {
            for_each_sample(edge, start_subpixel_y, end_subpixel_y, edge_extent, [&](int xi, int y, SampleType sample) {
                m_scanline[xi] |= sample;
                m_windings[xi].counts[y] += edge.winding;
            });
        };
        for (int scanline = min_scanline; scanline <= max_scanline; scanline++) {
            auto edge_extent = empty_edge_extent();
            active_edges = plot_edges_for_scanline(scanline, plot_edge, edge_extent, active_edges);

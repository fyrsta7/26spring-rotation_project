    if (fill_rule == "nonzero"sv)
        return Gfx::Painter::WindingRule::Nonzero;
    dbgln("Unrecognized fillRule for CRC2D.fill() - this problem goes away once we pass an enum instead of a string");
    return Gfx::Painter::WindingRule::Nonzero;
}

void CanvasRenderingContext2D::fill_internal(Gfx::Path& path, StringView fill_rule_value)
{
    draw_clipped([&](auto& painter) {
        path.close_all_subpaths();
        auto& drawing_state = this->drawing_state();
        auto fill_rule = parse_fill_rule(fill_rule_value);
        if (auto color = drawing_state.fill_style.as_color(); color.has_value()) {
            painter.fill_path(path, *color, fill_rule);

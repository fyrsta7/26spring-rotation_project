    if (fill_rule == "nonzero"sv)
        return Gfx::Painter::WindingRule::Nonzero;
    dbgln("Unrecognized fillRule for CRC2D.fill() - this problem goes away once we pass an enum instead of a string");
    return Gfx::Painter::WindingRule::Nonzero;
}

void CanvasRenderingContext2D::fill_internal(Gfx::Path& path, StringView fill_rule)
{

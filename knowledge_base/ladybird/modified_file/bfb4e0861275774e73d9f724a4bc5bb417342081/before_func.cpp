        position.set_x(position.x() - src_rect.x());
        src_rect.set_x(0);
    }
    if (src_rect.y() < 0) {
        position.set_y(position.y() - src_rect.y());
        src_rect.set_y(0);
    }
    blit(position, source, src_rect);
}

void Painter::blit(IntPoint const& position, Gfx::Bitmap const& source, IntRect const& a_src_rect, float opacity, bool apply_alpha)
{
    VERIFY(scale() >= source.scale() && "painter doesn't support downsampling scale factors");

    if (opacity < 1.0f || (source.has_alpha_channel() && apply_alpha))
        return blit_with_opacity(position, source, a_src_rect, opacity, apply_alpha);

    auto safe_src_rect = a_src_rect.intersected(source.rect());
    if (scale() != source.scale())
        return draw_scaled_bitmap({ position, safe_src_rect.size() }, source, safe_src_rect, opacity);

    // If we get here, the Painter might have a scale factor, but the source bitmap has the same scale factor.
    // We need to transform from logical to physical coordinates, but we can just copy pixels without resampling.
    auto dst_rect = IntRect(position, safe_src_rect.size()).translated(translation());
    auto clipped_rect = dst_rect.intersected(clip_rect());
    if (clipped_rect.is_empty())
        return;

    // All computations below are in physical coordinates.
    int scale = this->scale();
    auto src_rect = a_src_rect * scale;
    clipped_rect *= scale;
    dst_rect *= scale;

    int const first_row = clipped_rect.top() - dst_rect.top();
    int const last_row = clipped_rect.bottom() - dst_rect.top();
    int const first_column = clipped_rect.left() - dst_rect.left();
    ARGB32* dst = m_target->scanline(clipped_rect.y()) + clipped_rect.x();
    size_t const dst_skip = m_target->pitch() / sizeof(ARGB32);

    if (source.format() == BitmapFormat::BGRx8888 || source.format() == BitmapFormat::BGRA8888) {
        ARGB32 const* src = source.scanline(src_rect.top() + first_row) + src_rect.left() + first_column;
        size_t const src_skip = source.pitch() / sizeof(ARGB32);
        for (int row = first_row; row <= last_row; ++row) {
            fast_u32_copy(dst, src, clipped_rect.width());
            dst += dst_skip;
            src += src_skip;
        }
        return;
    }

    if (source.format() == BitmapFormat::RGBA8888) {
        u32 const* src = source.scanline(src_rect.top() + first_row) + src_rect.left() + first_column;
        size_t const src_skip = source.pitch() / sizeof(u32);
        for (int row = first_row; row <= last_row; ++row) {
            for (int i = 0; i < clipped_rect.width(); ++i) {
                u32 rgba = src[i];
                u32 bgra = (rgba & 0xff00ff00)
                    | ((rgba & 0x000000ff) << 16)
                    | ((rgba & 0x00ff0000) >> 16);
                dst[i] = bgra;
            }
            dst += dst_skip;
            src += src_skip;
        }
        return;
    }

    if (Bitmap::is_indexed(source.format())) {
        u8 const* src = source.scanline_u8(src_rect.top() + first_row) + src_rect.left() + first_column;
        size_t const src_skip = source.pitch();
        for (int row = first_row; row <= last_row; ++row) {

    RefPtr<Gfx::Bitmap> corner_bitmap = TRY(Gfx::Bitmap::create(Gfx::BitmapFormat::BGRA8888, corner_data.corners_bitmap_size));
    return try_make_ref_counted<BorderRadiusCornerClipper>(corner_data, corner_bitmap.release_nonnull(), corner_clip, border_rect);
}

void BorderRadiusCornerClipper::sample_under_corners(Gfx::Painter& page_painter)
{
    // Generate a mask for the corners:
    Gfx::Painter corner_painter { *m_corner_bitmap };
    Gfx::AntiAliasingPainter corner_aa_painter { corner_painter };
    corner_aa_painter.fill_rect_with_rounded_corners(m_corner_bitmap->rect(), Color::NamedColor::Black,
        m_data.corner_radii.top_left, m_data.corner_radii.top_right, m_data.corner_radii.bottom_right, m_data.corner_radii.bottom_left);

    auto copy_page_masked = [&](Gfx::IntRect const& mask_src, Gfx::IntPoint const& page_location) {
        for (int row = 0; row < mask_src.height(); ++row) {
            for (int col = 0; col < mask_src.width(); ++col) {
                auto corner_location = mask_src.location().translated(col, row);
                auto mask_pixel = m_corner_bitmap->get_pixel(corner_location);
                u8 mask_alpha = mask_pixel.alpha();
                if (m_corner_clip == CornerClip::Outside)
                    mask_alpha = ~mask_pixel.alpha();
                auto final_pixel = Color();
                if (mask_alpha > 0) {
                    auto page_pixel = page_painter.get_pixel(page_location.translated(col, row));
                    if (page_pixel.has_value())
                        final_pixel = page_pixel.value().with_alpha(mask_alpha);
                }
                m_corner_bitmap->set_pixel(corner_location, final_pixel);
            }
        }
    };

    // Copy the pixels under the corner mask (using the alpha of the mask):
    if (m_data.corner_radii.top_left)
        copy_page_masked(m_data.corner_radii.top_left.as_rect().translated(m_data.bitmap_locations.top_left), m_data.page_locations.top_left);
    if (m_data.corner_radii.top_right)
        copy_page_masked(m_data.corner_radii.top_right.as_rect().translated(m_data.bitmap_locations.top_right), m_data.page_locations.top_right);
    if (m_data.corner_radii.bottom_right)
        copy_page_masked(m_data.corner_radii.bottom_right.as_rect().translated(m_data.bitmap_locations.bottom_right), m_data.page_locations.bottom_right);
    if (m_data.corner_radii.bottom_left)

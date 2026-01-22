#include <LibGfx/Font/Font.h>

namespace Gfx {

GlyphRasterPosition GlyphRasterPosition::get_nearest_fit_for(FloatPoint position)
{
    constexpr auto subpixel_divisions = GlyphSubpixelOffset::subpixel_divisions();
    auto fit = [](float pos, int& blit_pos, u8& subpixel_offset) {
        blit_pos = floorf(pos);
        subpixel_offset = round_to<u8>((pos - blit_pos) / (1.0f / subpixel_divisions));
        if (subpixel_offset >= subpixel_divisions) {
            blit_pos += 1;
            subpixel_offset = 0;
        }
    };
    int blit_x, blit_y;
    u8 subpixel_x, subpixel_y;

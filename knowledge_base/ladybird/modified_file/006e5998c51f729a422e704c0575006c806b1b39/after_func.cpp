int BitmapFont::width(StringView const& view) const { return unicode_view_width(Utf8View(view)); }
int BitmapFont::width(Utf8View const& view) const { return unicode_view_width(view); }
int BitmapFont::width(Utf32View const& view) const { return unicode_view_width(view); }

template<typename T>
ALWAYS_INLINE int BitmapFont::unicode_view_width(T const& view) const
{
    if (view.is_empty())
        return 0;
    bool first = true;
    int width = 0;
    int longest_width = 0;

    for (u32 code_point : view) {
        if (code_point == '\n' || code_point == '\r') {
            first = true;
            longest_width = max(width, longest_width);
            width = 0;
            continue;
        }
        if (!first)
            width += glyph_spacing();
        first = false;
        width += glyph_or_emoji_width(code_point);

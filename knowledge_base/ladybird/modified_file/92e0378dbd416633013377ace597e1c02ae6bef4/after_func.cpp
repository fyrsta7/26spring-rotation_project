ALWAYS_INLINE bool Lexer::is_unicode_character() const
{
    return (m_current_char & 128) != 0;
}

ALWAYS_INLINE u32 Lexer::current_code_point() const
{
    static constexpr const u32 REPLACEMENT_CHARACTER = 0xFFFD;
    if (m_position == 0)
        return REPLACEMENT_CHARACTER;
    auto substring = m_source.substring_view(m_position - 1);
    if (substring.is_empty())
        return REPLACEMENT_CHARACTER;

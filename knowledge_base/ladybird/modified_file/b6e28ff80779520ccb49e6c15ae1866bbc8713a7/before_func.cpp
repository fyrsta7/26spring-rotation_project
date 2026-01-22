{
    auto decrease_count = min(m_buffer.size(), count);
    m_buffer.resize(m_buffer.size() - decrease_count);
}

ErrorOr<void> StringBuilder::try_append(StringView string)
{
    if (string.is_empty())

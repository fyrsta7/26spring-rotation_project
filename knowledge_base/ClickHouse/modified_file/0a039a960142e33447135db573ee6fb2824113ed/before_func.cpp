
    return {file.size(), State::END};
}

NextState InlineEscapingKeyStateHandler::readQuoted(std::string_view file, ElementType & key) const
{
    const auto quoting_character = extractor_configuration.quoting_character;

    key.clear();

    size_t pos = 0;
    while (const auto * p = find_first_symbols_or_null({file.begin() + pos, file.end()}, read_quoted_needles))
    {
        size_t character_position = p - file.begin();
        size_t next_pos = character_position + 1u;

        if (*p == '\\')
        {
            const size_t escape_seq_len = consumeWithEscapeSequence(file, pos, character_position, key);
            next_pos = character_position + escape_seq_len;

            if (escape_seq_len == 0)
            {
                return {next_pos, State::WAITING_KEY};
            }
        }
        else if (*p == quoting_character)
        {
            // todo try to optimize with resize and memcpy
            for (size_t i = pos; i < character_position; ++i)
            {
                key.push_back(file[i]);
            }

            if (key.empty())
            {
                return {next_pos, State::WAITING_KEY};
            }

            return {next_pos, State::READING_KV_DELIMITER};
        }

        pos = next_pos;

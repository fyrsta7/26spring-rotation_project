    }

    return substring_view(substring_start, substring_length);
}

Utf8CodePointIterator& Utf8CodePointIterator::operator++()
{
    VERIFY(m_length > 0);

    // OPTIMIZATION: Fast path for ASCII characters.
    if (*m_ptr <= 0x7F) {
        m_ptr += 1;
        m_length -= 1;
        return *this;
    }

    size_t code_point_length_in_bytes = underlying_code_point_length_in_bytes();
    if (code_point_length_in_bytes > m_length) {
        // We don't have enough data for the next code point. Skip one character and try again.
        // The rest of the code will output replacement characters as needed for any eventual extension bytes we might encounter afterwards.
        dbgln_if(UTF8_DEBUG, "Expected code point size {} is too big for the remaining length {}. Moving forward one byte.", code_point_length_in_bytes, m_length);
        m_ptr += 1;
        m_length -= 1;
        return *this;
    }

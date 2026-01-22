        return lhs;

    return vm.heap().allocate_without_realm<PrimitiveString>(lhs, rhs);
}

void PrimitiveString::resolve_rope_if_needed(EncodingPreference preference) const
{
    if (!m_is_rope)
        return;

    // This vector will hold all the pieces of the rope that need to be assembled
    // into the resolved string.
    Vector<PrimitiveString const*> pieces;

    // NOTE: We traverse the rope tree without using recursion, since we'd run out of
    //       stack space quickly when handling a long sequence of unresolved concatenations.
    Vector<PrimitiveString const*> stack;
    stack.append(m_rhs);
    stack.append(m_lhs);
    while (!stack.is_empty()) {
        auto const* current = stack.take_last();
        if (current->m_is_rope) {
            stack.append(current->m_rhs);
            stack.append(current->m_lhs);
            continue;
        }
        pieces.append(current);
    }

    if (preference == EncodingPreference::UTF16) {
        // The caller wants a UTF-16 string, so we can simply concatenate all the pieces
        // into a UTF-16 code unit buffer and create a Utf16String from it.

        Utf16Data code_units;
        for (auto const* current : pieces)
            code_units.extend(current->utf16_string().string());

        m_utf16_string = Utf16String::create(move(code_units));
        m_is_rope = false;
        m_lhs = nullptr;
        m_rhs = nullptr;
        return;
    }

    // Now that we have all the pieces, we can concatenate them using a StringBuilder.
    StringBuilder builder;

    // We keep track of the previous piece in order to handle surrogate pairs spread across two pieces.
    PrimitiveString const* previous = nullptr;
    for (auto const* current : pieces) {
        if (!previous) {
            // This is the very first piece, just append it and continue.
            builder.append(current->utf8_string());
            previous = current;
            continue;
        }

        // Get the UTF-8 representations for both strings.
        auto current_string_as_utf8 = current->utf8_string_view();
        auto previous_string_as_utf8 = previous->utf8_string_view();

        // NOTE: Now we need to look at the end of the previous string and the start
        //       of the current string, to see if they should be combined into a surrogate.

        // Surrogates encoded as UTF-8 are 3 bytes.
        if ((previous_string_as_utf8.length() < 3) || (current_string_as_utf8.length() < 3)) {
            builder.append(current_string_as_utf8);
            previous = current;
            continue;
        }

        // Might the previous string end with a UTF-8 encoded surrogate?
        if ((static_cast<u8>(previous_string_as_utf8[previous_string_as_utf8.length() - 3]) & 0xf0) != 0xe0) {
            // If not, just append the current string and continue.
            builder.append(current_string_as_utf8);
            previous = current;
            continue;
        }

        // Might the current string begin with a UTF-8 encoded surrogate?
        if ((static_cast<u8>(current_string_as_utf8[0]) & 0xf0) != 0xe0) {
            // If not, just append the current string and continue.
            builder.append(current_string_as_utf8);
            previous = current;
            continue;
        }

        auto high_surrogate = *Utf8View(previous_string_as_utf8.substring_view(previous_string_as_utf8.length() - 3)).begin();
        auto low_surrogate = *Utf8View(current_string_as_utf8).begin();

        if (!Utf16View::is_high_surrogate(high_surrogate) || !Utf16View::is_low_surrogate(low_surrogate)) {
            builder.append(current_string_as_utf8);
            previous = current;
            continue;
        }

        // Remove 3 bytes from the builder and replace them with the UTF-8 encoded code point.
        builder.trim(3);
        builder.append_code_point(Utf16View::decode_surrogate_pair(high_surrogate, low_surrogate));

        // Append the remaining part of the current string.
        builder.append(current_string_as_utf8.substring_view(3));
        previous = current;
    }

    // NOTE: We've already produced valid UTF-8 above, so there's no need for additional validation.

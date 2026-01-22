    VERIFY(m_rep0 < NumericLimits<u32>::max());
    return m_rep0 + 1;
}

ErrorOr<Bytes> LzmaDecompressor::read_some(Bytes bytes)
{
    while (m_dictionary->used_space() < bytes.size() && m_dictionary->empty_space() != 0) {
        if (m_found_end_of_stream_marker)
            break;

        if (has_reached_expected_data_size()) {
            // If the decoder is in a clean state, we assume that this is fine.
            if (is_range_decoder_in_clean_state())
                break;

            // Otherwise, we give it one last try to find the end marker in the remaining data.
        }

        // "The decoder calculates "state2" variable value to select exact variable from
        //  "IsMatch" and "IsRep0Long" arrays."
        u16 position_state = m_total_decoded_bytes & ((1 << m_options.position_bits) - 1);
        u16 state2 = (m_state << maximum_number_of_position_bits) + position_state;

        auto update_state_after_literal = [&] {
            if (m_state < 4)
                m_state = 0;
            else if (m_state < 10)
                m_state -= 3;
            else
                m_state -= 6;
        };

        auto update_state_after_match = [&] {
            if (m_state < 7)
                m_state = 7;
            else
                m_state = 10;
        };

        auto update_state_after_rep = [&] {
            if (m_state < 7)
                m_state = 8;
            else
                m_state = 11;
        };

        auto update_state_after_short_rep = [&] {
            if (m_state < 7)
                m_state = 9;
            else
                m_state = 11;
        };

        auto copy_match_to_buffer = [&](u16 real_length) -> ErrorOr<void> {
            VERIFY(!m_leftover_match_length.has_value());

            if (m_options.uncompressed_size.has_value() && m_options.uncompressed_size.value() < m_total_decoded_bytes + real_length)
                return Error::from_string_literal("Tried to copy match beyond expected uncompressed file size");

            while (real_length > 0) {
                if (m_dictionary->empty_space() == 0) {
                    m_leftover_match_length = real_length;
                    break;
                }

                u8 byte;
                auto read_bytes = TRY(m_dictionary->read_with_seekback({ &byte, sizeof(byte) }, current_repetition_offset()));
                VERIFY(read_bytes.size() == sizeof(byte));

                auto written_bytes = m_dictionary->write({ &byte, sizeof(byte) });
                VERIFY(written_bytes == sizeof(byte));
                m_total_decoded_bytes++;

                real_length--;
            }

            return {};
        };

        // If we have a leftover part of a repeating match, we should finish that first.
        if (m_leftover_match_length.has_value()) {
            TRY(copy_match_to_buffer(m_leftover_match_length.release_value()));
            continue;
        }

        // "The decoder uses the following code flow scheme to select exact
        //  type of LITERAL or MATCH:
        //
        //  IsMatch[state2] decode
        //   0 - the Literal"
        if (TRY(decode_bit_with_probability(m_is_match_probabilities[state2])) == 0) {
            // If we are already past the expected uncompressed size, we are already in "look for EOS only" mode.
            if (has_reached_expected_data_size())
                return Error::from_string_literal("Found literal after reaching expected uncompressed size");

            // "At first the LZMA decoder must check that it doesn't exceed
            //  specified uncompressed size."
            // This is already checked for at the beginning of the loop.

            // "Then it decodes literal value and puts it to sliding window."
            TRY(decode_literal_to_output_buffer());

            // "Then the decoder must update the "state" value."
            update_state_after_literal();
            continue;
        }

        // " 1 - the Match
        //     IsRep[state] decode
        //       0 - Simple Match"
        if (TRY(decode_bit_with_probability(m_is_rep_probabilities[m_state])) == 0) {
            // "The distance history table is updated with the following scheme:"
            m_rep3 = m_rep2;
            m_rep2 = m_rep1;
            m_rep1 = m_rep0;

            // "The zero-based length is decoded with "LenDecoder"."
            u16 normalized_length = TRY(decode_normalized_match_length(m_length_decoder));

            // "The state is update with UpdateState_Match function."
            update_state_after_match();

            // "and the new "rep0" value is decoded with DecodeDistance."
            m_rep0 = TRY(decode_normalized_match_distance(normalized_length));

            // "If the value of "rep0" is equal to 0xFFFFFFFF, it means that we have
            //  "End of stream" marker, so we can stop decoding and check finishing
            //  condition in Range Decoder"
            if (m_rep0 == 0xFFFFFFFF) {
                // If we should reject end-of-stream markers, do so now.
                // Note that this is not part of LZMA, as LZMA allows end-of-stream markers in all contexts, so pure LZMA should never set this option.
                if (m_options.reject_end_of_stream_marker)
                    return Error::from_string_literal("An end-of-stream marker was found, but the LZMA stream is configured to reject them");

                // The range decoder condition is checked after breaking out of the loop.
                m_found_end_of_stream_marker = true;
                continue;
            }

            // If we are looking for EOS, but haven't found it here, the stream is corrupted.
            if (has_reached_expected_data_size())
                return Error::from_string_literal("First simple match after the expected uncompressed size is not the EOS marker");

            // "If uncompressed size is defined, LZMA decoder must check that it doesn't
            //  exceed that specified uncompressed size."
            // This is being checked for in the common "copy to buffer" implementation.

            // "Also the decoder must check that "rep0" value is not larger than dictionary size
            //  and is not larger than the number of already decoded bytes."
            if (current_repetition_offset() > m_dictionary->seekback_limit())
                return Error::from_string_literal("rep0 value is larger than the possible lookback size");

            // "Then the decoder must copy match bytes as described in
            //  "The match symbols copying" section."
            TRY(copy_match_to_buffer(normalized_length + normalized_to_real_match_length_offset));

            continue;
        }

        // If we are looking for EOS, but find another match type, the stream is also corrupted.
        if (has_reached_expected_data_size())
            return Error::from_string_literal("First match type after the expected uncompressed size is not a simple match");

        // "     1 - Rep Match
        //         IsRepG0[state] decode
        //           0 - the distance is rep0"
        if (TRY(decode_bit_with_probability(m_is_rep_g0_probabilities[m_state])) == 0) {
            // "LZMA doesn't update the distance history."

            // "       IsRep0Long[state2] decode
            //           0 - Short Rep Match"
            if (TRY(decode_bit_with_probability(m_is_rep0_long_probabilities[state2])) == 0) {
                // "If the subtype is "Short Rep Match", the decoder updates the state, puts
                //  the one byte from window to current position in window and goes to next
                //  MATCH/LITERAL symbol."
                update_state_after_short_rep();

                TRY(copy_match_to_buffer(1));

                continue;
            }
            // "         1 - Rep Match 0"
            // Intentional fallthrough, we just need to make sure to not run the detection for other match types and to not switch around the distance history.
        } else {
            // "     1 -
            //         IsRepG1[state] decode
            //           0 - Rep Match 1"
            if (TRY(decode_bit_with_probability(m_is_rep_g1_probabilities[m_state])) == 0) {
                u32 distance = m_rep1;
                m_rep1 = m_rep0;
                m_rep0 = distance;
            }

            // "         1 -
            //             IsRepG2[state] decode
            //               0 - Rep Match 2"
            else if (TRY(decode_bit_with_probability(m_is_rep_g2_probabilities[m_state])) == 0) {
                u32 distance = m_rep2;
                m_rep2 = m_rep1;
                m_rep1 = m_rep0;
                m_rep0 = distance;
            }

            // "             1 - Rep Match 3"
            else {
                u32 distance = m_rep3;
                m_rep3 = m_rep2;
                m_rep2 = m_rep1;
                m_rep1 = m_rep0;
                m_rep0 = distance;
            }
        }

        // "In other cases (Rep Match 0/1/2/3), it decodes the zero-based
        //  length of match with "RepLenDecoder" decoder."
        u16 normalized_length = TRY(decode_normalized_match_length(m_rep_length_decoder));

        // "Then it updates the state."
        update_state_after_rep();

        // "Then the decoder must copy match bytes as described in
        //  "The Match symbols copying" section."
        TRY(copy_match_to_buffer(normalized_length + normalized_to_real_match_length_offset));
    }

    if (m_found_end_of_stream_marker || has_reached_expected_data_size()) {
        if (m_options.uncompressed_size.has_value() && m_total_decoded_bytes < m_options.uncompressed_size.value())
            return Error::from_string_literal("Found end-of-stream marker earlier than expected");

        if (!is_range_decoder_in_clean_state())
            return Error::from_string_literal("LZMA stream ends in an unclean state");

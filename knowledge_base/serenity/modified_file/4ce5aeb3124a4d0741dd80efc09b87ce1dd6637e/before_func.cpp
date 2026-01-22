static inline bool is_E(u32 code_point)
{
    return code_point == 0x45;
}

ErrorOr<Vector<Token>> Tokenizer::tokenize(StringView input, StringView encoding)
{
    // https://www.w3.org/TR/css-syntax-3/#css-filter-code-points
    auto filter_code_points = [](StringView input, auto encoding) -> ErrorOr<String> {
        auto decoder = TextCodec::decoder_for(encoding);
        VERIFY(decoder.has_value());

        StringBuilder builder { input.length() };
        bool last_was_carriage_return = false;

        // To filter code points from a stream of (unfiltered) code points input:
        TRY(decoder->process(input, [&builder, &last_was_carriage_return](u32 code_point) -> ErrorOr<void> {
            // Replace any U+000D CARRIAGE RETURN (CR) code points,
            // U+000C FORM FEED (FF) code points,
            // or pairs of U+000D CARRIAGE RETURN (CR) followed by U+000A LINE FEED (LF)
            // in input by a single U+000A LINE FEED (LF) code point.
            if (code_point == '\r') {
                if (last_was_carriage_return) {
                    TRY(builder.try_append('\n'));
                } else {
                    last_was_carriage_return = true;
                }
            } else {
                if (last_was_carriage_return)
                    TRY(builder.try_append('\n'));

                if (code_point == '\n') {
                    if (!last_was_carriage_return)
                        TRY(builder.try_append('\n'));

                } else if (code_point == '\f') {
                    TRY(builder.try_append('\n'));
                    // Replace any U+0000 NULL or surrogate code points in input with U+FFFD REPLACEMENT CHARACTER (�).
                } else if (code_point == 0x00 || (code_point >= 0xD800 && code_point <= 0xDFFF)) {
                    TRY(builder.try_append_code_point(REPLACEMENT_CHARACTER));
                } else {
                    TRY(builder.try_append_code_point(code_point));
                }

                last_was_carriage_return = false;
            }
            return {};
        }));
        return builder.to_string_without_validation();

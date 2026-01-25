// https://www.w3.org/TR/css-syntax-3/#convert-string-to-number
double Tokenizer::convert_a_string_to_a_number(StringView string)
{
    auto code_point_at = [&](size_t index) -> u32 {
        if (index < string.length())
            return string[index];
        return TOKENIZER_EOF;
    };

    // This algorithm does not do any verification to ensure that the string contains only a number.
    // Ensure that the string contains only a valid CSS number before calling this algorithm.

    // Divide the string into seven components, in order from left to right:
    size_t position = 0;

    // 1. A sign: a single U+002B PLUS SIGN (+) or U+002D HYPHEN-MINUS (-), or the empty string.
    //    Let s [sign] be the number -1 if the sign is U+002D HYPHEN-MINUS (-); otherwise, let s be the number 1.
    int sign = 1;
    if (is_plus_sign(code_point_at(position)) || is_hyphen_minus(code_point_at(position))) {
        sign = is_hyphen_minus(code_point_at(position)) ? -1 : 1;
        position++;
    }

    // 2. An integer part: zero or more digits.
    //    If there is at least one digit, let i [integer_part] be the number formed by interpreting the digits
    //    as a base-10 integer; otherwise, let i be the number 0.
    double integer_part = 0;
    while (is_ascii_digit(code_point_at(position))) {
        integer_part = (integer_part * 10) + (code_point_at(position) - '0');
        position++;
    }

    // 3. A decimal point: a single U+002E FULL STOP (.), or the empty string.
    if (is_full_stop(code_point_at(position)))
        position++;

    // 4. A fractional part: zero or more digits.
    //    If there is at least one digit, let f [fractional_part] be the number formed by interpreting the digits
    //    as a base-10 integer and d [fractional_digits] be the number of digits; otherwise, let f and d be the number 0.
    double fractional_part = 0;
    int fractional_digits = 0;
    while (is_ascii_digit(code_point_at(position))) {
        fractional_part = (fractional_part * 10) + (code_point_at(position) - '0');
        position++;
        fractional_digits++;
    }

    // 5. An exponent indicator: a single U+0045 LATIN CAPITAL LETTER E (E) or U+0065 LATIN SMALL LETTER E (e),
    //    or the empty string.
    if (is_e(code_point_at(position)) || is_E(code_point_at(position)))
        position++;

    // 6. An exponent sign: a single U+002B PLUS SIGN (+) or U+002D HYPHEN-MINUS (-), or the empty string.
    //    Let t [exponent_sign] be the number -1 if the sign is U+002D HYPHEN-MINUS (-); otherwise, let t be the number 1.
    int exponent_sign = 1;
    if (is_plus_sign(code_point_at(position)) || is_hyphen_minus(code_point_at(position))) {
        exponent_sign = is_hyphen_minus(code_point_at(position)) ? -1 : 1;
        position++;
    }

    // 7. An exponent: zero or more digits.
    //    If there is at least one digit, let e [exponent] be the number formed by interpreting the digits as a
    //    base-10 integer; otherwise, let e be the number 0.
    double exponent = 0;
    while (is_ascii_digit(code_point_at(position))) {
        exponent = (exponent * 10) + (code_point_at(position) - '0');
        position++;
    }

    // NOTE: We checked before calling this function that the string is a valid number,
    //       so if there is anything at the end, something has gone wrong!
    VERIFY(position == string.length());

    // Return the number s·(i + f·10^-d)·10^te.
    return sign * (integer_part + fractional_part * pow(10, -fractional_digits)) * pow(10, exponent_sign * exponent);
}
        || type == TokenType::StringLiteral
        || type == TokenType::TemplateLiteralEnd
        || type == TokenType::This;
}

Token Lexer::next()
{
    size_t trivia_start = m_position;
    auto in_template = !m_template_states.is_empty();
    bool line_has_token_yet = m_line_column > 1;
    bool unterminated_comment = false;

    if (!in_template || m_template_states.last().in_expr) {
        // consume whitespace and comments
        while (true) {
            if (is_line_terminator()) {
                line_has_token_yet = false;
                do {
                    consume();
                } while (is_line_terminator());
            } else if (is_whitespace()) {
                do {
                    consume();
                } while (is_whitespace());
            } else if (is_line_comment_start(line_has_token_yet)) {
                consume();
                do {
                    consume();
                } while (!is_eof() && !is_line_terminator());
            } else if (is_block_comment_start()) {
                consume();
                do {
                    consume();
                } while (!is_eof() && !is_block_comment_end());
                if (is_eof())
                    unterminated_comment = true;
                consume(); // consume *
                if (is_eof())
                    unterminated_comment = true;
                consume(); // consume /
            } else {
                break;
            }
        }
    }

    size_t value_start = m_position;
    size_t value_start_line_number = m_line_number;
    size_t value_start_column_number = m_line_column;
    auto token_type = TokenType::Invalid;
    auto did_consume_whitespace_or_comments = trivia_start != value_start;
    // This is being used to communicate info about invalid tokens to the parser, which then
    // can turn that into more specific error messages - instead of us having to make up a
    // bunch of Invalid* tokens (bad numeric literals, unterminated comments etc.)
    String token_message;

    Optional<FlyString> identifier;
    size_t identifier_length = 0;

    if (m_current_token.type() == TokenType::RegexLiteral && !is_eof() && is_ascii_alpha(m_current_char) && !did_consume_whitespace_or_comments) {
        token_type = TokenType::RegexFlags;
        while (!is_eof() && is_ascii_alpha(m_current_char))
            consume();
    } else if (m_current_char == '`') {
        consume();

        if (!in_template) {
            token_type = TokenType::TemplateLiteralStart;
            m_template_states.append({ false, 0 });
        } else {
            if (m_template_states.last().in_expr) {
                m_template_states.append({ false, 0 });
                token_type = TokenType::TemplateLiteralStart;
            } else {
                m_template_states.take_last();
                token_type = TokenType::TemplateLiteralEnd;
            }
        }
    } else if (in_template && m_template_states.last().in_expr && m_template_states.last().open_bracket_count == 0 && m_current_char == '}') {
        consume();
        token_type = TokenType::TemplateLiteralExprEnd;
        m_template_states.last().in_expr = false;
    } else if (in_template && !m_template_states.last().in_expr) {
        if (is_eof()) {
            token_type = TokenType::UnterminatedTemplateLiteral;
            m_template_states.take_last();
        } else if (match('$', '{')) {
            token_type = TokenType::TemplateLiteralExprStart;
            consume();
            consume();
            m_template_states.last().in_expr = true;
        } else {
            while (!match('$', '{') && m_current_char != '`' && !is_eof()) {
                if (match('\\', '$') || match('\\', '`'))
                    consume();
                consume();
            }
            if (is_eof() && !m_template_states.is_empty())
                token_type = TokenType::UnterminatedTemplateLiteral;
            else
                token_type = TokenType::TemplateLiteralString;
        }
    } else if (auto code_point = is_identifier_start(identifier_length); code_point.has_value()) {
        bool has_escaped_character = false;
        // identifier or keyword
        StringBuilder builder;
        do {
            builder.append_code_point(*code_point);
            for (size_t i = 0; i < identifier_length; ++i)
                consume();

            has_escaped_character |= identifier_length > 1;

            code_point = is_identifier_middle(identifier_length);
        } while (code_point.has_value());

        identifier = builder.build();
        m_parsed_identifiers->identifiers.set(*identifier);

        auto it = s_keywords.find(identifier->hash(), [&](auto& entry) { return entry.key == identifier; });
        if (it == s_keywords.end())
            token_type = TokenType::Identifier;
        else
            token_type = has_escaped_character ? TokenType::EscapedKeyword : it->value;
    } else if (is_numeric_literal_start()) {
        token_type = TokenType::NumericLiteral;
        bool is_invalid_numeric_literal = false;
        if (m_current_char == '0') {
            consume();
            if (m_current_char == '.') {
                // decimal
                consume();
                while (is_ascii_digit(m_current_char) || match_numeric_literal_separator_followed_by(is_ascii_digit))
                    consume();
                if (m_current_char == 'e' || m_current_char == 'E')
                    is_invalid_numeric_literal = !consume_exponent();
            } else if (m_current_char == 'e' || m_current_char == 'E') {
                is_invalid_numeric_literal = !consume_exponent();
            } else if (m_current_char == 'o' || m_current_char == 'O') {
                // octal
                is_invalid_numeric_literal = !consume_octal_number();
                if (m_current_char == 'n') {
                    consume();
                    token_type = TokenType::BigIntLiteral;
                }
            } else if (m_current_char == 'b' || m_current_char == 'B') {
                // binary
                is_invalid_numeric_literal = !consume_binary_number();
                if (m_current_char == 'n') {
                    consume();
                    token_type = TokenType::BigIntLiteral;
                }
            } else if (m_current_char == 'x' || m_current_char == 'X') {
                // hexadecimal
                is_invalid_numeric_literal = !consume_hexadecimal_number();
                if (m_current_char == 'n') {
                    consume();
                    token_type = TokenType::BigIntLiteral;
                }
            } else if (m_current_char == 'n') {
                consume();
                token_type = TokenType::BigIntLiteral;
            } else if (is_ascii_digit(m_current_char)) {
                // octal without '0o' prefix. Forbidden in 'strict mode'
                do {
                    consume();
                } while (is_ascii_digit(m_current_char) || match_numeric_literal_separator_followed_by(is_ascii_digit));
            }
        } else {
            // 1...9 or period
            while (is_ascii_digit(m_current_char) || match_numeric_literal_separator_followed_by(is_ascii_digit))
                consume();
            if (m_current_char == 'n') {
                consume();
                token_type = TokenType::BigIntLiteral;
            } else {
                if (m_current_char == '.') {
                    consume();
                    while (is_ascii_digit(m_current_char) || match_numeric_literal_separator_followed_by(is_ascii_digit))
                        consume();
                }
                if (m_current_char == 'e' || m_current_char == 'E')
                    is_invalid_numeric_literal = !consume_exponent();
            }
        }
        if (is_invalid_numeric_literal) {
            token_type = TokenType::Invalid;
            token_message = "Invalid numeric literal";
        }
    } else if (m_current_char == '"' || m_current_char == '\'') {
        char stop_char = m_current_char;
        consume();
        // Note: LS/PS line terminators are allowed in string literals.
        while (m_current_char != stop_char && m_current_char != '\r' && m_current_char != '\n' && !is_eof()) {
            if (m_current_char == '\\') {
                consume();
                if (m_current_char == '\r' && m_position < m_source.length() && m_source[m_position] == '\n') {
                    consume();
                }
            }
            consume();
        }
        if (m_current_char != stop_char) {
            token_type = TokenType::UnterminatedStringLiteral;
        } else {
            consume();
            token_type = TokenType::StringLiteral;
        }
    } else if (m_current_char == '/' && !slash_means_division()) {
        consume();
        token_type = consume_regex_literal();
    } else if (m_eof) {
        if (unterminated_comment) {
            token_type = TokenType::Invalid;
            token_message = "Unterminated multi-line comment";
        } else {
            token_type = TokenType::Eof;
        }
    } else {
        // There is only one four-char operator: >>>=
        bool found_four_char_token = false;
        if (match('>', '>', '>', '=')) {
            found_four_char_token = true;
            consume();
            consume();
            consume();
            consume();
            token_type = TokenType::UnsignedShiftRightEquals;
        }

        bool found_three_char_token = false;
        if (!found_four_char_token && m_position + 1 < m_source.length()) {
            auto three_chars_view = m_source.substring_view(m_position - 1, 3);
            auto it = s_three_char_tokens.find(three_chars_view.hash(), [&](auto& entry) { return entry.key == three_chars_view; });
            if (it != s_three_char_tokens.end()) {
                found_three_char_token = true;
                consume();
                consume();
                consume();
                token_type = it->value;
            }
        }

        bool found_two_char_token = false;
        if (!found_four_char_token && !found_three_char_token && m_position < m_source.length()) {
            auto two_chars_view = m_source.substring_view(m_position - 1, 2);
            auto it = s_two_char_tokens.find(two_chars_view.hash(), [&](auto& entry) { return entry.key == two_chars_view; });
            if (it != s_two_char_tokens.end()) {
                // OptionalChainingPunctuator :: ?. [lookahead ∉ DecimalDigit]
                if (!(it->value == TokenType::QuestionMarkPeriod && m_position + 1 < m_source.length() && is_ascii_digit(m_source[m_position + 1]))) {
                    found_two_char_token = true;
                    consume();
                    consume();
                    token_type = it->value;
                }
            }
        }

        bool found_one_char_token = false;
        if (!found_four_char_token && !found_three_char_token && !found_two_char_token) {
            auto it = s_single_char_tokens.find(m_current_char);
            if (it != s_single_char_tokens.end()) {
                found_one_char_token = true;
                consume();
                token_type = it->value;
            }
        }

        if (!found_four_char_token && !found_three_char_token && !found_two_char_token && !found_one_char_token) {
            consume();
            token_type = TokenType::Invalid;
        }
    }

    if (!m_template_states.is_empty() && m_template_states.last().in_expr) {
        if (token_type == TokenType::CurlyOpen) {
            m_template_states.last().open_bracket_count++;
        } else if (token_type == TokenType::CurlyClose) {
            m_template_states.last().open_bracket_count--;
        }
    }

    m_current_token = Token(
        token_type,
        token_message,
        m_source.substring_view(trivia_start - 1, value_start - trivia_start),
        m_source.substring_view(value_start - 1, m_position - value_start),
        m_filename,
        value_start_line_number,
        value_start_column_number,
        m_position);

    if (identifier.has_value())
        m_current_token.set_identifier_value(identifier.release_value());

    if constexpr (LEXER_DEBUG) {
        dbgln("------------------------------");
        dbgln("Token: {}", m_current_token.name());
        dbgln("Trivia: _{}_", m_current_token.trivia());
        dbgln("Value: _{}_", m_current_token.value());
        dbgln("Line: {}, Column: {}", m_current_token.line_number(), m_current_token.line_column());

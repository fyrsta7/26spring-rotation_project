}

bool Lexer::is_numeric_literal_start() const
{
    return isdigit(m_current_char) || (m_current_char == '.' && m_position < m_source.length() && isdigit(m_source[m_position]));
}

bool Lexer::slash_means_division() const
{
    auto type = m_current_token.type();
    return type == TokenType::BigIntLiteral
        || type == TokenType::BoolLiteral
        || type == TokenType::BracketClose
        || type == TokenType::CurlyClose
        || type == TokenType::Identifier
        || type == TokenType::NullLiteral
        || type == TokenType::NumericLiteral
        || type == TokenType::ParenClose
        || type == TokenType::RegexLiteral
        || type == TokenType::StringLiteral
        || type == TokenType::TemplateLiteralEnd
        || type == TokenType::This;
}

Token Lexer::next()
{
    size_t trivia_start = m_position;
    auto in_template = !m_template_states.is_empty();

    if (!in_template || m_template_states.last().in_expr) {
        // consume whitespace and comments
        while (true) {
            if (isspace(m_current_char)) {
                do {
                    consume();
                } while (isspace(m_current_char));
            } else if (is_line_comment_start()) {
                consume();
                do {
                    consume();
                } while (!is_eof() && m_current_char != '\n');
            } else if (is_block_comment_start()) {
                consume();
                do {
                    consume();
                } while (!is_eof() && !is_block_comment_end());
                consume(); // consume *
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

    if (m_current_token.type() == TokenType::RegexLiteral && !is_eof() && isalpha(m_current_char)) {
        token_type = TokenType::RegexFlags;
        while (!is_eof() && isalpha(m_current_char))
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

            token_type = TokenType::TemplateLiteralString;
        }
    } else if (is_identifier_start()) {
        // identifier or keyword
        do {
            consume();
        } while (is_identifier_middle());

        StringView value = m_source.substring_view(value_start - 1, m_position - value_start);
        auto it = s_keywords.find(value);
        if (it == s_keywords.end()) {
            token_type = TokenType::Identifier;
        } else {
            token_type = it->value;
        }
    } else if (is_numeric_literal_start()) {
        token_type = TokenType::NumericLiteral;
        if (m_current_char == '0') {
            consume();
            if (m_current_char == '.') {
                // decimal
                consume();
                while (isdigit(m_current_char))
                    consume();
                if (m_current_char == 'e' || m_current_char == 'E')
                    consume_exponent();
            } else if (m_current_char == 'e' || m_current_char == 'E') {
                consume_exponent();
            } else if (m_current_char == 'o' || m_current_char == 'O') {
                // octal
                consume();
                while (m_current_char >= '0' && m_current_char <= '7')
                    consume();
            } else if (m_current_char == 'b' || m_current_char == 'B') {
                // binary
                consume();
                while (m_current_char == '0' || m_current_char == '1')
                    consume();
            } else if (m_current_char == 'x' || m_current_char == 'X') {
                // hexadecimal
                consume();
                while (isxdigit(m_current_char))
                    consume();
            } else if (m_current_char == 'n') {
                consume();
                token_type = TokenType::BigIntLiteral;
            } else if (isdigit(m_current_char)) {
                // octal without 'O' prefix. Forbidden in 'strict mode'
                // FIXME: We need to make sure this produces a syntax error when in strict mode
                do {
                    consume();
                } while (isdigit(m_current_char));
            }
        } else {
            // 1...9 or period
            while (isdigit(m_current_char))
                consume();
            if (m_current_char == 'n') {
                consume();
                token_type = TokenType::BigIntLiteral;
            } else {
                if (m_current_char == '.') {
                    consume();
                    while (isdigit(m_current_char))
                        consume();
                }
                if (m_current_char == 'e' || m_current_char == 'E')
                    consume_exponent();
            }
        }
    } else if (m_current_char == '"' || m_current_char == '\'') {
        char stop_char = m_current_char;
        consume();
        while (m_current_char != stop_char && m_current_char != '\n' && !is_eof()) {
            if (m_current_char == '\\') {
                consume();
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
        token_type = TokenType::RegexLiteral;

        while (!is_eof()) {
            if (m_current_char == '[') {
                m_regex_is_in_character_class = true;
            } else if (m_current_char == ']') {
                m_regex_is_in_character_class = false;
            } else if (!m_regex_is_in_character_class && m_current_char == '/') {
                break;
            }

            if (match('\\', '/') || match('\\', '[') || match('\\', '\\') || (m_regex_is_in_character_class && match('\\', ']')))
                consume();
            consume();
        }

        if (is_eof()) {
            token_type = TokenType::UnterminatedRegexLiteral;
        } else {
            consume();
        }
    } else if (m_current_char == EOF) {
        token_type = TokenType::Eof;
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
            char second_char = m_source[m_position];
            char third_char = m_source[m_position + 1];
            char three_chars[] { (char)m_current_char, second_char, third_char, 0 };
            auto it = s_three_char_tokens.find(three_chars);
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
            char second_char = m_source[m_position];
            char two_chars[] { (char)m_current_char, second_char, 0 };
            auto it = s_two_char_tokens.find(two_chars);
            if (it != s_two_char_tokens.end()) {
                found_two_char_token = true;
                consume();
                consume();
                token_type = it->value;
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

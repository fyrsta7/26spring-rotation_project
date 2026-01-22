V8_INLINE Token::Value Scanner::ScanSingleToken() {
  Token::Value token;
  do {
    next().location.beg_pos = source_pos();

    if (V8_LIKELY(static_cast<unsigned>(c0_) <= kMaxAscii)) {
      token = one_char_tokens[c0_];

      switch (token) {
        case Token::LPAREN:
        case Token::RPAREN:
        case Token::LBRACE:
        case Token::RBRACE:
        case Token::LBRACK:
        case Token::RBRACK:
        case Token::COLON:
        case Token::SEMICOLON:
        case Token::COMMA:
        case Token::BIT_NOT:
        case Token::ILLEGAL:
          // One character tokens.
          return Select(token);

        case Token::CONDITIONAL:
          // ? ?. ??
          Advance();
          if (V8_UNLIKELY(allow_harmony_optional_chaining() && c0_ == '.')) {
            Advance();
            if (!IsDecimalDigit(c0_)) return Token::QUESTION_PERIOD;
            PushBack('.');
          } else if (V8_UNLIKELY(allow_harmony_nullish() && c0_ == '?')) {
            return Select(Token::NULLISH);
          }
          return Token::CONDITIONAL;

        case Token::STRING:
          return ScanString();

        case Token::LT:
          // < <= << <<= <!--
          Advance();
          if (c0_ == '=') return Select(Token::LTE);
          if (c0_ == '<') return Select('=', Token::ASSIGN_SHL, Token::SHL);
          if (c0_ == '!') {
            token = ScanHtmlComment();
            continue;
          }
          return Token::LT;

        case Token::GT:
          // > >= >> >>= >>> >>>=
          Advance();
          if (c0_ == '=') return Select(Token::GTE);
          if (c0_ == '>') {
            // >> >>= >>> >>>=
            Advance();
            if (c0_ == '=') return Select(Token::ASSIGN_SAR);
            if (c0_ == '>') return Select('=', Token::ASSIGN_SHR, Token::SHR);
            return Token::SAR;
          }
          return Token::GT;

        case Token::ASSIGN:
          // = == === =>
          Advance();
          if (c0_ == '=') return Select('=', Token::EQ_STRICT, Token::EQ);
          if (c0_ == '>') return Select(Token::ARROW);
          return Token::ASSIGN;

        case Token::NOT:
          // ! != !==
          Advance();
          if (c0_ == '=') return Select('=', Token::NE_STRICT, Token::NE);
          return Token::NOT;

        case Token::ADD:
          // + ++ +=
          Advance();
          if (c0_ == '+') return Select(Token::INC);
          if (c0_ == '=') return Select(Token::ASSIGN_ADD);
          return Token::ADD;

        case Token::SUB:
          // - -- --> -=
          Advance();
          if (c0_ == '-') {
            Advance();
            if (c0_ == '>' && next().after_line_terminator) {
              // For compatibility with SpiderMonkey, we skip lines that
              // start with an HTML comment end '-->'.
              token = SkipSingleHTMLComment();
              continue;
            }
            return Token::DEC;
          }
          if (c0_ == '=') return Select(Token::ASSIGN_SUB);
          return Token::SUB;

        case Token::MUL:
          // * *=
          Advance();
          if (c0_ == '*') return Select('=', Token::ASSIGN_EXP, Token::EXP);
          if (c0_ == '=') return Select(Token::ASSIGN_MUL);
          return Token::MUL;

        case Token::MOD:
          // % %=
          return Select('=', Token::ASSIGN_MOD, Token::MOD);

        case Token::DIV:
          // /  // /* /=
          Advance();
          if (c0_ == '/') {
            uc32 c = Peek();
            if (c == '#' || c == '@') {
              Advance();
              Advance();
              token = SkipSourceURLComment();
              continue;
            }
            token = SkipSingleLineComment();
            continue;
          }
          if (c0_ == '*') {
            token = SkipMultiLineComment();
            continue;
          }
          if (c0_ == '=') return Select(Token::ASSIGN_DIV);
          return Token::DIV;

        case Token::BIT_AND:
          // & && &=
          Advance();
          if (c0_ == '&') return Select(Token::AND);
          if (c0_ == '=') return Select(Token::ASSIGN_BIT_AND);
          return Token::BIT_AND;

        case Token::BIT_OR:
          // | || |=
          Advance();
          if (c0_ == '|') return Select(Token::OR);
          if (c0_ == '=') return Select(Token::ASSIGN_BIT_OR);
          return Token::BIT_OR;

        case Token::BIT_XOR:
          // ^ ^=
          return Select('=', Token::ASSIGN_BIT_XOR, Token::BIT_XOR);

        case Token::PERIOD:
          // . Number
          Advance();
          if (IsDecimalDigit(c0_)) return ScanNumber(true);
          if (c0_ == '.') {
            if (Peek() == '.') {
              Advance();
              Advance();
              return Token::ELLIPSIS;
            }
          }
          return Token::PERIOD;

        case Token::TEMPLATE_SPAN:
          Advance();
          return ScanTemplateSpan();

        case Token::PRIVATE_NAME:
          return ScanPrivateName();

        case Token::WHITESPACE:
          token = SkipWhiteSpace();
          continue;

        case Token::NUMBER:
          return ScanNumber(false);

        case Token::IDENTIFIER:
          return ScanIdentifierOrKeyword();

        default:
          UNREACHABLE();
      }
    }

    if (IsIdentifierStart(c0_) ||
        (CombineSurrogatePair() && IsIdentifierStart(c0_))) {
      return ScanIdentifierOrKeyword();
    }
    if (c0_ == kEndOfInput) {
      return source_->has_parser_error() ? Token::ILLEGAL : Token::EOS;
    }
    token = SkipWhiteSpace();

    // Continue scanning for tokens as long as we're just skipping whitespace.
  } while (token == Token::WHITESPACE);

  return token;
}

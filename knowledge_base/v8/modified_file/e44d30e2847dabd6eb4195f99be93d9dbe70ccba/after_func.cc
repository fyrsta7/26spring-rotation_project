Expression* Parser::ParseBinaryExpression(int prec, bool accept_IN, bool* ok) {
  ASSERT(prec >= 4);
  Expression* x = ParseUnaryExpression(CHECK_OK);
  for (int prec1 = Precedence(peek(), accept_IN); prec1 >= prec; prec1--) {
    // prec1 >= 4
    while (Precedence(peek(), accept_IN) == prec1) {
      Token::Value op = Next();
      Expression* y = ParseBinaryExpression(prec1 + 1, accept_IN, CHECK_OK);

      // Compute some expressions involving only number literals.
      if (x && x->AsLiteral() && x->AsLiteral()->handle()->IsNumber() &&
          y && y->AsLiteral() && y->AsLiteral()->handle()->IsNumber()) {
        double x_val = x->AsLiteral()->handle()->Number();
        double y_val = y->AsLiteral()->handle()->Number();

        switch (op) {
          case Token::ADD:
            x = NewNumberLiteral(x_val + y_val);
            continue;
          case Token::SUB:
            x = NewNumberLiteral(x_val - y_val);
            continue;
          case Token::MUL:
            x = NewNumberLiteral(x_val * y_val);
            continue;
          case Token::DIV:
            x = NewNumberLiteral(x_val / y_val);
            continue;
          case Token::BIT_OR:
            x = NewNumberLiteral(DoubleToInt32(x_val) | DoubleToInt32(y_val));
            continue;
          case Token::BIT_AND:
            x = NewNumberLiteral(DoubleToInt32(x_val) & DoubleToInt32(y_val));
            continue;
          case Token::BIT_XOR:
            x = NewNumberLiteral(DoubleToInt32(x_val) ^ DoubleToInt32(y_val));
            continue;
          case Token::SHL: {
            int value = DoubleToInt32(x_val) << (DoubleToInt32(y_val) & 0x1f);
            x = NewNumberLiteral(value);
            continue;
          }
          case Token::SHR: {
            uint32_t shift = DoubleToInt32(y_val) & 0x1f;
            uint32_t value = DoubleToUint32(x_val) >> shift;
            x = NewNumberLiteral(value);
            continue;
          }
          case Token::SAR: {
            uint32_t shift = DoubleToInt32(y_val) & 0x1f;
            int value = ArithmeticShiftRight(DoubleToInt32(x_val), shift);
            x = NewNumberLiteral(value);
            continue;
          }
          default:
            break;
        }
      }

      // Convert constant divisions to multiplications for speed.
      if (op == Token::DIV &&
          y && y->AsLiteral() && y->AsLiteral()->handle()->IsNumber()) {
        double y_val = y->AsLiteral()->handle()->Number();
        int64_t y_int = static_cast<int64_t>(y_val);
        // There are rounding issues with this optimization, but they don't
        // apply if the number to be divided with has a reciprocal that can
        // be precisely represented as a floating point number.  This is
        // the case if the number is an integer power of 2.
        if (static_cast<double>(y_int) == y_val && IsPowerOf2(y_int)) {
          y = NewNumberLiteral(1 / y_val);
          op = Token::MUL;
        }
      }

      // For now we distinguish between comparisons and other binary
      // operations.  (We could combine the two and get rid of this
      // code an AST node eventually.)
      if (Token::IsCompareOp(op)) {
        // We have a comparison.
        Token::Value cmp = op;
        switch (op) {
          case Token::NE: cmp = Token::EQ; break;
          case Token::NE_STRICT: cmp = Token::EQ_STRICT; break;
          default: break;
        }
        x = NEW(CompareOperation(cmp, x, y));
        if (cmp != op) {
          // The comparison was negated - add a NOT.
          x = NEW(UnaryOperation(Token::NOT, x));
        }

      } else {
        // We have a "normal" binary operation.
        x = NEW(BinaryOperation(op, x, y));
      }
    }
  }
  return x;
}

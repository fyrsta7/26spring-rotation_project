Expression* Parser::ParseObjectLiteral(bool* ok) {
  // ObjectLiteral ::
  //   '{' (
  //       ((IdentifierName | String | Number) ':' AssignmentExpression)
  //     | (('get' | 'set') (IdentifierName | String | Number) FunctionLiteral)
  //    )*[','] '}'

  ZoneList<ObjectLiteral::Property*>* properties =
      new(zone()) ZoneList<ObjectLiteral::Property*>(4);
  int number_of_boilerplate_properties = 0;
  bool has_function = false;

  ObjectLiteralPropertyChecker checker(this, top_scope_->language_mode());

  Expect(Token::LBRACE, CHECK_OK);

  while (peek() != Token::RBRACE) {
    if (fni_ != NULL) fni_->Enter();

    Literal* key = NULL;
    Token::Value next = peek();

    // Location of the property name token
    Scanner::Location loc = scanner().peek_location();

    switch (next) {
      case Token::FUTURE_RESERVED_WORD:
      case Token::FUTURE_STRICT_RESERVED_WORD:
      case Token::IDENTIFIER: {
        bool is_getter = false;
        bool is_setter = false;
        Handle<String> id =
            ParseIdentifierNameOrGetOrSet(&is_getter, &is_setter, CHECK_OK);
        if (fni_ != NULL) fni_->PushLiteralName(id);

        if ((is_getter || is_setter) && peek() != Token::COLON) {
            // Update loc to point to the identifier
            loc = scanner().peek_location();
            ObjectLiteral::Property* property =
                ParseObjectLiteralGetSet(is_getter, CHECK_OK);
            if (IsBoilerplateProperty(property)) {
              number_of_boilerplate_properties++;
            }
            // Validate the property.
            checker.CheckProperty(property, loc, CHECK_OK);
            properties->Add(property);
            if (peek() != Token::RBRACE) Expect(Token::COMMA, CHECK_OK);

            if (fni_ != NULL) {
              fni_->Infer();
              fni_->Leave();
            }
            continue;  // restart the while
        }
        // Failed to parse as get/set property, so it's just a property
        // called "get" or "set".
        key = NewLiteral(id);
        break;
      }
      case Token::STRING: {
        Consume(Token::STRING);
        Handle<String> string = GetSymbol(CHECK_OK);
        if (fni_ != NULL) fni_->PushLiteralName(string);
        uint32_t index;
        if (!string.is_null() && string->AsArrayIndex(&index)) {
          key = NewNumberLiteral(index);
          break;
        }
        key = NewLiteral(string);
        break;
      }
      case Token::NUMBER: {
        Consume(Token::NUMBER);
        ASSERT(scanner().is_literal_ascii());
        double value = StringToDouble(isolate()->unicode_cache(),
                                      scanner().literal_ascii_string(),
                                      ALLOW_HEX | ALLOW_OCTALS);
        key = NewNumberLiteral(value);
        break;
      }
      default:
        if (Token::IsKeyword(next)) {
          Consume(next);
          Handle<String> string = GetSymbol(CHECK_OK);
          key = NewLiteral(string);
        } else {
          // Unexpected token.
          Token::Value next = Next();
          ReportUnexpectedToken(next);
          *ok = false;
          return NULL;
        }
    }

    Expect(Token::COLON, CHECK_OK);
    Expression* value = ParseAssignmentExpression(true, CHECK_OK);

    ObjectLiteral::Property* property =
        new(zone()) ObjectLiteral::Property(key, value);

    // Mark top-level object literals that contain function literals and
    // pretenure the literal so it can be added as a constant function
    // property.
    if (top_scope_->DeclarationScope()->is_global_scope() &&
        value->AsFunctionLiteral() != NULL) {
      has_function = true;
      value->AsFunctionLiteral()->set_pretenure();
    }

    // Count CONSTANT or COMPUTED properties to maintain the enumeration order.
    if (IsBoilerplateProperty(property)) number_of_boilerplate_properties++;
    // Validate the property
    checker.CheckProperty(property, loc, CHECK_OK);
    properties->Add(property);

    // TODO(1240767): Consider allowing trailing comma.
    if (peek() != Token::RBRACE) Expect(Token::COMMA, CHECK_OK);

    if (fni_ != NULL) {
      fni_->Infer();
      fni_->Leave();
    }
  }
  Expect(Token::RBRACE, CHECK_OK);

  // Computation of literal_index must happen before pre parse bailout.
  int literal_index = current_function_state_->NextMaterializedLiteralIndex();

  Handle<FixedArray> constant_properties = isolate()->factory()->NewFixedArray(
      number_of_boilerplate_properties * 2, TENURED);

  bool is_simple = true;
  bool fast_elements = true;
  int depth = 1;
  BuildObjectLiteralConstantProperties(properties,
                                       constant_properties,
                                       &is_simple,
                                       &fast_elements,
                                       &depth);
  return new(zone()) ObjectLiteral(isolate(),
                                   constant_properties,
                                   properties,
                                   literal_index,
                                   is_simple,
                                   fast_elements,
                                   depth,
                                   has_function);
}

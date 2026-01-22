    return all_of(parameters, [](FunctionNode::Parameter const& parameter) {
        return !parameter.is_rest && parameter.default_value.is_null() && parameter.binding.has<FlyString>();
    });
}

RefPtr<FunctionExpression> Parser::try_parse_arrow_function_expression(bool expect_parens)
{
    if (!expect_parens) {
        // NOTE: This is a fast path where we try to fail early in case this can't possibly
        //       be a match. The idea is to avoid the expensive parser state save/load mechanism.
        //       The logic is duplicated below in the "real" !expect_parens branch.
        if (!match_identifier() && !match(TokenType::Yield) && !match(TokenType::Await))
            return nullptr;
        auto forked_lexer = m_state.lexer;
        auto token = forked_lexer.next();
        if (token.trivia_contains_line_terminator())
            return nullptr;
        if (token.type() != TokenType::Arrow)
            return nullptr;
    }

    save_state();
    auto rule_start = push_start();

    ArmedScopeGuard state_rollback_guard = [&] {
        load_state();
    };

    Vector<FunctionNode::Parameter> parameters;
    i32 function_length = -1;
    if (expect_parens) {
        // We have parens around the function parameters and can re-use the same parsing
        // logic used for regular functions: multiple parameters, default values, rest
        // parameter, maybe a trailing comma. If we have a new syntax error afterwards we
        // check if it's about a wrong token (something like duplicate parameter name must
        // not abort), know parsing failed and rollback the parser state.
        auto previous_syntax_errors = m_state.errors.size();
        parameters = parse_formal_parameters(function_length, FunctionNodeParseOptions::IsArrowFunction);
        if (m_state.errors.size() > previous_syntax_errors && m_state.errors[previous_syntax_errors].message.starts_with("Unexpected token"))
            return nullptr;
        if (!match(TokenType::ParenClose))
            return nullptr;
        consume();
    } else {
        // No parens - this must be an identifier followed by arrow. That's it.
        if (!match_identifier() && !match(TokenType::Yield) && !match(TokenType::Await))
            return nullptr;
        auto token = consume_identifier_reference();
        if (m_state.strict_mode && token.value().is_one_of("arguments"sv, "eval"sv))
            syntax_error("BindingIdentifier may not be 'arguments' or 'eval' in strict mode");
        parameters.append({ FlyString { token.value() }, {} });
    }
    // If there's a newline between the closing paren and arrow it's not a valid arrow function,
    // ASI should kick in instead (it'll then fail with "Unexpected token Arrow")
    if (m_state.current_token.trivia_contains_line_terminator())
        return nullptr;
    if (!match(TokenType::Arrow))
        return nullptr;
    consume();

    if (function_length == -1)
        function_length = parameters.size();

    m_state.function_parameters.append(parameters);

    auto old_labels_in_scope = move(m_state.labels_in_scope);
    ScopeGuard guard([&]() {
        m_state.labels_in_scope = move(old_labels_in_scope);
    });

    bool is_strict = false;

    auto function_body_result = [&]() -> RefPtr<BlockStatement> {
        TemporaryChange change(m_state.in_arrow_function_context, true);
        if (match(TokenType::CurlyOpen)) {
            // Parse a function body with statements
            ScopePusher scope(*this, ScopePusher::Var, Scope::Function);

            auto body = parse_block_statement(is_strict, !is_simple_parameter_list(parameters));
            scope.add_to_scope_node(body);
            return body;
        }
        if (match_expression()) {
            // Parse a function body which returns a single expression

            // FIXME: We synthesize a block with a return statement
            // for arrow function bodies which are a single expression.
            // Esprima generates a single "ArrowFunctionExpression"
            // with a "body" property.
            auto return_expression = parse_expression(2);
            auto return_block = create_ast_node<BlockStatement>({ m_state.current_token.filename(), rule_start.position(), position() });
            return_block->append<ReturnStatement>({ m_filename, rule_start.position(), position() }, move(return_expression));
            return return_block;
        }
        // Invalid arrow function body
        return nullptr;
    }();

    m_state.function_parameters.take_last();

    if (function_body_result.is_null())
        return nullptr;

    state_rollback_guard.disarm();
    discard_saved_state();
    auto body = function_body_result.release_nonnull();

    if (is_strict) {
        for (auto& parameter : parameters) {
            parameter.binding.visit(
                [&](FlyString const& name) {
                    check_identifier_name_for_assignment_validity(name, true);
                },
                [&](auto const&) {});
        }
    }

    return create_ast_node<FunctionExpression>(
        { m_state.current_token.filename(), rule_start.position(), position() }, "", move(body),
        move(parameters), function_length, FunctionKind::Regular, is_strict, true);
}

RefPtr<Statement> Parser::try_parse_labelled_statement(AllowLabelledFunction allow_function)
{
    save_state();
    auto rule_start = push_start();
    ArmedScopeGuard state_rollback_guard = [&] {
        load_state();
    };

    if (m_state.current_token.value() == "yield"sv && (m_state.strict_mode || m_state.in_generator_function_context)) {
        syntax_error("'yield' label not allowed in this context");
        return {};
    }

    auto identifier = consume_identifier_reference().value();
    if (!match(TokenType::Colon))
        return {};
    consume(TokenType::Colon);

    if (!match_statement())
        return {};

    if (match(TokenType::Function) && (allow_function == AllowLabelledFunction::No || m_state.strict_mode)) {
        syntax_error("Not allowed to declare a function here");
        return {};
    }

    if (m_state.labels_in_scope.contains(identifier))
        syntax_error(String::formatted("Label '{}' has already been declared", identifier));

    RefPtr<Statement> labelled_statement;

    if (match(TokenType::Function)) {
        m_state.labels_in_scope.set(identifier, false);
        auto function_declaration = parse_function_node<FunctionDeclaration>();
        m_state.current_scope->function_declarations.append(function_declaration);
        auto hoisting_target = m_state.current_scope->get_current_function_scope();
        hoisting_target->hoisted_function_declarations.append({ function_declaration, *m_state.current_scope });
        if (function_declaration->kind() == FunctionKind::Generator)
            syntax_error("Generator functions cannot be defined in labelled statements");

        labelled_statement = move(function_declaration);
    } else {
        auto is_iteration_statement = match(TokenType::For) || match(TokenType::Do) || match(TokenType::While);
        m_state.labels_in_scope.set(identifier, is_iteration_statement);
        labelled_statement = parse_statement();
    }

    m_state.labels_in_scope.remove(identifier);


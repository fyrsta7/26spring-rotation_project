
        m_scope_stack.last().variables.set(move(name), { js_undefined(), declaration_kind });
        break;
    }
}

void Interpreter::set_variable(const FlyString& name, Value value, bool first_assignment)
{
    for (ssize_t i = m_scope_stack.size() - 1; i >= 0; --i) {
        auto& scope = m_scope_stack.at(i);

        auto possible_match = scope.variables.get(name);
        if (possible_match.has_value()) {
            if (!first_assignment && possible_match.value().declaration_kind == DeclarationKind::Const)
                ASSERT_NOT_REACHED();

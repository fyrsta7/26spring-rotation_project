    lhs().dump(indent + 1);
    rhs().dump(indent + 1);
    body().dump(indent + 1);
}

Value Identifier::execute(Interpreter& interpreter, GlobalObject& global_object) const
{
    InterpreterNodeScope node_scope { interpreter, *this };

    if (m_argument_index.has_value())
        return interpreter.vm().argument(m_argument_index.value());

    auto value = interpreter.vm().get_variable(string(), global_object);
    if (value.is_empty()) {
        if (!interpreter.exception())

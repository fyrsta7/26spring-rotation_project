    }
    VERIFY_NOT_REACHED();
}

ThrowCompletionOr<void> Call::execute_impl(Bytecode::Interpreter& interpreter) const
{
    auto& vm = interpreter.vm();
    auto callee = interpreter.reg(m_callee);

    TRY(throw_if_needed_for_call(interpreter, callee, call_type(), expression_string()));

    if (m_builtin.has_value() && m_argument_count == Bytecode::builtin_argument_count(m_builtin.value()) && interpreter.realm().get_builtin_value(m_builtin.value()) == callee) {
        interpreter.accumulator() = TRY(dispatch_builtin_call(interpreter, m_builtin.value(), m_first_argument));
        return {};
    }

    MarkedVector<Value> argument_values(vm.heap());
    argument_values.ensure_capacity(m_argument_count);
    for (u32 i = 0; i < m_argument_count; ++i) {
        argument_values.unchecked_append(interpreter.reg(Register { m_first_argument.index() + i }));

    interpreter.do_return(interpreter.accumulator().value_or(js_undefined()));
    return {};
}

ThrowCompletionOr<void> Increment::execute_impl(Bytecode::Interpreter& interpreter) const
{
    auto& vm = interpreter.vm();
    auto old_value = TRY(interpreter.accumulator().to_numeric(vm));

    if (old_value.is_number())
        interpreter.accumulator() = Value(old_value.as_double() + 1);

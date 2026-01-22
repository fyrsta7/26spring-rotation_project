    interpreter.do_return(object);
    return {};
}

ThrowCompletionOr<void> GetByValue::execute_impl(Bytecode::Interpreter& interpreter) const
{
    auto& vm = interpreter.vm();

    // NOTE: Get the property key from the accumulator before side effects have a chance to overwrite it.
    auto property_key_value = interpreter.accumulator();

    auto base_value = interpreter.reg(m_base);
    auto object = TRY(base_object_for_get(interpreter, base_value));
    auto property_key = TRY(property_key_value.to_property_key(vm));

    if (base_value.is_string()) {
        auto string_value = TRY(base_value.as_string().get(vm, property_key));
        if (string_value.has_value()) {
            interpreter.accumulator() = *string_value;
            return {};
        }
    }

    // 5. Return unused.
    return {};
}

// 9.1.1.2.6 GetBindingValue ( N, S ), https://tc39.es/ecma262/#sec-object-environment-records-getbindingvalue-n-s
ThrowCompletionOr<Value> ObjectEnvironment::get_binding_value(VM&, FlyString const& name, bool strict)
{
    auto& vm = this->vm();

    // 1. Let bindingObject be envRec.[[BindingObject]].
    // 2. Let value be ? HasProperty(bindingObject, N).
    auto value = TRY(m_binding_object.has_property(name));

    // 3. If value is false, then
    if (!value) {
        // a. If S is false, return undefined; otherwise throw a ReferenceError exception.
        if (!strict)
            return js_undefined();
        return vm.throw_completion<ReferenceError>(ErrorType::UnknownIdentifier, name);

    // 5. Return unused.
    return {};
}

// 9.1.1.2.6 GetBindingValue ( N, S ), https://tc39.es/ecma262/#sec-object-environment-records-getbindingvalue-n-s
ThrowCompletionOr<Value> ObjectEnvironment::get_binding_value(VM&, FlyString const& name, bool strict)
{
    auto& vm = this->vm();

    // OPTIMIZATION: For non-with environments in non-strict mode, we don't need the separate HasProperty check
    //               since Get will return undefined for missing properties anyway. So we take advantage of this
    //               to avoid doing both HasProperty and Get.
    //               We can't do this for with environments, since it would be observable (e.g via a Proxy)
    // FIXME: We could combine HasProperty and Get in non-strict mode if Get would return a bit more failure information.
    if (!m_with_environment && !strict)
        return m_binding_object.get(name);

    // 1. Let bindingObject be envRec.[[BindingObject]].
    // 2. Let value be ? HasProperty(bindingObject, N).
    auto value = TRY(m_binding_object.has_property(name));

    // 3. If value is false, then
    if (!value) {
        // a. If S is false, return undefined; otherwise throw a ReferenceError exception.
        if (!strict)
            return js_undefined();
        return vm.throw_completion<ReferenceError>(ErrorType::UnknownIdentifier, name);

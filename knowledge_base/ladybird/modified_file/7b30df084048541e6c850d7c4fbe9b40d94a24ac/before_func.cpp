    // 2. Return unused.
    return {};
}

// 9.1.1.2.5 SetMutableBinding ( N, V, S ), https://tc39.es/ecma262/#sec-object-environment-records-setmutablebinding-n-v-s
ThrowCompletionOr<void> ObjectEnvironment::set_mutable_binding(VM&, FlyString const& name, Value value, bool strict)
{
    auto& vm = this->vm();

    // 1. Let bindingObject be envRec.[[BindingObject]].
    // 2. Let stillExists be ? HasProperty(bindingObject, N).
    auto still_exists = TRY(m_binding_object.has_property(name));

    // 3. If stillExists is false and S is true, throw a ReferenceError exception.
    if (!still_exists && strict)
        return vm.throw_completion<ReferenceError>(ErrorType::UnknownIdentifier, name);

    // 4. Perform ? Set(bindingObject, N, V, S).
    auto result_or_error = m_binding_object.set(name, value, strict ? Object::ShouldThrowExceptions::Yes : Object::ShouldThrowExceptions::No);

    // Note: Nothing like this in the spec, this is here to produce nicer errors instead of the generic one thrown by Object::set().
    if (result_or_error.is_error() && strict) {
        auto property_or_error = m_binding_object.internal_get_own_property(name);
        // Return the initial error instead of masking it with the new error
        if (property_or_error.is_error())
            return result_or_error.release_error();
        auto property = property_or_error.release_value();
        if (property.has_value() && !property->writable.value_or(true)) {
            return vm.throw_completion<TypeError>(ErrorType::DescWriteNonWritable, name);
        }
    }

    if (result_or_error.is_error())

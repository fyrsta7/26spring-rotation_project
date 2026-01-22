
    return {};
}

ThrowCompletionOr<void> GetGlobal::execute_impl(Bytecode::Interpreter& interpreter) const
{
    auto& vm = interpreter.vm();
    auto& realm = *vm.current_realm();

    auto const& name = interpreter.current_executable().get_identifier(m_identifier);

    auto& cache = interpreter.current_executable().global_variable_caches[m_cache_index];
    auto& binding_object = realm.global_environment().object_record().binding_object();
    auto& declarative_record = realm.global_environment().declarative_record();

    // OPTIMIZATION: If the shape of the object hasn't changed, we can use the cached property offset.
    // NOTE: Unique shapes don't change identity, so we compare their serial numbers instead.
    auto& shape = binding_object.shape();
    if (cache.environment_serial_number == declarative_record.environment_serial_number()
        && &shape == cache.shape
        && (!shape.is_unique() || shape.unique_shape_serial_number() == cache.unique_shape_serial_number)) {
        interpreter.accumulator() = binding_object.get_direct(cache.property_offset.value());
        return {};
    }

    cache.environment_serial_number = declarative_record.environment_serial_number();

    if (vm.running_execution_context().script_or_module.has<NonnullGCPtr<Module>>()) {
        // NOTE: GetGlobal is used to access variables stored in the module environment and global environment.
        //       The module environment is checked first since it precedes the global environment in the environment chain.
        auto& module_environment = *vm.running_execution_context().script_or_module.get<NonnullGCPtr<Module>>()->environment();
        if (TRY(module_environment.has_binding(name))) {
            // TODO: Cache offset of binding value
            interpreter.accumulator() = TRY(module_environment.get_binding_value(vm, name, vm.in_strict_mode()));
            return {};
        }
    }

    if (TRY(declarative_record.has_binding(name))) {
        // TODO: Cache offset of binding value
        interpreter.accumulator() = TRY(declarative_record.get_binding_value(vm, name, vm.in_strict_mode()));
        return {};
    }

    if (TRY(binding_object.has_property(name))) {
        CacheablePropertyMetadata cacheable_metadata;
        interpreter.accumulator() = js_undefined();
        interpreter.accumulator() = TRY(binding_object.internal_get(name, interpreter.accumulator(), &cacheable_metadata));
        if (cacheable_metadata.type == CacheablePropertyMetadata::Type::OwnProperty) {
            cache.shape = shape;
            cache.property_offset = cacheable_metadata.property_offset.value();
            cache.unique_shape_serial_number = shape.unique_shape_serial_number();
        }
        return {};

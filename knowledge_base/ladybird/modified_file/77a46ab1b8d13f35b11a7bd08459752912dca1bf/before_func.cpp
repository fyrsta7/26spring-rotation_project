
    return TRY(object->internal_get(property_key, base_value));
}

inline ThrowCompletionOr<Value> get_global(Interpreter& interpreter, IdentifierTableIndex identifier_index, GlobalVariableCache& cache)
{
    auto& vm = interpreter.vm();
    auto& binding_object = interpreter.global_object();
    auto& declarative_record = interpreter.global_declarative_environment();

    auto& shape = binding_object.shape();
    if (cache.environment_serial_number == declarative_record.environment_serial_number()) {

        // OPTIMIZATION: For global var bindings, if the shape of the global object hasn't changed,
        //               we can use the cached property offset.
        if (&shape == cache.shape) {
            auto value = binding_object.get_direct(cache.property_offset.value());
            if (value.is_accessor())
                return TRY(call(vm, value.as_accessor().getter(), js_undefined()));
        }

        // OPTIMIZATION: For global lexical bindings, if the global declarative environment hasn't changed,
        //               we can use the cached environment binding index.
        if (cache.environment_binding_index.has_value())
            return declarative_record.get_binding_value_direct(vm, cache.environment_binding_index.value());
    }

    cache.environment_serial_number = declarative_record.environment_serial_number();

    auto& identifier = interpreter.current_executable().get_identifier(identifier_index);

    if (vm.running_execution_context().script_or_module.has<NonnullGCPtr<Module>>()) {
        // NOTE: GetGlobal is used to access variables stored in the module environment and global environment.
        //       The module environment is checked first since it precedes the global environment in the environment chain.
        auto& module_environment = *vm.running_execution_context().script_or_module.get<NonnullGCPtr<Module>>()->environment();
        if (TRY(module_environment.has_binding(identifier))) {
            // TODO: Cache offset of binding value
            return TRY(module_environment.get_binding_value(vm, identifier, vm.in_strict_mode()));
        }
    }

    Optional<size_t> offset;
    if (TRY(declarative_record.has_binding(identifier, &offset))) {
        cache.environment_binding_index = static_cast<u32>(offset.value());
        return TRY(declarative_record.get_binding_value(vm, identifier, vm.in_strict_mode()));
    }

    if (TRY(binding_object.has_property(identifier))) {
        CacheablePropertyMetadata cacheable_metadata;
        auto value = TRY(binding_object.internal_get(identifier, js_undefined(), &cacheable_metadata));
        if (cacheable_metadata.type == CacheablePropertyMetadata::Type::OwnProperty) {
            cache.shape = shape;
            cache.property_offset = cacheable_metadata.property_offset.value();
        }
        return value;

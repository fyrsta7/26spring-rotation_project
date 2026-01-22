
    return base_value.to_object(vm);
}

inline ThrowCompletionOr<Value> get_by_id(VM& vm, DeprecatedFlyString const& property, Value base_value, Value this_value, PropertyLookupCache& cache)
{
    if (base_value.is_string()) {
        auto string_value = TRY(base_value.as_string().get(vm, property));
        if (string_value.has_value())
            return *string_value;
    }

    auto base_obj = TRY(base_object_for_get(vm, base_value));

    // OPTIMIZATION: If the shape of the object hasn't changed, we can use the cached property offset.
    auto& shape = base_obj->shape();
    if (&shape == cache.shape) {
        return base_obj->get_direct(cache.property_offset.value());
    }

    CacheablePropertyMetadata cacheable_metadata;
    auto value = TRY(base_obj->internal_get(property, this_value, &cacheable_metadata));

    if (cacheable_metadata.type == CacheablePropertyMetadata::Type::OwnProperty) {
        cache.shape = shape;
        cache.property_offset = cacheable_metadata.property_offset.value();

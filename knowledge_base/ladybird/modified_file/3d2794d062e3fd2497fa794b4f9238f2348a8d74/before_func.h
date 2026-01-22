
    return value;
}

inline ThrowCompletionOr<void> put_by_value(VM& vm, Value base, Value property_key_value, Value value, Op::PropertyKind kind)
{
    // OPTIMIZATION: Fast path for simple Int32 indexes in array-like objects.
    if ((kind == Op::PropertyKind::KeyValue || kind == Op::PropertyKind::DirectKeyValue)
        && base.is_object() && property_key_value.is_int32() && property_key_value.as_i32() >= 0) {
        auto& object = base.as_object();
        auto* storage = object.indexed_properties().storage();
        auto index = static_cast<u32>(property_key_value.as_i32());

        // For "non-typed arrays":
        if (storage
            && storage->is_simple_storage()
            && !object.may_interfere_with_indexed_property_access()
            && storage->has_index(index)) {
            auto existing_value = storage->get(index)->value;
            if (!existing_value.is_accessor()) {
                storage->put(index, value);
                return {};
            }
        }

        // For typed arrays:
        if (object.is_typed_array()) {
            auto& typed_array = static_cast<TypedArrayBase&>(object);
            auto canonical_index = CanonicalIndex { CanonicalIndex::Type::Index, index };

            if (value.is_int32() && is_valid_integer_index(typed_array, canonical_index)) {
                switch (typed_array.kind()) {
                case TypedArrayBase::Kind::Uint8Array:
                    fast_typed_array_set_element<u8>(typed_array, index, static_cast<u8>(value.as_i32()));
                    return {};
                case TypedArrayBase::Kind::Uint16Array:
                    fast_typed_array_set_element<u16>(typed_array, index, static_cast<u16>(value.as_i32()));
                    return {};
                case TypedArrayBase::Kind::Uint32Array:
                    fast_typed_array_set_element<u32>(typed_array, index, static_cast<u32>(value.as_i32()));
                    return {};
                case TypedArrayBase::Kind::Int8Array:
                    fast_typed_array_set_element<i8>(typed_array, index, static_cast<i8>(value.as_i32()));
                    return {};
                case TypedArrayBase::Kind::Int16Array:
                    fast_typed_array_set_element<i16>(typed_array, index, static_cast<i16>(value.as_i32()));
                    return {};
                case TypedArrayBase::Kind::Int32Array:
                    fast_typed_array_set_element<i32>(typed_array, index, value.as_i32());
                    return {};
                case TypedArrayBase::Kind::Uint8ClampedArray:
                    fast_typed_array_set_element<u8>(typed_array, index, clamp(value.as_i32(), 0, 255));
                    return {};
                default:
                    // FIXME: Support more TypedArray kinds.
                    break;
                }
            }

            switch (typed_array.kind()) {
#define __JS_ENUMERATE(ClassName, snake_name, PrototypeName, ConstructorName, Type) \
    case TypedArrayBase::Kind::ClassName:                                           \
        return typed_array_set_element<Type>(typed_array, canonical_index, value);
                JS_ENUMERATE_TYPED_ARRAYS
#undef __JS_ENUMERATE
            }
            return {};
        }
    }


{
    return TRY(to_number(global_object)).as_double();
}

// 7.1.19 ToPropertyKey ( argument ), https://tc39.es/ecma262/#sec-topropertykey
ThrowCompletionOr<PropertyKey> Value::to_property_key(GlobalObject& global_object) const
{
    if (type() == Type::Int32 && as_i32() >= 0)
        return PropertyKey { as_i32() };

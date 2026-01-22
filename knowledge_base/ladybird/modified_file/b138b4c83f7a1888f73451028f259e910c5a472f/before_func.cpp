{
    return TRY(to_number(global_object)).as_double();
}

// 7.1.19 ToPropertyKey ( argument ), https://tc39.es/ecma262/#sec-topropertykey
ThrowCompletionOr<PropertyKey> Value::to_property_key(GlobalObject& global_object) const
{

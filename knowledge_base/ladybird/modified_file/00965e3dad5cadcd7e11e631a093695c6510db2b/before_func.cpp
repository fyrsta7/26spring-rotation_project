
Value unsigned_right_shift(GlobalObject& global_object, Value lhs, Value rhs)
{
    // 6.1.6.1.11 Number::unsignedRightShift
    // https://tc39.es/ecma262/#sec-numeric-types-number-unsignedRightShift
    auto lhs_numeric = lhs.to_numeric(global_object);
    if (global_object.vm().exception())
        return {};
    auto rhs_numeric = rhs.to_numeric(global_object);
    if (global_object.vm().exception())
        return {};
    if (both_number(lhs_numeric, rhs_numeric)) {
        if (!lhs_numeric.is_finite_number())
            return Value(0);
        if (!rhs_numeric.is_finite_number())
            return lhs_numeric;
        // Ok, so this performs toNumber() again but that "can't" throw
        auto lhs_u32 = lhs_numeric.to_u32(global_object);
        auto rhs_u32 = rhs_numeric.to_u32(global_object) % 32;
        return Value(lhs_u32 >> rhs_u32);
    }
    global_object.vm().throw_exception<TypeError>(global_object, ErrorType::BigIntBadOperator, "unsigned right-shift");
    return {};
}

Value add(GlobalObject& global_object, Value lhs, Value rhs)
{
    auto& vm = global_object.vm();
    auto lhs_primitive = lhs.to_primitive(global_object);
    if (vm.exception())
        return {};
    auto rhs_primitive = rhs.to_primitive(global_object);
    if (vm.exception())
        return {};

    if (lhs_primitive.is_string() || rhs_primitive.is_string()) {

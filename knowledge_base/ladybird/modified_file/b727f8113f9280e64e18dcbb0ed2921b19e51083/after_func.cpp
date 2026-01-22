    if (is_int32())
        return as_i32();
    return to_i32_slow_case(vm);
}

// 7.1.7 ToUint32 ( argument ), https://tc39.es/ecma262/#sec-touint32
ThrowCompletionOr<u32> Value::to_u32(VM& vm) const
{
    // OPTIMIZATION: If this value is encoded as a positive i32, return it directly.
    if (is_int32() && as_i32() >= 0)
        return as_i32();

    // 1. Let number be ? ToNumber(argument).
    double number = TRY(to_number(vm)).as_double();

    // 2. If number is not finite or number is either +0𝔽 or -0𝔽, return +0𝔽.
    if (!isfinite(number) || number == 0)
        return 0;

    // 3. Let int be the mathematical value whose sign is the sign of number and whose magnitude is floor(abs(ℝ(number))).
    auto int_val = floor(fabs(number));
    if (signbit(number))
        int_val = -int_val;

    // 4. Let int32bit be int modulo 2^32.
    auto int32bit = modulo(int_val, NumericLimits<u32>::max() + 1.0);

    define_direct_property(vm.names.SQRT2, Value(M_SQRT2), 0);

    // 21.3.1.9 Math [ @@toStringTag ], https://tc39.es/ecma262/#sec-math-@@tostringtag
    define_direct_property(vm.well_known_symbol_to_string_tag(), PrimitiveString::create(vm, vm.names.Math.as_string()), Attribute::Configurable);
}

// 21.3.2.1 Math.abs ( x ), https://tc39.es/ecma262/#sec-math.abs
JS_DEFINE_NATIVE_FUNCTION(MathObject::abs)
{
    // Let n be ? ToNumber(x).
    auto number = TRY(vm.argument(0).to_number(vm));

    // 2. If n is NaN, return NaN.
    if (number.is_nan())
        return js_nan();

    // 3. If n is -0𝔽, return +0𝔽.
    if (number.is_negative_zero())
        return Value(0);

    // 4. If n is -∞𝔽, return +∞𝔽.

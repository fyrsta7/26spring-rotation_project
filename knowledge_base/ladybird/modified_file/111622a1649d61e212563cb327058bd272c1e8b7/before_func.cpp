    // 5. If Type(lnum) is different from Type(rnum), throw a TypeError exception.
    return vm.throw_completion<TypeError>(ErrorType::BigIntBadOperatorOtherType, "subtraction");
}

// 13.7 Multiplicative Operators, https://tc39.es/ecma262/#sec-multiplicative-operators
// MultiplicativeExpression : MultiplicativeExpression MultiplicativeOperator ExponentiationExpression
ThrowCompletionOr<Value> mul(VM& vm, Value lhs, Value rhs)
{
    // 13.15.3 ApplyStringOrNumericBinaryOperator ( lval, opText, rval ), https://tc39.es/ecma262/#sec-applystringornumericbinaryoperator
    // 1-2, 6. N/A.

    // 3. Let lnum be ? ToNumeric(lval).
    auto lhs_numeric = TRY(lhs.to_numeric(vm));

    // 4. Let rnum be ? ToNumeric(rval).
    auto rhs_numeric = TRY(rhs.to_numeric(vm));

    // 7. Let operation be the abstract operation associated with opText and Type(lnum) in the following table:
    // [...]
    // 8. Return operation(lnum, rnum).
    if (both_number(lhs_numeric, rhs_numeric)) {
        // 6.1.6.1.4 Number::multiply ( x, y ), https://tc39.es/ecma262/#sec-numeric-types-number-multiply
        auto x = lhs_numeric.as_double();
        auto y = rhs_numeric.as_double();
        return Value(x * y);
    }
    if (both_bigint(lhs_numeric, rhs_numeric)) {
        // 6.1.6.2.4 BigInt::multiply ( x, y ), https://tc39.es/ecma262/#sec-numeric-types-bigint-multiply
        auto x = lhs_numeric.as_bigint().big_integer();
        auto y = rhs_numeric.as_bigint().big_integer();
        // 1. Return the BigInt value that represents the product of x and y.

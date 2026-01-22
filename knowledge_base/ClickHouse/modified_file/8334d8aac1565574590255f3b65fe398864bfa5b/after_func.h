                break;
            }
        }
        return is_zero;
    }

    /// returns quotient as result and remainder in numerator.
    template <size_t Bits2>
    constexpr static integer<Bits2, unsigned> divide(integer<Bits2, unsigned> & numerator, integer<Bits2, unsigned> denominator)
    {
        static_assert(std::is_unsigned_v<Signed>);

        if constexpr (Bits == 128 && sizeof(base_type) == 8)
        {
            using CompilerUInt128 = unsigned __int128;

            CompilerUInt128 a = (CompilerUInt128(numerator.items[1]) << 64) + numerator.items[0];
            CompilerUInt128 b = (CompilerUInt128(denominator.items[1]) << 64) + denominator.items[0];
            CompilerUInt128 c = a / b;

            integer<Bits, Signed> res;
            res.items[0] = c;
            res.items[1] = c >> 64;

            CompilerUInt128 remainder = a - b * c;
            numerator.items[0] = remainder;
            numerator.items[1] = remainder >> 64;

            return res;
        }

        if (is_zero(denominator))
            throwError("Division by zero");

        integer<Bits2, unsigned> x = 1;
        integer<Bits2, unsigned> quotient = 0;

        while (!operator_greater(denominator, numerator) && is_zero(operator_amp(shift_right(denominator, Bits2 - 1), 1)))
        {
            x = shift_left(x, 1);
            denominator = shift_left(denominator, 1);
        }

        while (!is_zero(x))
        {
            if (!operator_greater(denominator, numerator))
            {
                numerator = operator_minus(numerator, denominator);
                quotient = operator_pipe(quotient, x);
            }

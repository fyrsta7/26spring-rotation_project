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

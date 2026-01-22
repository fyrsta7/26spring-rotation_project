    static inline ResultType apply(A a)
    {
        /// We count bits in the value representation in memory. For example, we support floats.
        /// We need to avoid sign-extension when converting signed numbers to larger type. So, uint8_t(-1) has 8 bits.

        return __builtin_popcountll(ext::bit_cast<unsigned long long>(a));
    }

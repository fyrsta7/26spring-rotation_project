    static inline ResultType apply(A a)
    {
        /// We count bits in the value representation in memory. For example, we support floats.
        /// We need to avoid sign-extension when converting signed numbers to larger type. So, uint8_t(-1) has 8 bits.

        if constexpr (std::is_same_v<A, UInt64> || std::is_same_v<A, Int64>)
            return __builtin_popcountll(a);
        if constexpr (std::is_same_v<A, UInt32> || std::is_same_v<A, Int32> || std::is_unsigned_v<A>)
            return __builtin_popcount(a);
        if constexpr (std::is_same_v<A, Int16>)
            return __builtin_popcount(static_cast<UInt16>(a));
        if constexpr (std::is_same_v<A, Int8>)
            return __builtin_popcount(static_cast<UInt8>(a));
        else
            return __builtin_popcountll(ext::bit_cast<unsigned long long>(a));
    }

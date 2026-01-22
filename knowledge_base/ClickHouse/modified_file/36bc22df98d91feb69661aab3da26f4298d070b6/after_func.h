    template <typename Value>
    void NO_SANITIZE_UNDEFINED NO_INLINE addManyNotNull(const Value * __restrict ptr, const UInt8 * __restrict null_map, size_t count)
    {
        const auto * end = ptr + count;

        if constexpr (
            (is_integer_v<T> && !is_big_int_v<T>)
            || (IsDecimalNumber<T> && !std::is_same_v<T, Decimal256> && !std::is_same_v<T, Decimal128>))
        {
            /// For integers we can vectorize the operation if we replace the null check using a multiplication (by 0 for null, 1 for not null)
            /// https://quick-bench.com/q/MLTnfTvwC2qZFVeWHfOBR3U7a8I
            T local_sum{};
            while (ptr < end)
            {
                T multiplier = !*null_map;
                Impl::add(local_sum, *ptr * multiplier);
                ++ptr;
                ++null_map;
            }
            Impl::add(sum, local_sum);
            return;
        }

        if constexpr (std::is_floating_point_v<T>)
        {
            constexpr size_t unroll_count = 128 / sizeof(T);
            T partial_sums[unroll_count]{};

            const auto * unrolled_end = ptr + (count / unroll_count * unroll_count);

            while (ptr < unrolled_end)
            {
                for (size_t i = 0; i < unroll_count; ++i)
                {
                    if (!null_map[i])
                    {
                        Impl::add(partial_sums[i], ptr[i]);
                    }
                }
                ptr += unroll_count;
                null_map += unroll_count;
            }

            for (size_t i = 0; i < unroll_count; ++i)
                Impl::add(sum, partial_sums[i]);
        }

        T local_sum{};
        while (ptr < end)
        {
            if (!*null_map)
                Impl::add(local_sum, *ptr);
            ++ptr;
            ++null_map;
        }
        Impl::add(sum, local_sum);
    }

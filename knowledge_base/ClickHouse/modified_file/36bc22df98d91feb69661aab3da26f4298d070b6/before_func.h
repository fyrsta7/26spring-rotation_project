    template <typename Value>
    void NO_SANITIZE_UNDEFINED NO_INLINE addManyNotNull(const Value * __restrict ptr, const UInt8 * __restrict null_map, size_t count)
    {
        const auto * end = ptr + count;

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

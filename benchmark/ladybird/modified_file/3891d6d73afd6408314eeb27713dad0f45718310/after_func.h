    static size_t copy(T* destination, const T* source, size_t count)
    {
        if (count == 0)
            return 0;

        if constexpr (Traits<T>::is_trivial()) {
            if (count == 1)
                *destination = *source;
            else
                __builtin_memmove(destination, source, count * sizeof(T));
            return count;
        }

        for (size_t i = 0; i < count; ++i) {
            if (destination <= source)
                new (&destination[i]) T(source[i]);
            else
                new (&destination[count - i - 1]) T(source[count - i - 1]);
        }

        return count;
    }
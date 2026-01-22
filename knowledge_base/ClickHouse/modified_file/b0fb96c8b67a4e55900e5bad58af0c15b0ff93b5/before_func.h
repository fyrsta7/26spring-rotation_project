    {
        update(x.data(), x.length());
    }

    ALWAYS_INLINE void update(const std::string_view x)
    {
        update(x.data(), x.size());
    }

    /// Get the result in some form. This can only be done once!


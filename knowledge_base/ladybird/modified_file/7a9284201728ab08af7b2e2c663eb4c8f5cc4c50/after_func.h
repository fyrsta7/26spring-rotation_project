        m_buffered_size = exchange(other.m_buffered_size, 0);
        return *this;
    }

    template<template<typename> typename BufferedType>
    static ErrorOr<NonnullOwnPtr<BufferedType<T>>> create_buffered(NonnullOwnPtr<T> stream, size_t buffer_size)
    {
        if (!buffer_size)
            return Error::from_errno(EINVAL);
        if (!stream->is_open())
            return Error::from_errno(ENOTCONN);

        auto maybe_buffer = ByteBuffer::create_uninitialized(buffer_size);
        if (!maybe_buffer.has_value())

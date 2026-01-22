
    void set_recoverable_error() const override { return m_stream.set_recoverable_error(); }
    void set_fatal_error() const override { return m_stream.set_fatal_error(); }

    size_t read(Bytes bytes) override
    {
        if (has_any_error())
            return 0;

        auto nread = buffer().trim(m_buffered).copy_trimmed_to(bytes);

        m_buffered -= nread;
        buffer().slice(nread, m_buffered).copy_to(buffer());

        if (nread < bytes.size()) {
            m_buffered = m_stream.read(buffer());

            if (m_buffered == 0)
                return nread;

            nread += read(bytes.slice(nread));

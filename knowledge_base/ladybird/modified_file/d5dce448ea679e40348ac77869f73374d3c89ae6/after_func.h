
    void set_recoverable_error() const override { return m_stream.set_recoverable_error(); }
    void set_fatal_error() const override { return m_stream.set_fatal_error(); }

    size_t read(Bytes bytes) override
    {
        if (has_any_error())
            return 0;

        auto nread = buffer().trim(m_buffered).copy_trimmed_to(bytes);

        m_buffered -= nread;
        if (m_buffered > 0)
            buffer().slice(nread, m_buffered).copy_to(buffer());

        if (nread < bytes.size()) {
            nread += m_stream.read(bytes.slice(nread));

            m_buffered = m_stream.read(buffer());

    explicit OutputBitStream(OutputStream& stream)
        : m_stream(stream)
    {
    }

    // WARNING: write aligns to the next byte boundary before writing, if unaligned writes are needed this should be rewritten
    size_t write(ReadonlyBytes bytes) override
    {
        if (has_any_error())
            return 0;
        align_to_byte_boundary();
        if (has_fatal_error()) // if align_to_byte_boundary failed
            return 0;
        return m_stream.write(bytes);
    }

    bool write_or_error(ReadonlyBytes bytes) override
    {
        if (write(bytes) < bytes.size()) {
            set_fatal_error();
            return false;
        }
        return true;
    }

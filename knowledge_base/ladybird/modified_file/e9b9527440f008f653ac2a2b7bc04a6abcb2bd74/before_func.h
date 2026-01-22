    bool is_empty() const
    {
        return all_of(m_chunks, [](auto& chunk) { return chunk.is_empty(); });
    }

    template<size_t InlineSize = 0>
    DisjointSpans<T, Vector<Span<T>, InlineSize>> spans() const&
    {
        Vector<Span<T>, InlineSize> spans;

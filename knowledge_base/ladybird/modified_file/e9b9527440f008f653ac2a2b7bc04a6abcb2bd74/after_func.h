    bool is_empty() const
    {
        return all_of(m_chunks, [](auto& chunk) { return chunk.is_empty(); });
    }

    template<size_t InlineSize = 0>
    DisjointSpans<T, Vector<Span<T>, InlineSize>> spans() const&
    {
        Vector<Span<T>, InlineSize> spans;
        spans.ensure_capacity(m_chunks.size());
        if (m_chunks.size() == 1) {
            spans.append(const_cast<ChunkType&>(m_chunks[0]).span());
            return DisjointSpans<T, Vector<Span<T>, InlineSize>> { move(spans) };
        }

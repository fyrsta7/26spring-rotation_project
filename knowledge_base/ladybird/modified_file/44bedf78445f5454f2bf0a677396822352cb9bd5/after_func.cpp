    VERIFY(chunk_index < m_chunks.size());

    return Bytes { m_chunks[chunk_index].data() + chunk_offset, write_size };
}

void AllocatingMemoryStream::cleanup_unused_chunks()
{
    // FIXME: Move these all at once.
    while (m_read_offset >= CHUNK_SIZE) {
        VERIFY(m_write_offset >= m_read_offset);


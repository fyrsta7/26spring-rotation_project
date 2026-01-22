bool DiskBackedFS::write_block(unsigned index, const ByteBuffer& data)
{
#ifdef DBFS_DEBUG
    kprintf("DiskBackedFileSystem::write_block %u, size=%u\n", index, data.size());
#endif
    ASSERT(data.size() == block_size());

    {
        LOCKER(block_cache().lock());
        if (auto* cached_block = block_cache().resource().get({ fsid(), index }))
            cached_block->m_buffer = data;
    }

    LOCKER(m_lock);
    m_write_cache.set(index, data.isolated_copy());

    if (m_write_cache.size() >= 32)
        flush_writes();

    return true;
}

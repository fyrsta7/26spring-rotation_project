    {
        m_clean_list.prepend(entry);
    }

    CacheEntry* get(BlockBasedFileSystem::BlockIndex block_index) const
    {
        auto it = m_hash.find(block_index);
        if (it == m_hash.end())
            return nullptr;

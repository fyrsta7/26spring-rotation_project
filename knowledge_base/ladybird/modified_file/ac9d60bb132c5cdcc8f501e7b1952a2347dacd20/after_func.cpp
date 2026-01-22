    {
        m_clean_list.prepend(entry);
    }

    CacheEntry* get(BlockBasedFileSystem::BlockIndex block_index) const
    {
        auto it = m_hash.find(block_index);
        if (it == m_hash.end())
            return nullptr;
        auto& entry = const_cast<CacheEntry&>(*it->value);
        VERIFY(entry.block_index == block_index);
        if (!entry_is_dirty(entry) && (m_clean_list.first() != &entry)) {
            // Cache hit! Promote the entry to the front of the list.

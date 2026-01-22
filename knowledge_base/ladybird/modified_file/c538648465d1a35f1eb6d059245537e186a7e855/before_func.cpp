void Ext2FS::flush_writes()
{
    LOCKER(m_lock);
    if (m_super_block_dirty) {
        flush_super_block();
        m_super_block_dirty = false;
    }
    if (m_block_group_descriptors_dirty) {
        flush_block_group_descriptor_table();
        m_block_group_descriptors_dirty = false;
    }
    for (auto& cached_bitmap : m_cached_bitmaps) {
        if (cached_bitmap->dirty) {
            write_block(cached_bitmap->bitmap_block_index, cached_bitmap->buffer.data());
            cached_bitmap->dirty = false;
#ifdef EXT2_DEBUG
            dbg() << "Flushed bitmap block " << cached_bitmap->bitmap_block_index;
#endif
        }
    }
    DiskBackedFS::flush_writes();
}

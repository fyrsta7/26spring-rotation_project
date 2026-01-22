KResult Ext2FSInode::resize(u64 new_size)
{
    u64 block_size = fs().block_size();
    u64 old_size = size();
    int blocks_needed_before = ceil_div(old_size, block_size);
    int blocks_needed_after = ceil_div(new_size, block_size);

#ifdef EXT2_DEBUG
    dbgprintf("Ext2FSInode::resize(): blocks needed before (size was %Q): %d\n", old_size, blocks_needed_before);
    dbgprintf("Ext2FSInode::resize(): blocks needed after  (size is  %Q): %d\n", new_size, blocks_needed_after);
#endif

    if (blocks_needed_after > blocks_needed_before) {
        u32 additional_blocks_needed = blocks_needed_after - blocks_needed_before;
        if (additional_blocks_needed > fs().super_block().s_free_blocks_count)
            return KResult(-ENOSPC);
    }


    auto block_list = fs().block_list_for_inode(m_raw_inode);
    if (blocks_needed_after > blocks_needed_before) {
        auto new_blocks = fs().allocate_blocks(fs().group_index_from_inode(index()), blocks_needed_after - blocks_needed_before);
        block_list.append(move(new_blocks));
    } else if (blocks_needed_after < blocks_needed_before) {
#ifdef EXT2_DEBUG
        dbgprintf("Ext2FSInode::resize(): Shrinking. Old block list is %d entries:\n", block_list.size());
        for (auto block_index : block_list) {
            dbgprintf("    # %u\n", block_index);
        }
#endif
        while (block_list.size() != blocks_needed_after) {
            auto block_index = block_list.take_last();
            fs().set_block_allocation_state(block_index, false);
        }
    }

    bool success = fs().write_block_list_for_inode(index(), m_raw_inode, block_list);
    if (!success)
        return KResult(-EIO);

    m_raw_inode.i_size = new_size;
    set_metadata_dirty(true);

    m_block_list = move(block_list);
    return KSuccess;
}

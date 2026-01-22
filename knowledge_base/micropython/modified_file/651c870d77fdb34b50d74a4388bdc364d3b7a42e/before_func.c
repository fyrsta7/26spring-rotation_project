DRESULT sflash_disk_init (void) {
    _i32 fileHandle;
    SlFsFileInfo_t FsFileInfo;

    if (!sflash_init_done) {
        // Allocate space for the block cache
        ASSERT ((sflash_block_cache = mem_Malloc(SFLASH_BLOCK_SIZE)) != NULL);
        sflash_init_done = true;

        // Proceed to format the memory if not done yet
        for (int i = 0; i < SFLASH_BLOCK_COUNT; i++) {
            print_block_name (i);
            sl_LockObjLock (&wlan_LockObj, SL_OS_WAIT_FOREVER);
            // Create the block file if it doesn't exist
            if (sl_FsGetInfo(sflash_block_name, 0, &FsFileInfo) < 0) {
                if (!sl_FsOpen(sflash_block_name, FS_MODE_OPEN_CREATE(SFLASH_BLOCK_SIZE, 0), NULL, &fileHandle)) {
                    sl_FsClose(fileHandle, NULL, NULL, 0);
                    sl_LockObjUnlock (&wlan_LockObj);
                    memset(sflash_block_cache, 0xFF, SFLASH_BLOCK_SIZE);
                    if (!sflash_access(FS_MODE_OPEN_WRITE, sl_FsWrite)) {
                        return RES_ERROR;
                    }
                }
                else {
                    // Unexpected failure while creating the file
                    sl_LockObjUnlock (&wlan_LockObj);
                    return RES_ERROR;
                }
            }
            sl_LockObjUnlock (&wlan_LockObj);
        }
        sflash_prblock = UINT32_MAX;
        sflash_cache_is_dirty = false;
    }
    return RES_OK;
}

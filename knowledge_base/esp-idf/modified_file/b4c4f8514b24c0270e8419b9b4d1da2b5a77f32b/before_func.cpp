esp_err_t Page::load(uint32_t sectorNumber)
{
    mBaseAddress = sectorNumber * SEC_SIZE;
    mUsedEntryCount = 0;
    mErasedEntryCount = 0;

    Header header;
    auto rc = spi_flash_read(mBaseAddress, &header, sizeof(header));
    if (rc != ESP_OK) {
        mState = PageState::INVALID;
        return rc;
    }
    if (header.mState == PageState::UNINITIALIZED) {
        mState = header.mState;
        // check if the whole page is really empty
        // reading the whole page takes ~40 times less than erasing it
        uint32_t line[8];
        for (uint32_t i = 0; i < SPI_FLASH_SEC_SIZE; i += sizeof(line)) {
            rc = spi_flash_read(mBaseAddress + i, line, sizeof(line));
            if (rc != ESP_OK) {
                mState = PageState::INVALID;
                return rc;
            }
            if (std::any_of(line, line + 4, [](uint32_t val) -> bool { return val != 0xffffffff; })) {
                // page isn't as empty after all, mark it as corrupted
                mState = PageState::CORRUPT;
                break;
            }
        }
    } else if (header.mCrc32 != header.calculateCrc32()) {
        header.mState = PageState::CORRUPT;
    } else {
        mState = header.mState;
        mSeqNumber = header.mSeqNumber;
        if(header.mVersion < NVS_VERSION) {
            return ESP_ERR_NVS_NEW_VERSION_FOUND;
        } else {
            mVersion = header.mVersion;
        }
    }

    switch (mState) {
    case PageState::UNINITIALIZED:
        break;

    case PageState::FULL:
    case PageState::ACTIVE:
    case PageState::FREEING:
        mLoadEntryTable();
        break;

    default:
        mState = PageState::CORRUPT;
        break;
    }

    return ESP_OK;
}

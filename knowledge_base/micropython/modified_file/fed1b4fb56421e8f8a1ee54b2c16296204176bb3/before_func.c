
uint64_t sdcard_get_capacity_in_bytes(void) {
    if (sd_handle.Instance == NULL) {
        return 0;
    }
    HAL_SD_CardInfoTypeDef cardinfo;
    HAL_SD_GetCardInfo(&sd_handle, &cardinfo);
    return (uint64_t)cardinfo.LogBlockNbr * (uint64_t)cardinfo.LogBlockSize;
}

void SDIO_IRQHandler(void) {
    IRQ_ENTER(SDIO_IRQn);
    HAL_SD_IRQHandler(&sd_handle);
    IRQ_EXIT(SDIO_IRQn);
}

#if defined(MCU_SERIES_F7)
void SDMMC2_IRQHandler(void) {
    IRQ_ENTER(SDMMC2_IRQn);
    HAL_SD_IRQHandler(&sd_handle);
    IRQ_EXIT(SDMMC2_IRQn);
}
#endif

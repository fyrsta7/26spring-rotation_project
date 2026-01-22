
/* Task spawned by timer callback */
static void stop_prov_timer_cb(void *arg)
{
    wifi_prov_mgr_stop_provisioning();
}

esp_err_t wifi_prov_mgr_disable_auto_stop(uint32_t cleanup_delay)
{
    if (!prov_ctx_lock) {
        ESP_LOGE(TAG, "Provisioning manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t ret = ESP_FAIL;
    ACQUIRE_LOCK(prov_ctx_lock);

    if (prov_ctx && prov_ctx->prov_state == WIFI_PROV_STATE_IDLE) {
        prov_ctx->mgr_info.capabilities.no_auto_stop = true;
        prov_ctx->cleanup_delay = cleanup_delay;
        ret = ESP_OK;
    } else {
        ret = ESP_ERR_INVALID_STATE;
    }

    RELEASE_LOCK(prov_ctx_lock);
    return ret;
}

/* Call this if provisioning is completed before the timeout occurs */
esp_err_t wifi_prov_mgr_done(void)
{
    if (!prov_ctx_lock) {
        ESP_LOGE(TAG, "Provisioning manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    bool auto_stop_enabled = false;
    ACQUIRE_LOCK(prov_ctx_lock);
    if (prov_ctx && !prov_ctx->mgr_info.capabilities.no_auto_stop) {
        auto_stop_enabled = true;
    }
    RELEASE_LOCK(prov_ctx_lock);

    /* Stop provisioning if auto stop is enabled */
    if (auto_stop_enabled) {
        wifi_prov_mgr_stop_provisioning();
    }
    return ESP_OK;
}

static esp_err_t update_wifi_scan_results(void)
{
    if (!prov_ctx->scanning) {
        return ESP_ERR_INVALID_STATE;
    }
    ESP_LOGD(TAG, "Scan finished");

    esp_err_t ret = ESP_FAIL;
    uint16_t count = 0;
    uint16_t curr_channel = prov_ctx->curr_channel;

    if (prov_ctx->ap_list[curr_channel]) {
        free(prov_ctx->ap_list[curr_channel]);
        prov_ctx->ap_list[curr_channel] = NULL;
        prov_ctx->ap_list_len[curr_channel] = 0;
    }

    if (esp_wifi_scan_get_ap_num(&count) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get count of scanned APs");
        goto exit;
    }

    if (!count) {
        ESP_LOGD(TAG, "Scan result empty");
        ret = ESP_OK;
        goto exit;
    }

    uint16_t get_count = MIN(count, MAX_SCAN_RESULTS);
    prov_ctx->ap_list[curr_channel] = (wifi_ap_record_t *) calloc(get_count, sizeof(wifi_ap_record_t));
    if (!prov_ctx->ap_list[curr_channel]) {
        ESP_LOGE(TAG, "Failed to allocate memory for AP list");
        goto exit;
    }
    if (esp_wifi_scan_get_ap_records(&get_count, prov_ctx->ap_list[curr_channel]) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get scanned AP records");
        goto exit;
    }
    prov_ctx->ap_list_len[curr_channel] = get_count;

    if (prov_ctx->channels_per_group) {
        ESP_LOGD(TAG, "Scan results for channel %d :", curr_channel);
    } else {
        ESP_LOGD(TAG, "Scan results :");
    }
    ESP_LOGD(TAG, "\tS.N. %-32s %-12s %s %s", "SSID", "BSSID", "RSSI", "AUTH");
    for (uint8_t i = 0; i < prov_ctx->ap_list_len[curr_channel]; i++) {
        ESP_LOGD(TAG, "\t[%2d] %-32s %02x%02x%02x%02x%02x%02x %4d %4d", i,
                 prov_ctx->ap_list[curr_channel][i].ssid,
                 prov_ctx->ap_list[curr_channel][i].bssid[0],
                 prov_ctx->ap_list[curr_channel][i].bssid[1],
                 prov_ctx->ap_list[curr_channel][i].bssid[2],
                 prov_ctx->ap_list[curr_channel][i].bssid[3],
                 prov_ctx->ap_list[curr_channel][i].bssid[4],
                 prov_ctx->ap_list[curr_channel][i].bssid[5],
                 prov_ctx->ap_list[curr_channel][i].rssi,
                 prov_ctx->ap_list[curr_channel][i].authmode);
    }

    /* Store results in sorted list */
    {
        int rc = get_count;

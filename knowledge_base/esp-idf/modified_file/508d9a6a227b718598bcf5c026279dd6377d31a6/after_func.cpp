    }
    return eraseEntryAndSpan(index);
}

esp_err_t Page::findItem(uint8_t nsIndex, ItemType datatype, const char* key, uint8_t chunkIdx, VerOffset chunkStart)
{

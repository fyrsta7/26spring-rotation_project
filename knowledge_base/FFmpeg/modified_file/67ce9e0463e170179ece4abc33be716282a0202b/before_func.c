
    avio_seek(pb, file_size - tag_bytes, SEEK_SET);

    if (val & APE_TAG_FLAG_CONTAINS_HEADER)
        tag_bytes += APE_TAG_HEADER_BYTES;

    tag_start = file_size - tag_bytes;

    for (i=0; i<fields; i++)
        if (ape_tag_read_field(s) < 0) break;

    return tag_start;
}

static int string_is_ascii(const uint8_t *str)
{
    while (*str && *str >= 0x20 && *str <= 0x7e ) str++;
    return !*str;
}

int ff_ape_write_tag(AVFormatContext *s)
{
    AVDictionaryEntry *e = NULL;
    int size, ret, count = 0;
    AVIOContext *dyn_bc = NULL;
    uint8_t *dyn_buf = NULL;

    if ((ret = avio_open_dyn_buf(&dyn_bc)) < 0)
        goto end;

    ff_standardize_creation_time(s);
    while ((e = av_dict_get(s->metadata, "", e, AV_DICT_IGNORE_SUFFIX))) {
        int val_len;

        if (!string_is_ascii(e->key)) {
            av_log(s, AV_LOG_WARNING, "Non ASCII keys are not allowed\n");
            continue;
        }

        val_len = strlen(e->value);
        avio_wl32(dyn_bc, val_len);            // value length
        avio_wl32(dyn_bc, 0);                  // item flags
        avio_put_str(dyn_bc, e->key);          // key
        avio_write(dyn_bc, e->value, val_len); // value
        count++;
    }
    if (!count)
        goto end;

    size = avio_close_dyn_buf(dyn_bc, &dyn_buf);
    if (size <= 0)
        goto end;
    size += APE_TAG_FOOTER_BYTES;

    // header
    avio_write(s->pb, "APETAGEX", 8);   // id
    avio_wl32(s->pb, APE_TAG_VERSION);  // version
    avio_wl32(s->pb, size);
    avio_wl32(s->pb, count);

    // flags
    avio_wl32(s->pb, APE_TAG_FLAG_CONTAINS_HEADER | APE_TAG_FLAG_IS_HEADER);
    ffio_fill(s->pb, 0, 8);             // reserved

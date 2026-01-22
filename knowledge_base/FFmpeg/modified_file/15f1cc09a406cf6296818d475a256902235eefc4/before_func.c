        metadata_size != FLAC_STREAMINFO_SIZE) {
        return AVERROR_INVALIDDATA;
    }
    ff_flac_parse_streaminfo(s->avctx, (FLACStreaminfo *)s, &buf[8]);
    ret = allocate_buffers(s);
    if (ret < 0)
        return ret;
    flac_set_bps(s);
    ff_flacdsp_init(&s->dsp, s->avctx->sample_fmt, s->bps);
    s->got_streaminfo = 1;

    return 0;
}

/**
 * Determine the size of an inline header.
 * @param buf input buffer, starting with the "fLaC" marker
 * @param buf_size buffer size
 * @return number of bytes in the header, or 0 if more data is needed
 */
static int get_metadata_size(const uint8_t *buf, int buf_size)
{
    int metadata_last, metadata_size;
    const uint8_t *buf_end = buf + buf_size;

    buf += 4;
    do {
        if (buf_end - buf < 4)
            return 0;
        flac_parse_block_header(buf, &metadata_last, NULL, &metadata_size);
        buf += 4;
        if (buf_end - buf < metadata_size) {
            /* need more data in order to read the complete header */
            return 0;
        }
        buf += metadata_size;
    } while (!metadata_last);

    return buf_size - (buf_end - buf);
}

static int decode_residuals(FLACContext *s, int32_t *decoded, int pred_order)
{

}

static int decode_blck(GetByteContext *gb, uint8_t *frame, int width, int height)
{
    memset(frame, 0, width * height);
    return 0;
}


typedef int (*chunk_decoder)(GetByteContext *gb, uint8_t *frame, int width, int height);

static const chunk_decoder decoder[8] = {
    decode_copy, decode_tsw1, decode_bdlt, decode_wdlt,
    decode_tdlt, decode_dsw1, decode_blck, decode_dds1,
};

static const char chunk_name[8][5] = {
    "COPY", "TSW1", "BDLT", "WDLT", "TDLT", "DSW1", "BLCK", "DDS1"
};

static int dfa_decode_frame(AVCodecContext *avctx,
                            void *data, int *got_frame,
                            AVPacket *avpkt)
{
    AVFrame *frame = data;
    DfaContext *s = avctx->priv_data;
    GetByteContext gb;
    const uint8_t *buf = avpkt->data;
    uint32_t chunk_type, chunk_size;
    uint8_t *dst;
    int ret;
    int i, pal_elems;
    int version = avctx->extradata_size==2 ? AV_RL16(avctx->extradata) : 0;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;

    bytestream2_init(&gb, avpkt->data, avpkt->size);
    while (bytestream2_get_bytes_left(&gb) > 0) {
        if (bytestream2_get_bytes_left(&gb) < 12)
            return AVERROR_INVALIDDATA;
        bytestream2_skip(&gb, 4);
        chunk_size = bytestream2_get_le32(&gb);
        chunk_type = bytestream2_get_le32(&gb);
        if (!chunk_type)
            break;
        if (chunk_type == 1) {
            pal_elems = FFMIN(chunk_size / 3, 256);
            for (i = 0; i < pal_elems; i++) {
                s->pal[i] = bytestream2_get_be24(&gb) << 2;
                s->pal[i] |= 0xFFU << 24 | (s->pal[i] >> 6) & 0x30303;
            }
            frame->palette_has_changed = 1;
        } else if (chunk_type <= 9) {
            if (decoder[chunk_type - 2](&gb, s->frame_buf, avctx->width, avctx->height)) {
                av_log(avctx, AV_LOG_ERROR, "Error decoding %s chunk\n",
                       chunk_name[chunk_type - 2]);
                return AVERROR_INVALIDDATA;
            }
        } else {
            av_log(avctx, AV_LOG_WARNING,
                   "Ignoring unknown chunk type %"PRIu32"\n",
                   chunk_type);
        }
        buf += chunk_size;
    }

    buf = s->frame_buf;
    dst = frame->data[0];
    for (i = 0; i < avctx->height; i++) {
        if(version == 0x100) {
            int j;
            const uint8_t *buf1 = buf + (i&3)*(avctx->width/4) + (i/4)*avctx->width;
            int stride = (avctx->height/4)*avctx->width;
            for(j = 0; j < avctx->width/4; j++) {
                dst[4*j+0] = buf1[j + 0*stride];

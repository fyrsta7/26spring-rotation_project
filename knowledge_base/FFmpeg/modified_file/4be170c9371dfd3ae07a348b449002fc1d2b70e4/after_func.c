/**
 * \brief extract quantization tables from codec data into our context
 */
static int get_quant(AVCodecContext *avctx, NuvContext *c,
                     const uint8_t *buf, int size) {
    int i;
    if (size < 2 * 64 * 4) {
        av_log(avctx, AV_LOG_ERROR, "insufficient rtjpeg quant data\n");
        return -1;
    }
    for (i = 0; i < 64; i++, buf += 4)
        c->lq[i] = AV_RL32(buf);
    for (i = 0; i < 64; i++, buf += 4)
        c->cq[i] = AV_RL32(buf);
    return 0;
}

/**
 * \brief set quantization tables from a quality value
 */
static void get_quant_quality(NuvContext *c, int quality) {

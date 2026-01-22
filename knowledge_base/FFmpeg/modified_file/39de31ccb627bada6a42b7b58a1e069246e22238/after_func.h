void ff_clean_mpeg4_qscales(MpegEncContext *s);
int ff_mpeg4_decode_partitions(Mpeg4DecContext *ctx);
int ff_mpeg4_get_video_packet_prefix_length(MpegEncContext *s);
int ff_mpeg4_decode_video_packet_header(Mpeg4DecContext *ctx);
void ff_mpeg4_init_direct_mv(MpegEncContext *s);
void ff_mpeg4videodec_static_init(void);
int ff_mpeg4_workaround_bugs(AVCodecContext *avctx);
int ff_mpeg4_frame_end(AVCodecContext *avctx, const uint8_t *buf, int buf_size);

/**
 *
 * @return the mb_type
 */
int ff_mpeg4_set_direct_mv(MpegEncContext *s, int mx, int my);

extern uint8_t ff_mpeg4_static_rl_table_store[3][2][2 * MAX_RUN + MAX_LEVEL + 3];

#if 0 //3IV1 is quite rare and it slows things down a tiny bit
#define IS_3IV1 s->codec_tag == AV_RL32("3IV1")
#else
#define IS_3IV1 0
#endif

/**
 * Predict the dc.
 * encoding quantized level -> quantized diff
 * decoding quantized diff -> quantized level
 * @param n block index (0-3 are luma, 4-5 are chroma)
 * @param dir_ptr pointer to an integer where the prediction direction will be stored
 */
static inline int ff_mpeg4_pred_dc(MpegEncContext *s, int n, int level,
                                   int *dir_ptr, int encoding)
{
    int a, b, c, wrap, pred, scale, ret;
    int16_t *dc_val;

    /* find prediction */
    if (n < 4)
        scale = s->y_dc_scale;
    else
        scale = s->c_dc_scale;
    if (IS_3IV1)
        scale = 8;

    wrap   = s->block_wrap[n];
    dc_val = s->dc_val[0] + s->block_index[n];

    /* B C
     * A X
     */
    a = dc_val[-1];
    b = dc_val[-1 - wrap];
    c = dc_val[-wrap];

    /* outside slice handling (we can't do that by memset as we need the
     * dc for error resilience) */
    if (s->first_slice_line && n != 3) {
        if (n != 2)
            b = c = 1024;
        if (n != 1 && s->mb_x == s->resync_mb_x)
            b = a = 1024;
    }
    if (s->mb_x == s->resync_mb_x && s->mb_y == s->resync_mb_y + 1) {
        if (n == 0 || n == 4 || n == 5)
            b = 1024;
    }

    if (abs(a - b) < abs(b - c)) {
        pred     = c;
        *dir_ptr = 1; /* top */
    } else {
        pred     = a;
        *dir_ptr = 0; /* left */

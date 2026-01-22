
    if ((ctx->height+15)>>4 == ctx->mb_height && ctx->picture.interlaced_frame)
        ctx->height <<= 1;

    if (ctx->mb_height > 68 ||
        (ctx->mb_height<<ctx->picture.interlaced_frame) > (ctx->height+15)>>4) {
        av_log(ctx->avctx, AV_LOG_ERROR, "mb height too big: %d\n", ctx->mb_height);
        return -1;
    }

    for (i = 0; i < ctx->mb_height; i++) {
        ctx->mb_scan_index[i] = AV_RB32(buf + 0x170 + (i<<2));
        av_dlog(ctx->avctx, "mb scan index %d\n", ctx->mb_scan_index[i]);
        if (buf_size < ctx->mb_scan_index[i] + 0x280) {
            av_log(ctx->avctx, AV_LOG_ERROR, "invalid mb scan index\n");
            return -1;
        }
    }

    return 0;
}

static av_always_inline void dnxhd_decode_dct_block(DNXHDContext *ctx,
                                                    DCTELEM *block, int n,
                                                    int qscale,
                                                    int index_bits,
                                                    int level_bias,
                                                    int level_shift)
{
    int i, j, index1, index2, len;
    int level, component, sign;
    const uint8_t *weight_matrix;
    OPEN_READER(bs, &ctx->gb);

    if (n&2) {
        component = 1 + (n&1);
        weight_matrix = ctx->cid_table->chroma_weight;
    } else {
        component = 0;
        weight_matrix = ctx->cid_table->luma_weight;
    }

    UPDATE_CACHE(bs, &ctx->gb);
    GET_VLC(len, bs, &ctx->gb, ctx->dc_vlc.table, DNXHD_DC_VLC_BITS, 1);
    if (len) {
        level = GET_CACHE(bs, &ctx->gb);
        LAST_SKIP_BITS(bs, &ctx->gb, len);
        sign  = ~level >> 31;
        level = (NEG_USR32(sign ^ level, len) ^ sign) - sign;
        ctx->last_dc[component] += level;
    }
    block[0] = ctx->last_dc[component];
    //av_log(ctx->avctx, AV_LOG_DEBUG, "dc %d\n", block[0]);

    for (i = 1; ; i++) {
        UPDATE_CACHE(bs, &ctx->gb);
        GET_VLC(index1, bs, &ctx->gb, ctx->ac_vlc.table,
                DNXHD_VLC_BITS, 2);
        //av_log(ctx->avctx, AV_LOG_DEBUG, "index %d\n", index1);
        level = ctx->cid_table->ac_level[index1];
        if (!level) { /* EOB */
            //av_log(ctx->avctx, AV_LOG_DEBUG, "EOB\n");
            break;
        }

        sign = SHOW_SBITS(bs, &ctx->gb, 1);
        SKIP_BITS(bs, &ctx->gb, 1);

        if (ctx->cid_table->ac_index_flag[index1]) {
            level += SHOW_UBITS(bs, &ctx->gb, index_bits) << 6;
            SKIP_BITS(bs, &ctx->gb, index_bits);
        }

        if (ctx->cid_table->ac_run_flag[index1]) {
            UPDATE_CACHE(bs, &ctx->gb);
            GET_VLC(index2, bs, &ctx->gb, ctx->run_vlc.table,
                    DNXHD_VLC_BITS, 2);

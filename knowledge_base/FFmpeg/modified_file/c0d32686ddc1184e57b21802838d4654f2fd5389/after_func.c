        h->mtype                      &= ~MB_TYPE_H261_FIL;

        ff_mpv_decode_mb(s, s->block);
    }

    return 0;
}

static const int mvmap[17] = {
    0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16
};

static int decode_mv_component(GetBitContext *gb, int v)
{
    int mv_diff = get_vlc2(gb, h261_mv_vlc.table, H261_MV_VLC_BITS, 2);

    /* check if mv_diff is valid */
    if (mv_diff < 0)
        return v;

    mv_diff = mvmap[mv_diff];

    if (mv_diff && !get_bits1(gb))
        mv_diff = -mv_diff;

    v += mv_diff;
    if (v <= -16)
        v += 32;
    else if (v >= 16)
        v -= 32;

    return v;
}

/**
 * Decode a macroblock.
 * @return <0 if an error occurred
 */
static int h261_decode_block(H261Context *h, int16_t *block, int n, int coded)
{
    MpegEncContext *const s = &h->s;
    int level, i, j, run;
    RLTable *rl = &ff_h261_rl_tcoeff;
    const uint8_t *scan_table;

    /* For the variable length encoding there are two code tables, one being
     * used for the first transmitted LEVEL in INTER, INTER + MC and
     * INTER + MC + FIL blocks, the second for all other LEVELs except the
     * first one in INTRA blocks which is fixed length coded with 8 bits.
     * NOTE: The two code tables only differ in one VLC so we handle that
     * manually. */
    scan_table = s->intra_scantable.permutated;
    if (s->mb_intra) {
        /* DC coef */
        level = get_bits(&s->gb, 8);
        // 0 (00000000b) and -128 (10000000b) are FORBIDDEN
        if ((level & 0x7F) == 0) {
            av_log(s->avctx, AV_LOG_ERROR, "illegal dc %d at %d %d\n",
                   level, s->mb_x, s->mb_y);
            return -1;
        }
        /* The code 1000 0000 is not used, the reconstruction level of 1024
         * being coded as 1111 1111. */
        if (level == 255)
            level = 128;
        block[0] = level;
        i        = 1;
    } else if (coded) {
        // Run  Level   Code
        // EOB          Not possible for first level when cbp is available (that's why the table is different)
        // 0    1       1s
        // *    *       0*
        int check = show_bits(&s->gb, 2);
        i = 0;
        if (check & 0x2) {
            skip_bits(&s->gb, 2);
            block[0] = (check & 0x1) ? -1 : 1;
            i        = 1;
        }
    } else {

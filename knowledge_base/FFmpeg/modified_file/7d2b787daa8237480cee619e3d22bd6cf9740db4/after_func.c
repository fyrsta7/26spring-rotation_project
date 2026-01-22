    for(i=0; i<3; i++){
        s->linesize[i]= s->picture.linesize[i] << s->interlaced;
    }

//    printf("%d %d %d %d %d %d\n", s->width, s->height, s->linesize[0], s->linesize[1], s->interlaced, s->avctx->height);

    if (len != (8+(3*nb_components)))
    {
        dprintf("decode_sof0: error, len(%d) mismatch\n", len);
    }

    return 0;
}

static inline int mjpeg_decode_dc(MJpegDecodeContext *s, int dc_index)
{
    int code;
    code = get_vlc2(&s->gb, s->vlcs[0][dc_index].table, 9, 2);
    if (code < 0)
    {
        dprintf("mjpeg_decode_dc: bad vlc: %d:%d (%p)\n", 0, dc_index,
                &s->vlcs[0][dc_index]);
        return 0xffff;
    }

    if(code)
        return get_xbits(&s->gb, code);
    else
        return 0;
}

/* decode block and dequantize */
static int decode_block(MJpegDecodeContext *s, DCTELEM *block,
                        int component, int dc_index, int ac_index, int16_t *quant_matrix)
{
    int code, i, j, level, val;
    VLC *ac_vlc;

    /* DC coef */
    val = mjpeg_decode_dc(s, dc_index);
    if (val == 0xffff) {
        dprintf("error dc\n");
        return -1;
    }
    val = val * quant_matrix[0] + s->last_dc[component];
    s->last_dc[component] = val;
    block[0] = val;
    /* AC coefs */
    ac_vlc = &s->vlcs[1][ac_index];
    i = 1;
    OPEN_READER(re, &s->gb)
    for(;;) {
        UPDATE_CACHE(re, &s->gb);
        GET_VLC(code, re, &s->gb, s->vlcs[1][ac_index].table, 9, 2)

        /* EOB */
        if (code == 0)

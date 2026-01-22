        else
            pre = av_clip( ((cabac_context_init_PB[h->cabac_init_idc][i][0] * s->qscale) >>4 ) + cabac_context_init_PB[h->cabac_init_idc][i][1], 1, 126 );

        if( pre <= 63 )
            h->cabac_state[i] = 2 * ( 63 - pre ) + 0;
        else
            h->cabac_state[i] = 2 * ( pre - 64 ) + 1;
    }
}

static int decode_cabac_field_decoding_flag(H264Context *h) {
    MpegEncContext * const s = &h->s;
    const long mba_xy = h->mb_xy - 1L;
    const long mbb_xy = h->mb_xy - 2L*s->mb_stride;

    unsigned long ctx = 0;

    ctx += (s->current_picture.mb_type[mba_xy]>>7)&(h->slice_table[mba_xy] == h->slice_num);
    ctx += (s->current_picture.mb_type[mbb_xy]>>7)&(h->slice_table[mbb_xy] == h->slice_num);

    return get_cabac_noinline( &h->cabac, &(h->cabac_state+70)[ctx] );
}

static int decode_cabac_intra_mb_type(H264Context *h, int ctx_base, int intra_slice) {
    uint8_t *state= &h->cabac_state[ctx_base];
    int mb_type;

    if(intra_slice){
        int ctx=0;

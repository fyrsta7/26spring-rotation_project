        {  31,  12 }, {  37,  23 }, {  31,  38 }, {  20,  64 },
    }
};

void ff_h264_init_cabac_states(H264Context *h) {
    MpegEncContext * const s = &h->s;
    int i;

    /* calculate pre-state */
    for( i= 0; i < 460; i++ ) {
        int pre;
        if( h->slice_type_nos == FF_I_TYPE )
            pre = av_clip( ((cabac_context_init_I[i][0] * s->qscale) >>4 ) + cabac_context_init_I[i][1], 1, 126 );
        else
            pre = av_clip( ((cabac_context_init_PB[h->cabac_init_idc][i][0] * s->qscale) >>4 ) + cabac_context_init_PB[h->cabac_init_idc][i][1], 1, 126 );

        if( pre <= 63 )
            h->cabac_state[i] = 2 * ( 63 - pre ) + 0;

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

    }
};

void ff_h264_init_cabac_states(H264Context *h) {
    MpegEncContext * const s = &h->s;
    int i;
    const int8_t (*tab)[2];

    if( h->slice_type_nos == FF_I_TYPE ) tab = cabac_context_init_I;
    else                                 tab = cabac_context_init_PB[h->cabac_init_idc];

    /* calculate pre-state */

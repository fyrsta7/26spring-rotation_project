
    if( h->slice_type_nos == FF_B_TYPE) {
        if( refa > 0 && !(h->direct_cache[scan8[n] - 1]&(MB_TYPE_DIRECT2>>1)) )
            ctx++;
        if( refb > 0 && !(h->direct_cache[scan8[n] - 8]&(MB_TYPE_DIRECT2>>1)) )
            ctx += 2;
    } else {
        if( refa > 0 )
            ctx++;
        if( refb > 0 )
            ctx += 2;
    }

    while( get_cabac( &h->cabac, &h->cabac_state[54+ctx] ) ) {
        ref++;
        ctx = (ctx>>2)+4;
        if(ref >= 32 /*h->ref_list[list]*/){
            return -1;
        }
    }
    return ref;
}

static int decode_cabac_mb_mvd( H264Context *h, int list, int n, int l ) {
    int amvd = h->mvd_cache[list][scan8[n] - 1][l] +
               h->mvd_cache[list][scan8[n] - 8][l];
    int ctxbase = (l == 0) ? 40 : 47;
    int mvd;

    if(!get_cabac(&h->cabac, &h->cabac_state[ctxbase+(amvd>2) + (amvd>32)]))
        return 0;

    mvd= 1;

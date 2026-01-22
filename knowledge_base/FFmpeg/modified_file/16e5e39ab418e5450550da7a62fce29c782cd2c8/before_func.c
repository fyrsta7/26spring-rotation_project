        //{ int i; for (i = 0; i < 4; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }
        if( dir == 0 ) {
            filter_mb_edgev( &img_y[4*edge], linesize, bS, qp, h );
            if( (edge&1) == 0 ) {
                int qp= ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;
                filter_mb_edgecv( &img_cb[2*edge], uvlinesize, bS, qp, h);
                if(h->pps.chroma_qp_diff)
                    qp= ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;
                filter_mb_edgecv( &img_cr[2*edge], uvlinesize, bS, qp, h);
            }
        } else {
            filter_mb_edgeh( &img_y[4*edge*linesize], linesize, bS, qp, h );
            if( (edge&1) == 0 ) {
                int qp= ( h->chroma_qp[0] + get_chroma_qp( h, 0, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;
                filter_mb_edgech( &img_cb[2*edge*uvlinesize], uvlinesize, bS, qp, h);
                if(h->pps.chroma_qp_diff)
                    qp= ( h->chroma_qp[1] + get_chroma_qp( h, 1, s->current_picture.qscale_table[mbn_xy] ) + 1 ) >> 1;
                filter_mb_edgech( &img_cr[2*edge*uvlinesize], uvlinesize, bS, qp, h);
            }
        }
    }
}

void ff_h264_filter_mb( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {
    MpegEncContext * const s = &h->s;
    const int mb_xy= mb_x + mb_y*s->mb_stride;
    const int mb_type = s->current_picture.mb_type[mb_xy];
    const int mvy_limit = IS_INTERLACED(mb_type) ? 2 : 4;
    int first_vertical_edge_done = 0;
    av_unused int dir;
    int list;

    if (FRAME_MBAFF
            // and current and left pair do not have the same interlaced type
            && IS_INTERLACED(mb_type^h->left_type[0])
            // and left mb is in the same slice if deblocking_filter == 2
            && h->left_type[0]) {
        /* First vertical edge is different in MBAFF frames
         * There are 8 different bS to compute and 2 different Qp
         */
        DECLARE_ALIGNED_8(int16_t, bS)[8];
        int qp[2];
        int bqp[2];
        int rqp[2];
        int mb_qp, mbn0_qp, mbn1_qp;
        int i;
        first_vertical_edge_done = 1;

        if( IS_INTRA(mb_type) )
            *(uint64_t*)&bS[0]=
            *(uint64_t*)&bS[4]= 0x0004000400040004ULL;
        else {
            for( i = 0; i < 8; i++ ) {
                int mbn_xy = MB_FIELD ? h->left_mb_xy[i>>2] : h->left_mb_xy[i&1];

                if( IS_INTRA( s->current_picture.mb_type[mbn_xy] ) )
                    bS[i] = 4;
                else if( h->non_zero_count_cache[12+8*(i>>1)] != 0 ||
                         ((!h->pps.cabac && IS_8x8DCT(s->current_picture.mb_type[mbn_xy])) ?
                            (h->cbp_table[mbn_xy] & ((MB_FIELD ? (i&2) : (mb_y&1)) ? 8 : 2))
                                                                       :
                            h->non_zero_count[mbn_xy][7+(MB_FIELD ? (i&3) : (i>>2)+(mb_y&1)*2)*8]))
                    bS[i] = 2;
                else
                    bS[i] = 1;
            }
        }

        mb_qp = s->current_picture.qscale_table[mb_xy];
        mbn0_qp = s->current_picture.qscale_table[h->left_mb_xy[0]];
        mbn1_qp = s->current_picture.qscale_table[h->left_mb_xy[1]];
        qp[0] = ( mb_qp + mbn0_qp + 1 ) >> 1;
        bqp[0] = ( get_chroma_qp( h, 0, mb_qp ) +
                   get_chroma_qp( h, 0, mbn0_qp ) + 1 ) >> 1;
        rqp[0] = ( get_chroma_qp( h, 1, mb_qp ) +
                   get_chroma_qp( h, 1, mbn0_qp ) + 1 ) >> 1;
        qp[1] = ( mb_qp + mbn1_qp + 1 ) >> 1;
        bqp[1] = ( get_chroma_qp( h, 0, mb_qp ) +
                   get_chroma_qp( h, 0, mbn1_qp ) + 1 ) >> 1;
        rqp[1] = ( get_chroma_qp( h, 1, mb_qp ) +
                   get_chroma_qp( h, 1, mbn1_qp ) + 1 ) >> 1;

        /* Filter edge */
        tprintf(s->avctx, "filter mb:%d/%d MBAFF, QPy:%d/%d, QPb:%d/%d QPr:%d/%d ls:%d uvls:%d", mb_x, mb_y, qp[0], qp[1], bqp[0], bqp[1], rqp[0], rqp[1], linesize, uvlinesize);
        { int i; for (i = 0; i < 8; i++) tprintf(s->avctx, " bS[%d]:%d", i, bS[i]); tprintf(s->avctx, "\n"); }

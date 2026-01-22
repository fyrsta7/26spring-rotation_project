    } else {
        h->s.dsp.h264_v_loop_filter_luma_intra(pix, stride, alpha, beta);
    }
}

static void av_always_inline filter_mb_edgech( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, H264Context *h ) {
    const unsigned int index_a = qp + h->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + h->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]]+1;
        tc[1] = tc0_table[index_a][bS[1]]+1;
        tc[2] = tc0_table[index_a][bS[2]]+1;
        tc[3] = tc0_table[index_a][bS[3]]+1;
        h->s.dsp.h264_v_loop_filter_chroma(pix, stride, alpha, beta, tc);
    } else {
        h->s.dsp.h264_v_loop_filter_chroma_intra(pix, stride, alpha, beta);
    }
}

void ff_h264_filter_mb_fast( H264Context *h, int mb_x, int mb_y, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {
    MpegEncContext * const s = &h->s;
    int mb_xy;
    int mb_type, left_type;
    int qp, qp0, qp1, qpc, qpc0, qpc1, qp_thresh;

    mb_xy = h->mb_xy;

    if(!h->top_type || !s->dsp.h264_loop_filter_strength || h->pps.chroma_qp_diff) {
        ff_h264_filter_mb(h, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize);
        return;
    }
    assert(!FRAME_MBAFF);
    left_type= h->left_type[0];

    mb_type = s->current_picture.mb_type[mb_xy];
    qp = s->current_picture.qscale_table[mb_xy];
    qp0 = s->current_picture.qscale_table[mb_xy-1];
    qp1 = s->current_picture.qscale_table[h->top_mb_xy];
    qpc = get_chroma_qp( h, 0, qp );
    qpc0 = get_chroma_qp( h, 0, qp0 );
    qpc1 = get_chroma_qp( h, 0, qp1 );
    qp0 = (qp + qp0 + 1) >> 1;
    qp1 = (qp + qp1 + 1) >> 1;
    qpc0 = (qpc + qpc0 + 1) >> 1;
    qpc1 = (qpc + qpc1 + 1) >> 1;
    qp_thresh = 15+52 - h->slice_alpha_c0_offset;
    if(qp <= qp_thresh && qp0 <= qp_thresh && qp1 <= qp_thresh &&
       qpc <= qp_thresh && qpc0 <= qp_thresh && qpc1 <= qp_thresh)
        return;

    if( IS_INTRA(mb_type) ) {
        int16_t bS4[4] = {4,4,4,4};
        int16_t bS3[4] = {3,3,3,3};
        int16_t *bSH = FIELD_PICTURE ? bS3 : bS4;
        if(left_type)
            filter_mb_edgev( &img_y[4*0], linesize, bS4, qp0, h);
        if( IS_8x8DCT(mb_type) ) {
            filter_mb_edgev( &img_y[4*2], linesize, bS3, qp, h);
            filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, h);
            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, h);
        } else {
            filter_mb_edgev( &img_y[4*1], linesize, bS3, qp, h);
            filter_mb_edgev( &img_y[4*2], linesize, bS3, qp, h);
            filter_mb_edgev( &img_y[4*3], linesize, bS3, qp, h);
            filter_mb_edgeh( &img_y[4*0*linesize], linesize, bSH, qp1, h);
            filter_mb_edgeh( &img_y[4*1*linesize], linesize, bS3, qp, h);
            filter_mb_edgeh( &img_y[4*2*linesize], linesize, bS3, qp, h);
            filter_mb_edgeh( &img_y[4*3*linesize], linesize, bS3, qp, h);
        }
        if(left_type){
            filter_mb_edgecv( &img_cb[2*0], uvlinesize, bS4, qpc0, h);
            filter_mb_edgecv( &img_cr[2*0], uvlinesize, bS4, qpc0, h);
        }
        filter_mb_edgecv( &img_cb[2*2], uvlinesize, bS3, qpc, h);
        filter_mb_edgecv( &img_cr[2*2], uvlinesize, bS3, qpc, h);
        filter_mb_edgech( &img_cb[2*0*uvlinesize], uvlinesize, bSH, qpc1, h);
        filter_mb_edgech( &img_cb[2*2*uvlinesize], uvlinesize, bS3, qpc, h);
        filter_mb_edgech( &img_cr[2*0*uvlinesize], uvlinesize, bSH, qpc1, h);
        filter_mb_edgech( &img_cr[2*2*uvlinesize], uvlinesize, bS3, qpc, h);
        return;
    } else {
        DECLARE_ALIGNED_8(int16_t, bS)[2][4][4];
        uint64_t (*bSv)[4] = (uint64_t(*)[4])bS;
        int edges;
        if( IS_8x8DCT(mb_type) && (h->cbp&7) == 7 ) {
            edges = 4;
            bSv[0][0] = bSv[0][2] = bSv[1][0] = bSv[1][2] = 0x0002000200020002ULL;
        } else {
            int mask_edge1 = (3*(((5*mb_type)>>5)&1)) | (mb_type>>4); //(mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : (mb_type & MB_TYPE_16x8) ? 1 : 0;
            int mask_edge0 = 3*((mask_edge1>>1) & ((5*h->left_type[0])>>5)&1); // (mb_type & (MB_TYPE_16x16 | MB_TYPE_8x16)) && (h->left_type[0] & (MB_TYPE_16x16 | MB_TYPE_8x16)) ? 3 : 0;
            int step =  1+(mb_type>>24); //IS_8x8DCT(mb_type) ? 2 : 1;
            edges = 4 - 3*((mb_type>>3) & !(h->cbp & 15)); //(mb_type & MB_TYPE_16x16) && !(h->cbp & 15) ? 1 : 4;
            s->dsp.h264_loop_filter_strength( bS, h->non_zero_count_cache, h->ref_cache, h->mv_cache,
                                              h->list_count==2, edges, step, mask_edge0, mask_edge1, FIELD_PICTURE);
        }
        if( IS_INTRA(h->left_type[0]) )
            bSv[0][0] = 0x0004000400040004ULL;
        if( IS_INTRA(h->top_type) )
            bSv[1][0] = FIELD_PICTURE ? 0x0003000300030003ULL : 0x0004000400040004ULL;

#define FILTER(hv,dir,edge)\
        if(bSv[dir][edge]) {\
            filter_mb_edge##hv( &img_y[4*edge*(dir?linesize:1)], linesize, bS[dir][edge], edge ? qp : qp##dir, h );\
            if(!(edge&1)) {\
                filter_mb_edgec##hv( &img_cb[2*edge*(dir?uvlinesize:1)], uvlinesize, bS[dir][edge], edge ? qpc : qpc##dir, h );\

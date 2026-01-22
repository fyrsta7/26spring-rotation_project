        s->dsp.put_qpel_pixels_tab[0][dxy](dest_y    , ptr    , stride);

        dxy = ((motion_by & 3) << 2) | (motion_bx & 3);
        src_x = motion_bx >> 2;
        src_y = motion_by >> 2;

        ptr = ref2_data[0] + (src_y * stride) + src_x;
        s->dsp.avg_qpel_pixels_tab[size][dxy](dest_y    , ptr    , stride);
    }else{
        dxy = ((motion_fy & 1) << 1) | (motion_fx & 1);
        src_x = motion_fx >> 1;
        src_y = motion_fy >> 1;

        ptr = ref_data[0] + (src_y * stride) + src_x;
        s->dsp.put_pixels_tab[size][dxy](dest_y    , ptr    , stride, h);

        dxy = ((motion_by & 1) << 1) | (motion_bx & 1);
        src_x = motion_bx >> 1;
        src_y = motion_by >> 1;

        ptr = ref2_data[0] + (src_y * stride) + src_x;
        s->dsp.avg_pixels_tab[size][dxy](dest_y    , ptr    , stride, h);
    }

    fbmin = (mv_penalty_f[motion_fx-pred_fx] + mv_penalty_f[motion_fy-pred_fy])*c->mb_penalty_factor
           +(mv_penalty_b[motion_bx-pred_bx] + mv_penalty_b[motion_by-pred_by])*c->mb_penalty_factor
           + s->dsp.mb_cmp[size](s, src_data[0], dest_y, stride, h); //FIXME new_pic

    if(c->avctx->mb_cmp&FF_CMP_CHROMA){
    }
    //FIXME CHROMA !!!

    return fbmin;
}

/* refine the bidir vectors in hq mode and return the score in both lq & hq mode*/
static inline int bidir_refine(MpegEncContext * s, int mb_x, int mb_y)
{
    MotionEstContext * const c= &s->me;
    const int mot_stride = s->mb_stride;
    const int xy = mb_y *mot_stride + mb_x;
    int fbmin;
    int pred_fx= s->b_bidir_forw_mv_table[xy-1][0];
    int pred_fy= s->b_bidir_forw_mv_table[xy-1][1];
    int pred_bx= s->b_bidir_back_mv_table[xy-1][0];
    int pred_by= s->b_bidir_back_mv_table[xy-1][1];
    int motion_fx= s->b_bidir_forw_mv_table[xy][0]= s->b_forw_mv_table[xy][0];
    int motion_fy= s->b_bidir_forw_mv_table[xy][1]= s->b_forw_mv_table[xy][1];
    int motion_bx= s->b_bidir_back_mv_table[xy][0]= s->b_back_mv_table[xy][0];
    int motion_by= s->b_bidir_back_mv_table[xy][1]= s->b_back_mv_table[xy][1];
    const int flags= c->sub_flags;
    const int qpel= flags&FLAG_QPEL;
    const int shift= 1+qpel;
    const int xmin= c->xmin<<shift;
    const int ymin= c->ymin<<shift;
    const int xmax= c->xmax<<shift;
    const int ymax= c->ymax<<shift;
#define HASH(fx,fy,bx,by) ((fx)+17*(fy)+63*(bx)+117*(by))
    int hashidx= HASH(motion_fx,motion_fy, motion_bx, motion_by);
    uint8_t map[256];

    memset(map,0,sizeof(map));
    map[hashidx&255] = 1;

    fbmin= check_bidir_mv(s, motion_fx, motion_fy,
                          motion_bx, motion_by,
                          pred_fx, pred_fy,
                          pred_bx, pred_by,
                          0, 16);

    if(s->avctx->bidir_refine){
        int score, end;
#define CHECK_BIDIR(fx,fy,bx,by)\
    if( !map[(hashidx+HASH(fx,fy,bx,by))&255]\
       &&(fx<=0 || motion_fx+fx<=xmax) && (fy<=0 || motion_fy+fy<=ymax) && (bx<=0 || motion_bx+bx<=xmax) && (by<=0 || motion_by+by<=ymax)\
       &&(fx>=0 || motion_fx+fx>=xmin) && (fy>=0 || motion_fy+fy>=ymin) && (bx>=0 || motion_bx+bx>=xmin) && (by>=0 || motion_by+by>=ymin)){\
        map[(hashidx+HASH(fx,fy,bx,by))&255] = 1;\
        score= check_bidir_mv(s, motion_fx+fx, motion_fy+fy, motion_bx+bx, motion_by+by, pred_fx, pred_fy, pred_bx, pred_by, 0, 16);\
        if(score < fbmin){\
            hashidx += HASH(fx,fy,bx,by);\
            fbmin= score;\
            motion_fx+=fx;\
            motion_fy+=fy;\
            motion_bx+=bx;\
            motion_by+=by;\
            end=0;\
        }\
    }
#define CHECK_BIDIR2(a,b,c,d)\
CHECK_BIDIR(a,b,c,d)\
CHECK_BIDIR(-(a),-(b),-(c),-(d))

#define CHECK_BIDIRR(a,b,c,d)\
CHECK_BIDIR2(a,b,c,d)\
CHECK_BIDIR2(b,c,d,a)\
CHECK_BIDIR2(c,d,a,b)\
CHECK_BIDIR2(d,a,b,c)

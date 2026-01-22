    if(f->ac>1){
        for(i=1; i<256; i++){
            put_symbol(c, state, f->state_transition[i] - c->one_state[i], 1);
        }
    }
    put_symbol(c, state, f->colorspace, 0); //YUV cs type
    put_symbol(c, state, f->avctx->bits_per_raw_sample, 0);
    put_rac(c, state, 1); //chroma planes
        put_symbol(c, state, f->chroma_h_shift, 0);
        put_symbol(c, state, f->chroma_v_shift, 0);
    put_rac(c, state, 0); //no transparency plane
    put_symbol(c, state, f->num_h_slices-1, 0);
    put_symbol(c, state, f->num_v_slices-1, 0);

    put_symbol(c, state, f->quant_table_count, 0);
    for(i=0; i<f->quant_table_count; i++)
        write_quant_tables(c, f->quant_tables[i]);

    f->avctx->extradata_size= ff_rac_terminate(c);

    return 0;
}

static av_cold int encode_init(AVCodecContext *avctx)
{
    FFV1Context *s = avctx->priv_data;
    int i, j, i2;

    common_init(avctx);

    s->version=0;
    s->ac= avctx->coder_type ? 2:0;

    if(s->ac>1)
        for(i=1; i<256; i++)
            s->state_transition[i]=ver2_state[i];

    s->plane_count=2;
    for(i=0; i<256; i++){
        s->quant_table_count=2;
        if(avctx->bits_per_raw_sample <=8){
            s->quant_tables[0][0][i]=           quant11[i];
            s->quant_tables[0][1][i]=        11*quant11[i];
            s->quant_tables[0][2][i]=     11*11*quant11[i];
            s->quant_tables[1][0][i]=           quant11[i];
            s->quant_tables[1][1][i]=        11*quant11[i];
            s->quant_tables[1][2][i]=     11*11*quant5 [i];
            s->quant_tables[1][3][i]=   5*11*11*quant5 [i];
            s->quant_tables[1][4][i]= 5*5*11*11*quant5 [i];
        }else{
            s->quant_tables[0][0][i]=           quant9_10bit[i];
            s->quant_tables[0][1][i]=        11*quant9_10bit[i];
            s->quant_tables[0][2][i]=     11*11*quant9_10bit[i];
            s->quant_tables[1][0][i]=           quant9_10bit[i];
            s->quant_tables[1][1][i]=        11*quant9_10bit[i];
            s->quant_tables[1][2][i]=     11*11*quant5_10bit[i];
            s->quant_tables[1][3][i]=   5*11*11*quant5_10bit[i];
            s->quant_tables[1][4][i]= 5*5*11*11*quant5_10bit[i];
        }
    }
    memcpy(s->quant_table, s->quant_tables[avctx->context_model], sizeof(s->quant_table));

    for(i=0; i<s->plane_count; i++){
        PlaneContext * const p= &s->plane[i];

        memcpy(p->quant_table, s->quant_table, sizeof(p->quant_table));
        if(avctx->context_model==0){
            p->context_count= (11*11*11+1)/2;
        }else{
            p->context_count= (11*11*5*5*5+1)/2;
        }
    }

    avctx->coded_frame= &s->picture;
    switch(avctx->pix_fmt){
    case PIX_FMT_YUV444P16:
    case PIX_FMT_YUV422P16:
    case PIX_FMT_YUV420P16:
        if(avctx->bits_per_raw_sample <=8){
            av_log(avctx, AV_LOG_ERROR, "bits_per_raw_sample invalid\n");
            return -1;
        }
        if(!s->ac){
            av_log(avctx, AV_LOG_ERROR, "bits_per_raw_sample of more than 8 needs -coder 1 currently\n");
            return -1;
        }
        s->version= FFMAX(s->version, 1);
    case PIX_FMT_YUV444P:
    case PIX_FMT_YUV422P:
    case PIX_FMT_YUV420P:
    case PIX_FMT_YUV411P:
    case PIX_FMT_YUV410P:
        s->colorspace= 0;
        break;
    case PIX_FMT_RGB32:
        s->colorspace= 1;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "format not supported\n");
        return -1;
    }
    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_h_shift, &s->chroma_v_shift);

    s->picture_number=0;

    if(avctx->stats_in){
        char *p= avctx->stats_in;
        int changed;

        for(;;){
            for(j=0; j<256; j++){
                for(i=0; i<2; i++){
                    char *next;
                    s->rc_stat[j][i]= strtol(p, &next, 0);
                    if(next==p){
                        av_log(avctx, AV_LOG_ERROR, "2Pass file invalid at %d %d [%s]\n", j,i,p);
                        return -1;
                    }
                    p=next;
                }
            }
            while(*p=='\n' || *p==' ') p++;
            if(p[0]==0) break;
        }

        do{
            changed=0;
            for(i=12; i<244; i++){
                for(i2=i+1; i2<245; i2++){
#define COST(old, new) \
    s->rc_stat[old][0]*-log2((256-(new))/256.0)\
   +s->rc_stat[old][1]*-log2(     (new) /256.0)

#define COST2(old, new) \
    COST(old, new)\
   +COST(256-(old), 256-(new))

                    double size0= COST2(i, i ) + COST2(i2, i2);
                    double sizeX= COST2(i, i2) + COST2(i2, i );
                    if(sizeX < size0 && i!=128 && i2!=128){ //128 is special we cant swap it around FIXME 127<->129 swap
                        int j;
                        FFSWAP(int, s->state_transition[    i], s->state_transition[    i2]);
                        FFSWAP(int, s->state_transition[256-i], s->state_transition[256-i2]);
                        FFSWAP(int, s->rc_stat[i    ][0],s->rc_stat[    i2][0]);
                        FFSWAP(int, s->rc_stat[i    ][1],s->rc_stat[    i2][1]);
                        FFSWAP(int, s->rc_stat[256-i][0],s->rc_stat[256-i2][0]);
                        FFSWAP(int, s->rc_stat[256-i][1],s->rc_stat[256-i2][1]);
                        for(j=1; j<256; j++){
                            if     (s->state_transition[j] == i ) s->state_transition[j] = i2;
                            else if(s->state_transition[j] == i2) s->state_transition[j] = i ;
                            if     (s->state_transition[256-j] == 256-i ) s->state_transition[256-j] = 256-i2;
                            else if(s->state_transition[256-j] == 256-i2) s->state_transition[256-j] = 256-i ;

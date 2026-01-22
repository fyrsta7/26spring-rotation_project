
        /* Obtain average GOB size for RTP */
        if (s->rtp_mode) {
            if (!mb_y)
                s->mb_line_avgsize = pbBufPtr(&s->pb) - s->ptr_last_mb_line;
            else if (!(mb_y % s->gob_index)) {    
                s->mb_line_avgsize = (s->mb_line_avgsize + pbBufPtr(&s->pb) - s->ptr_last_mb_line) >> 1;
                s->ptr_last_mb_line = pbBufPtr(&s->pb);
            }
            //fprintf(stderr, "\nMB line: %d\tSize: %u\tAvg. Size: %u", s->mb_y, 
            //                    (s->pb.buf_ptr - s->ptr_last_mb_line), s->mb_line_avgsize);
            if(s->codec_id!=CODEC_ID_MPEG4) s->first_slice_line = 0; //FIXME clean
        }
    }
    emms_c();

    if(s->codec_id==CODEC_ID_MPEG4 && s->data_partitioning && s->pict_type!=B_TYPE)
        ff_mpeg4_merge_partitions(s);

    if (s->msmpeg4_version && s->msmpeg4_version<4 && s->pict_type == I_TYPE)
        msmpeg4_encode_ext_header(s);

    if(s->codec_id==CODEC_ID_MPEG4) 
        ff_mpeg4_stuffing(&s->pb);

    //if (s->gob_number)
    //    fprintf(stderr,"\nNumber of GOB: %d", s->gob_number);
    
    /* Send the last GOB if RTP */    
    if (s->rtp_mode) {
        flush_put_bits(&s->pb);
        pdif = pbBufPtr(&s->pb) - s->ptr_lastgob;
        /* Call the RTP callback to send the last GOB */
        if (s->rtp_callback)
            s->rtp_callback(s->ptr_lastgob, pdif, s->gob_number);
        s->ptr_lastgob = pbBufPtr(&s->pb);
        //fprintf(stderr,"\nGOB: %2d size: %d (last)", s->gob_number, pdif);
    }
}

static int dct_quantize_c(MpegEncContext *s, 
                        DCTELEM *block, int n,
                        int qscale, int *overflow)
{
    int i, j, level, last_non_zero, q;
    const int *qmat;
    int bias;
    int max=0;
    unsigned int threshold1, threshold2;
    
    s->fdct (block);

    /* we need this permutation so that we correct the IDCT
       permutation. will be moved into DCT code */
    block_permute(block);

    if (s->mb_intra) {
        if (!s->h263_aic) {
            if (n < 4)
                q = s->y_dc_scale;
            else
                q = s->c_dc_scale;
            q = q << 3;
        } else
            /* For AIC we skip quant/dequant of INTRADC */
            q = 1 << 3;

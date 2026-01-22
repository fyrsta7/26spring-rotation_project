    s->dsp.idct_put(dest_y              + 8, linesize, block[1]);
    s->dsp.idct_put(dest_y + 8*linesize    , linesize, block[2]);
    s->dsp.idct_put(dest_y + 8*linesize + 8, linesize, block[3]);
    if(!(s->avctx->flags&CODEC_FLAG_GRAY)) {
        s->dsp.idct_put(dest_cb, t->frame.linesize[1], block[4]);
        s->dsp.idct_put(dest_cr, t->frame.linesize[2], block[5]);
    }
}

static void tqi_calculate_qtable(MpegEncContext *s, int quant)
{
    const int qscale = (215 - 2*quant)*5;
    int i;
    if (s->avctx->idct_algo==FF_IDCT_EA) {
        s->intra_matrix[0] = (ff_inv_aanscales[0]*ff_mpeg1_default_intra_matrix[0])>>11;
        for(i=1; i<64; i++)
            s->intra_matrix[i] = (ff_inv_aanscales[i]*ff_mpeg1_default_intra_matrix[i]*qscale + 32)>>14;
    }else{
        s->intra_matrix[0] = ff_mpeg1_default_intra_matrix[0];
        for(i=1; i<64; i++)
            s->intra_matrix[i] = (ff_mpeg1_default_intra_matrix[i]*qscale + 32)>>3;
    }
}

static int tqi_decode_frame(AVCodecContext *avctx,
                            void *data, int *data_size,
                            AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    const uint8_t *buf_end = buf+buf_size;
    TqiContext *t = avctx->priv_data;
    MpegEncContext *s = &t->s;

    s->width  = AV_RL16(&buf[0]);
    s->height = AV_RL16(&buf[2]);
    tqi_calculate_qtable(s, buf[4]);
    buf += 8;

    if (t->frame.data[0])
        avctx->release_buffer(avctx, &t->frame);

    if (s->avctx->width!=s->width || s->avctx->height!=s->height)
        avcodec_set_dimensions(s->avctx, s->width, s->height);

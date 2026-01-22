    int linesize= a->picture.linesize[0];
    int i;

    uint8_t *ptr_y  = a->picture.data[0] + (mb_y * 16* linesize              ) + mb_x * 16;
    uint8_t *ptr_cb = a->picture.data[1] + (mb_y * 8 * a->picture.linesize[1]) + mb_x * 8;
    uint8_t *ptr_cr = a->picture.data[2] + (mb_y * 8 * a->picture.linesize[2]) + mb_x * 8;

    a->dsp.get_pixels(block[0], ptr_y                 , linesize);
    a->dsp.get_pixels(block[1], ptr_y              + 8, linesize);
    a->dsp.get_pixels(block[2], ptr_y + 8*linesize    , linesize);
    a->dsp.get_pixels(block[3], ptr_y + 8*linesize + 8, linesize);
    for(i=0; i<4; i++)
        a->dsp.fdct(block[i]);

    if(!(a->avctx->flags&CODEC_FLAG_GRAY)){
        a->dsp.get_pixels(block[4], ptr_cb, a->picture.linesize[1]);
        a->dsp.get_pixels(block[5], ptr_cr, a->picture.linesize[2]);
        for(i=4; i<6; i++)
            a->dsp.fdct(block[i]);
    }
}

static int decode_frame(AVCodecContext *avctx,
                        void *data, int *data_size,
                        AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    ASV1Context * const a = avctx->priv_data;
    AVFrame *picture = data;
    AVFrame * const p= &a->picture;
    int mb_x, mb_y;

    if(p->data[0])
        avctx->release_buffer(avctx, p);

    p->reference= 0;
    if(avctx->get_buffer(avctx, p) < 0){
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return -1;
    }
    p->pict_type= AV_PICTURE_TYPE_I;
    p->key_frame= 1;

    av_fast_padded_malloc(&a->bitstream_buffer, &a->bitstream_buffer_size,
                          buf_size);
    if (!a->bitstream_buffer)
        return AVERROR(ENOMEM);

    if(avctx->codec_id == CODEC_ID_ASV1)
        a->dsp.bswap_buf((uint32_t*)a->bitstream_buffer, (const uint32_t*)buf, buf_size/4);
    else{
        int i;
        for(i=0; i<buf_size; i++)
            a->bitstream_buffer[i]= av_reverse[ buf[i] ];
    }

    init_get_bits(&a->gb, a->bitstream_buffer, buf_size*8);

    for(mb_y=0; mb_y<a->mb_height2; mb_y++){
        for(mb_x=0; mb_x<a->mb_width2; mb_x++){
            if( decode_mb(a, a->block) <0)
                return -1;

            idct_put(a, mb_x, mb_y);
        }
    }

    if(a->mb_width2 != a->mb_width){
        mb_x= a->mb_width2;
        for(mb_y=0; mb_y<a->mb_height2; mb_y++){
            if( decode_mb(a, a->block) <0)
                return -1;

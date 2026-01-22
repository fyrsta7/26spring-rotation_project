}

/* generic function for encode/decode called after a frame has been coded/decoded */
void MPV_frame_end(MpegEncContext *s)
{
    /* draw edge for correct motion prediction if outside */
    if (s->pict_type != B_TYPE && !s->intra_only) {
      if(s->avctx==NULL || s->avctx->codec->id!=CODEC_ID_MPEG4){
        draw_edges(s->current_picture[0], s->linesize, s->mb_width*16, s->mb_height*16, EDGE_WIDTH);
        draw_edges(s->current_picture[1], s->linesize/2, s->mb_width*8, s->mb_height*8, EDGE_WIDTH/2);
        draw_edges(s->current_picture[2], s->linesize/2, s->mb_width*8, s->mb_height*8, EDGE_WIDTH/2);
      }else{
        /* OpenDivx, but i dunno how to distinguish it from mpeg4 */
        draw_edges(s->current_picture[0], s->linesize, s->width, s->height, EDGE_WIDTH);
        draw_edges(s->current_picture[1], s->linesize/2, s->width/2, s->height/2, EDGE_WIDTH/2);
        draw_edges(s->current_picture[2], s->linesize/2, s->width/2, s->height/2, EDGE_WIDTH/2);
      }
    }
    emms_c();
}

int MPV_encode_picture(AVCodecContext *avctx,
                       unsigned char *buf, int buf_size, void *data)
{
    MpegEncContext *s = avctx->priv_data;
    AVPicture *pict = data;
    int i, j;

    if (s->fixed_qscale) 
        s->qscale = avctx->quality;

    init_put_bits(&s->pb, buf, buf_size, NULL, NULL);

    if (!s->intra_only) {
        /* first picture of GOP is intra */
        if ((s->picture_number % s->gop_size) == 0)
            s->pict_type = I_TYPE;
        else
            s->pict_type = P_TYPE;
    } else {
        s->pict_type = I_TYPE;
    }
    avctx->key_frame = (s->pict_type == I_TYPE);
    
    MPV_frame_start(s);
    
    for(i=0;i<3;i++) {
        UINT8 *src = pict->data[i];
        UINT8 *dest = s->current_picture[i];
        int src_wrap = pict->linesize[i];
        int dest_wrap = s->linesize;
        int w = s->width;
        int h = s->height;

        if (i >= 1) {
            dest_wrap >>= 1;
            w >>= 1;
            h >>= 1;
        }

	if(s->intra_only && dest_wrap==src_wrap){
	    s->current_picture[i] = pict->data[i];
	}else {
            for(j=0;j<h;j++) {

            /* unknown fields */
            skip_bits1(&gb);
            skip_bits1(&gb);
            skip_bits1(&gb);
            skip_bits1(&gb);

            s->low_delay = get_bits1(&gb);

            /* unknown field */
            skip_bits1(&gb);

            while (get_bits1(&gb)) {
                skip_bits(&gb, 8);
            }

            svq3->unknown_flag = get_bits1(&gb);
            avctx->has_b_frames = !s->low_delay;
            if (svq3->unknown_flag) {
#if CONFIG_ZLIB
                unsigned watermark_width  = svq3_get_ue_golomb(&gb);
                unsigned watermark_height = svq3_get_ue_golomb(&gb);
                int u1 = svq3_get_ue_golomb(&gb);
                int u2 = get_bits(&gb, 8);
                int u3 = get_bits(&gb, 2);
                int u4 = svq3_get_ue_golomb(&gb);
                unsigned long buf_len = watermark_width*watermark_height*4;
                int offset = (get_bits_count(&gb)+7)>>3;
                uint8_t *buf;

                if ((uint64_t)watermark_width*4 > UINT_MAX/watermark_height)
                    return -1;

                buf = av_malloc(buf_len);
                av_log(avctx, AV_LOG_DEBUG, "watermark size: %dx%d\n", watermark_width, watermark_height);
                av_log(avctx, AV_LOG_DEBUG, "u1: %x u2: %x u3: %x compressed data size: %d offset: %d\n", u1, u2, u3, u4, offset);
                if (uncompress(buf, &buf_len, extradata + 8 + offset, size - offset) != Z_OK) {
                    av_log(avctx, AV_LOG_ERROR, "could not uncompress watermark logo\n");
                    av_free(buf);
                    return -1;
                }
                svq3->watermark_key = ff_svq1_packet_checksum(buf, buf_len, 0);
                svq3->watermark_key = svq3->watermark_key << 16 | svq3->watermark_key;
                av_log(avctx, AV_LOG_DEBUG, "watermark key %#x\n", svq3->watermark_key);
                av_free(buf);
#else
                av_log(avctx, AV_LOG_ERROR, "this svq3 file contains watermark which need zlib support compiled in\n");
                return -1;
#endif
            }
        }
    }

    return 0;
}

static int svq3_decode_frame(AVCodecContext *avctx,
                             void *data, int *data_size,
                             AVPacket *avpkt)
{
    SVQ3Context *svq3 = avctx->priv_data;
    H264Context *h = &svq3->h;
    MpegEncContext *s = &h->s;
    int buf_size = avpkt->size;
    int m, mb_type, left;
    uint8_t *buf;

    /* special case for last picture */
    if (buf_size == 0) {
        if (s->next_picture_ptr && !s->low_delay) {
            *(AVFrame *) data = *(AVFrame *) &s->next_picture;
            s->next_picture_ptr = NULL;
            *data_size = sizeof(AVFrame);
        }
        return 0;
    }

    s->mb_x = s->mb_y = h->mb_xy = 0;

    if (svq3->watermark_key) {
        av_fast_malloc(&svq3->buf, &svq3->buf_size,
                       buf_size+FF_INPUT_BUFFER_PADDING_SIZE);
        if (!svq3->buf)
            return AVERROR(ENOMEM);
        memcpy(svq3->buf, avpkt->data, buf_size);
        buf = svq3->buf;
    } else {
        buf = avpkt->data;
    }

    init_get_bits(&s->gb, buf, 8*buf_size);

    if (svq3_decode_slice_header(avctx))
        return -1;

    s->pict_type = h->slice_type;
    s->picture_number = h->slice_num;

    if (avctx->debug&FF_DEBUG_PICT_INFO){
        av_log(h->s.avctx, AV_LOG_DEBUG, "%c hpel:%d, tpel:%d aqp:%d qp:%d, slice_num:%02X\n",
               av_get_picture_type_char(s->pict_type), svq3->halfpel_flag, svq3->thirdpel_flag,
               s->adaptive_quant, s->qscale, h->slice_num);
    }

    /* for skipping the frame */
    s->current_picture.pict_type = s->pict_type;
    s->current_picture.key_frame = (s->pict_type == AV_PICTURE_TYPE_I);

    /* Skip B-frames if we do not have reference frames. */
    if (s->last_picture_ptr == NULL && s->pict_type == AV_PICTURE_TYPE_B)
        return 0;
    if (  (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type == AV_PICTURE_TYPE_B)
        ||(avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type != AV_PICTURE_TYPE_I)
        || avctx->skip_frame >= AVDISCARD_ALL)
        return 0;

    if (s->next_p_frame_damaged) {
        if (s->pict_type == AV_PICTURE_TYPE_B)
            return 0;
        else
            s->next_p_frame_damaged = 0;
    }

    if (ff_h264_frame_start(h) < 0)
        return -1;

    if (s->pict_type == AV_PICTURE_TYPE_B) {
        h->frame_num_offset = (h->slice_num - h->prev_frame_num);

        if (h->frame_num_offset < 0) {
            h->frame_num_offset += 256;
        }
        if (h->frame_num_offset == 0 || h->frame_num_offset >= h->prev_frame_num_offset) {
            av_log(h->s.avctx, AV_LOG_ERROR, "error in B-frame picture id\n");
            return -1;
        }
    } else {
        h->prev_frame_num = h->frame_num;
        h->frame_num = h->slice_num;
        h->prev_frame_num_offset = (h->frame_num - h->prev_frame_num);

        if (h->prev_frame_num_offset < 0) {
            h->prev_frame_num_offset += 256;
        }
    }

    for (m = 0; m < 2; m++){
        int i;
        for (i = 0; i < 4; i++){
            int j;
            for (j = -1; j < 4; j++)
                h->ref_cache[m][scan8[0] + 8*i + j]= 1;
            if (i < 3)
                h->ref_cache[m][scan8[0] + 8*i + j]= PART_NOT_AVAILABLE;
        }
    }

    for (s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {
        for (s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {
            h->mb_xy = s->mb_x + s->mb_y*s->mb_stride;

            if ( (get_bits_count(&s->gb) + 7) >= s->gb.size_in_bits &&
                ((get_bits_count(&s->gb) & 7) == 0 || show_bits(&s->gb, (-get_bits_count(&s->gb) & 7)) == 0)) {

                skip_bits(&s->gb, svq3->next_slice_index - get_bits_count(&s->gb));
                s->gb.size_in_bits = 8*buf_size;

                if (svq3_decode_slice_header(avctx))
                    return -1;


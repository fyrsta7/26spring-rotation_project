            return AVERROR_INVALIDDATA;
        }

        if ((v->sprite_width&1) || (v->sprite_height&1)) {
            avpriv_request_sample(avctx, "odd sprites support");
            return AVERROR_PATCHWELCOME;
        }
    }
    return 0;
}

/** Close a VC1/WMV3 decoder
 * @warning Initial try at using MpegEncContext stuff
 */
av_cold int ff_vc1_decode_end(AVCodecContext *avctx)
{
    VC1Context *v = avctx->priv_data;
    int i;

    av_frame_free(&v->sprite_output_frame);

    for (i = 0; i < 4; i++)
        av_freep(&v->sr_rows[i >> 1][i & 1]);
    ff_mpv_common_end(&v->s);
    av_freep(&v->mv_type_mb_plane);
    av_freep(&v->direct_mb_plane);
    av_freep(&v->forward_mb_plane);
    av_freep(&v->fieldtx_plane);
    av_freep(&v->acpred_plane);
    av_freep(&v->over_flags_plane);
    av_freep(&v->mb_type_base);
    av_freep(&v->blk_mv_type_base);
    av_freep(&v->mv_f_base);
    av_freep(&v->mv_f_next_base);
    av_freep(&v->block);
    av_freep(&v->cbp_base);
    av_freep(&v->ttblk_base);
    av_freep(&v->is_intra_base); // FIXME use v->mb_type[]
    av_freep(&v->luma_mv_base);
    ff_intrax8_common_end(&v->x8);
    return 0;
}


/** Decode a VC1/WMV3 frame
 * @todo TODO: Handle VC-1 IDUs (Transport level?)
 */
static int vc1_decode_frame(AVCodecContext *avctx, AVFrame *pict,
                            int *got_frame, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size, n_slices = 0, i, ret;
    VC1Context *v = avctx->priv_data;
    MpegEncContext *s = &v->s;
    uint8_t *buf2 = NULL;
    const uint8_t *buf_start = buf, *buf_start_second_field = NULL;
    int mb_height, n_slices1=-1;
    struct {
        uint8_t *buf;
        GetBitContext gb;
        int mby_start;
        const uint8_t *rawbuf;
        int raw_size;
    } *slices = NULL, *tmp;
    unsigned slices_allocated = 0;

    v->second_field = 0;

    if(s->avctx->flags & AV_CODEC_FLAG_LOW_DELAY)
        s->low_delay = 1;

    /* no supplementary picture */
    if (buf_size == 0 || (buf_size == 4 && AV_RB32(buf) == VC1_CODE_ENDOFSEQ)) {
        /* special case for last picture */
        if (s->low_delay == 0 && s->next_picture_ptr) {
            if ((ret = av_frame_ref(pict, s->next_picture_ptr->f)) < 0)
                return ret;
            s->next_picture_ptr = NULL;

            *got_frame = 1;
        }

        return buf_size;
    }

    //for advanced profile we may need to parse and unescape data
    if (avctx->codec_id == AV_CODEC_ID_VC1 || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {
        int buf_size2 = 0;
        size_t next_allocated = 0;
        buf2 = av_mallocz(buf_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!buf2)
            return AVERROR(ENOMEM);

        if (IS_MARKER(AV_RB32(buf))) { /* frame starts with marker and needs to be parsed */
            const uint8_t *start, *end, *next;
            int size;

            next = buf;
            for (start = buf, end = buf + buf_size; next < end; start = next) {
                next = find_next_marker(start + 4, end);
                size = next - start - 4;
                if (size <= 0) continue;
                switch (AV_RB32(start)) {
                case VC1_CODE_FRAME:
                    if (avctx->hwaccel)
                        buf_start = start;
                    buf_size2 = v->vc1dsp.vc1_unescape_buffer(start + 4, size, buf2);
                    break;
                case VC1_CODE_FIELD: {
                    int buf_size3;
                    if (avctx->hwaccel)
                        buf_start_second_field = start;
                    av_size_mult(sizeof(*slices), n_slices+1, &next_allocated);
                    tmp = next_allocated ? av_fast_realloc(slices, &slices_allocated, next_allocated) : NULL;
                    if (!tmp) {
                        ret = AVERROR(ENOMEM);
                        goto err;
                    }
                    slices = tmp;
                    slices[n_slices].buf = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);
                    if (!slices[n_slices].buf) {
                        ret = AVERROR(ENOMEM);
                        goto err;
                    }
                    buf_size3 = v->vc1dsp.vc1_unescape_buffer(start + 4, size,
                                                              slices[n_slices].buf);
                    init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                                  buf_size3 << 3);
                    slices[n_slices].mby_start = avctx->coded_height + 31 >> 5;
                    slices[n_slices].rawbuf = start;
                    slices[n_slices].raw_size = size + 4;
                    n_slices1 = n_slices - 1; // index of the last slice of the first field
                    n_slices++;
                    break;
                }
                case VC1_CODE_ENTRYPOINT: /* it should be before frame data */
                    buf_size2 = v->vc1dsp.vc1_unescape_buffer(start + 4, size, buf2);
                    init_get_bits(&s->gb, buf2, buf_size2 * 8);
                    ff_vc1_decode_entry_point(avctx, v, &s->gb);
                    break;
                case VC1_CODE_SLICE: {
                    int buf_size3;
                    av_size_mult(sizeof(*slices), n_slices+1, &next_allocated);
                    tmp = next_allocated ? av_fast_realloc(slices, &slices_allocated, next_allocated) : NULL;
                    if (!tmp) {
                        ret = AVERROR(ENOMEM);
                        goto err;
                    }
                    slices = tmp;
                    slices[n_slices].buf = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);
                    if (!slices[n_slices].buf) {
                        ret = AVERROR(ENOMEM);
                        goto err;
                    }
                    buf_size3 = v->vc1dsp.vc1_unescape_buffer(start + 4, size,
                                                              slices[n_slices].buf);
                    init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                                  buf_size3 << 3);
                    slices[n_slices].mby_start = get_bits(&slices[n_slices].gb, 9);
                    slices[n_slices].rawbuf = start;
                    slices[n_slices].raw_size = size + 4;
                    n_slices++;
                    break;
                }
                }
            }
        } else if (v->interlace && ((buf[0] & 0xC0) == 0xC0)) { /* WVC1 interlaced stores both fields divided by marker */
            const uint8_t *divider;
            int buf_size3;

            divider = find_next_marker(buf, buf + buf_size);
            if ((divider == (buf + buf_size)) || AV_RB32(divider) != VC1_CODE_FIELD) {
                av_log(avctx, AV_LOG_ERROR, "Error in WVC1 interlaced frame\n");
                ret = AVERROR_INVALIDDATA;
                goto err;
            } else { // found field marker, unescape second field
                if (avctx->hwaccel)
                    buf_start_second_field = divider;
                av_size_mult(sizeof(*slices), n_slices+1, &next_allocated);
                tmp = next_allocated ? av_fast_realloc(slices, &slices_allocated, next_allocated) : NULL;
                if (!tmp) {
                    ret = AVERROR(ENOMEM);
                    goto err;
                }
                slices = tmp;
                slices[n_slices].buf = av_mallocz(buf_size + AV_INPUT_BUFFER_PADDING_SIZE);
                if (!slices[n_slices].buf) {
                    ret = AVERROR(ENOMEM);
                    goto err;
                }
                buf_size3 = v->vc1dsp.vc1_unescape_buffer(divider + 4, buf + buf_size - divider - 4, slices[n_slices].buf);
                init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                              buf_size3 << 3);
                slices[n_slices].mby_start = s->mb_height + 1 >> 1;
                slices[n_slices].rawbuf = divider;
                slices[n_slices].raw_size = buf + buf_size - divider;
                n_slices1 = n_slices - 1;
                n_slices++;
            }
            buf_size2 = v->vc1dsp.vc1_unescape_buffer(buf, divider - buf, buf2);
        } else {
            buf_size2 = v->vc1dsp.vc1_unescape_buffer(buf, buf_size, buf2);
        }
        init_get_bits(&s->gb, buf2, buf_size2*8);
    } else{
        ret = init_get_bits8(&s->gb, buf, buf_size);
        if (ret < 0)
            return ret;
    }

    if (v->res_sprite) {
        v->new_sprite  = !get_bits1(&s->gb);
        v->two_sprites =  get_bits1(&s->gb);
        /* res_sprite means a Windows Media Image stream, AV_CODEC_ID_*IMAGE means
           we're using the sprite compositor. These are intentionally kept separate
           so you can get the raw sprites by using the wmv3 decoder for WMVP or
           the vc1 one for WVP2 */
        if (avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {
            if (v->new_sprite) {
                // switch AVCodecContext parameters to those of the sprites
                avctx->width  = avctx->coded_width  = v->sprite_width;
                avctx->height = avctx->coded_height = v->sprite_height;
            } else {
                goto image;
            }
        }
    }

    if (s->context_initialized &&
        (s->width  != avctx->coded_width ||
         s->height != avctx->coded_height)) {
        ff_vc1_decode_end(avctx);
    }

    if (!s->context_initialized) {
        ret = ff_vc1_decode_init(avctx);
        if (ret < 0)
            goto err;

        s->low_delay = !avctx->has_b_frames || v->res_sprite;

        if (v->profile == PROFILE_ADVANCED) {
            if(avctx->coded_width<=1 || avctx->coded_height<=1) {
                ret = AVERROR_INVALIDDATA;
                goto err;
            }
            s->h_edge_pos = avctx->coded_width;
            s->v_edge_pos = avctx->coded_height;
        }
    }

    // do parse frame header
    v->pic_header_flag = 0;
    v->first_pic_header_flag = 1;
    if (v->profile < PROFILE_ADVANCED) {
        if ((ret = ff_vc1_parse_frame_header(v, &s->gb)) < 0) {
            goto err;
        }
    } else {
        if ((ret = ff_vc1_parse_frame_header_adv(v, &s->gb)) < 0) {
            goto err;
        }
    }
    v->first_pic_header_flag = 0;

    if (avctx->debug & FF_DEBUG_PICT_INFO)
        av_log(v->s.avctx, AV_LOG_DEBUG, "pict_type: %c\n", av_get_picture_type_char(s->pict_type));

    if ((avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE)
        && s->pict_type != AV_PICTURE_TYPE_I) {
        av_log(v->s.avctx, AV_LOG_ERROR, "Sprite decoder: expected I-frame\n");
        ret = AVERROR_INVALIDDATA;
        goto err;
    }
    if ((avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE)
        && v->field_mode) {
        av_log(v->s.avctx, AV_LOG_ERROR, "Sprite decoder: expected Frames not Fields\n");
        ret = AVERROR_INVALIDDATA;
        goto err;
    }
    if ((s->mb_height >> v->field_mode) == 0) {
        av_log(v->s.avctx, AV_LOG_ERROR, "image too short\n");
        ret = AVERROR_INVALIDDATA;
        goto err;
    }

    // for skipping the frame
    s->current_picture.f->pict_type = s->pict_type;
    s->current_picture.f->key_frame = s->pict_type == AV_PICTURE_TYPE_I;

    /* skip B-frames if we don't have reference frames */
    if (!s->last_picture_ptr && s->pict_type == AV_PICTURE_TYPE_B) {
        av_log(v->s.avctx, AV_LOG_DEBUG, "Skipping B frame without reference frames\n");
        goto end;
    }
    if ((avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type == AV_PICTURE_TYPE_B) ||
        (avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type != AV_PICTURE_TYPE_I) ||
         avctx->skip_frame >= AVDISCARD_ALL) {
        goto end;
    }

    if ((ret = ff_mpv_frame_start(s, avctx)) < 0) {
        goto err;
    }

    v->s.current_picture_ptr->field_picture = v->field_mode;
    v->s.current_picture_ptr->f->interlaced_frame = (v->fcm != PROGRESSIVE);
    v->s.current_picture_ptr->f->top_field_first  = v->tff;

    // process pulldown flags
    s->current_picture_ptr->f->repeat_pict = 0;
    // Pulldown flags are only valid when 'broadcast' has been set.
    // So ticks_per_frame will be 2
    if (v->rff) {
        // repeat field
        s->current_picture_ptr->f->repeat_pict = 1;
    } else if (v->rptfrm) {
        // repeat frames
        s->current_picture_ptr->f->repeat_pict = v->rptfrm * 2;
    }

    if (avctx->hwaccel) {
        s->mb_y = 0;
        if (v->field_mode && buf_start_second_field) {
            // decode first field
            s->picture_structure = PICT_BOTTOM_FIELD - v->tff;
            if ((ret = avctx->hwaccel->start_frame(avctx, buf_start, buf_start_second_field - buf_start)) < 0)
                goto err;

            if (n_slices1 == -1) {
                // no slices, decode the field as-is
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start, buf_start_second_field - buf_start)) < 0)
                    goto err;
            } else {
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start, slices[0].rawbuf - buf_start)) < 0)
                    goto err;

                for (i = 0 ; i < n_slices1 + 1; i++) {
                    s->gb = slices[i].gb;
                    s->mb_y = slices[i].mby_start;

                    v->pic_header_flag = get_bits1(&s->gb);
                    if (v->pic_header_flag) {
                        if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                            av_log(v->s.avctx, AV_LOG_ERROR, "Slice header damaged\n");
                            ret = AVERROR_INVALIDDATA;
                            if (avctx->err_recognition & AV_EF_EXPLODE)
                                goto err;
                            continue;
                        }
                    }

                    if ((ret = avctx->hwaccel->decode_slice(avctx, slices[i].rawbuf, slices[i].raw_size)) < 0)
                        goto err;
                }
            }

            if ((ret = avctx->hwaccel->end_frame(avctx)) < 0)
                goto err;

            // decode second field
            s->gb = slices[n_slices1 + 1].gb;
            s->mb_y = slices[n_slices1 + 1].mby_start;
            s->picture_structure = PICT_TOP_FIELD + v->tff;
            v->second_field = 1;
            v->pic_header_flag = 0;
            if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                av_log(avctx, AV_LOG_ERROR, "parsing header for second field failed");
                ret = AVERROR_INVALIDDATA;
                goto err;
            }
            v->s.current_picture_ptr->f->pict_type = v->s.pict_type;

            if ((ret = avctx->hwaccel->start_frame(avctx, buf_start_second_field, (buf + buf_size) - buf_start_second_field)) < 0)
                goto err;

            if (n_slices - n_slices1 == 2) {
                // no slices, decode the field as-is
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start_second_field, (buf + buf_size) - buf_start_second_field)) < 0)
                    goto err;
            } else {
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start_second_field, slices[n_slices1 + 2].rawbuf - buf_start_second_field)) < 0)
                    goto err;

                for (i = n_slices1 + 2; i < n_slices; i++) {
                    s->gb = slices[i].gb;
                    s->mb_y = slices[i].mby_start;

                    v->pic_header_flag = get_bits1(&s->gb);
                    if (v->pic_header_flag) {
                        if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                            av_log(v->s.avctx, AV_LOG_ERROR, "Slice header damaged\n");
                            ret = AVERROR_INVALIDDATA;
                            if (avctx->err_recognition & AV_EF_EXPLODE)
                                goto err;
                            continue;
                        }
                    }

                    if ((ret = avctx->hwaccel->decode_slice(avctx, slices[i].rawbuf, slices[i].raw_size)) < 0)
                        goto err;
                }
            }

            if ((ret = avctx->hwaccel->end_frame(avctx)) < 0)
                goto err;
        } else {
            s->picture_structure = PICT_FRAME;
            if ((ret = avctx->hwaccel->start_frame(avctx, buf_start, (buf + buf_size) - buf_start)) < 0)
                goto err;

            if (n_slices == 0) {
                // no slices, decode the frame as-is
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start, (buf + buf_size) - buf_start)) < 0)
                    goto err;
            } else {
                // decode the frame part as the first slice
                if ((ret = avctx->hwaccel->decode_slice(avctx, buf_start, slices[0].rawbuf - buf_start)) < 0)
                    goto err;

                // and process the slices as additional slices afterwards
                for (i = 0 ; i < n_slices; i++) {
                    s->gb = slices[i].gb;
                    s->mb_y = slices[i].mby_start;

                    v->pic_header_flag = get_bits1(&s->gb);
                    if (v->pic_header_flag) {
                        if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                            av_log(v->s.avctx, AV_LOG_ERROR, "Slice header damaged\n");
                            ret = AVERROR_INVALIDDATA;
                            if (avctx->err_recognition & AV_EF_EXPLODE)
                                goto err;
                            continue;
                        }
                    }

                    if ((ret = avctx->hwaccel->decode_slice(avctx, slices[i].rawbuf, slices[i].raw_size)) < 0)
                        goto err;
                }
            }
            if ((ret = avctx->hwaccel->end_frame(avctx)) < 0)
                goto err;
        }
    } else {
        int header_ret = 0;

        ff_mpeg_er_frame_start(s);

        v->end_mb_x = s->mb_width;
        if (v->field_mode) {
            s->current_picture.f->linesize[0] <<= 1;
            s->current_picture.f->linesize[1] <<= 1;
            s->current_picture.f->linesize[2] <<= 1;
            s->linesize                      <<= 1;
            s->uvlinesize                    <<= 1;
        }
        mb_height = s->mb_height >> v->field_mode;

        av_assert0 (mb_height > 0);

        for (i = 0; i <= n_slices; i++) {
            if (i > 0 &&  slices[i - 1].mby_start >= mb_height) {
                if (v->field_mode <= 0) {
                    av_log(v->s.avctx, AV_LOG_ERROR, "Slice %d starts beyond "
                           "picture boundary (%d >= %d)\n", i,
                           slices[i - 1].mby_start, mb_height);
                    continue;
                }
                v->second_field = 1;
                av_assert0((s->mb_height & 1) == 0);
                v->blocks_off   = s->b8_stride * (s->mb_height&~1);
                v->mb_off       = s->mb_stride * s->mb_height >> 1;
            } else {
                v->second_field = 0;
                v->blocks_off   = 0;
                v->mb_off       = 0;
            }
            if (i) {
                v->pic_header_flag = 0;
                if (v->field_mode && i == n_slices1 + 2) {
                    if ((header_ret = ff_vc1_parse_frame_header_adv(v, &s->gb)) < 0) {
                        av_log(v->s.avctx, AV_LOG_ERROR, "Field header damaged\n");
                        ret = AVERROR_INVALIDDATA;
                        if (avctx->err_recognition & AV_EF_EXPLODE)
                            goto err;
                        continue;
                    }
                } else if (get_bits1(&s->gb)) {
                    v->pic_header_flag = 1;
                    if ((header_ret = ff_vc1_parse_frame_header_adv(v, &s->gb)) < 0) {
                        av_log(v->s.avctx, AV_LOG_ERROR, "Slice header damaged\n");
                        ret = AVERROR_INVALIDDATA;
                        if (avctx->err_recognition & AV_EF_EXPLODE)
                            goto err;
                        continue;
                    }
                }
            }
            if (header_ret < 0)
                continue;
            s->start_mb_y = (i == 0) ? 0 : FFMAX(0, slices[i-1].mby_start % mb_height);
            if (!v->field_mode || v->second_field)
                s->end_mb_y = (i == n_slices     ) ? mb_height : FFMIN(mb_height, slices[i].mby_start % mb_height);
            else {
                if (i >= n_slices) {
                    av_log(v->s.avctx, AV_LOG_ERROR, "first field slice count too large\n");
                    continue;
                }
                s->end_mb_y = (i == n_slices1 + 1) ? mb_height : FFMIN(mb_height, slices[i].mby_start % mb_height);
            }
            if (s->end_mb_y <= s->start_mb_y) {
                av_log(v->s.avctx, AV_LOG_ERROR, "end mb y %d %d invalid\n", s->end_mb_y, s->start_mb_y);
                continue;
            }
            if (((s->pict_type == AV_PICTURE_TYPE_P && !v->p_frame_skipped) ||
                 (s->pict_type == AV_PICTURE_TYPE_B && !v->bi_type)) &&
                !v->cbpcy_vlc) {
                av_log(v->s.avctx, AV_LOG_ERROR, "missing cbpcy_vlc\n");
                continue;
            }
            ff_vc1_decode_blocks(v);
            if (i != n_slices) {
                s->gb = slices[i].gb;
            }
        }
        if (v->field_mode) {
            v->second_field = 0;
            s->current_picture.f->linesize[0] >>= 1;
            s->current_picture.f->linesize[1] >>= 1;
            s->current_picture.f->linesize[2] >>= 1;
            s->linesize                      >>= 1;
            s->uvlinesize                    >>= 1;
            if (v->s.pict_type != AV_PICTURE_TYPE_BI && v->s.pict_type != AV_PICTURE_TYPE_B) {
                FFSWAP(uint8_t *, v->mv_f_next[0], v->mv_f[0]);
                FFSWAP(uint8_t *, v->mv_f_next[1], v->mv_f[1]);
            }
        }
        ff_dlog(s->avctx, "Consumed %i/%i bits\n",
                get_bits_count(&s->gb), s->gb.size_in_bits);
//  if (get_bits_count(&s->gb) > buf_size * 8)
//      return -1;
        if(s->er.error_occurred && s->pict_type == AV_PICTURE_TYPE_B) {
            ret = AVERROR_INVALIDDATA;
            goto err;
        }
        if (   !v->field_mode
            && avctx->codec_id != AV_CODEC_ID_WMV3IMAGE
            && avctx->codec_id != AV_CODEC_ID_VC1IMAGE)
            ff_er_frame_end(&s->er);
    }


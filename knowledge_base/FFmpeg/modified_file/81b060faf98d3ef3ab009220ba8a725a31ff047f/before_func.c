        ti1 = 0.01;

    if (verbose || is_last_report) {
        bitrate = (double)(total_size * 8) / ti1 / 1000.0;

        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf),
            "size=%8.0fkB time=%0.2f bitrate=%6.1fkbits/s",
            (double)total_size / 1024, ti1, bitrate);

        if (verbose > 1)
          snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " dup=%d drop=%d",
                  nb_frames_dup, nb_frames_drop);

        if (verbose >= 0)
            fprintf(stderr, "%s    \r", buf);

        fflush(stderr);
    }

    if (is_last_report && verbose >= 0){
        int64_t raw= audio_size + video_size + extra_size;
        fprintf(stderr, "\n");
        fprintf(stderr, "video:%1.0fkB audio:%1.0fkB global headers:%1.0fkB muxing overhead %f%%\n",
                video_size/1024.0,
                audio_size/1024.0,
                extra_size/1024.0,
                100.0*(total_size - raw)/raw
        );
    }
}

/* pkt = NULL means EOF (needed to flush decoder buffers) */
static int output_packet(AVInputStream *ist, int ist_index,
                         AVOutputStream **ost_table, int nb_ostreams,
                         const AVPacket *pkt)
{
    AVFormatContext *os;
    AVOutputStream *ost;
    uint8_t *ptr;
    int len, ret, i;
    uint8_t *data_buf;
    int data_size, got_picture;
    AVFrame picture;
    void *buffer_to_free;
    static unsigned int samples_size= 0;
    static short *samples= NULL;
    AVSubtitle subtitle, *subtitle_to_free;
    int got_subtitle;

    if(ist->next_pts == AV_NOPTS_VALUE)
        ist->next_pts= ist->pts;

    if (pkt == NULL) {
        /* EOF handling */
        ptr = NULL;
        len = 0;
        goto handle_eof;
    }

    if(pkt->dts != AV_NOPTS_VALUE)
        ist->next_pts = ist->pts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);

    len = pkt->size;
    ptr = pkt->data;

    //while we have more to decode or while the decoder did output something on EOF
    while (len > 0 || (!pkt && ist->next_pts != ist->pts)) {
    handle_eof:
        ist->pts= ist->next_pts;

        if(len && len != pkt->size && verbose>0)
            fprintf(stderr, "Multiple frames in a packet from stream %d\n", pkt->stream_index);

        /* decode the packet if needed */
        data_buf = NULL; /* fail safe */
        data_size = 0;
        subtitle_to_free = NULL;
        if (ist->decoding_needed) {
            switch(ist->st->codec->codec_type) {
            case CODEC_TYPE_AUDIO:{
                if(pkt)
                    samples= av_fast_realloc(samples, &samples_size, FFMAX(pkt->size*sizeof(*samples), AVCODEC_MAX_AUDIO_FRAME_SIZE));
                data_size= samples_size;
                    /* XXX: could avoid copy if PCM 16 bits with same
                       endianness as CPU */
                ret = avcodec_decode_audio2(ist->st->codec, samples, &data_size,
                                           ptr, len);
                if (ret < 0)
                    goto fail_decode;
                ptr += ret;
                len -= ret;
                /* Some bug in mpeg audio decoder gives */
                /* data_size < 0, it seems they are overflows */
                if (data_size <= 0) {
                    /* no audio frame */
                    continue;
                }
                data_buf = (uint8_t *)samples;
                ist->next_pts += ((int64_t)AV_TIME_BASE/2 * data_size) /
                    (ist->st->codec->sample_rate * ist->st->codec->channels);
                break;}
            case CODEC_TYPE_VIDEO:
                    data_size = (ist->st->codec->width * ist->st->codec->height * 3) / 2;
                    /* XXX: allocate picture correctly */
                    avcodec_get_frame_defaults(&picture);

                    ret = avcodec_decode_video(ist->st->codec,
                                               &picture, &got_picture, ptr, len);
                    ist->st->quality= picture.quality;
                    if (ret < 0)
                        goto fail_decode;
                    if (!got_picture) {
                        /* no picture yet */
                        goto discard_packet;
                    }
                    if (ist->st->codec->time_base.num != 0) {
                        ist->next_pts += ((int64_t)AV_TIME_BASE *
                                          ist->st->codec->time_base.num) /
                            ist->st->codec->time_base.den;
                    }
                    len = 0;
                    break;
            case CODEC_TYPE_SUBTITLE:
                ret = avcodec_decode_subtitle(ist->st->codec,
                                              &subtitle, &got_subtitle, ptr, len);
                if (ret < 0)
                    goto fail_decode;
                if (!got_subtitle) {
                    goto discard_packet;
                }
                subtitle_to_free = &subtitle;
                len = 0;
                break;
            default:
                goto fail_decode;
            }
        } else {
            switch(ist->st->codec->codec_type) {
            case CODEC_TYPE_AUDIO:
                ist->next_pts += ((int64_t)AV_TIME_BASE * ist->st->codec->frame_size) /
                    ist->st->codec->sample_rate;
                break;
            case CODEC_TYPE_VIDEO:
                if (ist->st->codec->time_base.num != 0) {
                    ist->next_pts += ((int64_t)AV_TIME_BASE *
                                      ist->st->codec->time_base.num) /
                        ist->st->codec->time_base.den;
                }
                break;
            }
            data_buf = ptr;
            data_size = len;
            ret = len;
            len = 0;
        }

        buffer_to_free = NULL;
        if (ist->st->codec->codec_type == CODEC_TYPE_VIDEO) {
            pre_process_video_frame(ist, (AVPicture *)&picture,
                                    &buffer_to_free);
        }

        // preprocess audio (volume)
        if (ist->st->codec->codec_type == CODEC_TYPE_AUDIO) {
            if (audio_volume != 256) {
                short *volp;
                volp = samples;
                for(i=0;i<(data_size / sizeof(short));i++) {
                    int v = ((*volp) * audio_volume + 128) >> 8;
                    if (v < -32768) v = -32768;
                    if (v >  32767) v = 32767;
                    *volp++ = v;
                }
            }
        }

        /* frame rate emulation */
        if (ist->st->codec->rate_emu) {
            int64_t pts = av_rescale((int64_t) ist->frame * ist->st->codec->time_base.num, 1000000, ist->st->codec->time_base.den);
            int64_t now = av_gettime() - ist->start;
            if (pts > now)
                usleep(pts - now);

            ist->frame++;
        }

#if 0
        /* mpeg PTS deordering : if it is a P or I frame, the PTS
           is the one of the next displayed one */
        /* XXX: add mpeg4 too ? */
        if (ist->st->codec->codec_id == CODEC_ID_MPEG1VIDEO) {
            if (ist->st->codec->pict_type != B_TYPE) {
                int64_t tmp;
                tmp = ist->last_ip_pts;
                ist->last_ip_pts  = ist->frac_pts.val;
                ist->frac_pts.val = tmp;
            }
        }
#endif
        /* if output time reached then transcode raw format,
           encode packets and output them */
        if (start_time == 0 || ist->pts >= start_time)
            for(i=0;i<nb_ostreams;i++) {
                int frame_size;

                ost = ost_table[i];
                if (ost->source_index == ist_index) {
                    os = output_files[ost->file_index];

#if 0
                    printf("%d: got pts=%0.3f %0.3f\n", i,
                           (double)pkt->pts / AV_TIME_BASE,
                           ((double)ist->pts / AV_TIME_BASE) -
                           ((double)ost->st->pts.val * ost->st->time_base.num / ost->st->time_base.den));
#endif
                    /* set the input output pts pairs */
                    //ost->sync_ipts = (double)(ist->pts + input_files_ts_offset[ist->file_index] - start_time)/ AV_TIME_BASE;

                    if (ost->encoding_needed) {
                        switch(ost->st->codec->codec_type) {
                        case CODEC_TYPE_AUDIO:
                            do_audio_out(os, ost, ist, data_buf, data_size);
                            break;
                        case CODEC_TYPE_VIDEO:
                            do_video_out(os, ost, ist, &picture, &frame_size);
                            if (vstats_filename && frame_size)
                                do_video_stats(os, ost, frame_size);
                            break;
                        case CODEC_TYPE_SUBTITLE:
                            do_subtitle_out(os, ost, ist, &subtitle,
                                            pkt->pts);
                            break;
                        default:
                            abort();
                        }
                    } else {
                        AVFrame avframe; //FIXME/XXX remove this
                        AVPacket opkt;
                        av_init_packet(&opkt);

                        if (!ost->frame_number && !(pkt->flags & PKT_FLAG_KEY))
                            continue;

                        /* no reencoding needed : output the packet directly */
                        /* force the input stream PTS */

                        avcodec_get_frame_defaults(&avframe);
                        ost->st->codec->coded_frame= &avframe;
                        avframe.key_frame = pkt->flags & PKT_FLAG_KEY;

                        if(ost->st->codec->codec_type == CODEC_TYPE_AUDIO)
                            audio_size += data_size;
                        else if (ost->st->codec->codec_type == CODEC_TYPE_VIDEO) {
                            video_size += data_size;
                            ost->sync_opts++;
                        }

                        opkt.stream_index= ost->index;
                        if(pkt->pts != AV_NOPTS_VALUE)
                            opkt.pts= av_rescale_q(pkt->pts, ist->st->time_base, ost->st->time_base);
                        else
                            opkt.pts= AV_NOPTS_VALUE;

                        if (pkt->dts == AV_NOPTS_VALUE)
                            opkt.dts = av_rescale_q(ist->pts, AV_TIME_BASE_Q, ost->st->time_base);
                        else
                            opkt.dts = av_rescale_q(pkt->dts, ist->st->time_base, ost->st->time_base);

                        opkt.duration = av_rescale_q(pkt->duration, ist->st->time_base, ost->st->time_base);
                        opkt.flags= pkt->flags;

                        //FIXME remove the following 2 lines they shall be replaced by the bitstream filters
                        if(av_parser_change(ist->st->parser, ost->st->codec, &opkt.data, &opkt.size, data_buf, data_size, pkt->flags & PKT_FLAG_KEY))
                            opkt.destruct= av_destruct_packet;

                        write_frame(os, &opkt, ost->st->codec, bitstream_filters[ost->file_index][opkt.stream_index]);
                        ost->st->codec->frame_number++;
                        ost->frame_number++;
                        av_free_packet(&opkt);
                    }
                }
            }
        av_free(buffer_to_free);
        /* XXX: allocate the subtitles in the codec ? */
        if (subtitle_to_free) {
            if (subtitle_to_free->rects != NULL) {
                for (i = 0; i < subtitle_to_free->num_rects; i++) {
                    av_free(subtitle_to_free->rects[i].bitmap);
                    av_free(subtitle_to_free->rects[i].rgba_palette);
                }
                av_freep(&subtitle_to_free->rects);
            }
            subtitle_to_free->num_rects = 0;
            subtitle_to_free = NULL;
        }
    }
 discard_packet:
    if (pkt == NULL) {
        /* EOF handling */

        for(i=0;i<nb_ostreams;i++) {
            ost = ost_table[i];
            if (ost->source_index == ist_index) {
                AVCodecContext *enc= ost->st->codec;
                os = output_files[ost->file_index];

                if(ost->st->codec->codec_type == CODEC_TYPE_AUDIO && enc->frame_size <=1)
                    continue;
                if(ost->st->codec->codec_type == CODEC_TYPE_VIDEO && (os->oformat->flags & AVFMT_RAWPICTURE))
                    continue;

                if (ost->encoding_needed) {
                    for(;;) {
                        AVPacket pkt;
                        int fifo_bytes;
                        av_init_packet(&pkt);
                        pkt.stream_index= ost->index;

                        switch(ost->st->codec->codec_type) {
                        case CODEC_TYPE_AUDIO:
                            fifo_bytes = av_fifo_size(&ost->fifo);
                            ret = 0;
                            /* encode any samples remaining in fifo */
                            if(fifo_bytes > 0 && enc->codec->capabilities & CODEC_CAP_SMALL_LAST_FRAME) {
                                int fs_tmp = enc->frame_size;
                                enc->frame_size = fifo_bytes / (2 * enc->channels);
                                av_fifo_read(&ost->fifo, (uint8_t *)samples, fifo_bytes);
                                    ret = avcodec_encode_audio(enc, bit_buffer, bit_buffer_size, samples);
                                enc->frame_size = fs_tmp;
                            }
                            if(ret <= 0) {

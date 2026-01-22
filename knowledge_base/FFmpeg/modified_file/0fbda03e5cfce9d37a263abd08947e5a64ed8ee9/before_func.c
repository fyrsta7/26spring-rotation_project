            if (track->nb_frag_info >= track->frag_info_capacity) {
                unsigned new_capacity = track->nb_frag_info + MOV_FRAG_INFO_ALLOC_INCREMENT;
                if (av_reallocp_array(&track->frag_info,
                                      new_capacity,
                                      sizeof(*track->frag_info)))
                    return AVERROR(ENOMEM);
                track->frag_info_capacity = new_capacity;
            }
            info = &track->frag_info[track->nb_frag_info - 1];
            info->offset   = avio_tell(s->pb);
            info->time     = mov->tracks[i].frag_start;
            info->duration = duration;
            mov_write_tfrf_tags(s->pb, mov, track);

            mov_write_moof_tag(s->pb, mov, moof_tracks);
            info->tfrf_offset = track->tfrf_offset;
            mov->fragments++;

            avio_wb32(s->pb, mdat_size + 8);
            ffio_wfourcc(s->pb, "mdat");
        }

        if (track->entry)
            track->frag_start += duration;
        track->entry = 0;
        if (!track->mdat_buf)
            continue;
        buf_size = avio_close_dyn_buf(track->mdat_buf, &buf);
        track->mdat_buf = NULL;

        avio_write(s->pb, buf, buf_size);
        av_free(buf);
    }

    mov->mdat_size = 0;

    avio_flush(s->pb);
    return 0;
}

int ff_mov_write_packet(AVFormatContext *s, AVPacket *pkt)
{
    MOVMuxContext *mov = s->priv_data;
    AVIOContext *pb = s->pb;
    MOVTrack *trk = &mov->tracks[pkt->stream_index];
    AVCodecContext *enc = trk->enc;
    unsigned int samples_in_chunk = 0;
    int size = pkt->size;
    uint8_t *reformatted_data = NULL;

    if (mov->flags & FF_MOV_FLAG_FRAGMENT) {
        int ret;
        if (mov->fragments > 0) {
            if (!trk->mdat_buf) {
                if ((ret = avio_open_dyn_buf(&trk->mdat_buf)) < 0)
                    return ret;
            }
            pb = trk->mdat_buf;
        } else {
            if (!mov->mdat_buf) {
                if ((ret = avio_open_dyn_buf(&mov->mdat_buf)) < 0)
                    return ret;
            }
            pb = mov->mdat_buf;
        }
    }

    if (enc->codec_id == AV_CODEC_ID_AMR_NB) {
        /* We must find out how many AMR blocks there are in one packet */
        static uint16_t packed_size[16] =
            {13, 14, 16, 18, 20, 21, 27, 32, 6, 0, 0, 0, 0, 0, 0, 1};
        int len = 0;

        while (len < size && samples_in_chunk < 100) {
            len += packed_size[(pkt->data[len] >> 3) & 0x0F];
            samples_in_chunk++;
        }
        if (samples_in_chunk > 1) {
            av_log(s, AV_LOG_ERROR, "fatal error, input is not a single packet, implement a AVParser for it\n");
            return -1;
        }
    } else if (trk->sample_size)
        samples_in_chunk = size / trk->sample_size;
    else
        samples_in_chunk = 1;

    /* copy extradata if it exists */
    if (trk->vos_len == 0 && enc->extradata_size > 0) {
        trk->vos_len  = enc->extradata_size;
        trk->vos_data = av_malloc(trk->vos_len);
        memcpy(trk->vos_data, enc->extradata, trk->vos_len);
    }

    if (enc->codec_id == AV_CODEC_ID_H264 && trk->vos_len > 0 && *(uint8_t *)trk->vos_data != 1) {
        /* from x264 or from bytestream h264 */
        /* nal reformating needed */
        if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams) {
            ff_avc_parse_nal_units_buf(pkt->data, &reformatted_data,
                                       &size);
            avio_write(pb, reformatted_data, size);
        } else {
            size = ff_avc_parse_nal_units(pb, pkt->data, pkt->size);
        }
    } else {
        avio_write(pb, pkt->data, size);
    }

    if ((enc->codec_id == AV_CODEC_ID_DNXHD ||
         enc->codec_id == AV_CODEC_ID_AC3) && !trk->vos_len) {
        /* copy frame to create needed atoms */
        trk->vos_len  = size;
        trk->vos_data = av_malloc(size);
        if (!trk->vos_data)
            return AVERROR(ENOMEM);
        memcpy(trk->vos_data, pkt->data, size);
    }

    if (trk->entry >= trk->cluster_capacity) {
        unsigned new_capacity = 2 * (trk->entry + MOV_INDEX_CLUSTER_SIZE);
        if (av_reallocp_array(&trk->cluster, new_capacity,
                              sizeof(*trk->cluster)))
            return AVERROR(ENOMEM);
        trk->cluster_capacity = new_capacity;
    }

    trk->cluster[trk->entry].pos              = avio_tell(pb) - size;
    trk->cluster[trk->entry].samples_in_chunk = samples_in_chunk;
    trk->cluster[trk->entry].size             = size;
    trk->cluster[trk->entry].entries          = samples_in_chunk;
    trk->cluster[trk->entry].dts              = pkt->dts;
    if (!trk->entry && trk->start_dts != AV_NOPTS_VALUE) {
        /* First packet of a new fragment. We already wrote the duration

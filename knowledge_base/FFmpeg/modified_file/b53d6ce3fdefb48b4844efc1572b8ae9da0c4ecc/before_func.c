               pa_strerror(ret));
        return AVERROR(EIO);
    }
    /* take real parameters */
    st->codec->codec_type  = AVMEDIA_TYPE_AUDIO;
    st->codec->codec_id    = codec_id;
    st->codec->sample_rate = pd->sample_rate;
    st->codec->channels    = pd->channels;
    avpriv_set_pts_info(st, 64, 1, pd->sample_rate);  /* 64 bits pts in us */

    pd->pts = AV_NOPTS_VALUE;
    sample_bytes = (av_get_bits_per_sample(codec_id) >> 3) * pd->channels;

    if (pd->frame_size % sample_bytes) {
        av_log(s, AV_LOG_WARNING, "frame_size %i is not divisible by %i "
            "(channels * bytes_per_sample) \n", pd->frame_size, sample_bytes);
    }

    pd->frame_duration = pd->frame_size / sample_bytes;

    return 0;
}

static int pulse_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    PulseData *pd  = s->priv_data;
    int res;
    pa_usec_t latency;

    if (av_new_packet(pkt, pd->frame_size) < 0) {
        return AVERROR(ENOMEM);
    }


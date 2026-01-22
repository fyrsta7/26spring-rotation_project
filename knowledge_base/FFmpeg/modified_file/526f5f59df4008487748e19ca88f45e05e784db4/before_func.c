    case AV_CODEC_ID_VC1:
    case AV_CODEC_ID_VP8:
    case AV_CODEC_ID_VP9:
        st->need_parsing = AVSTREAM_PARSE_FULL;
        break;
    default:
        break;
    }
    return 0;
}

static int mov_skip_multiple_stsd(MOVContext *c, AVIOContext *pb,
                                  int codec_tag, int format,
                                  int64_t size)
{
    int video_codec_id = ff_codec_get_id(ff_codec_movvideo_tags, format);

    if (codec_tag &&
         (codec_tag != format &&
          // AVID 1:1 samples with differing data format and codec tag exist
          (codec_tag != AV_RL32("AV1x") || format != AV_RL32("AVup")) &&
          // prores is allowed to have differing data format and codec tag
          codec_tag != AV_RL32("apcn") && codec_tag != AV_RL32("apch") &&
          // so is dv (sigh)
          codec_tag != AV_RL32("dvpp") && codec_tag != AV_RL32("dvcp") &&
          (c->fc->video_codec_id ? video_codec_id != c->fc->video_codec_id
                                 : codec_tag != MKTAG('j','p','e','g')))) {
        /* Multiple fourcc, we skip JPEG. This is not correct, we should
         * export it as a separate AVStream but this needs a few changes
         * in the MOV demuxer, patch welcome. */

        av_log(c->fc, AV_LOG_WARNING, "multiple fourcc not supported\n");
        avio_skip(pb, size);
        return 1;
    }

    return 0;
}

int ff_mov_read_stsd_entries(MOVContext *c, AVIOContext *pb, int entries)
{
    AVStream *st;
    MOVStreamContext *sc;
    int pseudo_stream_id;

    av_assert0 (c->fc->nb_streams >= 1);
    st = c->fc->streams[c->fc->nb_streams-1];
    sc = st->priv_data;

    for (pseudo_stream_id = 0;
         pseudo_stream_id < entries && !pb->eof_reached;
         pseudo_stream_id++) {
        //Parsing Sample description table
        enum AVCodecID id;
        int ret, dref_id = 1;
        MOVAtom a = { AV_RL32("stsd") };
        int64_t start_pos = avio_tell(pb);
        int64_t size    = avio_rb32(pb); /* size */
        uint32_t format = avio_rl32(pb); /* data format */

        if (size >= 16) {
            avio_rb32(pb); /* reserved */
            avio_rb16(pb); /* reserved */
            dref_id = avio_rb16(pb);
        } else if (size <= 7) {
            av_log(c->fc, AV_LOG_ERROR,
                   "invalid size %"PRId64" in stsd\n", size);
            return AVERROR_INVALIDDATA;
        }

        if (mov_skip_multiple_stsd(c, pb, st->codecpar->codec_tag, format,
                                   size - (avio_tell(pb) - start_pos))) {
            sc->stsd_count++;
            continue;
        }

        sc->pseudo_stream_id = st->codecpar->codec_tag ? -1 : pseudo_stream_id;
        sc->dref_id= dref_id;
        sc->format = format;

        id = mov_codec_id(st, format);

        av_log(c->fc, AV_LOG_TRACE,
               "size=%"PRId64" 4CC=%s codec_type=%d\n", size,
               av_fourcc2str(format), st->codecpar->codec_type);

        st->codecpar->codec_id = id;
        if (st->codecpar->codec_type==AVMEDIA_TYPE_VIDEO) {
            mov_parse_stsd_video(c, pb, st, sc);
        } else if (st->codecpar->codec_type==AVMEDIA_TYPE_AUDIO) {
            mov_parse_stsd_audio(c, pb, st, sc);
            if (st->codecpar->sample_rate < 0) {
                av_log(c->fc, AV_LOG_ERROR, "Invalid sample rate %d\n", st->codecpar->sample_rate);
                return AVERROR_INVALIDDATA;
            }

                return ret;
            }
        }

        return mov_write_single_packet(s, pkt);
    }
}

// QuickTime chapters involve an additional text track with the chapter names
// as samples, and a tref pointing from the other tracks to the chapter one.
static int mov_create_chapter_track(AVFormatContext *s, int tracknum)
{
    AVIOContext *pb;

    MOVMuxContext *mov = s->priv_data;
    MOVTrack *track = &mov->tracks[tracknum];
    AVPacket pkt = { .stream_index = tracknum, .flags = AV_PKT_FLAG_KEY };
    int i, len;

    track->mode = mov->mode;
    track->tag = MKTAG('t','e','x','t');
    track->timescale = MOV_TIMESCALE;
    track->par = avcodec_parameters_alloc();
    if (!track->par)
        return AVERROR(ENOMEM);
    track->par->codec_type = AVMEDIA_TYPE_SUBTITLE;
#if 0
    // These properties are required to make QT recognize the chapter track
    uint8_t chapter_properties[43] = { 0, 0, 0, 0, 0, 0, 0, 1, };
    if (ff_alloc_extradata(track->par, sizeof(chapter_properties)))
        return AVERROR(ENOMEM);
    memcpy(track->par->extradata, chapter_properties, sizeof(chapter_properties));
#else

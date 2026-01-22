    mp3->frames = 0;
    mp3->header_filesize   = 0;

    mp3_parse_info_tag(s, st, &c, spf);
    mp3_parse_vbri_tag(s, st, base);

    if (!mp3->frames && !mp3->header_filesize)
        return -1;

    /* Skip the vbr tag frame */
    avio_seek(s->pb, base + vbrtag_size, SEEK_SET);

    if (mp3->frames)
        st->duration = av_rescale_q(mp3->frames, (AVRational){spf, c.sample_rate},
                                    st->time_base);
    if (mp3->header_filesize && mp3->frames && !mp3->is_cbr)
        st->codecpar->bit_rate = av_rescale(mp3->header_filesize, 8 * c.sample_rate, mp3->frames * (int64_t)spf);

    return 0;
}

static int mp3_read_header(AVFormatContext *s)
{
    FFFormatContext *const si = ffformatcontext(s);
    MP3DecContext *mp3 = s->priv_data;
    AVStream *st;
    FFStream *sti;
    int64_t off;
    int ret;
    int i;

    s->metadata = si->id3v2_meta;
    si->id3v2_meta = NULL;

    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    sti = ffstream(st);

    st->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    st->codecpar->codec_id = AV_CODEC_ID_MP3;
    sti->need_parsing = AVSTREAM_PARSE_FULL_RAW;
    st->start_time = 0;

    // lcm of all mp3 sample rates
    avpriv_set_pts_info(st, 64, 1, 14112000);

    ffiocontext(s->pb)->maxsize = -1;
    off = avio_tell(s->pb);

    if (!av_dict_count(s->metadata))
        ff_id3v1_read(s);

    if (s->pb->seekable & AVIO_SEEKABLE_NORMAL)
        mp3->filesize = avio_size(s->pb);

    if (mp3_parse_vbr_tags(s, st, off) < 0)
        avio_seek(s->pb, off, SEEK_SET);

    ret = ff_replaygain_export(st, s->metadata);
    if (ret < 0)
        return ret;

    off = avio_tell(s->pb);
    for (i = 0; i < 64 * 1024; i++) {
        uint32_t header, header2;
        int frame_size;
        if (!(i&1023))
            ffio_ensure_seekback(s->pb, i + 1024 + 4);
        frame_size = check(s->pb, off + i, &header);
        if (frame_size > 0) {
            ffio_ensure_seekback(s->pb, i + 1024 + frame_size + 4);
            ret = check(s->pb, off + i + frame_size, &header2);
            if (ret >= 0 &&
                (header & MP3_MASK) == (header2 & MP3_MASK))
            {
                break;
            } else if (ret == CHECK_SEEK_FAILED) {
                av_log(s, AV_LOG_ERROR, "Invalid frame size (%d): Could not seek to %"PRId64".\n", frame_size, off + i + frame_size);
                return AVERROR(EINVAL);
            }
        } else if (frame_size == CHECK_SEEK_FAILED) {

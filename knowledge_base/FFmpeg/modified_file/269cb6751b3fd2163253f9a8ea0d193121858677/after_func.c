    else                                    return -1;
#else
    /* only use the extension for safer guess */
    if (av_match_ext(p->filename, "ts"))
        return AVPROBE_SCORE_MAX;
    else
        return 0;
#endif
}

/* return the 90kHz PCR and the extension for the 27MHz PCR. return
   (-1) if not available */
static int parse_pcr(int64_t *ppcr_high, int *ppcr_low,
                     const uint8_t *packet)
{
    int afc, len, flags;
    const uint8_t *p;
    unsigned int v;

    afc = (packet[3] >> 4) & 3;
    if (afc <= 1)
        return -1;
    p = packet + 4;
    len = p[0];
    p++;
    if (len == 0)
        return -1;
    flags = *p++;
    len--;
    if (!(flags & 0x10))
        return -1;
    if (len < 6)
        return -1;
    v = AV_RB32(p);
    *ppcr_high = ((int64_t)v << 1) | (p[4] >> 7);
    *ppcr_low = ((p[4] & 1) << 8) | p[5];
    return 0;
}

static int mpegts_read_header(AVFormatContext *s)
{
    MpegTSContext *ts = s->priv_data;
    AVIOContext *pb = s->pb;
    uint8_t buf[5*1024];
    int len;
    int64_t pos;

    /* read the first 1024 bytes to get packet size */
    pos = avio_tell(pb);
    len = avio_read(pb, buf, sizeof(buf));
    if (len != sizeof(buf))
        goto fail;
    ts->raw_packet_size = get_packet_size(buf, sizeof(buf));
    if (ts->raw_packet_size <= 0)
        goto fail;
    ts->stream = s;
    ts->auto_guess = 0;

    if (s->iformat == &ff_mpegts_demuxer) {
        /* normal demux */

        /* first do a scan to get all the services */
        if (avio_seek(pb, pos, SEEK_SET) < 0 && pb->seekable)
            av_log(s, AV_LOG_ERROR, "Unable to seek back to the start\n");

        mpegts_open_section_filter(ts, SDT_PID, sdt_cb, ts, 1);

        mpegts_open_section_filter(ts, PAT_PID, pat_cb, ts, 1);

        handle_packets(ts, s->probesize / ts->raw_packet_size);
        /* if could not find service, enable auto_guess */

        ts->auto_guess = 1;

        av_dlog(ts->stream, "tuning done\n");

        s->ctx_flags |= AVFMTCTX_NOHEADER;
    } else {
        AVStream *st;
        int pcr_pid, pid, nb_packets, nb_pcrs, ret, pcr_l;
        int64_t pcrs[2], pcr_h;
        int packet_count[2];
        uint8_t packet[TS_PACKET_SIZE];

        /* only read packets */

        st = avformat_new_stream(s, NULL);
        if (!st)
            goto fail;
        avpriv_set_pts_info(st, 60, 1, 27000000);
        st->codec->codec_type = AVMEDIA_TYPE_DATA;

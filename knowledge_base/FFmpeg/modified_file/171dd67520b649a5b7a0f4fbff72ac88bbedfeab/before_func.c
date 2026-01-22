
    /* note: we need to modify the packet size here to handle the last
       packet */
    pkt->size = ret;
    return ret;
}

static int check(AVFormatContext *s, int64_t pos)
{
    int64_t ret = avio_seek(s->pb, pos, SEEK_SET);
    unsigned header;
    MPADecodeHeader sd;
    if (ret < 0)
        return ret;
    header = avio_rb32(s->pb);
    if (ff_mpa_check_header(header) < 0)
        return -1;
    if (avpriv_mpegaudio_decode_header(&sd, header) == 1)
        return -1;
    return sd.frame_size;
}

static int mp3_seek(AVFormatContext *s, int stream_index, int64_t timestamp,
                    int flags)
{
    MP3DecContext *mp3 = s->priv_data;
    AVIndexEntry *ie, ie1;
    AVStream *st = s->streams[0];
    int64_t ret  = av_index_search_timestamp(st, timestamp, flags);
    int i, j;
    int dir = (flags&AVSEEK_FLAG_BACKWARD) ? -1 : 1;

    if (mp3->is_cbr && st->duration > 0 && mp3->header_filesize > s->data_offset) {
        int64_t filesize = avio_size(s->pb);
        int64_t duration;
        if (filesize <= s->data_offset)
            filesize = mp3->header_filesize;
        filesize -= s->data_offset;
        duration = av_rescale(st->duration, filesize, mp3->header_filesize - s->data_offset);
        ie = &ie1;
        timestamp = av_clip64(timestamp, 0, duration);
        ie->timestamp = timestamp;
        ie->pos       = av_rescale(timestamp, filesize, duration) + s->data_offset;
    } else if (mp3->xing_toc) {
        if (ret < 0)
            return ret;

        ie = &st->index_entries[ret];
    } else {
        st->skip_samples = timestamp <= 0 ? mp3->start_pad + 528 + 1 : 0;

        return -1;
    }

    ret = avio_seek(s->pb, ie->pos, SEEK_SET);
    if (ret < 0)
        return ret;


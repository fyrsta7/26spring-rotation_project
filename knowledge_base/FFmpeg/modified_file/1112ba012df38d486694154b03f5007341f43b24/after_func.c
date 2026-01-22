
    num_bytes = (entries*field_size+4)>>3;

    buf = av_malloc(num_bytes+AV_INPUT_BUFFER_PADDING_SIZE);
    if (!buf) {
        av_freep(&sc->sample_sizes);
        return AVERROR(ENOMEM);
    }

    ret = ffio_read_size(pb, buf, num_bytes);
    if (ret < 0) {
        av_freep(&sc->sample_sizes);
        av_free(buf);
        return ret;
    }

    init_get_bits(&gb, buf, 8*num_bytes);

    for (i = 0; i < entries && !pb->eof_reached; i++) {
        sc->sample_sizes[i] = get_bits_long(&gb, field_size);
        sc->data_size += sc->sample_sizes[i];
    }

    sc->sample_count = i;

    av_free(buf);

    if (pb->eof_reached) {
        av_log(c->fc, AV_LOG_WARNING, "reached eof, corrupted STSZ atom\n");
        return AVERROR_EOF;
    }

    return 0;
}

static int mov_read_stts(MOVContext *c, AVIOContext *pb, MOVAtom atom)
{
    AVStream *st;
    MOVStreamContext *sc;
    unsigned int i, entries, alloc_size = 0;
    int64_t duration=0;
    int64_t total_sample_count=0;

    if (c->fc->nb_streams < 1)
        return 0;
    st = c->fc->streams[c->fc->nb_streams-1];
    sc = st->priv_data;

    avio_r8(pb); /* version */
    avio_rb24(pb); /* flags */
    entries = avio_rb32(pb);

    av_log(c->fc, AV_LOG_TRACE, "track[%u].stts.entries = %u\n",
            c->fc->nb_streams-1, entries);

    if (sc->stts_data)
        av_log(c->fc, AV_LOG_WARNING, "Duplicated STTS atom\n");
    av_freep(&sc->stts_data);
    sc->stts_count = 0;
    if (entries >= INT_MAX / sizeof(*sc->stts_data))
        return AVERROR(ENOMEM);

    for (i = 0; i < entries && !pb->eof_reached; i++) {
        int sample_duration;
        unsigned int sample_count;
        unsigned min_entries = FFMIN(FFMAX(i, 1024 * 1024), entries);
        MOVStts *stts_data = av_fast_realloc(sc->stts_data, &alloc_size,
                                             min_entries * sizeof(*sc->stts_data));
        if (!stts_data) {
            av_freep(&sc->stts_data);
            sc->stts_count = 0;
            return AVERROR(ENOMEM);
        }
        sc->stts_count = min_entries;
        sc->stts_data = stts_data;


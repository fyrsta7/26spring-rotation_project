
static int64_t dv_frame_offset(AVFormatContext *s, DVDemuxContext *c,
                               int64_t timestamp, int flags)
{
    // FIXME: sys may be wrong if last dv_read_packet() failed (buffer is junk)
    const int frame_size = c->sys->frame_size;
    int64_t offset;
    int64_t size       = avio_size(s->pb) - s->internal->data_offset;
    int64_t max_offset = ((size - 1) / frame_size) * frame_size;

    offset = frame_size * timestamp;

    if (size >= 0 && offset > max_offset)
        offset = max_offset;
    else if (offset < 0)
        offset = 0;

    return offset + s->internal->data_offset;
}

void ff_dv_offset_reset(DVDemuxContext *c, int64_t frame_offset)
{
    c->frames = frame_offset;
    if (c->ach) {
        if (c->sys) {
        c->abytes = av_rescale_q(c->frames, c->sys->time_base,
                                 (AVRational) { 8, c->ast[0]->codecpar->bit_rate });
        } else
            av_log(c->fctx, AV_LOG_ERROR, "cannot adjust audio bytes\n");

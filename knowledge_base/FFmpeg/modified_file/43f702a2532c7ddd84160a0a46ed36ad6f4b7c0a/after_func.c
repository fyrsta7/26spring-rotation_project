    av_assert0(frame);
    pts = av_rescale_q(frame->pts, fs->in[in].time_base, fs->time_base);
    frame->pts = pts;
    fs->in[in].frame_next = frame;
    fs->in[in].pts_next   = pts;
    fs->in[in].have_next  = 1;
}

static void framesync_inject_status(FFFrameSync *fs, unsigned in, int status, int64_t pts)
{
    av_assert0(!fs->in[in].have_next);
    pts = fs->in[in].state != STATE_RUN || fs->in[in].after == EXT_INFINITY
        ? INT64_MAX : framesync_pts_extrapolate(fs, in, fs->in[in].pts);
    fs->in[in].sync = 0;
    framesync_sync_level_update(fs);
    fs->in[in].frame_next = NULL;
    fs->in[in].pts_next   = pts;
    fs->in[in].have_next  = 1;
}

int ff_framesync_get_frame(FFFrameSync *fs, unsigned in, AVFrame **rframe,
                            unsigned get)
{
    AVFrame *frame;
    unsigned need_copy = 0, i;
    int64_t pts_next;

    if (!fs->in[in].frame) {
        *rframe = NULL;
        return 0;

        for (i = 0; i < nb_frames; i++)
            avg_hist[j] += (double)s->frames[i].histogram[j];
        avg_hist[j] /= nb_frames;
    }

    // find the frame closer to the average using the sum of squared errors
    for (i = 0; i < nb_frames; i++) {
        sq_err = frame_sum_square_err(s->frames[i].histogram, avg_hist);
        if (i == 0 || sq_err < min_sq_err)
            best_frame_idx = i, min_sq_err = sq_err;
    }

    // free and reset everything (except the best frame buffer)
    for (i = 0; i < nb_frames; i++) {
        memset(s->frames[i].histogram, 0, sizeof(s->frames[i].histogram));
        if (i != best_frame_idx)
            av_frame_free(&s->frames[i].buf);
    }
    s->n = 0;

    // raise the chosen one
    picref = s->frames[best_frame_idx].buf;
    av_log(ctx, AV_LOG_INFO, "frame id #%d (pts_time=%f) selected "
           "from a set of %d images\n", best_frame_idx,
           picref->pts * av_q2d(s->tb), nb_frames);
    s->frames[best_frame_idx].buf = NULL;

    return picref;
}

static int do_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ThumbContext *s = ctx->priv;
    AVFrame *frame = arg;
    int *hist = s->thread_histogram + HIST_SIZE * jobnr;
    const int h = frame->height;
    const int w = frame->width;
    const int slice_start = (h * jobnr) / nb_jobs;
    const int slice_end = (h * (jobnr+1)) / nb_jobs;
    const uint8_t *p = frame->data[0] + slice_start * frame->linesize[0];

    memset(hist, 0, sizeof(*hist) * HIST_SIZE);

    switch (frame->format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        for (int j = slice_start; j < slice_end; j++) {
            for (int i = 0; i < w; i++) {
                hist[0*256 + p[i*3    ]]++;
                hist[1*256 + p[i*3 + 1]]++;
                hist[2*256 + p[i*3 + 2]]++;
            }
            p += frame->linesize[0];
        }
        break;
    case AV_PIX_FMT_RGB0:
    case AV_PIX_FMT_BGR0:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGRA:
        for (int j = slice_start; j < slice_end; j++) {
            for (int i = 0; i < w; i++) {
                hist[0*256 + p[i*4    ]]++;
                hist[1*256 + p[i*4 + 1]]++;
                hist[2*256 + p[i*4 + 2]]++;
            }
            p += frame->linesize[0];
        }
        break;

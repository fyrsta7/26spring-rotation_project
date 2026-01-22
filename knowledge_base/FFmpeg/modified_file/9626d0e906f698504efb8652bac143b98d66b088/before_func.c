        return NULL;

    frame->extended_data = NULL;
    get_frame_defaults(frame);

    return frame;
}

void av_frame_free(AVFrame **frame)
{
    if (!frame || !*frame)
        return;

    av_frame_unref(*frame);
    av_freep(frame);
}

static int get_video_buffer(AVFrame *frame, int align)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);
    int ret, i;

    if (!desc)
        return AVERROR(EINVAL);

    if ((ret = av_image_check_size(frame->width, frame->height, 0, NULL)) < 0)
        return ret;

    if (!frame->linesize[0]) {
        ret = av_image_fill_linesizes(frame->linesize, frame->format,
                                      frame->width);
        if (ret < 0)
            return ret;

        for (i = 0; i < 4 && frame->linesize[i]; i++)
            frame->linesize[i] = FFALIGN(frame->linesize[i], align);
    }

    for (i = 0; i < 4 && frame->linesize[i]; i++) {
        int h = FFALIGN(frame->height, 32);
        if (i == 1 || i == 2)
            h = FF_CEIL_RSHIFT(h, desc->log2_chroma_h);

        frame->buf[i] = av_buffer_alloc(frame->linesize[i] * h + 16);
        if (!frame->buf[i])
            goto fail;


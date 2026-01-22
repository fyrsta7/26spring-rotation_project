{
    AVFilterContext *ctx = inlink->dst;
    CropDetectContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    av_image_fill_max_pixsteps(s->max_pixsteps, NULL, desc);

    if (s->limit < 1.0)
        s->limit *= (1 << desc->comp[0].depth) - 1;

    s->x1 = inlink->w - 1;
    s->y1 = inlink->h - 1;
    s->x2 = 0;
    s->y2 = 0;

    return 0;
}

#define SET_META(key, value) \
    av_dict_set_int(metadata, key, value, 0)

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    CropDetectContext *s = ctx->priv;
    int bpp = s->max_pixsteps[0];
    int w, h, x, y, shrink_by;
    AVDictionary **metadata;
    int outliers, last_y;
    int limit = round(s->limit);

    // ignore first 2 frames - they may be empty
    if (++s->frame_nb > 0) {
        metadata = avpriv_frame_get_metadatap(frame);

        // Reset the crop area every reset_count frames, if reset_count is > 0
        if (s->reset_count > 0 && s->frame_nb > s->reset_count) {
            s->x1 = frame->width  - 1;
            s->y1 = frame->height - 1;
            s->x2 = 0;
            s->y2 = 0;
            s->frame_nb = 1;
        }

#define FIND(DST, FROM, NOEND, INC, STEP0, STEP1, LEN) \
        outliers = 0;\
        for (last_y = y = FROM; NOEND; y = y INC) {\
            if (checkline(ctx, frame->data[0] + STEP0 * y, STEP1, LEN, bpp) > limit) {\
                if (++outliers > s->max_outliers) { \
                    DST = last_y;\
                    break;\
                }\
            } else\
                last_y = y INC;\
        }

        FIND(s->y1,                 0,               y < s->y1, +1, frame->linesize[0], bpp, frame->width);
        FIND(s->y2, frame->height - 1, y > FFMAX(s->y2, s->y1), -1, frame->linesize[0], bpp, frame->width);
        FIND(s->x1,                 0,               y < s->x1, +1, bpp, frame->linesize[0], frame->height);
        FIND(s->x2,  frame->width - 1, y > FFMAX(s->x2, s->x1), -1, bpp, frame->linesize[0], frame->height);


        // round x and y (up), important for yuv colorspaces
        // make sure they stay rounded!
        x = (s->x1+1) & ~1;
        y = (s->y1+1) & ~1;

        w = s->x2 - x + 1;
        h = s->y2 - y + 1;

        // w and h must be divisible by 2 as well because of yuv
        // colorspace problems.
        if (s->round <= 1)
            s->round = 16;
        if (s->round % 2)
            s->round *= 2;

        shrink_by = w % s->round;
        w -= shrink_by;
        x += (shrink_by/2 + 1) & ~1;

        shrink_by = h % s->round;

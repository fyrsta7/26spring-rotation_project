static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    ColorLevelsContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    const int step = s->step;
    AVFrame *out;
    int x, y, i;

    if (av_frame_is_writable(in)) {
        out = in;
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
    }

    switch (s->bpp) {
    case 1:
        for (i = 0; i < s->nb_comp; i++) {
            Range *r = &s->range[i];
            const uint8_t offset = s->rgba_map[i];
            const uint8_t *srcrow = in->data[0];
            uint8_t *dstrow = out->data[0];
            int imin = round(r->in_min  * UINT8_MAX);
            int imax = round(r->in_max  * UINT8_MAX);
            int omin = round(r->out_min * UINT8_MAX);
            int omax = round(r->out_max * UINT8_MAX);
            double coeff;

            if (imin < 0) {
                imin = UINT8_MAX;
                for (y = 0; y < inlink->h; y++) {
                    const uint8_t *src = srcrow;

                    for (x = 0; x < s->linesize; x += step)
                        imin = FFMIN(imin, src[x + offset]);
                    srcrow += in->linesize[0];
                }
            }
            if (imax < 0) {
                srcrow = in->data[0];
                imax = 0;
                for (y = 0; y < inlink->h; y++) {
                    const uint8_t *src = srcrow;

                    for (x = 0; x < s->linesize; x += step)
                        imax = FFMAX(imax, src[x + offset]);
                    srcrow += in->linesize[0];
                }
            }

            srcrow = in->data[0];
            coeff = (omax - omin) / (double)(imax - imin);
            for (y = 0; y < inlink->h; y++) {
                const uint8_t *src = srcrow;
                uint8_t *dst = dstrow;

                for (x = 0; x < s->linesize; x += step)
                    dst[x + offset] = av_clip_uint8((src[x + offset] - imin) * coeff + omin);
                dstrow += out->linesize[0];
                srcrow += in->linesize[0];
            }
        }
        break;
    case 2:
        for (i = 0; i < s->nb_comp; i++) {
            Range *r = &s->range[i];
            const uint8_t offset = s->rgba_map[i];
            const uint8_t *srcrow = in->data[0];
            uint8_t *dstrow = out->data[0];
            int imin = round(r->in_min  * UINT16_MAX);
            int imax = round(r->in_max  * UINT16_MAX);
            int omin = round(r->out_min * UINT16_MAX);
            int omax = round(r->out_max * UINT16_MAX);
            double coeff;

            if (imin < 0) {
                imin = UINT16_MAX;
                for (y = 0; y < inlink->h; y++) {
                    const uint16_t *src = (const uint16_t *)srcrow;

                    for (x = 0; x < s->linesize; x += step)
                        imin = FFMIN(imin, src[x + offset]);
                    srcrow += in->linesize[0];
                }
            }
            if (imax < 0) {
                srcrow = in->data[0];
                imax = 0;
                for (y = 0; y < inlink->h; y++) {
                    const uint16_t *src = (const uint16_t *)srcrow;

                    for (x = 0; x < s->linesize; x += step)
                        imax = FFMAX(imax, src[x + offset]);
                    srcrow += in->linesize[0];
                }
            }

            srcrow = in->data[0];
            coeff = (omax - omin) / (double)(imax - imin);
            for (y = 0; y < inlink->h; y++) {
                const uint16_t *src = (const uint16_t*)srcrow;
                uint16_t *dst = (uint16_t *)dstrow;

                for (x = 0; x < s->linesize; x += step)
                    dst[x + offset] = av_clip_uint16((src[x + offset] - imin) * coeff + omin);
                dstrow += out->linesize[0];
                srcrow += in->linesize[0];
            }
        }
    }

    if (in != out)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}
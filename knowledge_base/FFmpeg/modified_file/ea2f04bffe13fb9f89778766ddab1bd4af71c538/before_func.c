{
    AVFilterContext *ctx = outlink->src;
    HistogramContext *h = ctx->priv;
    int ncomp = 0, i;

    for (i = 0; i < h->ncomp; i++) {
        if ((1 << i) & h->components)
            ncomp++;
    }
    outlink->w = h->histogram_size;
    outlink->h = (h->level_height + h->scale_height) * FFMAX(ncomp * h->display_mode, 1);

    h->odesc = av_pix_fmt_desc_get(outlink->format);
    outlink->sample_aspect_ratio = (AVRational){1,1};

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    HistogramContext *h   = inlink->dst->priv;
    AVFilterContext *ctx  = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;
    int i, j, k, l, m;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    out->pts = in->pts;

    for (k = 0; k < 4 && out->data[k]; k++) {
        const int is_chroma = (k == 1 || k == 2);
        const int dst_h = FF_CEIL_RSHIFT(outlink->h, (is_chroma ? h->odesc->log2_chroma_h : 0));
        const int dst_w = FF_CEIL_RSHIFT(outlink->w, (is_chroma ? h->odesc->log2_chroma_w : 0));

        if (h->histogram_size <= 256) {
            for (i = 0; i < dst_h ; i++)
                memset(out->data[h->odesc->comp[k].plane] +
                       i * out->linesize[h->odesc->comp[k].plane],
                       h->bg_color[k], dst_w);
        } else {
            const int mult = h->mult;

            for (i = 0; i < dst_h ; i++)
                for (j = 0; j < dst_w; j++)
                    AV_WN16(out->data[h->odesc->comp[k].plane] +
                        i * out->linesize[h->odesc->comp[k].plane] + j * 2,
                        h->bg_color[k] * mult);
        }
    }

    for (m = 0, k = 0; k < h->ncomp; k++) {
        const int p = h->desc->comp[k].plane;
        const int height = h->planeheight[p];
        const int width = h->planewidth[p];
        double max_hval_log;
        unsigned max_hval = 0;
        int start;

        if (!((1 << k) & h->components))
            continue;
        start = m++ * (h->level_height + h->scale_height) * h->display_mode;

        if (h->histogram_size <= 256) {
            for (i = 0; i < height; i++) {
                const uint8_t *src = in->data[p] + i * in->linesize[p];
                for (j = 0; j < width; j++)
                    h->histogram[src[j]]++;
            }
        } else {
            for (i = 0; i < height; i++) {
                const uint16_t *src = (const uint16_t *)(in->data[p] + i * in->linesize[p]);
                for (j = 0; j < width; j++)
                    h->histogram[src[j]]++;
            }
        }

        for (i = 0; i < h->histogram_size; i++)
            max_hval = FFMAX(max_hval, h->histogram[i]);
        max_hval_log = log2(max_hval + 1);

        for (i = 0; i < outlink->w; i++) {
            int col_height;

            if (h->levels_mode)
                col_height = round(h->level_height * (1. - (log2(h->histogram[i] + 1) / max_hval_log)));
            else
                col_height = h->level_height - (h->histogram[i] * (int64_t)h->level_height + max_hval - 1) / max_hval;

            if (h->histogram_size <= 256) {
                for (j = h->level_height - 1; j >= col_height; j--) {
                    if (h->display_mode) {
                        for (l = 0; l < h->ncomp; l++)
                            out->data[l][(j + start) * out->linesize[l] + i] = h->fg_color[l];
                    } else {
                        out->data[p][(j + start) * out->linesize[p] + i] = 255;
                    }
                }
                for (j = h->level_height + h->scale_height - 1; j >= h->level_height; j--)
                    out->data[p][(j + start) * out->linesize[p] + i] = i;
            } else {
                const int mult = h->mult;

                for (j = h->level_height - 1; j >= col_height; j--) {

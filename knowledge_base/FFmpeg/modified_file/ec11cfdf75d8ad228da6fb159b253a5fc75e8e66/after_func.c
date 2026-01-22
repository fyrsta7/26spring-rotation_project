    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold int init(AVFilterContext *ctx)
{
    DelogoContext *s = ctx->priv;

#define CHECK_UNSET_OPT(opt)                                            \
    if (s->opt == -1) {                                            \
        av_log(s, AV_LOG_ERROR, "Option %s was not set.\n", #opt); \
        return AVERROR(EINVAL);                                         \
    }
    CHECK_UNSET_OPT(x);
    CHECK_UNSET_OPT(y);
    CHECK_UNSET_OPT(w);
    CHECK_UNSET_OPT(h);

#if LIBAVFILTER_VERSION_MAJOR < 7
    if (s->band == 0) { /* Unset, use default */
        av_log(ctx, AV_LOG_WARNING, "Note: default band value was changed from 4 to 1.\n");
        s->band = 1;
    } else if (s->band != 1) {
        av_log(ctx, AV_LOG_WARNING, "Option band is deprecated.\n");
    }
#else
    s->band = 1;
#endif
    av_log(ctx, AV_LOG_VERBOSE, "x:%d y:%d, w:%d h:%d band:%d show:%d\n",
           s->x, s->y, s->w, s->h, s->band, s->show);

    s->w += s->band*2;
    s->h += s->band*2;
    s->x -= s->band;
    s->y -= s->band;

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    DelogoContext *s = inlink->dst->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFrame *out;
    int hsub0 = desc->log2_chroma_w;
    int vsub0 = desc->log2_chroma_h;
    int direct = 0;

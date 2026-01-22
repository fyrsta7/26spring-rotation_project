        AV_PIX_FMT_YUV420P12,
        AV_PIX_FMT_YUV422P12,
        AV_PIX_FMT_YUV444P12,
        AV_PIX_FMT_YUV420P14,
        AV_PIX_FMT_YUV422P14,
        AV_PIX_FMT_YUV444P14,
        AV_PIX_FMT_YUV420P16,
        AV_PIX_FMT_YUV422P16,
        AV_PIX_FMT_YUV444P16,
        AV_PIX_FMT_YUVA420P,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold int init(AVFilterContext *ctx)
{

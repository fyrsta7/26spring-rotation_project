        if (depth != desc->comp[0].depth_minus1 ||
            be    != (desc->flags & AV_PIX_FMT_FLAG_BE)) {
            return AVERROR(EAGAIN);
        }
    }

    if (depth == 7)
        out_pixfmts = out8_pixfmts;
    else if (be)
        out_pixfmts = out16be_pixfmts;
    else
        out_pixfmts = out16le_pixfmts;

    for (i = 0; i < ctx->nb_outputs; i++)
        ff_formats_ref(ff_make_format_list(out_pixfmts), &ctx->outputs[i]->in_formats);
    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    ExtractPlanesContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    int plane_avail, ret, i;
    uint8_t rgba_map[4];

    plane_avail = ((desc->flags & AV_PIX_FMT_FLAG_RGB) ? PLANE_R|PLANE_G|PLANE_B :
                                                 PLANE_Y |
                                ((desc->nb_components > 2) ? PLANE_U|PLANE_V : 0)) |
                  ((desc->flags & AV_PIX_FMT_FLAG_ALPHA) ? PLANE_A : 0);
    if (s->requested_planes & ~plane_avail) {
